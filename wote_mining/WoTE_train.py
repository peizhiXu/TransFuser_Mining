"""Training graph for the HD465 WoTE adaptation.

The orchestration follows the authors' ``AgentLightningModule`` and
``WoTE_loss.py`` (Apache-2.0), but remains plain PyTorch so it fits this
TransFuser checkout without adding NAVSIM/Hydra/Lightning dependencies.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from wote_mining.WoTE_loss import source_style_core_losses
from wote_mining.WoTE_model import METRIC_NAMES, current_semantic_map_loss


class WoTEMiningTrainingModule(nn.Module):
    """Prepare mining batches and combine the source-style WoTE objectives."""

    def __init__(self, planner, config):
        super().__init__()
        self.planner = planner
        self.config = config

    def _future_map_indices(self, future_poses):
        batch = future_poses.shape[0]
        count = self.planner.trajectory_head.anchors.shape[0]
        sampled = int(self.config.wote_num_future_map_candidates)
        if sampled < 1 or sampled > self.planner.world_model.candidate_chunk:
            raise ValueError(
                "wote_num_future_map_candidates must be between 1 and candidate_chunk"
            )
        if self.training:
            return torch.randint(count, (batch, sampled), device=future_poses.device)
        # Stable validation: supervise the nearest fixed anchor first, then
        # deterministic neighboring IDs if more than one map is requested.
        anchors = self.planner.trajectory_head.anchors.to(future_poses)
        distance = torch.linalg.vector_norm(
            (anchors[None] - future_poses[:, None]).flatten(2), dim=-1
        )
        nearest = distance.argmin(dim=1, keepdim=True)
        offsets = torch.arange(sampled, device=future_poses.device)[None]
        return (nearest + offsets) % count

    def forward(self, batch):
        rgb = batch["rgb"].float()
        lidar = batch["lidar"].float()
        if self.config.use_target_point_image:
            lidar = torch.cat((lidar, batch["target_point_image"].float()), dim=1)
        speed = batch["speed"].float().reshape(-1, 1)
        target_point = batch["target_point"].float()
        future_poses = batch["wote_future_poses"].float()
        future_map_indices = self._future_map_indices(future_poses)
        return self.planner(
            rgb, lidar, speed, target_point,
            augmentation_degrees=batch["wote_augmentation_degrees"].float(),
            predict_future_map=True,
            future_map_candidate_indices=future_map_indices,
            # Both train and validation targets were generated for the fixed
            # 256 anchors.  Refined trajectories are used only by the online
            # CARLA agent, matching the source WoTE train/eval split.
            use_refined_world=False,
        )

    def compute_losses(self, batch, outputs):
        core = source_style_core_losses(
            outputs,
            batch["wote_future_poses"].float(),
            batch["wote_metric_targets"].float(),
            batch["wote_metric_valid"].float(),
        )
        core.pop("matched_anchor")
        raw = dict(core)
        raw["loss_current_map"] = current_semantic_map_loss(
            outputs["current_map_logits"],
            batch["wote_current_scene"].float(),
            batch["wote_current_valid"].float(),
        )
        raw.update(self.planner.current_agent_head.matching_loss(
            outputs,
            batch["wote_current_agent_boxes"].float(),
            batch["wote_current_agent_mask"],
        ))
        raw["loss_future_map"] = self.planner.world_model.future_map_loss(
            outputs,
            batch["wote_future_scene"].float(),
            batch["wote_future_valid"].float(),
            augmentation_degrees=batch["wote_augmentation_degrees"].float(),
        )
        weights = {
            "loss_traj_offset": self.config.wote_traj_offset_loss_weight,
            "loss_offset_imitation": self.config.wote_offset_imitation_loss_weight,
            "loss_imitation_reward": self.config.wote_imitation_reward_loss_weight,
            "loss_metric_reward": self.config.wote_metric_reward_loss_weight,
            "loss_current_map": self.config.wote_current_map_loss_weight,
            "loss_future_map": self.config.wote_future_map_loss_weight,
            "loss_current_agent_presence": self.config.wote_agent_presence_loss_weight,
            "loss_current_agent_box": self.config.wote_agent_box_loss_weight,
        }
        weighted = {}
        for name, value in raw.items():
            weighted[name] = value * float(weights[name])
        weighted["loss_total"] = sum(weighted.values())
        return weighted

    @torch.no_grad()
    def compute_diagnostics(self, batch, outputs):
        """Return detached training diagnostics which never enter loss_total."""
        logits = outputs["metric_logits"].float()
        targets = batch["wote_metric_targets"].to(logits).float()
        valid = batch["wote_metric_valid"].to(logits).float()
        if logits.shape != targets.shape or logits.shape[-1] != len(METRIC_NAMES):
            raise ValueError("metric diagnostic tensors have incompatible shapes")
        if valid.shape != targets.shape:
            raise ValueError("metric diagnostic validity has incompatible shape")

        probabilities = torch.sigmoid(logits)
        elementwise_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        elementwise_mae = (probabilities - targets).abs()
        diagnostics = {}
        diagnostic_weights = {}
        for metric_index, metric_name in enumerate(METRIC_NAMES):
            metric_valid = valid[..., metric_index]
            valid_count = metric_valid.sum()
            denominator = valid_count.clamp_min(1.0)
            prefix = "metric_%s" % metric_name
            diagnostics[prefix + "_bce"] = (
                elementwise_bce[..., metric_index] * metric_valid
            ).sum() / denominator
            diagnostics[prefix + "_mae"] = (
                elementwise_mae[..., metric_index] * metric_valid
            ).sum() / denominator
            diagnostics[prefix + "_pred_mean"] = (
                probabilities[..., metric_index] * metric_valid
            ).sum() / denominator
            diagnostics[prefix + "_target_mean"] = (
                targets[..., metric_index] * metric_valid
            ).sum() / denominator
            diagnostics[prefix + "_valid_fraction"] = metric_valid.mean()
            for suffix in ("_bce", "_mae", "_pred_mean", "_target_mean"):
                diagnostic_weights[prefix + suffix] = valid_count
            diagnostic_weights[prefix + "_valid_fraction"] = valid_count.new_tensor(
                metric_valid.numel()
            )

        future = batch["wote_future_poses"].to(outputs["trajectories"]).float()
        anchors = outputs["anchors"]
        batch_size, candidate_count = anchors.shape[:2]
        if future.shape != (batch_size, 8, 3):
            raise ValueError("future pose diagnostics require shape [B,8,3]")
        nearest = torch.linalg.vector_norm(
            (anchors - future[:, None]).reshape(batch_size, candidate_count, -1),
            dim=-1,
        ).argmin(dim=1)
        batch_index = torch.arange(batch_size, device=anchors.device)
        matched_trajectory = outputs["trajectories"][batch_index, nearest]

        # During training the reward labels correspond to fixed anchors. Use
        # the selected anchor ID to inspect the refined trajectory that would
        # be passed to the world model during online inference.
        selected_index = outputs["selected_index"]
        selected_trajectory = outputs["trajectories"][batch_index, selected_index]
        for prefix, trajectory in (
            ("traj_matched", matched_trajectory),
            ("traj_selected", selected_trajectory),
        ):
            displacement = torch.linalg.vector_norm(
                trajectory[..., :2] - future[..., :2], dim=-1
            )
            diagnostics[prefix + "_ade_m"] = displacement.mean()
            diagnostics[prefix + "_fde_m"] = displacement[:, -1].mean()
            diagnostic_weights[prefix + "_ade_m"] = future.new_tensor(batch_size)
            diagnostic_weights[prefix + "_fde_m"] = future.new_tensor(batch_size)
        return diagnostics, diagnostic_weights


def build_optimizer(module, config):
    """AdamW with the source WoTE 0.1x image-encoder learning rate."""
    image_parameters = []
    other_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if "planner.backbone.image_encoder" in name:
            image_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": other_parameters, "lr_scale": 1.0},
            {
                "params": image_parameters,
                "lr_scale": float(config.wote_image_encoder_lr_multiplier),
            },
        ],
        lr=float(config.wote_lr),
        weight_decay=float(config.wote_weight_decay),
    )


def set_warmup_cosine_lr(optimizer, epoch, config):
    """Set the source WoTE epoch-level warmup/cosine schedule."""
    warmup = int(config.wote_warmup_epochs)
    epochs = int(config.wote_max_epochs)
    base = float(config.wote_lr)
    minimum = float(config.wote_min_lr)
    if epoch < warmup:
        lr = base * float(epoch + 1) / max(warmup, 1)
    else:
        progress = float(epoch - warmup) / max(epochs - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        lr = minimum + 0.5 * (base - minimum) * (
            1.0 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = lr * float(group.get("lr_scale", 1.0))
    return lr

# Command-line training entry point. The HD465 TransFuser baseline remains
# unchanged; this entry supports one GPU directly and multi-GPU via torchrun.

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEAM_CODE = ROOT / "team_code_transfuser"
if str(TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE))

from team_code_transfuser.data import CARLA_Data
from team_code_transfuser.transfuser import TransfuserBackbone
from wote_mining.WoTE_config import WoTEMiningConfig
from wote_mining.WoTE_model import WoTEMiningPlanner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train WoTE adapted to the HD465 mining TransFuser dataset"
    )
    parser.add_argument(
        "--root-dir", required=True,
        help=("HD465 raw dataset root, or a prepared root containing "
              "train/ and val/"),
    )
    parser.add_argument(
        "--anchors", default=str(
            ROOT / "wote_mining/assets/anchors/trajectory_anchors_256.npy"
        ),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "log/wote-mining"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--image-architecture", default="regnety_032")
    parser.add_argument("--lidar-architecture", default="regnety_032")
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--use-velocity", action="store_true")
    parser.add_argument("--no-target-point-image", action="store_true")
    parser.add_argument("--future-map-candidates", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument(
        "--save-every", type=int, default=5,
        help=("save a numbered checkpoint every N epochs and at the final "
              "epoch; 0 disables periodic numbered checkpoints"),
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--tensorboard", action="store_true",
        help="write all train/validation metrics for TensorBoard",
    )
    parser.add_argument(
        "--tensorboard-dir", default=None,
        help="TensorBoard log directory (default: OUTPUT_DIR/tensorboard)",
    )
    parser.add_argument(
        "--amp", action="store_true", help="enable CUDA automatic mixed precision"
    )
    return parser.parse_args()


def seed_everything(seed, rank=0):
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def seed_worker(worker_id):
    value = torch.initial_seed() % (2 ** 32)
    random.seed(value)
    np.random.seed(value)


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("multi-process WoTE training requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device(
        "cuda:%d" % local_rank if torch.cuda.is_available() else "cpu"
    )
    return distributed, rank, local_rank, world_size, device


def route_dirs_from_manifest(raw_root, manifest_path):
    """Resolve one cache split against raw data without copying the data."""
    raw_root = Path(raw_root)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested = set(manifest["routes"])
    resolved = {}
    for lidar_dir in raw_root.rglob("lidar"):
        route_dir = lidar_dir.parent
        if route_dir.name in requested:
            if route_dir.name in resolved:
                raise ValueError("duplicate raw route name: %s" % route_dir.name)
            resolved[route_dir.name] = route_dir
    missing = sorted(requested.difference(resolved))
    if missing:
        raise FileNotFoundError(
            "%d manifest routes are missing below %s; first: %s"
            % (len(missing), raw_root, missing[0])
        )
    return [str(resolved[name]) for name in sorted(requested)]


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def reduce_loss_sums(loss_sums, batches, device, distributed):
    names = sorted(loss_sums)
    values = torch.tensor(
        [loss_sums[name] for name in names] + [float(batches)],
        dtype=torch.float64, device=device,
    )
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    denominator = max(values[-1].item(), 1.0)
    return {name: values[index].item() / denominator for index, name in enumerate(names)}


def reduce_weighted_sums(value_sums, weight_sums, device, distributed):
    """Reduce detached diagnostic numerators and denominators across ranks."""
    names = sorted(value_sums)
    if set(names) != set(weight_sums):
        raise ValueError("diagnostic values and weights must have the same names")
    values = torch.tensor(
        [value_sums[name] for name in names]
        + [weight_sums[name] for name in names],
        dtype=torch.float64, device=device,
    )
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    split = len(names)
    result = {}
    for index, name in enumerate(names):
        denominator = values[split + index].item()
        result[name] = (
            values[index].item() / denominator if denominator > 0.0 else 0.0
        )
    return result


def run_epoch(model, loader, optimizer, scaler, device, config, epoch,
              training, distributed, rank, max_batches, grad_clip):
    model.train(training)
    loss_sums = {}
    diagnostic_sums = {}
    diagnostic_weight_sums = {}
    batches = 0
    progress_total = (
        min(len(loader), max_batches) if max_batches else len(loader)
    )
    iterator = tqdm(
        loader, total=progress_total, disable=rank != 0,
        desc=("train" if training else "val"),
    )
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in enumerate(iterator):
            if max_batches and batch_index >= max_batches:
                break
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=bool(config.wote_amp and device.type == "cuda")
            ):
                outputs = model(batch)
                module = model.module if isinstance(model, DistributedDataParallel) else model
                losses = module.compute_losses(batch, outputs)
                total = losses["loss_total"]
            diagnostics, diagnostic_weights = module.compute_diagnostics(batch, outputs)
            if training:
                scaler.scale(total).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            batches += 1
            if set(losses).intersection(diagnostics):
                raise RuntimeError("loss and diagnostic names must be disjoint")
            for name, value in losses.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach())
            for name, value in diagnostics.items():
                weight = float(diagnostic_weights[name].detach())
                diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + (
                    float(value.detach()) * weight
                )
                diagnostic_weight_sums[name] = (
                    diagnostic_weight_sums.get(name, 0.0) + weight
                )
            if rank == 0:
                iterator.set_postfix(
                    loss="%.4f" % float(total.detach()),
                    selected_ADE="%.2fm" % float(
                        diagnostics["traj_selected_ade_m"].detach()
                    ),
                )
    epoch_results = reduce_loss_sums(
        loss_sums, batches, device, distributed
    )
    epoch_results.update(reduce_weighted_sums(
        diagnostic_sums, diagnostic_weight_sums, device, distributed
    ))
    return epoch_results


def save_checkpoint(path, model, optimizer, scaler, epoch, args, config,
                    best_val_loss):
    module = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save({
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
        "config": {
            "reward_weights": tuple(config.wote_reward_weights),
            "loss_weights": {
                name: getattr(config, name) for name in (
                    "wote_traj_offset_loss_weight",
                    "wote_offset_imitation_loss_weight",
                    "wote_imitation_reward_loss_weight",
                    "wote_metric_reward_loss_weight",
                    "wote_current_map_loss_weight",
                    "wote_future_map_loss_weight",
                    "wote_agent_presence_loss_weight",
                    "wote_agent_box_loss_weight",
                )
            },
        },
    }, path)


def write_tensorboard_epoch(writer, epoch, learning_rate, train_metrics,
                            val_metrics=None):
    """Write one compact epoch record while retaining every scalar metric."""
    step = int(epoch) + 1
    writer.add_scalar("optimizer/learning_rate", float(learning_rate), step)
    for name, value in train_metrics.items():
        writer.add_scalar("train/%s" % name, float(value), step)
    if val_metrics is not None:
        for name, value in val_metrics.items():
            writer.add_scalar("val/%s" % name, float(value), step)
    writer.flush()


def backfill_tensorboard(writer, metrics_path, before_epoch):
    """Import completed JSONL epochs when TensorBoard is enabled on resume."""
    if not metrics_path.is_file():
        return
    with metrics_path.open("r") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            epoch = int(record["epoch"])
            if epoch >= before_epoch:
                continue
            write_tensorboard_epoch(
                writer, epoch, record["lr"], record["train"], record.get("val")
            )


def main():
    args = parse_args()
    if args.save_every < 0:
        raise ValueError("--save-every must be non-negative")
    distributed, rank, local_rank, world_size, device = distributed_context()
    seed_everything(args.seed, rank)

    prepared = (
        (Path(args.root_dir) / "train").is_dir()
        and (Path(args.root_dir) / "val").is_dir()
    )
    config = WoTEMiningConfig(
        root_dir=args.root_dir, setting=("mining" if prepared else "eval")
    )
    if not prepared:
        cache_root = Path(config.wote_metric_cache_dir)
        config.train_data = route_dirs_from_manifest(
            args.root_dir, cache_root / "train/manifest.json"
        )
        config.val_data = route_dirs_from_manifest(
            args.root_dir, cache_root / "val/manifest.json"
        )
    config.backbone = "transFuser"
    config.multitask = False
    config.use_point_pillars = False
    config.use_target_point_image = not args.no_target_point_image
    config.n_layer = args.transformer_layers
    config.wote_num_future_map_candidates = args.future_map_candidates
    config.wote_lr = args.lr
    config.wote_min_lr = args.min_lr
    config.wote_weight_decay = args.weight_decay
    config.wote_warmup_epochs = args.warmup_epochs
    config.wote_max_epochs = args.epochs
    config.wote_amp = args.amp

    if not Path(args.anchors).is_file():
        raise FileNotFoundError("trajectory anchors not found: %s" % args.anchors)
    if config.augment:
        raise ValueError("cached WoTE metric labels require config.augment=False")

    backbone = TransfuserBackbone(
        config,
        image_architecture=args.image_architecture,
        lidar_architecture=args.lidar_architecture,
        use_velocity=args.use_velocity,
    )
    planner = WoTEMiningPlanner(backbone, args.anchors)
    model = WoTEMiningTrainingModule(planner, config).to(device)
    optimizer = build_optimizer(model, config)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))

    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False,
        )

    train_set = CARLA_Data(root=config.train_data, config=config)
    val_set = CARLA_Data(root=config.val_data, config=config)
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True
    ) if distributed else None
    val_sampler = DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=False
    ) if distributed else None
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(args.workers > 0),
    )
    train_loader = DataLoader(
        train_set, sampler=train_sampler, shuffle=train_sampler is None,
        **loader_options
    )
    val_loader = DataLoader(
        val_set, sampler=val_sampler, shuffle=False, **loader_options
    )

    output_dir = Path(args.output_dir)
    writer = None
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "args.json").open("w") as stream:
            json.dump(vars(args), stream, indent=2, ensure_ascii=False)
        if args.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError(
                    "TensorBoard is not installed; run: pip install tensorboard"
                ) from error
            tensorboard_dir = (
                Path(args.tensorboard_dir)
                if args.tensorboard_dir
                else output_dir / "tensorboard"
            )
            writer = SummaryWriter(
                log_dir=str(tensorboard_dir), purge_step=start_epoch + 1
            )
            backfill_tensorboard(
                writer, output_dir / "metrics.jsonl", start_epoch
            )
            print("tensorboard_logdir=%s" % tensorboard_dir)
        print("device=%s world_size=%d train=%d val=%d" % (
            device, world_size, len(train_set), len(val_set)
        ))

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        learning_rate = set_warmup_cosine_lr(optimizer, epoch, config)
        train_losses = run_epoch(
            model, train_loader, optimizer, scaler, device, config, epoch,
            True, distributed, rank, args.max_train_batches, args.grad_clip,
        )
        val_losses = None
        if args.val_every > 0 and (epoch + 1) % args.val_every == 0:
            val_losses = run_epoch(
                model, val_loader, optimizer, scaler, device, config, epoch,
                False, distributed, rank, args.max_val_batches, args.grad_clip,
            )
        if rank == 0:
            is_best = (
                val_losses is not None
                and val_losses["loss_total"] < best_val_loss
            )
            if is_best:
                best_val_loss = val_losses["loss_total"]
            record = {
                "epoch": epoch, "lr": learning_rate,
                "train": train_losses, "val": val_losses,
            }
            summary = (
                "epoch=%03d/%03d lr=%.3e train_loss=%.4f "
                "train_selected_ADE=%.2fm"
                % (
                    epoch + 1, args.epochs, learning_rate,
                    train_losses["loss_total"],
                    train_losses["traj_selected_ade_m"],
                )
            )
            if val_losses is not None:
                summary += (
                    " val_loss=%.4f val_selected_ADE=%.2fm "
                    "val_selected_FDE=%.2fm best_val=%.4f%s"
                    % (
                        val_losses["loss_total"],
                        val_losses["traj_selected_ade_m"],
                        val_losses["traj_selected_fde_m"],
                        best_val_loss,
                        " new_best" if is_best else "",
                    )
                )
            print(summary, flush=True)
            with (output_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if writer is not None:
                write_tensorboard_epoch(
                    writer, epoch, learning_rate, train_losses, val_losses
                )
            save_checkpoint(
                output_dir / "latest.pth", model, optimizer, scaler,
                epoch, args, config, best_val_loss,
            )
            periodic = args.save_every > 0 and (epoch + 1) % args.save_every == 0
            final_epoch = (epoch + 1) == args.epochs
            if periodic or final_epoch:
                save_checkpoint(
                    output_dir / ("checkpoint_%03d.pth" % (epoch + 1)),
                    model, optimizer, scaler, epoch, args, config, best_val_loss,
                )
            if is_best:
                save_checkpoint(
                    output_dir / "best.pth", model, optimizer, scaler,
                    epoch, args, config, best_val_loss,
                )
        if distributed:
            dist.barrier()

    if writer is not None:
        writer.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
