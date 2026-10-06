"""Compare fixed-anchor validation selection with online trajectory scoring.

Read-only model evaluation, not CARLA simulation or PDMS. Supports the residual
AdaLN checkpoints before/after the stable-read auxiliary head was introduced.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUXILIARY_KEYS = {
    "trajectory_head.future_read_metric_head.weight",
    "trajectory_head.future_read_metric_head.bias",
}


def load_planner_weights(planner, training_state):
    """Permit only the unused auxiliary head missing from legacy checkpoints.

    Validate all names/shapes before loading; do not silently accept unrelated
    architecture differences or a partially saved auxiliary head.
    """
    if not training_state or any(not key.startswith("planner.") for key in training_state):
        raise ValueError("expected WoTE training state with planner.* keys")
    state = {key[len("planner."):]: value for key, value in training_state.items()}
    expected = planner.state_dict()
    missing = set(expected).difference(state)
    unexpected = set(state).difference(expected)
    if unexpected or missing not in (set(), AUXILIARY_KEYS):
        raise ValueError("incompatible checkpoint; missing=%s unexpected=%s" % (
            sorted(missing), sorted(unexpected)))
    mismatched = [key for key in state if state[key].shape != expected[key].shape]
    if mismatched:
        raise ValueError("checkpoint tensor shapes differ: %s" % sorted(mismatched))
    # The saved bank defines candidate IDs. Never silently substitute anchors.
    anchor_key = "trajectory_head.anchors"
    if anchor_key in expected and not torch.equal(state[anchor_key].cpu(), expected[anchor_key].cpu()):
        raise ValueError("anchor file differs from checkpoint's saved anchor bank")
    if missing:
        # Not called when predict_auxiliary=False. Deterministic zero tensors
        # keep strict loading possible without allowing any other missing key.
        state.update({key: torch.zeros_like(expected[key]) for key in missing})
    planner.load_state_dict(state, strict=True)
    return sorted(missing)


def trajectory_metrics(predicted, expert):
    """Per-sample geometry and the exact PID's learned desired-speed formula."""
    distances = torch.linalg.vector_norm(predicted[..., :2] - expert[..., :2], dim=-1)
    velocity = (predicted[:, 1, :2] - predicted[:, 0, :2]) * 2.0
    truth_velocity = (expert[:, 1, :2] - expert[:, 0, :2]) * 2.0
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    truth_speed = torch.linalg.vector_norm(truth_velocity, dim=-1)
    cross = velocity[:, 0] * truth_velocity[:, 1] - velocity[:, 1] * truth_velocity[:, 0]
    dot = (velocity * truth_velocity).sum(-1)
    direction_error = torch.atan2(cross, dot).abs() * (180.0 / np.pi)
    # Direction is undefined for either nearly stationary segment. Keep these
    # samples in ADE/speed metrics and explicitly count direction-valid ones.
    direction_valid = (speed > 0.1) & (truth_speed > 0.1)
    return {
        "ade_m": distances.mean(-1),
        "fde_m": distances[:, -1],
        "first_second_ade_m": distances[:, :2].mean(-1),
        "desired_speed_mps": speed,
        "desired_speed_abs_error_mps": (speed - truth_speed).abs(),
        "first_segment_velocity_error_mps": torch.linalg.vector_norm(velocity - truth_velocity, dim=-1),
        "first_segment_direction_error_deg": torch.where(
            direction_valid, direction_error, torch.full_like(direction_error, float("nan"))),
        "direction_valid": direction_valid,
    }


def compare_selections(fixed, online, expert):
    """Compare two selections of the SAME final decoded candidate bank.

    fixed.selected_trajectory is an anchor, so deliberately gather the decoded
    trajectory by fixed.selected_index instead (the TensorBoard proxy).
    """
    trajectories = online["trajectories"].float()
    expert = expert.float()
    if trajectories.ndim != 4 or trajectories.shape[2:] != (8, 3):
        raise ValueError("decoded trajectories must have shape [B,K,8,3]")
    if expert.shape != (trajectories.shape[0], 8, 3):
        raise ValueError("expert trajectories must have shape [B,8,3]")
    if not torch.isfinite(trajectories).all() or not torch.isfinite(expert).all():
        raise ValueError("nonfinite trajectory or expert")
    if not torch.allclose(fixed["trajectories"].float(), trajectories, rtol=1e-4, atol=1e-4):
        raise ValueError("decoded banks differ: both evaluation passes must be deterministic")
    rows = torch.arange(trajectories.shape[0], device=trajectories.device)
    proxy_id = fixed["selected_index"].long()
    online_id = online["selected_index"].long()
    for output in (fixed, online):
        if not torch.isfinite(output["final_rewards"]).all():
            raise ValueError("nonfinite predicted rewards")
        if not torch.equal(output["selected_index"], output["final_rewards"].argmax(-1)):
            raise ValueError("selection is not the full-bank reward argmax")
    chosen = trajectories[rows, online_id]
    if not torch.allclose(chosen, online["selected_trajectory"].float(), rtol=1e-4, atol=1e-4):
        raise ValueError("online selected trajectory does not match decoded bank")
    distances = torch.linalg.vector_norm(trajectories[..., :2] - expert[:, None, :, :2], dim=-1)
    ade, fde = distances.mean(-1), distances[..., -1]
    oracle_id = ade.argmin(-1)
    result = {
        "proxy_index": proxy_id,
        "online_index": online_id,
        "oracle_by_ade_index": oracle_id,
        "selection_agreement": proxy_id == online_id,
        "oracle_min_ade_m": ade[rows, oracle_id],
        "oracle_by_ade_fde_m": fde[rows, oracle_id],
        "oracle_min_fde_m": fde.min(-1).values,
        "online_selection_regret_ade_m": ade[rows, online_id] - ade[rows, oracle_id],
        "online_minus_proxy_ade_m": ade[rows, online_id] - ade[rows, proxy_id],
        "online_oracle_reward_rank": 1 + (
            online["final_rewards"] > online["final_rewards"][rows, oracle_id, None]
        ).sum(-1),
        "expert_first_segment_speed_mps": torch.linalg.vector_norm(
            (expert[:, 1, :2] - expert[:, 0, :2]) * 2.0, dim=-1),
    }
    topk_ids = online["final_rewards"].topk(min(4, trajectories.shape[1]), dim=-1).indices
    result["online_reward_top4_min_ade_m"] = ade.gather(1, topk_ids).min(-1).values
    for name, predicted in (("proxy", trajectories[rows, proxy_id]), ("online", chosen)):
        result.update({name + "_" + key: value for key, value in trajectory_metrics(predicted, expert).items()})
    return result


def summarize_records(records):
    """Sample-weighted means; undefined directions have explicit denominators."""
    if not records:
        raise ValueError("no evaluation samples")
    excluded = {"sample_index", "route", "frame", "proxy_index", "online_index", "oracle_by_ade_index"}
    means, counts = {}, {}
    for key in records[0]:
        if key in excluded:
            continue
        values = [row[key] for row in records if row[key] is not None]
        means[key] = float(np.mean(values)) if values else None
        counts[key] = len(values)
    return {"samples": len(records), "means": means, "valid_counts": counts}


def make_config(root_dir, saved_args, saved_config):
    from wote_mining.WoTE_config import WoTEMiningConfig
    from wote_mining.WoTE_train import route_dirs_from_manifest
    config = WoTEMiningConfig(setting="eval")
    val_root = Path(root_dir) / "val"
    if val_root.is_dir():
        config.val_data = [str(path) for path in sorted(val_root.iterdir()) if (path / "lidar").is_dir()]
        if not config.val_data:
            raise ValueError("no route folders with lidar found under %s" % val_root)
    else:
        config.val_data = route_dirs_from_manifest(
            root_dir, Path(config.wote_metric_cache_dir) / "val/manifest.json")
    config.backbone = "transFuser"
    config.multitask = False
    config.use_point_pillars = False
    config.augment = False
    config.use_target_point_image = not saved_args.get("no_target_point_image", False)
    config.n_layer = saved_args.get("transformer_layers", 4)
    config.wote_reward_weights = tuple(saved_config.get("reward_weights", config.wote_reward_weights))
    # Only sensor inputs and expert poses are needed; do not load unused map,
    # actor, cached reward labels, or dense-route supervision during evaluation.
    config.wote_future_scene_targets = False
    config.wote_metric_cache_dir = None
    config.wote_dense_routes_dir = None
    return config


def build_planner(config, saved_args, state, anchors_override=None):
    from team_code_transfuser import transfuser
    from wote_mining.WoTE_model import WoTEMiningPlanner
    anchors = Path(anchors_override or saved_args.get("anchors", ""))
    if not anchors.is_file() and anchors_override is None:
        anchors = ROOT / "wote_mining/assets/anchors/trajectory_anchors_256.npy"
    if not anchors.is_file():
        raise FileNotFoundError("anchor file not found: %s" % anchors)
    # Every backbone weight comes from the checkpoint. Disable pretrained
    # downloads only in this constructor, preserving the training defaults.
    create_model = transfuser.timm.create_model

    def without_pretraining(*args, **kwargs):
        kwargs["pretrained"] = False
        return create_model(*args, **kwargs)

    with patch.object(transfuser.timm, "create_model", side_effect=without_pretraining):
        backbone = transfuser.TransfuserBackbone(
            config, image_architecture=saved_args.get("image_architecture", "regnety_032"),
            lidar_architecture=saved_args.get("lidar_architecture", "regnety_032"),
            use_velocity=saved_args.get("use_velocity", False))
    planner = WoTEMiningPlanner(backbone, anchors)
    missing = load_planner_weights(planner, state)
    return planner, missing


@torch.no_grad()
def snapshot(planner, batch, device, use_fused_world, amp):
    rgb = batch["rgb"].to(device).float()
    lidar = batch["lidar"].to(device).float()
    if planner.backbone.config.use_target_point_image:
        lidar = torch.cat((lidar, batch["target_point_image"].to(device).float()), dim=1)
    speed = batch["speed"].to(device).float().reshape(-1, 1)
    with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
        output = planner(
            rgb, lidar, speed, batch["target_point"].to(device).float(),
            augmentation_degrees=speed.new_zeros(speed.shape[0]),
            predict_future_map=False, predict_auxiliary=False,
            use_fused_world=use_fused_world)
    # Do not retain a full B*256*64*256 world twice on GPU.
    return {key: output[key].detach().cpu() for key in
            ("trajectories", "selected_index", "selected_trajectory", "final_rewards")}


def evaluate(planner, loader, sample_metadata, device, amp):
    from tqdm import tqdm
    planner.eval()
    records = []
    cursor = 0
    for batch in tqdm(loader, desc="inference evaluation"):
        fixed = snapshot(planner, batch, device, use_fused_world=False, amp=amp)
        online = snapshot(planner, batch, device, use_fused_world=True, amp=amp)
        values = compare_selections(fixed, online, batch["wote_future_poses"])
        count = batch["rgb"].shape[0]
        for row_index in range(count):
            record = dict(sample_metadata[cursor + row_index])
            for key, tensor in values.items():
                value = tensor[row_index].item()
                record[key] = None if isinstance(value, float) and not np.isfinite(value) else value
            records.append(record)
        cursor += count
    if cursor != len(sample_metadata):
        raise ValueError("evaluation sample count changed")
    return records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH",
                        help="repeat to compare checkpoints on the same validation samples")
    parser.add_argument("--output-dir", required=True, help="new directory; existing results are never overwritten")
    parser.add_argument("--anchors", help="override anchor path; must exactly match the checkpoint bank")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true",
                        help="optional faster mixed precision; default FP32 matches the current CARLA agent")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-samples", type=int, help="debug only: first N sorted validation frames")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0 or (args.max_samples is not None and args.max_samples < 1):
        raise ValueError("invalid batch-size, workers, or max-samples")
    checkpoints = []
    for spec in args.checkpoint:
        name, separator, path = spec.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in [item[0] for item in checkpoints]:
            raise ValueError("checkpoints require unique NAME=PATH with a simple filename-safe name")
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        checkpoints.append((name, path))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for a CPU check")
    # Same legacy absolute imports as the existing training/agent entrypoints.
    sys.path.insert(0, str(ROOT / "team_code_transfuser"))
    from team_code_transfuser.data import CARLA_Data
    from wote_mining.WoTE_train import seed_everything, seed_worker
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.amp:
        print("NOTE: AMP may change near-tied reward rankings; the current CARLA agent uses FP32.", flush=True)
    dataset = None
    summaries = {}
    preprocessing = None
    for name, path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu")
        saved_args, saved_config = checkpoint.get("args", {}), checkpoint.get("config", {})
        state = checkpoint["model"]
        epoch = checkpoint.get("epoch")
        del checkpoint  # Free optimizer/scaler before constructing another model.
        config = make_config(args.root_dir, saved_args, saved_config)
        signature = (config.use_target_point_image, config.img_resolution, config.img_width,
                     config.scale, config.seq_len, config.pred_len, config.lidar_pos)
        if preprocessing is not None and signature != preprocessing:
            raise ValueError("checkpoints use different sensor preprocessing")
        preprocessing = signature
        if dataset is None:
            dataset = CARLA_Data(root=config.val_data, config=config)
            indices = sorted(range(len(dataset)), key=lambda index: dataset.lidars[index][-1])
            if args.max_samples is not None:
                indices = indices[:args.max_samples]
            if not indices:
                raise ValueError("validation split has no usable frames")
            metadata = []
            for index in indices:
                lidar_path = Path(dataset.lidars[index][-1].decode("utf-8"))
                metadata.append(dict(sample_index=index, route=str(lidar_path.parent.parent), frame=lidar_path.stem))
            with (output_dir / "samples.json").open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2)
        seed_everything(args.seed)
        planner, missing = build_planner(config, saved_args, state, args.anchors)
        del state
        planner.to(device).eval()
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                            shuffle=False, drop_last=False, num_workers=args.workers,
                            worker_init_fn=seed_worker, generator=generator,
                            pin_memory=device.type == "cuda")
        print("%s: epoch=%s samples=%d legacy_auxiliary_keys=%s" % (name, epoch, len(indices), missing), flush=True)
        records = evaluate(planner, loader, metadata, device, args.amp)
        grouped = defaultdict(list)
        for record in records:
            grouped[record["route"]].append(record)
        summary = summarize_records(records)
        summary.update(checkpoint=str(Path(path).resolve()), epoch=epoch, legacy_auxiliary_keys=missing,
                       by_route={route: summarize_records(rows) for route, rows in sorted(grouped.items())})
        summaries[name] = summary
        with (output_dir / (name + "_samples.jsonl")).open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        with (output_dir / (name + "_summary.json")).open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps(summary["means"], ensure_ascii=False, indent=2), flush=True)
        del planner, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    first_name = checkpoints[0][0]
    baseline = summaries[first_name]["means"]
    comparison = {"settings": vars(args), "reference": first_name, "runs": summaries,
                  "mean_differences_vs_reference": {
                      name: {key: value - baseline[key] for key, value in summary["means"].items()
                             if value is not None and baseline[key] is not None}
                      for name, summary in summaries.items() if name != first_name},
                  "limitations": "Expert-distance diagnostics only; not PDMS, safety ground truth, or a CARLA closed-loop score."}
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(comparison, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("Saved inference diagnostics to %s" % output_dir, flush=True)


if __name__ == "__main__":
    main()
