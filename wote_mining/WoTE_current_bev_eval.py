"""FP32 open-loop evaluation of the score-selected current-BEV baseline.

Uses the same CARLA_Data vehicle-matrix expert poses and 2 Hz pointwise
metrics as the retained WoTE inference evaluation. No reward output is used.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sample_metrics(outputs, expert):
    """Measure only the score-selected decoded trajectory, never the oracle."""
    trajectories = outputs['trajectories'].float()
    expert = expert.to(trajectories).float()
    scores = outputs['scores'].float()
    if trajectories.ndim != 4 or trajectories.shape[2:] != (8, 3):
        raise ValueError('expected candidate trajectories [B,K,8,3]')
    if expert.shape != (trajectories.shape[0], 8, 3):
        raise ValueError('expected expert poses [B,8,3]')
    if scores.shape != trajectories.shape[:2]:
        raise ValueError('expected candidate scores [B,K]')
    if not all(torch.isfinite(x).all() for x in (trajectories, scores, expert)):
        raise ValueError('nonfinite predictions or expert poses')
    indices = scores.argmax(dim=1)
    rows = torch.arange(trajectories.shape[0], device=trajectories.device)
    selected = trajectories[rows, indices]
    if not torch.equal(indices, outputs['selected_index']):
        raise ValueError('selection differs from score_head argmax')
    if not torch.allclose(selected, outputs['selected_trajectory'].float(), atol=1e-5, rtol=1e-5):
        raise ValueError('selected trajectory differs from decoded candidate bank')
    distances = torch.linalg.vector_norm(selected[..., :2] - expert[..., :2], dim=-1)
    candidate_ade = torch.linalg.vector_norm(
        trajectories[..., :2] - expert[:, None, :, :2], dim=-1
    ).mean(dim=-1)
    velocity = 2.0 * (selected[:, 1, :2] - selected[:, 0, :2])
    expert_velocity = 2.0 * (expert[:, 1, :2] - expert[:, 0, :2])
    return {
        'online_l2_1s_m': distances[:, 1],
        'online_l2_2s_m': distances[:, 3],
        'online_l2_3s_m': distances[:, 5],
        'online_l2_4s_m': distances[:, 7],
        'online_ade_2s_m': distances[:, :4].mean(dim=-1),
        'online_ade_m': distances.mean(dim=-1),
        'online_fde_m': distances[:, -1],
        'online_first_second_ade_m': distances[:, :2].mean(dim=-1),
        'online_desired_speed_abs_error_mps': (
            torch.linalg.vector_norm(velocity, dim=-1)
            - torch.linalg.vector_norm(expert_velocity, dim=-1)
        ).abs(),
        'online_first_segment_velocity_error_mps': torch.linalg.vector_norm(
            velocity - expert_velocity, dim=-1
        ),
        'oracle_min_ade_m': candidate_ade.min(dim=1).values,
        'online_selection_regret_ade_m': distances.mean(dim=-1) - candidate_ade.min(dim=1).values,
    }


def load_planner(checkpoint_path, root_dir, anchors_override=None):
    from wote_mining.WoTE_config import WoTEMiningConfig
    from wote_mining.WoTE_model import WoTEMiningPlanner
    from team_code_transfuser import transfuser

    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    saved = checkpoint.get('args', {})
    mode = checkpoint.get('config', {}).get('planner_mode', saved.get('planner_mode'))
    if mode != 'current_only':
        raise ValueError('this evaluator requires a current_only checkpoint')
    state = checkpoint['model']
    epoch = int(checkpoint['epoch']) + 1
    del checkpoint  # Release saved optimizer/scaler tensors before building model.
    config = WoTEMiningConfig(root_dir=str(root_dir), setting='mining')
    config.set_planner_mode('current_only')
    config.backbone = 'transFuser'
    config.multitask = False
    config.use_point_pillars = False
    config.augment = False
    config.use_target_point_image = not saved.get('no_target_point_image', False)
    config.n_layer = saved.get('transformer_layers', 4)
    anchors = Path(anchors_override or saved.get('anchors', ''))
    if not anchors.is_file() and anchors_override is None:
        anchors = ROOT / 'wote_mining/assets/anchors/trajectory_anchors_256.npy'
    if not anchors.is_file():
        raise FileNotFoundError(str(anchors))
    create_model = transfuser.timm.create_model

    def no_download(*args, **kwargs):
        kwargs['pretrained'] = False
        return create_model(*args, **kwargs)

    with patch.object(transfuser.timm, 'create_model', side_effect=no_download):
        backbone = transfuser.TransfuserBackbone(
            config, image_architecture=saved.get('image_architecture', 'regnety_032'),
            lidar_architecture=saved.get('lidar_architecture', 'regnety_032'),
            use_velocity=saved.get('use_velocity', False),
        )
    planner = WoTEMiningPlanner(backbone, anchors)
    if not state or any(not name.startswith('planner.') for name in state):
        raise ValueError('expected training checkpoint with planner.* keys')
    state = {name[len('planner.'):]: value for name, value in state.items()}
    if not torch.equal(state['trajectory_head.anchors'], planner.trajectory_head.anchors):
        raise ValueError('anchor file differs from checkpoint bank')
    planner.load_state_dict(state, strict=True)
    return planner, config, epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--anchors')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-samples', type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or (args.max_samples is not None and args.max_samples < 1):
        raise ValueError('invalid batch size, workers or max samples')
    root = Path(args.root_dir)
    if not (root / 'train').is_dir() or not (root / 'val').is_dir():
        raise ValueError('root must contain train/ and val/')
    sys.path.insert(0, str(ROOT / 'team_code_transfuser'))
    from team_code_transfuser.data import CARLA_Data
    from wote_mining.WoTE_train import seed_everything, seed_worker
    from tqdm import tqdm

    seed_everything(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    planner, config, epoch = load_planner(args.checkpoint, root, args.anchors)
    # Same route and frame ordering as the existing inference evaluator.
    config.val_data = [str(p) for p in sorted((root / 'val').iterdir()) if (p / 'lidar').is_dir()]
    dataset = CARLA_Data(root=config.val_data, config=config)
    indices = sorted(range(len(dataset)), key=lambda i: dataset.lidars[i][-1])
    if args.max_samples is not None:
        indices = indices[:args.max_samples]
    if not indices:
        raise ValueError('no validation samples')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=args.workers, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(2026), pin_memory=device.type == 'cuda',
    )
    planner.to(device).eval()
    sums = {}
    cursor = 0
    sample_ids = []
    print('current_only: epoch=%d samples=%d selection=score_head world_passes=0 FP32' % (epoch, len(indices)), flush=True)
    with (output / 'samples.jsonl').open('w', encoding='utf-8') as stream, torch.no_grad():
        for batch in tqdm(loader, desc='current-BEV open-loop'):
            rgb, lidar = batch['rgb'].to(device).float(), batch['lidar'].to(device).float()
            if config.use_target_point_image:
                lidar = torch.cat((lidar, batch['target_point_image'].to(device).float()), dim=1)
            predictions = planner(
                rgb, lidar, batch['speed'].to(device).float().reshape(-1, 1),
                batch['target_point'].to(device).float(), predict_auxiliary=False,
            )
            expert = batch['wote_future_poses'].to(device).float()
            values = {key: val.cpu().tolist() for key, val in sample_metrics(predictions, expert).items()}
            selected = predictions['selected_trajectory'].cpu().tolist()
            ids = predictions['selected_index'].cpu().tolist()
            expert_rows = expert.cpu().tolist()
            for j in range(len(ids)):
                index = indices[cursor]
                path = Path(dataset.lidars[index][-1].decode('utf-8'))
                identity = dict(sample_index=index, route=str(path.parent.parent), frame=path.stem)
                sample_ids.append(identity)
                record = dict(identity, selected_index=ids[j],
                              trajectory=selected[j], expert_trajectory=expert_rows[j])
                for key, vector in values.items():
                    record[key] = vector[j]
                    sums[key] = sums.get(key, 0.0) + vector[j]
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                cursor += 1
    if cursor != len(indices):
        raise ValueError('sample count changed')
    (output / 'samples.json').write_text(json.dumps(sample_ids, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = dict(
        samples=cursor, means={key: value / cursor for key, value in sums.items()},
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_epoch_number=epoch,
        planner_mode='current_only', selection_source='score_head', deployment_world_passes=0,
        precision='fp32', ground_truth='CARLA_Data vehicle-matrix poses',
        sample_period_s=0.5, prediction_horizon_s=4.0,
        settings=vars(args),
        model_source_sha256=hashlib.sha256((ROOT / 'wote_mining/WoTE_model.py').read_bytes()).hexdigest(),
    )
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
