"""Read-only candidate telemetry, independent of CARLA and control logic."""

import torch

from wote_mining.WoTE_model import METRIC_NAMES


@torch.no_grad()
def reward_topk_diagnostics(outputs, topk=4):
    """Describe batch 0 of full-candidate inference without changing selection.

    Indices refer to the full anchor bank, as used by forward_ego. When rewards
    tie, list the actual selected candidate first, then ascending anchor IDs.
    Desired speed is the controller's unadjusted 2 Hz trajectory speed, before
    slope, safety, or stuck-recovery adjustments.
    """
    rewards = outputs["final_rewards"][0].detach().cpu().tolist()
    selected = int(outputs["selected_index"][0].detach().cpu())
    indices = sorted(
        range(len(rewards)),
        key=lambda index: (-rewards[index], index != selected, index),
    )[:max(0, int(topk))]
    # Transfer only the four trajectories/metric vectors, not the full bank.
    device_indices = torch.tensor(
        indices, device=outputs["world_trajectories"].device, dtype=torch.long
    )
    trajectories = outputs["world_trajectories"][0].index_select(
        0, device_indices
    ).detach().cpu()
    metrics = outputs["metric_scores"][0].index_select(
        0, device_indices
    ).detach().cpu().tolist()
    candidates = []
    for rank, index in enumerate(indices):
        trajectory = trajectories[rank]
        speed = float(torch.norm(trajectory[1, :2] - trajectory[0, :2]) * 2.0)
        candidates.append({
            "rank": rank + 1,
            "anchor_index": index,
            "selected": index == selected,
            "reward": float(rewards[index]),
            "learned_desired_speed_mps": speed,
            "metric_scores": dict(zip(METRIC_NAMES, metrics[rank])),
            "trajectory": trajectory.tolist(),
        })
    return {"reward_topk_candidates": candidates}
