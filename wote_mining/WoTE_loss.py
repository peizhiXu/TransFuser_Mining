"""Source-style core objectives for the mining WoTE model graph.

Reference: WoTE/navsim/agents/WoTE/WoTE_loss.py (Apache-2.0). This version
uses the local [B,K,8,3] trajectory convention and stable logits. The five
candidate metric labels are deliberately an optional *input*: this file does
not invent counterfactual mining scores from one recorded expert future.
"""

import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def select_reward_topk_training_candidates(
        final_rewards, anchors, trajectories, future_poses,
        reward_topk=4, endpoint_max_m=2.0, ade_max_m=1.0):
    """Select extra expert-compatible modes, leaving the WTA oracle alone.

    Port the rewardtopk eligibility/ranking rule, not its refiner. Rank by
    detached reward-head scores (not score_head); gate on detached decoded
    trajectories. Training scores still describe fixed anchors, so this is
    a candidate-ID preference, not an online evaluation of decoded motions.
    Invalid slots are masked; they never supervise an unrelated mode.
    """
    if final_rewards.ndim != 2:
        raise ValueError("final_rewards must have shape [B,K]")
    batch, count = final_rewards.shape
    expected = (batch, count, 8, 3)
    if anchors.shape != expected or trajectories.shape != expected:
        raise ValueError("anchors/trajectories must have shape [B,K,8,3]")
    if future_poses.shape != (batch, 8, 3):
        raise ValueError("future_poses must have shape [B,8,3]")
    if not isinstance(reward_topk, int) or not 0 <= reward_topk < count:
        raise ValueError("reward_topk must be an integer between 0 and K-1")
    if any(not math.isfinite(value) or value <= 0.0
           for value in (endpoint_max_m, ade_max_m)):
        raise ValueError("reward top-k distance thresholds must be finite and positive")

    anchor_distance = torch.linalg.vector_norm(
        (anchors.float() - future_poses.float()[:, None]).flatten(2), dim=-1
    )
    oracle = anchor_distance.argmin(dim=1)
    xy_distance = torch.linalg.vector_norm(
        trajectories.float()[..., :2] - future_poses.float()[:, None, :, :2],
        dim=-1,
    )
    eligible = (
        (xy_distance.mean(dim=-1) <= float(ade_max_m))
        & (xy_distance[:, :, -1] <= float(endpoint_max_m))
        & torch.isfinite(trajectories).all(dim=-1).all(dim=-1)
        & torch.isfinite(final_rewards)
    )
    # The existing oracle L1 loss is retained unchanged; do not double-count it.
    eligible.scatter_(1, oracle[:, None], False)
    ranked_rewards = final_rewards.float().masked_fill(~eligible, float("-inf"))
    top_values, indices = ranked_rewards.topk(reward_topk, dim=1)
    valid = torch.isfinite(top_values)
    # Pad with a known ID for safe gathering, but padded slots have zero loss.
    indices = torch.where(valid, indices, oracle[:, None].expand_as(indices))
    reward_selected = final_rewards.argmax(dim=1)
    selected_supervised = (
        (reward_selected == oracle)
        | ((indices == reward_selected[:, None]) & valid).any(dim=1)
    )
    return {
        "reward_topk_training_indices": indices,
        "reward_topk_training_valid": valid,
        "reward_topk_oracle_index": oracle,
        "reward_topk_eligible_count": eligible.sum(dim=1),
        "reward_topk_selected_supervised": selected_supervised,
    }


def reward_topk_trajectory_loss(outputs, future_poses):
    """Light auxiliary regression on selected final AdaLN trajectories.

    Normalize within each observation, then average over the batch. Samples
    with no eligible extras contribute a graph-connected zero. This loss has
    no gradient into the discrete candidate selection or reward-head scores.
    """
    trajectories = outputs["trajectories"]
    if trajectories.ndim != 4 or trajectories.shape[2:] != (8, 3):
        raise ValueError("trajectories must have shape [B,K,8,3]")
    batch = trajectories.shape[0]
    indices = outputs["reward_topk_training_indices"]
    valid = outputs["reward_topk_training_valid"]
    if (indices.ndim != 2 or indices.shape[0] != batch
            or valid.shape != indices.shape or valid.dtype != torch.bool
            or indices.dtype != torch.long
            or future_poses.shape != (batch, 8, 3)):
        raise ValueError("reward top-k loss tensors have incompatible shapes or dtypes")
    selected = trajectories.gather(
        1, indices[:, :, None, None].expand(-1, -1, 8, 3)
    ).float()
    target = future_poses.float()[:, None].expand_as(selected)
    # Mask before computing the loss as well: padding must not turn 0*NaN
    # into a NaN loss when an invalid slot contains a non-finite prediction.
    selected = torch.where(valid[:, :, None, None], selected, target)
    per_candidate = F.smooth_l1_loss(selected, target, reduction="none").mean(
        dim=(2, 3)
    )
    weights = valid.to(per_candidate)
    return ((per_candidate * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0)).mean()


def future_read_metric_loss(logits, metric_targets, metric_valid):
    """Teach the detached-future reader NC/DAC/EP using fixed-anchor labels.

    Each metric is averaged over its own valid entries, then the three are
    summed. Missing labels contribute a graph-connected zero for DDP. This
    is not a second reward used to select trajectories during inference.
    """
    if logits.ndim != 3 or logits.shape[-1] != 3:
        raise ValueError("future read logits must have shape [B,K,3]")
    if (metric_targets.shape != logits.shape[:2] + (5,)
            or metric_valid.shape != metric_targets.shape):
        raise ValueError("future read targets/validity must have shape [B,K,5]")
    logits = logits.float()
    target = metric_targets[..., :3].to(logits)
    valid = metric_valid[..., :3].to(logits)
    if not torch.isfinite(valid).all() or ((valid < 0) | (valid > 1)).any():
        raise ValueError("future read validity must be finite values in [0,1]")
    active = valid > 0
    if (not torch.isfinite(target[active]).all()
            or ((target[active] < 0) | (target[active] > 1)).any()):
        raise ValueError("valid future read targets must be finite values in [0,1]")
    # Mask before BCE, so missing targets/predictions cannot introduce NaNs.
    target = torch.where(active, target, torch.zeros_like(target))
    safe_logits = torch.where(active, logits, torch.zeros_like(logits))
    elementwise = F.binary_cross_entropy_with_logits(
        safe_logits, target, reduction="none"
    )
    return ((elementwise * valid).sum(dim=(0, 1))
            / valid.sum(dim=(0, 1)).clamp_min(1.0)).sum()


def source_style_core_losses(outputs, future_poses, metric_targets=None,
                             metric_valid=None):
    """WTA offset + soft imitation objectives, optionally five metric heads.

    ``metric_targets`` [B,K,5], if supplied, must be independently defined
    candidate-level values in [0,1]. ``metric_valid`` masks any unavailable
    candidate/metric labels. The function does not include map/agent losses;
    their matching and spatial masks live with those modules.
    """
    anchors = outputs["anchors"]
    offsets = outputs["offsets"]
    offset_logits = outputs["scores"]
    imitation_logits = outputs["imitation_logits"]
    if anchors.ndim != 4 or anchors.shape[2:] != (8, 3):
        raise ValueError("anchors must have shape [B,K,8,3]")
    batch, count = anchors.shape[:2]
    if offsets.shape != anchors.shape or future_poses.shape != (batch, 8, 3):
        raise ValueError("offsets/future poses have incompatible shapes")
    if offset_logits.shape != (batch, count):
        raise ValueError("offset scores must have shape [B,K]")
    if imitation_logits.shape != (batch, count):
        raise ValueError("full candidate imitation logits must have shape [B,K]")

    # Match the source's Euclidean distance in flattened 24-D pose space.
    distances = torch.linalg.vector_norm(
        (anchors - future_poses[:, None]).reshape(batch, count, -1), dim=-1
    )
    winner = distances.argmin(dim=1)
    batch_index = torch.arange(batch, device=anchors.device)
    desired_offset = future_poses - anchors[batch_index, winner]
    predicted_offset = offsets[batch_index, winner]
    soft_target = torch.softmax(-distances.detach(), dim=-1)
    losses = {
        "loss_traj_offset": F.l1_loss(predicted_offset, desired_offset),
        "loss_offset_imitation": -(
            soft_target * F.log_softmax(offset_logits, dim=-1)
        ).sum(dim=-1).mean(),
        "loss_imitation_reward": -(
            soft_target * F.log_softmax(imitation_logits, dim=-1)
        ).sum(dim=-1).mean(),
        "matched_anchor": winner,
    }

    if metric_targets is not None:
        if metric_targets.shape != (batch, count, 5):
            raise ValueError("metric_targets must have shape [B,K,5]")
        if not torch.isfinite(metric_targets).all() or (
            (metric_targets < 0) | (metric_targets > 1)
        ).any():
            raise ValueError("metric_targets must be finite values in [0,1]")
        metric_logits = outputs["metric_logits"]
        if metric_logits.shape != metric_targets.shape:
            raise ValueError("metric logits/targets have incompatible shapes")
        target = metric_targets.to(device=metric_logits.device, dtype=metric_logits.dtype)
        elementwise = F.binary_cross_entropy_with_logits(
            metric_logits, target, reduction="none"
        )
        if metric_valid is None:
            # Source WoTE adds the five independently averaged metric losses.
            # Keeping the heads separate prevents a missing label in one head
            # from changing the effective weight of every other head.
            losses["loss_metric_reward"] = elementwise.mean(dim=(0, 1)).sum()
        else:
            if metric_valid.shape != metric_targets.shape:
                raise ValueError("metric_valid must have shape [B,K,5]")
            valid = metric_valid.to(device=metric_logits.device, dtype=metric_logits.dtype)
            valid_count = valid.sum(dim=(0, 1))
            per_metric = (elementwise * valid).sum(dim=(0, 1)) / valid_count.clamp_min(1)
            losses["loss_metric_reward"] = (
                per_metric * (valid_count > 0).to(per_metric.dtype)
            ).sum()
    elif metric_valid is not None:
        raise ValueError("metric_valid requires metric_targets")
    return losses
