"""Source-style core objectives for the mining WoTE model graph.

Reference: WoTE/navsim/agents/WoTE/WoTE_loss.py (Apache-2.0). This version
uses the local [B,K,8,3] trajectory convention and stable logits. The five
candidate metric labels are deliberately an optional *input*: this file does
not invent counterfactual mining scores from one recorded expert future.
"""

import torch
from torch.nn import functional as F


def future_refinement_loss(outputs, future_poses, identity_loss_weight=0.1):
    """Refine expert-compatible modes while preserving accurate coarse paths."""
    if identity_loss_weight < 0:
        raise ValueError("identity_loss_weight must be nonnegative")
    anchors = outputs["anchors"]
    coarse = outputs["coarse_trajectories"]
    if anchors.ndim != 4 or anchors.shape[2:] != (8, 3):
        raise ValueError("anchors must have shape [B,K,8,3]")
    batch, count = anchors.shape[:2]
    if coarse.shape != anchors.shape or future_poses.shape != (batch, 8, 3):
        raise ValueError("refined trajectories/future poses have incompatible shapes")
    indices = outputs["refinement_training_indices"]
    valid = outputs["refinement_training_valid"]
    predicted_residual = outputs["refinement_training_offsets"]
    selected_count = indices.shape[1]
    if (indices.shape != (batch, selected_count)
            or valid.shape != indices.shape
            or predicted_residual.shape != (batch, selected_count, 8, 3)):
        raise ValueError("refinement training tensors have incompatible shapes")
    gather_index = indices[:, :, None, None].expand(-1, -1, 8, 3)
    selected_coarse = coarse.gather(1, gather_index).detach()
    desired_residual = future_poses[:, None] - selected_coarse
    per_candidate = F.smooth_l1_loss(
        predicted_residual, desired_residual, reduction="none"
    ).mean(dim=(2, 3))
    # The oracle and the highest-ranked compatible reward choice matter most.
    # Other reward modes contribute without overpowering a good oracle path.
    rank_weights = per_candidate.new_full((selected_count,), 0.25)
    rank_weights[0] = 2.0
    if selected_count > 1:
        rank_weights[1] = 1.0
    weights = valid.to(per_candidate) * rank_weights[None]
    normalizer = weights.sum(dim=1).clamp_min(1.0)
    imitation = (per_candidate * weights).sum(dim=1) / normalizer

    coarse_xy_ade = torch.linalg.vector_norm(
        selected_coarse[..., :2] - future_poses[:, None, :, :2], dim=-1
    ).mean(dim=-1)
    near_expert = (1.0 - coarse_xy_ade / 0.5).clamp(0.0, 1.0).detach()
    correction_xy = predicted_residual[..., :2].square().mean(dim=(2, 3))
    identity = (correction_xy * near_expert * weights).sum(dim=1) / normalizer
    return (imitation + float(identity_loss_weight) * identity).mean()


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
