"""Single-step (t+4 s) latent BEV world model for the HD465.

Architecture reference: WoTE/navsim/agents/WoTE/WoTE_model.py, especially
``extract_reward_feature`` and ``_latent_world_model_processing`` (Apache-2.0).
Reimplemented for the mining TransFuser's 8x8 forward 32 m / lateral 32 m
feature map. The NAVSIM-specific geometry and target heads are not copied.
This module predicts candidate-conditioned future scene features and maps;
The trajectory, world-model, scene-decoding, and reward stages are kept
together in this module so the complete WoTE model is easy to locate.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


BEV_SIDE = 8
BEV_METERS = 32.0
MAP_SIDE = 160
SOURCE_FOCAL_ALPHA = 0.5
SOURCE_FOCAL_GAMMA = 2.0


def masked_sigmoid_focal_loss(logits, target, valid,
                              alpha=SOURCE_FOCAL_ALPHA,
                              gamma=SOURCE_FOCAL_GAMMA):
    """Multi-label focal loss with WoTE's alpha/gamma and a spatial mask.

    Source WoTE uses focal loss for semantic BEV supervision. Its NAVSIM map
    is categorical, whereas the mining map layers may overlap, so this port
    uses the sigmoid form rather than forcing them through a softmax.
    """
    if logits.shape != target.shape:
        raise ValueError("focal logits and target must have identical shapes")
    if not 0.0 <= alpha <= 1.0 or gamma < 0.0:
        raise ValueError("focal alpha must be in [0,1] and gamma nonnegative")
    truth = target.to(device=logits.device, dtype=logits.dtype)
    mask = valid.to(device=logits.device, dtype=logits.dtype)
    try:
        mask = mask.expand_as(logits)
    except RuntimeError as error:
        raise ValueError("focal validity mask is not broadcastable to logits") from error
    binary_ce = F.binary_cross_entropy_with_logits(
        logits, truth, reduction="none"
    )
    probability = torch.sigmoid(logits)
    probability_t = probability * truth + (1.0 - probability) * (1.0 - truth)
    alpha_t = alpha * truth + (1.0 - alpha) * (1.0 - truth)
    focal = alpha_t * (1.0 - probability_t).pow(gamma) * binary_ce
    return (focal * mask).sum() / mask.sum().clamp_min(1)


def vehicle_points_to_bev_grid(points, lidar_x=3.5, augmentation_degrees=None):
    """Map augmented vehicle-frame [x forward,y right] to 8x8 cell centers.

    Row zero is far ahead and column zero is left. ``inside`` uses actual BEV
    boundaries, not just cell-center bounds. Current vehicle origin is outside
    the forward-only crop with a forward-mounted LiDAR; the action token still
    carries it through the transformer, so we do not force it into a BEV cell.
    """
    if points.shape[-1] != 2:
        raise ValueError("points must end in two coordinates")
    if augmentation_degrees is None:
        angle = points.new_zeros(points.shape[:-1])
    else:
        angle = torch.as_tensor(
            augmentation_degrees, dtype=points.dtype, device=points.device
        ) * (math.pi / 180.0)
        while angle.ndim < points.ndim - 1:
            angle = angle.unsqueeze(-1)
    mount_x = float(lidar_x) * torch.cos(angle)
    mount_y = -float(lidar_x) * torch.sin(angle)
    x_lidar = points[..., 0] - mount_x
    y_lidar = points[..., 1] - mount_y
    meters_per_cell = BEV_METERS / BEV_SIDE
    row = (BEV_METERS - x_lidar) / meters_per_cell - 0.5
    col = (y_lidar + BEV_METERS / 2) / meters_per_cell - 0.5
    inside = (
        (x_lidar >= 0) & (x_lidar < BEV_METERS)
        & (y_lidar >= -BEV_METERS / 2) & (y_lidar < BEV_METERS / 2)
    )
    return row, col, inside


def inject_trajectory_feature(scene_tokens, features, points, lidar_x=3.5,
                              augmentation_degrees=None):
    """Bilinearly add one action feature to its future position in BEV.

    ``scene_tokens`` is [N,64,C], ``features`` [N,C], and ``points`` [N,2].
    Weights are renormalized at the crop edge. An out-of-range point causes
    no spatial update, while its action token remains available to the model.
    """
    if scene_tokens.ndim != 3 or scene_tokens.shape[1] != BEV_SIDE ** 2:
        raise ValueError("scene_tokens must have shape [N,64,C]")
    count, _, channels = scene_tokens.shape
    if features.shape != (count, channels) or points.shape != (count, 2):
        raise ValueError("features or points have incompatible shape")
    row, col, inside = vehicle_points_to_bev_grid(
        points, lidar_x, augmentation_degrees
    )
    r0, c0 = row.floor().long(), col.floor().long()
    dr, dc = row - r0, col - c0
    neighbors = (
        (r0, c0, (1 - dr) * (1 - dc)),
        (r0, c0 + 1, (1 - dr) * dc),
        (r0 + 1, c0, dr * (1 - dc)),
        (r0 + 1, c0 + 1, dr * dc),
    )
    normalizer = torch.zeros_like(row)
    for rr, cc, weight in neighbors:
        valid = inside & (rr >= 0) & (rr < BEV_SIDE) & (cc >= 0) & (cc < BEV_SIDE)
        normalizer = normalizer + torch.where(valid, weight, torch.zeros_like(weight))
    flat = scene_tokens.reshape(count * BEV_SIDE ** 2, channels)
    batch_index = torch.arange(count, device=scene_tokens.device)
    for rr, cc, weight in neighbors:
        valid = inside & (rr >= 0) & (rr < BEV_SIDE) & (cc >= 0) & (cc < BEV_SIDE)
        if valid.any():
            index = batch_index[valid] * BEV_SIDE ** 2 + rr[valid] * BEV_SIDE + cc[valid]
            contribution = features[valid] * (
                weight[valid] / normalizer[valid].clamp_min(1e-6)
            ).unsqueeze(-1)
            flat = flat.index_add(0, index, contribution)
    return flat.reshape(count, BEV_SIDE ** 2, channels)


class WoTEMiningWorldModel(nn.Module):
    """Candidate-conditioned latent scene transition plus selected map head."""

    def __init__(self, hidden_dim=256, layers=2, heads=8, lidar_x=3.5,
                 candidate_chunk=16, ego_length=9.3969, ego_width=5.3645):
        super().__init__()
        if candidate_chunk < 1:
            raise ValueError("candidate_chunk must be positive")
        self.hidden_dim = hidden_dim
        self.lidar_x = float(lidar_x)
        self.candidate_chunk = int(candidate_chunk)
        self.ego_length = float(ego_length)
        self.ego_width = float(ego_width)
        self.action_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 24, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scene_position = nn.Embedding(BEV_SIDE ** 2 + 1, hidden_dim)
        self.transition = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_dim, heads, hidden_dim * 2, dropout=0.1, batch_first=True
            ),
            num_layers=layers,
        )
        self.map_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(size=(MAP_SIDE, MAP_SIDE), mode="bilinear", align_corners=False),
            nn.Conv2d(64, 5, 1),
        )

    def forward(self, bev_tokens, trajectory_features, trajectories,
                candidate_indices=None, augmentation_degrees=None,
                predict_future_map=False):
        if bev_tokens.ndim != 3 or bev_tokens.shape[1:] != (64, self.hidden_dim):
            raise ValueError("bev_tokens must have shape [B,64,C]")
        batch, count, channels = trajectory_features.shape
        if batch != bev_tokens.shape[0] or channels != self.hidden_dim:
            raise ValueError("trajectory_features must have shape [B,K,C]")
        if trajectories.shape != (batch, count, 8, 3):
            raise ValueError("trajectories must have shape [B,K,8,3]")
        if candidate_indices is not None:
            if candidate_indices.ndim != 2 or candidate_indices.shape[0] != batch:
                raise ValueError("candidate_indices must have shape [B,S]")
            if candidate_indices.dtype != torch.long or candidate_indices.shape[1] < 1:
                raise ValueError("candidate_indices must be nonempty int64 indices")
            if (candidate_indices < 0).any() or (candidate_indices >= count).any():
                raise ValueError("candidate index is out of range")
            select_features = candidate_indices.unsqueeze(-1).expand(-1, -1, channels)
            select_trajectories = candidate_indices[:, :, None, None].expand(-1, -1, 8, 3)
            trajectory_features = trajectory_features.gather(1, select_features)
            trajectories = trajectories.gather(1, select_trajectories)
        selected_count = trajectories.shape[1]
        # Decoding every one of the 256 maps at once is expensive. Training
        # selects a small candidate subset, while all candidates can still
        # pass through the latent world model for reward prediction.
        if predict_future_map and selected_count > self.candidate_chunk:
            raise ValueError("future map decoding requires at most candidate_chunk candidates")

        if augmentation_degrees is None:
            degrees = bev_tokens.new_zeros(batch)
        else:
            degrees = torch.as_tensor(
                augmentation_degrees, dtype=bev_tokens.dtype, device=bev_tokens.device
            ).reshape(batch)
        future_scene_chunks = []
        future_action_chunks = []
        for start in range(0, selected_count, self.candidate_chunk):
            stop = min(start + self.candidate_chunk, selected_count)
            chunk = stop - start
            traj = trajectories[:, start:stop]
            action_features = trajectory_features[:, start:stop]
            action = self.action_encoder(
                torch.cat((action_features, traj.flatten(2)), dim=-1)
            )
            scene = bev_tokens[:, None].expand(-1, chunk, -1, -1)
            tokens = torch.cat((action.unsqueeze(2), scene), dim=2)
            tokens = tokens.reshape(batch * chunk, 65, channels)
            tokens = tokens + self.scene_position.weight.unsqueeze(0)
            future = self.transition(tokens)
            future_action = future[:, 0]
            future_scene = inject_trajectory_feature(
                future[:, 1:], future_action, traj[:, :, -1, :2].reshape(-1, 2),
                self.lidar_x, degrees[:, None].expand(-1, chunk).reshape(-1),
            )
            future_scene_chunks.append(future_scene.reshape(batch, chunk, 64, channels))
            future_action_chunks.append(future_action.reshape(batch, chunk, channels))
        result = {
            "future_bev_tokens": torch.cat(future_scene_chunks, dim=1),
            "future_action_features": torch.cat(future_action_chunks, dim=1),
            "world_trajectories": trajectories,
        }
        if predict_future_map:
            result.update(self.decode_future_map(result))
        return result

    def decode_future_map(self, world_outputs, candidate_indices=None):
        """Decode maps for a small subset after scoring every candidate.

        Source WoTE transitions all anchors through the latent world model,
        while its expensive semantic-map supervision samples only one anchor.
        Keeping selection here avoids running the transition twice and keeps
        the reward heads trained on the complete candidate set.
        """
        tokens = world_outputs["future_bev_tokens"]
        trajectories = world_outputs["world_trajectories"]
        if tokens.ndim != 4 or tokens.shape[2:] != (BEV_SIDE ** 2, self.hidden_dim):
            raise ValueError("future_bev_tokens must have shape [B,K,64,C]")
        batch, count = tokens.shape[:2]
        if trajectories.shape != (batch, count, 8, 3):
            raise ValueError("world_trajectories must have shape [B,K,8,3]")
        if candidate_indices is not None:
            if (candidate_indices.ndim != 2 or candidate_indices.shape[0] != batch
                    or candidate_indices.dtype != torch.long
                    or candidate_indices.shape[1] < 1):
                raise ValueError("candidate_indices must have shape [B,S] and dtype int64")
            if (candidate_indices < 0).any() or (candidate_indices >= count).any():
                raise ValueError("candidate index is out of range")
            selected_count = candidate_indices.shape[1]
            tokens = tokens.gather(
                1, candidate_indices[:, :, None, None].expand(
                    -1, -1, BEV_SIDE ** 2, self.hidden_dim
                )
            )
            trajectories = trajectories.gather(
                1, candidate_indices[:, :, None, None].expand(-1, -1, 8, 3)
            )
        else:
            selected_count = count
        if selected_count > self.candidate_chunk:
            raise ValueError("future map decoding requires at most candidate_chunk candidates")
        scene = tokens.reshape(
            batch * selected_count, BEV_SIDE ** 2, self.hidden_dim
        ).transpose(1, 2)
        logits = self.map_head(
            scene.reshape(
                batch * selected_count, self.hidden_dim, BEV_SIDE, BEV_SIDE
            )
        ).reshape(batch, selected_count, 5, MAP_SIDE, MAP_SIDE)
        return {
            "future_map_logits": logits,
            "future_map_trajectories": trajectories,
            "future_map_candidate_indices": candidate_indices,
        }

    def future_map_loss(self, outputs, future_scene, future_valid,
                        augmentation_degrees=None):
        """Five-layer map loss for a sampled set of candidate trajectories.

        The first four layers replay the one recorded future scene. The fifth
        rasterizes each candidate's own future ego truck. This is a NAVSIM-
        style *non-reactive* target: other actors do not respond to the
        changed ego path, so it is not a fully counterfactual ground truth.
        """
        logits = outputs["future_map_logits"]
        trajectories = outputs.get(
            "future_map_trajectories", outputs["world_trajectories"]
        )
        target, validity = self.future_map_targets(
            trajectories, future_scene, future_valid,
            augmentation_degrees=augmentation_degrees,
        )
        if logits.shape != target.shape:
            raise ValueError("future_map_logits and targets have incompatible shapes")
        return masked_sigmoid_focal_loss(logits, target, validity)

    def future_map_targets(self, trajectories, future_scene, future_valid,
                           augmentation_degrees=None):
        """Build per-candidate map targets from one recorded non-reactive scene."""
        batch, count = trajectories.shape[:2]
        if trajectories.shape != (batch, count, 8, 3) or count < 1:
            raise ValueError("trajectories must have shape [B,S,8,3]")
        if future_scene.shape != (batch, 4, MAP_SIDE, MAP_SIDE):
            raise ValueError("future_scene must have shape [B,4,160,160]")
        if future_valid.shape != (batch, MAP_SIDE, MAP_SIDE):
            raise ValueError("future_valid must have shape [B,160,160]")
        device, dtype = trajectories.device, trajectories.dtype
        if augmentation_degrees is None:
            degrees = trajectories.new_zeros(batch)
        else:
            degrees = torch.as_tensor(
                augmentation_degrees, dtype=dtype, device=device
            ).reshape(batch)
        ego = self._rasterize_ego(
            trajectories[:, :, -1].reshape(batch * count, 3),
            degrees[:, None].expand(-1, count).reshape(-1),
        ).reshape(batch, count, MAP_SIDE, MAP_SIDE)
        scene = future_scene.to(device=device, dtype=dtype)
        target = torch.cat((
            scene[:, None].expand(-1, count, -1, -1, -1), ego[:, :, None],
        ), dim=2)
        validity = future_valid.to(device=device, dtype=dtype)
        validity = validity[:, None, None]
        return target, validity

    def _rasterize_ego(self, endpoint, degrees):
        batch = endpoint.shape[0]
        dtype, device = endpoint.dtype, endpoint.device
        row, col = torch.meshgrid(
            torch.arange(MAP_SIDE, dtype=dtype, device=device),
            torch.arange(MAP_SIDE, dtype=dtype, device=device),
            indexing="ij",
        )
        x_lidar = (MAP_SIDE - row) / 5.0
        y_lidar = (col - MAP_SIDE / 2) / 5.0
        angle = degrees.to(dtype) * (math.pi / 180.0)
        shift = math.floor(self.lidar_x * 5.0 + 0.5) / 5.0
        x_vehicle = x_lidar[None] + shift * torch.cos(angle)[:, None, None]
        y_vehicle = y_lidar[None] - shift * torch.sin(angle)[:, None, None]
        dx = x_vehicle - endpoint[:, 0, None, None]
        dy = y_vehicle - endpoint[:, 1, None, None]
        yaw = endpoint[:, 2, None, None]
        local_x = torch.cos(yaw) * dx + torch.sin(yaw) * dy
        local_y = -torch.sin(yaw) * dx + torch.cos(yaw) * dy
        inside = (local_x.abs() <= self.ego_length / 2) & (
            local_y.abs() <= self.ego_width / 2
        )
        return inside.to(dtype).reshape(batch, MAP_SIDE, MAP_SIDE)

"""WoTE-style per-candidate reward features and trajectory selection.

Architecture reference: WoTE/navsim/agents/WoTE/WoTE_model.py, including
``RewardConvNet`` and ``weighted_reward_calculation`` (Apache-2.0).
This mining adaptation uses the local current/future 8x8 BEV tokens and
the HD465 trajectory-query features. The five metric heads are learned model
outputs, not geometric oracle scores.
"""

import torch
from torch import nn
from torch.nn import functional as F


METRIC_NAMES = (
    "no_collision",
    "drivable_compliance",
    "ego_progress",
    "time_to_collision",
    "comfort",
)


class WoTEMiningRewardHead(nn.Module):
    """Score [B,K] candidates from current/future BEV and action features."""

    def __init__(self, hidden_dim=256, weights=(0.1, 0.5, 0.5, 1.0)):
        super().__init__()
        if len(weights) != 4 or any(weight < 0 for weight in weights):
            raise ValueError("reward weights must contain four nonnegative values")
        self.hidden_dim = hidden_dim
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.reward_conv = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.reward_fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.imitation_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1),
        )
        self.metric_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1))
            for _ in METRIC_NAMES
        ])

    def forward(self, current_bev, future_bev, current_action, future_action):
        if current_bev.ndim != 3 or current_bev.shape[1:] != (64, self.hidden_dim):
            raise ValueError("current_bev must have shape [B,64,C]")
        if future_bev.ndim != 4 or future_bev.shape[2:] != (64, self.hidden_dim):
            raise ValueError("future_bev must have shape [B,K,64,C]")
        batch, count = future_bev.shape[:2]
        if current_bev.shape[0] != batch:
            raise ValueError("current and future BEV batch sizes differ")
        if current_action.shape != (batch, count, self.hidden_dim):
            raise ValueError("current_action must have shape [B,K,C]")
        if future_action.shape != current_action.shape:
            raise ValueError("future_action must have shape [B,K,C]")

        current = current_bev[:, None].expand(-1, count, -1, -1)
        paired = torch.cat((current, future_bev), dim=-1)
        paired = paired.reshape(batch * count, 8, 8, 2 * self.hidden_dim)
        paired = paired.permute(0, 3, 1, 2)
        pooled = self.reward_conv(paired).flatten(1).reshape(batch, count, -1)
        reward_features = self.reward_fusion(
            torch.cat((current_action, future_action, pooled), dim=-1)
        )
        imitation_logits = self.imitation_head(reward_features).squeeze(-1)
        metric_logits = torch.cat(
            [head(reward_features) for head in self.metric_heads], dim=-1
        )
        final_rewards, imitation_probs, metric_scores = self.compose_rewards(
            imitation_logits, metric_logits
        )
        return {
            "reward_features": reward_features,
            "imitation_logits": imitation_logits,
            "imitation_probs": imitation_probs,
            "metric_logits": metric_logits,
            "metric_scores": metric_scores,
            "final_rewards": final_rewards,
        }

    def compose_rewards(self, imitation_logits, metric_logits):
        """Use WoTE's four-term reward composition with stable logarithms.

        Metric order: no collision, drivable compliance, progress, TTC,
        comfort. A higher final reward is better.
        """
        if imitation_logits.ndim != 2 or metric_logits.shape != (
            imitation_logits.shape[0], imitation_logits.shape[1], 5
        ):
            raise ValueError("expected imitation [B,K] and metrics [B,K,5]")
        if imitation_logits.shape[1] < 1:
            raise ValueError("at least one candidate is required")
        metric_scores = torch.sigmoid(metric_logits)
        log_imitation = F.log_softmax(imitation_logits, dim=-1)
        # Match source WoTE's NC and DAC log barriers, while clamping values
        # only where a logarithm of a weighted sum is required.
        log_no_collision = F.logsigmoid(metric_logits[..., 0])
        log_drivable = F.logsigmoid(metric_logits[..., 1])
        progress = metric_scores[..., 2]
        ttc = metric_scores[..., 3]
        comfort = metric_scores[..., 4]
        combined = (5.0 * ttc + 2.0 * comfort + 5.0 * progress).clamp_min(1e-8)
        w = self.weights.to(dtype=imitation_logits.dtype)
        final_rewards = (
            w[0] * log_imitation + w[1] * log_no_collision
            + w[2] * log_drivable + w[3] * torch.log(combined)
        )
        return final_rewards, log_imitation.exp(), metric_scores


def select_best_trajectory(final_rewards, trajectories, candidate_indices=None):
    """Select the highest-scoring candidate and preserve its global ID."""
    if final_rewards.ndim != 2 or trajectories.shape != (
        final_rewards.shape[0], final_rewards.shape[1], 8, 3
    ):
        raise ValueError("expected rewards [B,K] and trajectories [B,K,8,3]")
    batch = final_rewards.shape[0]
    local_index = final_rewards.argmax(dim=1)
    batch_index = torch.arange(batch, device=final_rewards.device)
    selected = trajectories[batch_index, local_index]
    if candidate_indices is None:
        anchor_index = local_index
    else:
        if candidate_indices.shape != final_rewards.shape:
            raise ValueError("candidate_indices must have shape [B,K]")
        anchor_index = candidate_indices[batch_index, local_index]
    return {
        "selected_index": local_index,
        "selected_anchor_index": anchor_index,
        "selected_trajectory": selected,
    }

"""Current-scene auxiliary heads adapted from the WoTE source model.

Reference: WoTE/navsim/agents/WoTE/WoTE_model.py, ``_process_agent`` and
``AgentHead`` (Apache-2.0). The mining version uses 20 queries and its
forward-only 32 m x 32 m LiDAR region. The current map decoder is shared
with the future world-map head by ``WoTEMiningPlanner``.
"""

import math

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


class WoTEMiningAgentHead(nn.Module):
    """Decode current non-ego vehicles from 64 fused LiDAR BEV tokens."""

    def __init__(self, hidden_dim=256, num_agents=20, lidar_x=3.5,
                 layers=2, heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.lidar_x = float(lidar_x)
        self.query = nn.Embedding(num_agents, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                hidden_dim, heads, hidden_dim * 4, dropout=0.1,
                batch_first=True,
            ),
            num_layers=layers,
        )
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )
        self.presence_head = nn.Linear(hidden_dim, 1)

    def forward(self, bev_tokens):
        if bev_tokens.ndim != 3 or bev_tokens.shape[1:] != (64, self.hidden_dim):
            raise ValueError("bev_tokens must have shape [B,64,C]")
        batch = bev_tokens.shape[0]
        queries = self.query.weight.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(queries, bev_tokens)
        raw = self.box_head(decoded)
        boxes = torch.stack(
            (
                self.lidar_x + 32.0 * torch.sigmoid(raw[..., 0]),
                16.0 * torch.tanh(raw[..., 1]),
                math.pi * torch.tanh(raw[..., 2]),
                F.softplus(raw[..., 3]) + 1.0,
                F.softplus(raw[..., 4]) + 1.0,
            ),
            dim=-1,
        )
        return {
            "current_agent_boxes_pred": boxes,
            "current_agent_logits": self.presence_head(decoded).squeeze(-1),
        }

    @staticmethod
    def matching_loss(predictions, target_boxes, target_mask):
        """Hungarian-match unordered vehicle boxes, as in source WoTE.

        Targets are [x_forward,y_right,yaw,length,width] in current vehicle
        coordinates. Returns presence BCE and normalized box Smooth L1.
        """
        pred_boxes = predictions["current_agent_boxes_pred"]
        pred_logits = predictions["current_agent_logits"]
        batch, queries, dimensions = pred_boxes.shape
        if dimensions != 5 or pred_logits.shape != (batch, queries):
            raise ValueError("invalid predicted agent shapes")
        if target_boxes.shape != pred_boxes.shape or target_mask.shape != (batch, queries):
            raise ValueError("target boxes/mask must match query dimensions")

        normalizer = pred_boxes.new_tensor([32.0, 16.0, math.pi, 10.0, 5.0])
        presence_targets = torch.zeros_like(pred_logits)
        box_total = pred_boxes.sum() * 0.0
        matched_total = 0
        for batch_index in range(batch):
            valid_targets = target_boxes[batch_index, target_mask[batch_index].bool()]
            if not valid_targets.numel():
                continue
            cost = torch.cdist(
                pred_boxes[batch_index] / normalizer,
                valid_targets / normalizer,
                p=1,
            )
            row, column = linear_sum_assignment(cost.detach().cpu().numpy())
            pred_index = torch.as_tensor(row, device=pred_boxes.device, dtype=torch.long)
            target_index = torch.as_tensor(column, device=pred_boxes.device, dtype=torch.long)
            presence_targets[batch_index, pred_index] = 1.0
            selected_pred = pred_boxes[batch_index, pred_index]
            selected_target = valid_targets[target_index]
            delta = selected_pred - selected_target
            delta_yaw = torch.atan2(torch.sin(delta[:, 2]), torch.cos(delta[:, 2]))
            delta = torch.cat((delta[:, :2], delta_yaw[:, None], delta[:, 3:]), dim=-1)
            box_total = box_total + F.smooth_l1_loss(
                delta / normalizer, torch.zeros_like(delta), reduction="sum"
            )
            matched_total += len(row)
        presence_loss = F.binary_cross_entropy_with_logits(
            pred_logits, presence_targets
        )
        box_loss = box_total / max(matched_total * 5, 1)
        return {
            "loss_current_agent_presence": presence_loss,
            "loss_current_agent_box": box_loss,
        }


def current_semantic_map_loss(logits, target, valid):
    """Masked multi-label focal loss for the four current-scene layers."""
    if logits.ndim != 4 or logits.shape[1:] != (4, 160, 160):
        raise ValueError("current map logits must have shape [B,4,160,160]")
    if target.shape != logits.shape or valid.shape != (logits.shape[0], 160, 160):
        raise ValueError("current map target or validity mask has wrong shape")
    truth = target.to(device=logits.device, dtype=logits.dtype)
    mask = valid.to(device=logits.device, dtype=logits.dtype)[:, None]
    return masked_sigmoid_focal_loss(logits, truth, mask)

"""HD465 trajectory-query front end adapted from the WoTE design.

Architecture reference: WoTE/navsim/agents/WoTE/WoTE_model.py (Apache-2.0).
This is a mining-specific reimplementation, not a verbatim NAVSIM port. It
uses the local TransFuser backbone and the local 256x256 / 8x8 LiDAR geometry.
The reward stage is implemented below in the same module.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class WoTEMiningTrajectoryHead(nn.Module):
    """Refine fixed [K,8,3] anchors against fused 8x8 LiDAR features."""

    def __init__(self, anchors_path, hidden_dim=256, layers=2, heads=8):
        super().__init__()
        anchors = np.load(Path(anchors_path), allow_pickle=False)
        if anchors.ndim != 3 or anchors.shape[1:] != (8, 3):
            raise ValueError("anchors must have shape [K,8,3]")
        if not np.isfinite(anchors).all():
            raise ValueError("anchors contain non-finite values")
        self.register_buffer("anchors", torch.from_numpy(anchors.astype(np.float32)))
        self.bev_downscale = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.bev_position = nn.Embedding(64, hidden_dim)
        self.anchor_encoder = nn.Sequential(
            nn.Linear(24, 128), nn.ReLU(), nn.Linear(128, hidden_dim)
        )
        self.anchor_context = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_dim, heads, hidden_dim * 2, dropout=0.1, batch_first=True
            ),
            num_layers=layers,
        )
        # Local mining status: speed (m/s) and route target (forward/right m).
        self.status_encoder = nn.Linear(3, hidden_dim)
        self.query_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.offset_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                hidden_dim, heads, hidden_dim * 2, dropout=0.1, batch_first=True
            ),
            num_layers=layers,
        )
        self.offset_head = nn.Linear(hidden_dim, 24)
        self.score_head = nn.Linear(hidden_dim, 1)
        # Start exactly on the fixed anchor distribution before training.
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def encode_trajectory_features(self, trajectories, speed, target_point):
        """Encode fixed or refined trajectories without BEV offset decoding.

        Source WoTE uses fixed-anchor features to train its world/reward
        models and re-encodes refined trajectories only for online inference.
        Keeping this operation explicit prevents cached fixed-anchor labels
        from being paired with a moving ``anchor + offset`` input.
        """
        batch, count = trajectories.shape[:2]
        trajectory_features = self.anchor_context(
            self.anchor_encoder(trajectories.flatten(2))
        )
        status = torch.cat((speed.reshape(batch, 1), target_point), dim=1)
        status_feature = self.status_encoder(status).unsqueeze(1).expand(
            -1, count, -1
        )
        return self.query_fusion(
            torch.cat((trajectory_features, status_feature), dim=-1)
        )

    def forward(self, fused_lidar, speed, target_point):
        if fused_lidar.ndim != 4 or fused_lidar.shape[1:] != (512, 8, 8):
            raise ValueError("fused_lidar must have shape [B,512,8,8]")
        batch = fused_lidar.shape[0]
        speed = speed.reshape(batch, 1)
        if target_point.shape != (batch, 2):
            raise ValueError("target_point must have shape [B,2]")

        bev_tokens = self.bev_downscale(fused_lidar).flatten(2).transpose(1, 2)
        bev_tokens = bev_tokens + self.bev_position.weight.unsqueeze(0)
        anchors = self.anchors.unsqueeze(0).expand(batch, -1, -1, -1)
        anchor_features = self.encode_trajectory_features(
            anchors, speed, target_point
        )
        decoded = self.offset_decoder(anchor_features, bev_tokens)
        offsets = self.offset_head(decoded).reshape_as(anchors)
        trajectories = anchors + offsets
        scores = self.score_head(decoded).squeeze(-1)
        return {
            "bev_tokens": bev_tokens,
            "anchors": anchors,
            "offsets": offsets,
            "trajectories": trajectories,
            "scores": scores,
            "anchor_features": anchor_features,
            "offset_features": decoded,
        }

    @staticmethod
    def imitation_loss(outputs, future_poses, yaw_weight=1.0):
        """Oracle-matched trajectory supervision for the first training stage.

        This is intentionally *not* WoTE's world-model/reward objective. It
        only trains candidate ranking and offsets against expert ego poses.
        """
        trajectories = outputs["trajectories"]
        anchors = outputs["anchors"]
        scores = outputs["scores"]
        if future_poses.shape != trajectories.shape[:1] + (8, 3):
            raise ValueError("future_poses must have shape [B,8,3]")
        weights = future_poses.new_tensor([1.0, 1.0, yaw_weight])
        distances = (((anchors - future_poses[:, None]) * weights) ** 2).sum((2, 3))
        labels = distances.argmin(dim=1)
        batch_indices = torch.arange(labels.shape[0], device=labels.device)
        chosen = trajectories[batch_indices, labels]
        return {
            "loss_trajectory": F.smooth_l1_loss(chosen * weights, future_poses * weights),
            "loss_selection": F.cross_entropy(scores, labels),
            "matched_anchor": labels,
        }


class WoTEMiningPlanner(nn.Module):
    """Connect the unchanged mining TransFuser fusion backbone to WoTE queries."""

    def __init__(self, backbone, anchors_path):
        super().__init__()
        self.backbone = backbone
        # These legacy TransFuser heads are downstream of the 8x8 fused LiDAR
        # map consumed by WoTE.  Freezing them makes the WoTE computation graph
        # valid under DDP without find_unused_parameters=True and also keeps
        # them out of AdamW's state (roughly 6 MiB saved per checkpoint).
        unused_output_heads = (
            backbone.change_channel_conv_image,
            backbone.c5_conv,
            backbone.up_conv5,
            backbone.up_conv4,
            backbone.up_conv3,
        )
        for module in unused_output_heads:
            module.requires_grad_(False)
        self.trajectory_head = WoTEMiningTrajectoryHead(anchors_path)
        self.world_model = WoTEMiningWorldModel(
            lidar_x=backbone.config.lidar_pos[0]
        )
        self.reward_head = WoTEMiningRewardHead(
            weights=getattr(backbone.config, "wote_reward_weights", (0.1, 0.5, 0.5, 1.0))
        )
        self.current_agent_head = WoTEMiningAgentHead(
            lidar_x=backbone.config.lidar_pos[0]
        )

    def forward(self, rgb, lidar_bev, speed, target_point,
                world_candidate_indices=None, augmentation_degrees=None,
                predict_future_map=False, future_map_candidate_indices=None,
                predict_auxiliary=True, use_refined_world=True):
        # WoTE asks the local backbone for its fused 512x8x8 LiDAR map before
        # the legacy TransFuser output heads.
        fused_lidar = self.backbone(
            rgb, lidar_bev, speed, return_fused_lidar=True
        )
        result = self.trajectory_head(fused_lidar, speed, target_point)
        if predict_auxiliary:
            current_map = result["bev_tokens"].transpose(1, 2).reshape(
                result["bev_tokens"].shape[0], 256, 8, 8
            )
            result["current_map_logits"] = self.world_model.map_head(current_map)[:, :4]
            result.update(self.current_agent_head(result["bev_tokens"]))
        if use_refined_world:
            world_trajectories = result["trajectories"]
            world_action_features = self.trajectory_head.encode_trajectory_features(
                world_trajectories, speed, target_point
            )
        else:
            # Training and validation labels are cached for the fixed anchors.
            world_trajectories = result["anchors"]
            world_action_features = result["anchor_features"]
        world = self.world_model(
            result["bev_tokens"], world_action_features,
            world_trajectories, candidate_indices=world_candidate_indices,
            augmentation_degrees=augmentation_degrees,
            predict_future_map=(
                predict_future_map and future_map_candidate_indices is None
            ),
        )
        result.update(world)
        if predict_future_map and future_map_candidate_indices is not None:
            result.update(self.world_model.decode_future_map(
                world, future_map_candidate_indices
            ))
        if world_candidate_indices is None:
            scored_action_features = world_action_features
        else:
            scored_action_features = world_action_features.gather(
                1, world_candidate_indices.unsqueeze(-1).expand(
                    -1, -1, world_action_features.shape[-1]
                )
            )
        rewards = self.reward_head(
            result["bev_tokens"], world["future_bev_tokens"],
            scored_action_features, world["future_action_features"],
        )
        result.update(rewards)
        result.update(select_best_trajectory(
            rewards["final_rewards"], world["world_trajectories"],
            world_candidate_indices,
        ))
        return result
