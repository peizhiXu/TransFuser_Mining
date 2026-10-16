"""WoTE mining settings layered on the existing HD465 TransFuser settings."""

from pathlib import Path

from team_code_transfuser.config import GlobalConfig


class WoTEMiningConfig(GlobalConfig):
    # This branch's baseline uses only current BEV, offsets and candidate scores.
    wote_planner_mode = "current_only"
    pred_len = 8  # 4 seconds at the dataset's 2 Hz recording rate
    # Cached candidate scores correspond to the unrotated 256 anchors. The
    # legacy TransFuser rotation augmentation changes the scene/trajectory
    # coordinate frame but cannot rotate those cached per-anchor labels.
    # Disable it until labels are recomputed online under augmentation.
    augment = False
    wote_future_poses = True
    wote_future_scene_targets = True
    wote_dense_routes_dir = str(
        Path(__file__).resolve().parent / "assets/routes"
    )
    # WoTE-style offline simulator scores for all split-v2 training frames.
    wote_metric_cache_dir = str(
        Path(__file__).resolve().parent / "assets/metric_cache"
    )
    num_traj_anchors = 256
    wote_hidden_dim = 256
    wote_reward_weights = (0.1, 0.5, 0.5, 1.0)
    # Preserve existing runs; explicitly opt into the four-head experiment
    # with --reward-scheme mining_boundary after checking label statistics.
    wote_reward_scheme = "source"
    wote_boundary_target_m = 0.6  # pilot setting, to be checked on train labels
    wote_boundary_reward_weight = 0.2  # bounded, soft preference (not a veto)
    wote_ttc_reward_weight = 0.2  # mining_boundary_ttc only; bounded safety bonus
    # Source WoTE training defaults (configs/default.py).
    wote_num_future_map_candidates = 1
    wote_traj_offset_loss_weight = 1.0
    # Keep the original WTA oracle loss; add at most four compatible,
    # reward-preferred modes directly on the final AdaLN decoded trajectories.
    wote_reward_topk_loss_weight = 0.25
    wote_reward_topk = 4
    wote_reward_topk_endpoint_max_m = 2.0
    wote_reward_topk_ade_max_m = 1.0
    wote_offset_imitation_loss_weight = 0.1
    wote_imitation_reward_loss_weight = 1.0
    wote_metric_reward_loss_weight = 1.0
    wote_current_map_loss_weight = 10.0
    wote_future_map_loss_weight = 0.1
    # The source defaults these auxiliary agent weights to zero unless an
    # experiment overrides them. Keep the head and losses available without
    # silently changing the paper's optimization objective.
    wote_agent_presence_loss_weight = 0.0
    wote_agent_box_loss_weight = 0.0
    wote_lr = 1e-4
    wote_min_lr = 1e-6
    wote_weight_decay = 1e-4
    wote_image_encoder_lr_multiplier = 0.1
    wote_warmup_epochs = 3
    wote_max_epochs = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_planner_mode(self.wote_planner_mode)

    def set_planner_mode(self, mode):
        if mode not in ("current_only", "future_reward"):
            raise ValueError("unknown planner mode: %s" % mode)
        self.wote_planner_mode = mode
        for name, disabled in (
            ("wote_future_scene_targets", False),
            ("wote_metric_cache_dir", None),
            ("wote_dense_routes_dir", None),
            ("wote_reward_topk", 0),
            ("wote_reward_topk_loss_weight", 0.0),
            ("wote_imitation_reward_loss_weight", 0.0),
            ("wote_metric_reward_loss_weight", 0.0),
            ("wote_current_map_loss_weight", 0.0),
            ("wote_future_map_loss_weight", 0.0),
            ("wote_agent_presence_loss_weight", 0.0),
            ("wote_agent_box_loss_weight", 0.0),
        ):
            setattr(self, name, disabled if mode == "current_only" else getattr(type(self), name))
