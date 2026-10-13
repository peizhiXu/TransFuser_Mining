"""Explicit reward schemas; legacy checkpoints always keep their old meaning."""

import math


SOURCE_METRIC_NAMES = (
    "no_collision", "drivable_compliance", "ego_progress",
    "time_to_collision", "comfort",
)
BOUNDARY_METRIC_NAMES = SOURCE_METRIC_NAMES[:3] + ("road_boundary_margin",)
BOUNDARY_TTC_METRIC_NAMES = BOUNDARY_METRIC_NAMES + ("time_to_collision",)
REWARD_SCHEMES = ("source", "mining_boundary", "mining_boundary_ttc")
BOUNDARY_SCHEMES = ("mining_boundary", "mining_boundary_ttc")
BOUNDARY_LABEL_VERSION = "future_footprint_saturated_min_v2"
TTC_LABEL_VERSION = "recorded_agents_ego_extrapolation_1s_v1"


def metric_names_for_scheme(scheme):
    if scheme == "source":
        return SOURCE_METRIC_NAMES
    if scheme == "mining_boundary":
        return BOUNDARY_METRIC_NAMES
    if scheme == "mining_boundary_ttc":
        return BOUNDARY_TTC_METRIC_NAMES
    raise ValueError("unknown reward scheme: %s" % scheme)


def validate_boundary_settings(target_m, weight):
    if not math.isfinite(float(target_m)) or target_m <= 0:
        raise ValueError("boundary target clearance must be finite and positive")
    if not math.isfinite(float(weight)) or weight < 0:
        raise ValueError("boundary reward weight must be finite and nonnegative")


def validate_ttc_weight(weight):
    if not math.isfinite(float(weight)) or weight < 0:
        raise ValueError("TTC reward weight must be finite and nonnegative")


def restore_reward_settings(config, saved_args, saved_config=None):
    """No scheme in an old checkpoint means source, never the CLI default."""
    saved = (saved_config or {}).get("reward_scheme_settings", {})
    config.wote_reward_scheme = saved.get("scheme", saved_args.get("reward_scheme", "source"))
    config.wote_boundary_target_m = float(saved.get(
        "boundary_target_m", saved_args.get("boundary_target_m", 0.6)))
    config.wote_boundary_reward_weight = float(saved.get(
        "boundary_reward_weight", saved_args.get("boundary_reward_weight", 0.2)))
    metric_names_for_scheme(config.wote_reward_scheme)
    validate_boundary_settings(config.wote_boundary_target_m, config.wote_boundary_reward_weight)
    if config.wote_reward_scheme in BOUNDARY_SCHEMES:
        version = saved.get("label_version", saved_args.get("boundary_label_version"))
        if version != BOUNDARY_LABEL_VERSION:
            raise ValueError("boundary checkpoint label version mismatch: %r, expected %r" % (
                version, BOUNDARY_LABEL_VERSION))
    config.wote_ttc_reward_weight = float(saved.get(
        "ttc_reward_weight", saved_args.get("ttc_reward_weight", 0.2)))
    validate_ttc_weight(config.wote_ttc_reward_weight)
    if config.wote_reward_scheme == "mining_boundary_ttc":
        version = saved.get("ttc_label_version", saved_args.get("ttc_label_version"))
        if version != TTC_LABEL_VERSION:
            raise ValueError("TTC checkpoint label version mismatch: %r" % version)


def reward_settings(config):
    settings = dict(
        scheme=getattr(config, "wote_reward_scheme", "source"),
        boundary_target_m=getattr(config, "wote_boundary_target_m", 0.6),
        boundary_reward_weight=getattr(config, "wote_boundary_reward_weight", 0.2),
    )
    if settings["scheme"] in BOUNDARY_SCHEMES:
        settings["label_version"] = BOUNDARY_LABEL_VERSION
    if settings["scheme"] == "mining_boundary_ttc":
        settings["ttc_reward_weight"] = getattr(config, "wote_ttc_reward_weight", 0.2)
        settings["ttc_label_version"] = TTC_LABEL_VERSION
    return settings
