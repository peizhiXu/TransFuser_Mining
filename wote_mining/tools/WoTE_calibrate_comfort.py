#!/usr/bin/env python3
"""Calibrate WoTE-style Comfort limits from HD465 training trajectories."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wote_mining.tools.WoTE_build_anchors import (
    extract_route_trajectories, load_assignments, route_key,
)
from wote_mining.WoTE_simulator import (
    SOURCE_WOTE_COMFORT_LIMITS, evaluate_comfort,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-percent", type=float, default=0.5)
    parser.add_argument("--max-train-routes", type=int, default=None)
    return parser.parse_args()


def _round_upper(value):
    return math.ceil(float(value) * 100.0) / 100.0


def _round_lower(value):
    return math.floor(float(value) * 100.0) / 100.0


def main():
    args = parse_args()
    if not 0.0 < args.tail_percent < 10.0:
        raise ValueError("tail-percent must be in (0,10)")
    if args.max_train_routes is not None and args.max_train_routes < 1:
        raise ValueError("max-train-routes must be positive")
    data_root = args.data_root.resolve(strict=True)
    assignments_path = args.assignments.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite %s" % output)

    assignments = load_assignments(assignments_path)
    routes = sorted(path for path in data_root.glob("*/*/*") if path.is_dir())
    found = {route_key(route): route for route in routes}
    if set(found) != set(assignments):
        raise ValueError("raw route folders do not match split assignments")
    train_routes = [
        found[key] for key, split in sorted(assignments.items()) if split == "train"
    ]
    if args.max_train_routes is not None:
        train_routes = train_routes[:args.max_train_routes]

    keys = (
        "min_longitudinal_acceleration_mps2",
        "max_longitudinal_acceleration_mps2",
        "max_abs_lateral_acceleration_mps2",
        "max_abs_magnitude_jerk_mps3",
        "max_abs_longitudinal_jerk_mps3",
        "max_abs_yaw_acceleration_radps2",
        "max_abs_yaw_rate_radps",
    )
    values = {key: [] for key in keys}
    for index, route in enumerate(train_routes, start=1):
        trajectories, _ = extract_route_trajectories(route)
        result = evaluate_comfort(
            trajectories, limits=SOURCE_WOTE_COMFORT_LIMITS
        )
        for key in keys:
            values[key].append(result[key])
        print("[%d/%d] %s: %d windows" % (
            index, len(train_routes), route.name, len(trajectories)
        ), flush=True)
    values = {key: np.concatenate(parts) for key, parts in values.items()}
    lower = args.tail_percent
    upper = 100.0 - args.tail_percent
    raw_limits = {
        "min_longitudinal_acceleration_mps2": float(np.percentile(
            values["min_longitudinal_acceleration_mps2"], lower
        )),
        "max_longitudinal_acceleration_mps2": float(np.percentile(
            values["max_longitudinal_acceleration_mps2"], upper
        )),
        "max_abs_lateral_acceleration_mps2": float(np.percentile(
            values["max_abs_lateral_acceleration_mps2"], upper
        )),
        "max_abs_magnitude_jerk_mps3": float(np.percentile(
            values["max_abs_magnitude_jerk_mps3"], upper
        )),
        "max_abs_longitudinal_jerk_mps3": float(np.percentile(
            values["max_abs_longitudinal_jerk_mps3"], upper
        )),
        "max_abs_yaw_acceleration_radps2": float(np.percentile(
            values["max_abs_yaw_acceleration_radps2"], upper
        )),
        "max_abs_yaw_rate_radps": float(np.percentile(
            values["max_abs_yaw_rate_radps"], upper
        )),
    }
    recommended = {
        key: (_round_lower(value) if key.startswith("min_") else _round_upper(value))
        for key, value in raw_limits.items()
    }
    component_compliance = np.stack((
        ((values["min_longitudinal_acceleration_mps2"]
          > recommended["min_longitudinal_acceleration_mps2"])
         & (values["max_longitudinal_acceleration_mps2"]
            < recommended["max_longitudinal_acceleration_mps2"])),
        (values["max_abs_lateral_acceleration_mps2"]
         < recommended["max_abs_lateral_acceleration_mps2"]),
        (values["max_abs_magnitude_jerk_mps3"]
         < recommended["max_abs_magnitude_jerk_mps3"]),
        (values["max_abs_longitudinal_jerk_mps3"]
         < recommended["max_abs_longitudinal_jerk_mps3"]),
        (values["max_abs_yaw_acceleration_radps2"]
         < recommended["max_abs_yaw_acceleration_radps2"]),
        (values["max_abs_yaw_rate_radps"]
         < recommended["max_abs_yaw_rate_radps"]),
    ), axis=1)
    comfortable = component_compliance.all(axis=1)
    trajectories_count = len(next(iter(values.values())))
    summary = {
        "data_root": str(data_root),
        "assignments": str(assignments_path),
        "assignments_sha256": hashlib.sha256(
            assignments_path.read_bytes()
        ).hexdigest(),
        "train_routes": len(train_routes),
        "expert_windows_4s": trajectories_count,
        "sampling_seconds": 0.5,
        "tail_percent": args.tail_percent,
        "source_wote_limits": SOURCE_WOTE_COMFORT_LIMITS,
        "raw_training_quantile_limits": raw_limits,
        "recommended_mining_limits": recommended,
        "recommended_training_comfortable_windows": int(comfortable.sum()),
        "recommended_training_comfortable_fraction": float(comfortable.mean()),
        "recommended_training_failed_by_component": (
            (~component_compliance).sum(axis=0).tolist()
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
