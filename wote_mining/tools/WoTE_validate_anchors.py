#!/usr/bin/env python3
"""Measure held-out WoTE anchor coverage on HD465 split-v2 routes.

This is an oracle *anchor coverage* check, not a model prediction score. Each
expert trajectory is matched to its nearest fixed anchor using the same
24-dimensional distance used during clustering.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

from WoTE_build_anchors import (
    FUTURE_STEPS,
    extract_route_trajectories,
    load_assignments,
    route_key,
)


def describe(values):
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def evaluate(trajectories, anchors, heading_weight):
    expert_flat = trajectories.reshape(len(trajectories), -1).copy()
    anchor_flat = anchors.reshape(len(anchors), -1).copy()
    expert_flat[:, 2::3] *= heading_weight
    anchor_flat[:, 2::3] *= heading_weight
    nearest_indices, _ = pairwise_distances_argmin_min(expert_flat, anchor_flat)
    matched = anchors[nearest_indices]
    position_error = np.linalg.norm(trajectories[:, :, :2] - matched[:, :, :2], axis=-1)
    yaw_difference = trajectories[:, :, 2] - matched[:, :, 2]
    yaw_error = np.abs(np.arctan2(np.sin(yaw_difference), np.cos(yaw_difference)))
    endpoint_displacement = np.linalg.norm(trajectories[:, -1, :2], axis=-1)
    endpoint_yaw = np.abs(trajectories[:, -1, 2])
    # These groups overlap: a slow trajectory can also turn.
    groups = {
        "all": np.ones(len(trajectories), dtype=bool),
        "near_stationary_endpoint_lt_1m": endpoint_displacement < 1.0,
        "turn_abs_final_yaw_ge_0p2rad": endpoint_yaw >= 0.2,
        "large_lateral_abs_final_y_ge_2m": np.abs(trajectories[:, -1, 1]) >= 2.0,
        "other": (endpoint_displacement >= 1.0)
        & (endpoint_yaw < 0.2)
        & (np.abs(trajectories[:, -1, 1]) < 2.0),
    }
    report = {}
    for name, mask in groups.items():
        if not mask.any():
            report[name] = {"count": 0}
            continue
        mean_step_error = position_error[mask].mean(axis=1)
        endpoint_error = position_error[mask, -1]
        report[name] = {
            "count": int(mask.sum()),
            "mean_step_position_error_m": describe(mean_step_error),
            "endpoint_position_error_m": describe(endpoint_error),
            "mean_step_yaw_error_deg": describe(np.degrees(yaw_error[mask].mean(axis=1))),
            "endpoint_error_lt_1m_fraction": float(np.mean(endpoint_error < 1.0)),
            "mean_step_error_lt_1m_fraction": float(np.mean(mean_step_error < 1.0)),
        }
    return report, nearest_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--heading-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.heading_weight <= 0:
        raise ValueError("heading-weight must be positive")

    data_root = args.data_root.resolve(strict=True)
    assignments_path = args.assignments.resolve(strict=True)
    anchors_path = args.anchors.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite %s" % output_path)
    if output_path == data_root or data_root in output_path.parents:
        raise ValueError("output must not be inside the raw dataset")

    anchors = np.load(anchors_path, allow_pickle=False)
    if anchors.ndim != 3 or anchors.shape[1:] != (FUTURE_STEPS, 3):
        raise ValueError("anchors must have shape [K, 8, 3]")
    if not np.isfinite(anchors).all():
        raise ValueError("non-finite anchor")

    assignments = load_assignments(assignments_path)
    routes = sorted(path for path in data_root.glob("*/*/*") if path.is_dir())
    found = {route_key(route): route for route in routes}
    if len(routes) != len(assignments) or set(found) != set(assignments):
        raise ValueError("raw route folders do not match split-v2 assignments")
    split_routes = [found[key] for key, split in sorted(assignments.items()) if split == args.split]
    trajectory_parts = []
    route_windows = []
    for index, route in enumerate(split_routes, start=1):
        part, frames = extract_route_trajectories(route)
        if not len(part):
            raise ValueError("route has no 4-second windows: %s" % route)
        trajectory_parts.append(part)
        route_windows.append((route.name, len(part), frames))
        print("[%d/%d] %s: %d windows" %
              (index, len(split_routes), route.name, len(part)), flush=True)
    trajectories = np.concatenate(trajectory_parts)
    group_metrics, nearest_indices = evaluate(trajectories, anchors, args.heading_weight)
    anchor_usage = np.bincount(nearest_indices, minlength=len(anchors))
    route_metrics = []
    offset = 0
    for name, count, frames in route_windows:
        metrics, _ = evaluate(trajectories[offset:offset + count], anchors, args.heading_weight)
        route_metrics.append({
            "route": name,
            "frames": frames,
            "windows": count,
            "mean_endpoint_error_m": metrics["all"]["endpoint_position_error_m"]["mean"],
            "p95_endpoint_error_m": metrics["all"]["endpoint_position_error_m"]["p95"],
        })
        offset += count

    report = {
        "split": args.split,
        "route_count": len(split_routes),
        "window_count": len(trajectories),
        "anchor_count": len(anchors),
        "matching_distance": "flattened 8 x (x, y, heading_weight * yaw), Euclidean",
        "heading_weight_m_per_rad": args.heading_weight,
        "assignments_sha256": hashlib.sha256(assignments_path.read_bytes()).hexdigest(),
        "anchors_sha256": hashlib.sha256(anchors_path.read_bytes()).hexdigest(),
        "anchor_usage": {
            "used_anchors": int(np.count_nonzero(anchor_usage)),
            "unused_anchors": int(np.count_nonzero(anchor_usage == 0)),
            "top_five_counts": sorted(anchor_usage.tolist(), reverse=True)[:5],
        },
        "groups": group_metrics,
        "routes_by_mean_endpoint_error_desc": sorted(
            route_metrics, key=lambda row: row["mean_endpoint_error_m"], reverse=True
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("split", "route_count", "window_count", "anchor_usage", "groups")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
