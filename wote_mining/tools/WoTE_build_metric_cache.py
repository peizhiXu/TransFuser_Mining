#!/usr/bin/env python3
"""Precompute WoTE-style five-metric labels for HD465 trajectory anchors."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wote_mining.tools.WoTE_build_anchors import (
    load_assignments, route_key,
)
from wote_mining.WoTE_targets import recorded_vehicle_tracks, METRIC_NAMES
from wote_mining.WoTE_simulator import evaluate_mining_candidates
from wote_mining.WoTE_simulator import (
    dense_route_file_for_collection, local_route_window,
)
from wote_mining.WoTE_targets import decode_topdown_scene


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--routes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--max-routes", type=int, default=None)
    parser.add_argument("--max-frames-per-route", type=int, default=None)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Frames evaluated concurrently within each route.",
    )
    return parser.parse_args()


def _load_labels(route, frame):
    labels = []
    for index in range(frame, frame + 9):
        path = route / "label_raw" / ("%04d.json" % index)
        labels.append(json.loads(path.read_text(encoding="utf-8")))
    return labels


def _valid_frames(route):
    # Match CARLA_Data exactly: seq_len=1, pred_len=8, two discarded frames at
    # each end plus its historical first-two-frame exclusion.
    count = len(list((route / "lidar").glob("*.npy")))
    return list(range(2, count - 8 - 1 - 2))


def _evaluate_frame(route, frame, anchors, route_world_xy):
    labels = _load_labels(route, frame)
    ego_id = labels[0][0]["id"]
    ego = next(item for item in labels[0] if item.get("id") == ego_id)
    ego_matrix = np.asarray(ego["ego_matrix"], dtype=np.float64)
    route_window = local_route_window(route_world_xy, ego_matrix)
    route_xy = route_window["route_xy"][route_window["route_mask"] > 0]
    topdown_path = route / "topdown" / ("encoded_%04d.png" % frame)
    topdown = cv2.imread(str(topdown_path), cv2.IMREAD_COLOR)
    if topdown is None:
        raise FileNotFoundError(str(topdown_path))
    road = decode_topdown_scene(topdown)[0]
    tracks = recorded_vehicle_tracks(ego_matrix, labels)
    result = evaluate_mining_candidates(
        anchors, road, route_xy, recorded_tracks=tracks
    )
    return result["metric_targets"], result["metric_valid"]


def _write_manifest(output_dir, args, anchors_sha256, routes):
    total_frames = sum(item["frames"] for item in routes.values())
    target_counts = np.sum(
        [np.asarray(item["target_positive_counts"], dtype=np.int64)
         for item in routes.values()], axis=0,
    ) if routes else np.zeros(5, dtype=np.int64)
    target_sums = np.sum(
        [np.asarray(item["target_sums"], dtype=np.float64)
         for item in routes.values()], axis=0,
    ) if routes else np.zeros(5, dtype=np.float64)
    valid_counts = np.sum(
        [np.asarray(item["valid_counts"], dtype=np.int64)
         for item in routes.values()], axis=0,
    ) if routes else np.zeros(5, dtype=np.int64)
    manifest = {
        "format_version": 1,
        "split": args.split,
        "data_root": str(args.data_root.resolve()),
        "assignments": str(args.assignments.resolve()),
        "assignments_sha256": hashlib.sha256(
            args.assignments.read_bytes()
        ).hexdigest(),
        "anchors": str(args.anchors.resolve()),
        "anchors_sha256": anchors_sha256,
        "anchor_count": 256,
        "metric_names": list(METRIC_NAMES),
        "route_count": len(routes),
        "frame_count": total_frames,
        "valid_counts": valid_counts.tolist(),
        "target_positive_counts": target_counts.tolist(),
        "target_sums": target_sums.tolist(),
        "routes": routes,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.max_routes is not None and args.max_routes < 1:
        raise ValueError("max-routes must be positive")
    if args.max_frames_per_route is not None and args.max_frames_per_route < 1:
        raise ValueError("max-frames-per-route must be positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.data_root = args.data_root.resolve(strict=True)
    args.assignments = args.assignments.resolve(strict=True)
    args.anchors = args.anchors.resolve(strict=True)
    args.routes_dir = args.routes_dir.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty %s" % args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    anchors = np.load(args.anchors).astype(np.float32)
    if anchors.shape != (256, 8, 3) or not np.isfinite(anchors).all():
        raise ValueError("anchors must be finite [256,8,3]")
    anchors_sha256 = hashlib.sha256(args.anchors.read_bytes()).hexdigest()
    assignments = load_assignments(args.assignments)
    raw_routes = sorted(
        path for path in args.data_root.glob("*/*/*") if path.is_dir()
    )
    found = {route_key(route): route for route in raw_routes}
    if set(found) != set(assignments):
        raise ValueError("raw route folders do not match split assignments")
    selected = [
        found[key] for key, split in sorted(assignments.items())
        if split == args.split
    ]
    if args.max_routes is not None:
        selected = selected[:args.max_routes]

    route_entries = {}
    for route_index, route in enumerate(selected, start=1):
        started = time.monotonic()
        route_artifact = dense_route_file_for_collection(route, args.routes_dir)
        route_world_xy = np.load(route_artifact)["world_xy"].astype(np.float32)
        frames = _valid_frames(route)
        if args.max_frames_per_route is not None:
            frames = frames[:args.max_frames_per_route]
        targets, valid = [], []
        evaluate = partial(
            _evaluate_frame, route, anchors=anchors,
            route_world_xy=route_world_xy,
        )
        if args.workers == 1:
            evaluated = map(evaluate, frames)
        else:
            executor = ThreadPoolExecutor(max_workers=args.workers)
            evaluated = executor.map(evaluate, frames)
        for item_index, (item_targets, item_valid) in enumerate(
                evaluated, start=1):
            targets.append(item_targets)
            valid.append(item_valid)
            if item_index == 1 or item_index % 25 == 0 or item_index == len(frames):
                print("[%d/%d] %s frame %d/%d" % (
                    route_index, len(selected), route.name,
                    item_index, len(frames),
                ), flush=True)
        if args.workers != 1:
            executor.shutdown()
        targets = np.asarray(targets, dtype=np.float32)
        valid = np.asarray(valid, dtype=np.uint8)
        shard_name = route.name + ".npz"
        np.savez_compressed(
            args.output_dir / shard_name,
            frames=np.asarray(frames, dtype=np.int32),
            metric_targets=targets,
            metric_valid=valid,
        )
        route_entries[route.name] = {
            "file": shard_name,
            "frames": len(frames),
            "valid_counts": valid.sum(axis=(0, 1)).astype(int).tolist(),
            "target_positive_counts": (
                ((targets > 0.5) & valid.astype(bool))
                .sum(axis=(0, 1)).astype(int).tolist()
            ),
            "target_sums": (
                (targets * valid).sum(axis=(0, 1)).astype(float).tolist()
            ),
            "seconds": round(time.monotonic() - started, 3),
        }
        _write_manifest(
            args.output_dir, args, anchors_sha256, route_entries
        )
    print((args.output_dir / "manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
