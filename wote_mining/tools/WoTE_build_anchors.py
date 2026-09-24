#!/usr/bin/env python3
"""Build WoTE-style trajectory anchors from HD465 training routes only.

Trajectories contain eight future poses at 0.5 s intervals. Each pose is
relative to the current vehicle frame, not the virtual LiDAR frame used by the
existing TransFuser waypoint target. Positive x is forward and positive y is
right in the CARLA vehicle frame.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans


EXPECTED_SPLITS = {"train": 120, "val": 20, "test": 20}
FUTURE_STEPS = 8
INTERVAL_SECONDS = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-anchors", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--heading-weight", type=float, default=1.0,
        help="Meters per radian used only for K-means distance (WoTE-like default: 1).",
    )
    parser.add_argument(
        "--max-train-routes", type=int, default=None,
        help="Smoke-test limit; only the first N training routes are used.",
    )
    return parser.parse_args()


def load_assignments(path):
    assignments = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["source_xml_file"], int(row["source_route_ordinal"]))
            if key in assignments:
                raise ValueError("duplicate assignment: %r" % (key,))
            assignments[key] = row["split"]
    counts = Counter(assignments.values())
    if dict(counts) != EXPECTED_SPLITS:
        raise ValueError("unexpected split sizes: %r" % dict(counts))
    return assignments


def route_key(route):
    match = re.search(r"(?:^|_)route(\d+)(?:_|$)", route.name)
    if match is None:
        raise ValueError("route ordinal missing: %s" % route)
    return (route.parent.parent.name + route.parent.name + ".xml", int(match.group(1)))


def load_ego_matrices(route):
    files = sorted((route / "label_raw").glob("*.json"))
    matrices = []
    ids = []
    for path in files:
        if not path.stem.isdigit():
            raise ValueError("non-numeric label filename: %s" % path)
        with path.open(encoding="utf-8") as stream:
            labels = json.load(stream)
        if not labels or "ego_matrix" not in labels[0]:
            raise ValueError("ego_matrix missing: %s" % path)
        matrix = np.asarray(labels[0]["ego_matrix"], dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("invalid ego_matrix: %s" % path)
        ids.append(int(path.stem))
        matrices.append(matrix)
    return ids, np.asarray(matrices, dtype=np.float64)


def extract_route_trajectories(route):
    ids, matrices = load_ego_matrices(route)
    if not ids:
        raise ValueError("empty label directory: %s" % route)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate frame numbers: %s" % route)
    windows = []
    for start in range(len(ids) - FUTURE_STEPS):
        if ids[start + FUTURE_STEPS] - ids[start] != FUTURE_STEPS:
            continue
        relative = np.linalg.inv(matrices[start]) @ matrices[start + 1:start + FUTURE_STEPS + 1]
        poses = np.empty((FUTURE_STEPS, 3), dtype=np.float32)
        poses[:, :2] = relative[:, :2, 3]
        headings = np.arctan2(relative[:, 1, 0], relative[:, 0, 0])
        poses[:, 2] = np.unwrap(np.r_[0.0, headings])[1:]
        if not np.isfinite(poses).all():
            raise ValueError("non-finite trajectory: %s frame %d" % (route, ids[start]))
        windows.append(poses)
    return np.asarray(windows, dtype=np.float32), len(ids)


def draw_trajectories(trajectories, title, path, color, max_lines=None, seed=0):
    if max_lines is not None and len(trajectories) > max_lines:
        indices = np.random.default_rng(seed).choice(len(trajectories), max_lines, replace=False)
        trajectories = trajectories[indices]
    fig, ax = plt.subplots(figsize=(8, 8))
    for poses in trajectories:
        xy = np.vstack((np.zeros((1, 2), dtype=np.float32), poses[:, :2]))
        ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.25, linewidth=0.8)
    ax.scatter([0], [0], c="black", s=35, label="ego at t=0")
    ax.set(xlabel="forward x (m)", ylabel="right y (m)", title=title)
    ax.axis("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    if args.num_anchors < 1 or args.batch_size < 1 or args.heading_weight <= 0:
        raise ValueError("num-anchors, batch-size and heading-weight must be positive")
    data_root = args.data_root.resolve(strict=True)
    assignments_path = args.assignments.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir == data_root or data_root in output_dir.parents:
        raise ValueError("output-dir must not be inside the raw dataset")
    output_names = (
        "trajectory_anchors_%d.npy" % args.num_anchors,
        "trajectory_anchors_%d.png" % args.num_anchors,
        "trajectory_examples.png",
        "trajectory_anchor_summary.json",
    )
    if any((output_dir / name).exists() for name in output_names):
        raise FileExistsError("refusing to overwrite files in %s" % output_dir)

    assignments = load_assignments(assignments_path)
    routes = sorted(path for path in data_root.glob("*/*/*") if path.is_dir())
    found = {route_key(route): route for route in routes}
    if len(routes) != len(assignments) or set(found) != set(assignments):
        raise ValueError("raw route folders do not match the split-v2 assignment")
    train_routes = [found[key] for key, split in sorted(assignments.items()) if split == "train"]
    if args.max_train_routes is not None:
        if args.max_train_routes < 1:
            raise ValueError("max-train-routes must be positive")
        train_routes = train_routes[:args.max_train_routes]

    all_trajectories = []
    total_frames = 0
    for index, route in enumerate(train_routes, start=1):
        trajectories, frames = extract_route_trajectories(route)
        all_trajectories.append(trajectories)
        total_frames += frames
        print("[%d/%d] %s: %d frames, %d windows" %
              (index, len(train_routes), route.name, frames, len(trajectories)), flush=True)
    trajectories = np.concatenate(all_trajectories, axis=0)
    if len(trajectories) < args.num_anchors:
        raise ValueError("fewer trajectories than anchors")

    fit_data = trajectories.reshape(len(trajectories), -1).copy()
    fit_data[:, 2::3] *= args.heading_weight
    kmeans = MiniBatchKMeans(
        n_clusters=args.num_anchors,
        random_state=args.seed,
        batch_size=args.batch_size,
        n_init=3,
    )
    kmeans.fit(fit_data)
    centers = kmeans.cluster_centers_.reshape(args.num_anchors, FUTURE_STEPS, 3)
    centers[:, :, 2] /= args.heading_weight
    centers = centers.astype(np.float32)

    nearest = centers[kmeans.labels_]
    position_errors = np.linalg.norm(trajectories[:, :, :2] - nearest[:, :, :2], axis=-1)
    endpoint_errors = position_errors[:, -1]
    cluster_counts = np.bincount(kmeans.labels_, minlength=args.num_anchors)
    summary = {
        "data_root": str(data_root),
        "assignments": str(assignments_path),
        "assignments_sha256": hashlib.sha256(assignments_path.read_bytes()).hexdigest(),
        "train_routes": len(train_routes),
        "train_frames": total_frames,
        "train_windows_4s": len(trajectories),
        "coordinate_frame": "current vehicle, x forward, y right, yaw radians",
        "sampling_seconds": INTERVAL_SECONDS,
        "future_steps": FUTURE_STEPS,
        "horizon_seconds": FUTURE_STEPS * INTERVAL_SECONDS,
        "num_anchors": args.num_anchors,
        "seed": args.seed,
        "heading_weight_m_per_rad": args.heading_weight,
        "cluster_size_min": int(cluster_counts.min()),
        "cluster_size_median": float(np.median(cluster_counts)),
        "cluster_size_max": int(cluster_counts.max()),
        "mean_position_error_m": float(position_errors.mean()),
        "p95_position_error_m": float(np.percentile(position_errors, 95)),
        "mean_endpoint_error_m": float(endpoint_errors.mean()),
        "p95_endpoint_error_m": float(np.percentile(endpoint_errors, 95)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / output_names[0], centers)
    draw_trajectories(centers, "%d HD465 trajectory anchors (4 s)" % args.num_anchors,
                      output_dir / output_names[1], "#1767b1")
    draw_trajectories(trajectories, "Sample training trajectories (4 s)",
                      output_dir / output_names[2], "#b45f21", max_lines=300, seed=args.seed)
    (output_dir / output_names[3]).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
