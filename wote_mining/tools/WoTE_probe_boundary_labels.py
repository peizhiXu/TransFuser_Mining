#!/usr/bin/env python3
"""Read-only, route-stratified boundary-label probe before expensive training."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wote_mining.WoTE_simulator import evaluate_road_compliance
from wote_mining.WoTE_targets import decode_topdown_scene, future_ego_poses


def select_probe_frames(frames, targets, valid, count, selection):
    """Edge sampling is deliberately biased; never report it as prevalence."""
    if selection == "uniform":
        return np.unique(np.linspace(0, len(frames)-1, min(len(frames), count), dtype=int))
    if selection != "edge":
        raise ValueError("unknown frame selection")
    # Reuse SOURCE DAC labels only to locate potentially constrained maps.
    known = valid[..., 1].astype(bool)
    failed = known & (targets[..., 1] < 0.5)
    fraction = failed.sum(axis=1) / np.maximum(known.sum(axis=1), 1)
    return np.argsort(-fraction, kind="stable")[:min(len(frames), count)]


def compare_boundary_aggregation(clearance, target):
    """Compare temporal mean with worst FUTURE pose, keeping spatial min."""
    per_pose = np.clip(np.asarray(clearance) / target, 0, 1)
    return per_pose.mean(axis=1), per_pose.min(axis=1)


def save_boundary_preview(path, road, trajectory, clearances, target):
    """Plot data geometry, not a claim of alignment to physical road shoulders."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(road, cmap="gray", vmin=0, vmax=1,
                   extent=(-50, 50, -50, 50), origin="upper")
    # Plot horizontal right-y, vertical forward-x, matching the topdown raster.
    axes[0].plot(trajectory[:, 1], trajectory[:, 0], "c.-", label="candidate")
    worst = int(np.argmin(clearances))
    # Same interpolation as evaluate_road_compliance: four samples per interval.
    from wote_mining.WoTE_simulator import _interpolate_ego
    pose = _interpolate_ego(trajectory[None], 4)[0, 1:][worst]
    corners = np.array([[a, b] for a, b in [
        (-9.3969/2, -5.3645/2), (9.3969/2, -5.3645/2),
        (9.3969/2, 5.3645/2), (-9.3969/2, 5.3645/2)]])
    ca, sa = np.cos(pose[2]), np.sin(pose[2])
    corners = corners @ np.array([[ca, sa], [-sa, ca]]) + pose[:2]
    axes[0].add_patch(Polygon(corners[:, ::-1], fill=False, edgecolor="red", lw=1.5))
    points = np.r_[trajectory[:, :2], corners, [[0, 0]]]
    axes[0].set_xlim(points[:, 1].min()-5, points[:, 1].max()+5)
    axes[0].set_ylim(points[:, 0].min()-5, points[:, 0].max()+5)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("right y (m)")
    axes[0].set_ylabel("forward x (m)")
    axes[0].set_title("Road mask + HD465 footprint at minimum clearance")
    axes[1].plot(np.arange(1, len(clearances)+1)*0.125, clearances)
    axes[1].axhline(target, color="red", linestyle="--", label="trial margin")
    axes[1].set_xlabel("future time (s)")
    axes[1].set_ylabel("minimum footprint clearance (m)")
    axes[1].legend()
    mean, minimum = compare_boundary_aggregation(clearances[None], target)
    axes[1].set_title("mean score %.3f; worst-pose score %.3f" % (mean[0], minimum[0]))
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True, help="existing source cache root")
    parser.add_argument("--anchors", type=Path, default=ROOT / "wote_mining/assets/anchors/trajectory_anchors_256.npy")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--max-routes", type=int, default=20)
    parser.add_argument("--frames-per-route", type=int, default=3)
    parser.add_argument("--target-m", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--frame-selection", choices=("uniform", "edge"), default="uniform")
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--preview-limit", type=int, default=4)
    args = parser.parse_args()
    if (args.max_routes < 1 or args.frames_per_route < 1
            or any(not np.isfinite(t) or t <= 0 for t in args.target_m)):
        raise ValueError("sample counts and target clearances must be positive")
    manifest = json.loads((args.cache_dir / args.split / "manifest.json").read_text())
    if hashlib.sha256(args.anchors.read_bytes()).hexdigest() != manifest["anchors_sha256"]:
        raise ValueError("probe anchors differ from cache anchors")
    anchors = np.load(args.anchors)
    routes = {p.name: p for p in args.data_root.glob("*/*/*") if p.is_dir()}
    names = sorted(manifest["routes"])
    rng = np.random.RandomState(args.seed)
    names = [names[i] for i in rng.choice(len(names), min(len(names), args.max_routes), replace=False)]
    observations = {str(t): [] for t in args.target_m}
    minima = {str(t): [] for t in args.target_m}
    expert_clearances, expert_scores = [], {str(t): [] for t in args.target_m}
    frames_seen = 0
    preview_count = 0
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        route = routes[name]
        with np.load(args.cache_dir / args.split / manifest["routes"][name]["file"]) as shard:
            frames = shard["frames"]
            selected = select_probe_frames(frames, shard["metric_targets"], shard["metric_valid"],
                                           args.frames_per_route, args.frame_selection)
        for index in selected:
            frame = int(frames[index])
            road = decode_topdown_scene(cv2.imread(str(route / "topdown" / ("encoded_%04d.png" % frame))))[0]
            matrices = []
            for offset in range(9):
                labels = json.loads((route / "label_raw" / ("%04d.json" % (frame+offset))).read_text())
                matrices.append(labels[0]["ego_matrix"])
            expert = future_ego_poses(np.asarray(matrices))[None]
            result = evaluate_road_compliance(np.concatenate((anchors, expert)), road)
            eligible = ((result["drivable_compliance"] > 0) & (result["drivable_valid"] > 0)
                        & (result["road_boundary_valid"] > 0))
            clearance = result["future_road_clearance_m"]
            candidate_clearance = clearance[:-1][eligible[:-1]]
            frames_seen += 1
            if eligible[-1]:
                expert_clearances.append(clearance[-1].min())
            for target in args.target_m:
                scores, minimum = compare_boundary_aggregation(candidate_clearance, target)
                observations[str(target)].append(scores)
                minima[str(target)].append(minimum)
                if eligible[-1]:
                    expert_scores[str(target)].append(float(np.clip(clearance[-1]/target, 0, 1).mean()))
            if (args.preview_dir is not None and preview_count < args.preview_limit
                    and len(candidate_clearance)):
                anchor_ids = np.flatnonzero(eligible[:-1])
                nearest = int(anchor_ids[candidate_clearance.min(axis=1).argmin()])
                if clearance[nearest].min() < max(args.target_m):
                    filename = "%s_frame%04d_anchor%03d.png" % (name, frame, nearest)
                    save_boundary_preview(args.preview_dir / filename, road, anchors[nearest],
                                          clearance[nearest], args.target_m[0])
                    preview_count += 1
        print("probed %s (%d maps)" % (name, frames_seen), flush=True)
    summaries = {}
    for target, arrays in observations.items():
        values = np.concatenate(arrays)
        stds = [a.std() for a in arrays if len(a) >= 2]
        worst = np.concatenate(minima[target])
        worst_stds = [a.std() for a in minima[target] if len(a) >= 2]
        summaries[target] = dict(
            compliant_candidates=len(values),
            score_mean=float(values.mean()) if len(values) else None,
            saturated_fraction=float((values >= 1-1e-6).mean()) if len(values) else None,
            partial_fraction=float(((values > 0) & (values < 1-1e-6)).mean()) if len(values) else None,
            compliant_candidate_std_mean=float(np.mean(stds)) if stds else None,
            expert_score_mean=float(np.mean(expert_scores[target])) if expert_scores[target] else None,
            worst_pose_score_mean=float(worst.mean()) if len(worst) else None,
            worst_pose_candidate_std_mean=float(np.mean(worst_stds)) if worst_stds else None,
            mean_hides_low_clearance_fraction=float(((values >= .95) & (worst < .5)).mean()) if len(values) else None,
        )
    print(json.dumps(dict(split=args.split, frame_selection=args.frame_selection,
                          edge_sampling_is_biased=args.frame_selection == "edge",
                          routes=len(names), maps=frames_seen,
                          expert_min_clearance_quantiles_m=np.quantile(expert_clearances, [0, .1, .5, .9, 1]).tolist() if expert_clearances else [],
                          targets=summaries), indent=2))
    print("Use training split to choose a physically justified target; do not optimize it on test outcomes.")


if __name__ == "__main__":
    main()
