#!/usr/bin/env python3
"""Check one HD465 TransFuser collection batch.

Exit codes:
  0: the Leaderboard checkpoint is complete and all route folders are aligned
  1: malformed checkpoint or malformed saved data
  2: collection is incomplete and can be resumed
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


FRAME_DIRS = {
    "rgb": ("*.png", re.compile(r"^(\d+)$")),
    "lidar": ("*.npy", re.compile(r"^(\d+)$")),
    "depth": ("*.png", re.compile(r"^(\d+)$")),
    "semantics": ("*.png", re.compile(r"^(\d+)$")),
    "topdown": ("*.png", re.compile(r"^encoded_(\d+)$")),
    "label_raw": ("*.json", re.compile(r"^(\d+)$")),
    "measurements": ("*.json", re.compile(r"^(\d+)$")),
}


def frame_ids(directory, glob_pattern, name_pattern):
    ids = set()
    bad_names = []
    for path in directory.glob(glob_pattern):
        match = name_pattern.match(path.stem)
        if match:
            ids.add(int(match.group(1)))
        else:
            bad_names.append(path.name)
    return ids, bad_names


def inspect_route(route_dir):
    errors = []
    all_ids = {}
    for name, (glob_pattern, name_pattern) in FRAME_DIRS.items():
        directory = route_dir / name
        if not directory.is_dir():
            errors.append("missing directory: {}".format(name))
            continue
        ids, bad_names = frame_ids(directory, glob_pattern, name_pattern)
        all_ids[name] = ids
        if bad_names:
            errors.append(
                "unexpected filenames in {}: {}".format(
                    name, ", ".join(sorted(bad_names)[:5])
                )
            )

    if len(all_ids) == len(FRAME_DIRS):
        reference = all_ids["measurements"]
        if not reference:
            errors.append("no saved frames")
        for name, ids in all_ids.items():
            if ids != reference:
                errors.append(
                    "frame mismatch: measurements={} {}={}".format(
                        len(reference), name, len(ids)
                    )
                )
        if reference:
            expected = set(range(min(reference), max(reference) + 1))
            if reference != expected:
                errors.append("measurement frame indices are not contiguous")

            # Decode the boundary samples as well as counting filenames.  This
            # catches truncated writes and accidental sensor/layout changes
            # without making validation scale with every frame in the batch.
            for frame in sorted((min(reference), max(reference))):
                stem = "%04d" % frame
                expected_images = {
                    "rgb": ((160, 960, 3), route_dir / "rgb" / (stem + ".png")),
                    "depth": ((160, 960, 3), route_dir / "depth" / (stem + ".png")),
                    "semantics": ((160, 960), route_dir / "semantics" / (stem + ".png")),
                    "topdown": ((500, 500, 3), route_dir / "topdown" / ("encoded_" + stem + ".png")),
                }
                for name, (expected_shape, path) in expected_images.items():
                    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        errors.append("cannot decode {} frame {}".format(name, stem))
                    elif image.shape != expected_shape:
                        errors.append(
                            "unexpected {} shape at frame {}: {} expected {}".format(
                                name, stem, image.shape, expected_shape
                            )
                        )

                lidar_path = route_dir / "lidar" / (stem + ".npy")
                try:
                    lidar = np.load(str(lidar_path), allow_pickle=True)
                    points = lidar[1]
                    if points.ndim != 2 or points.shape[1] < 3:
                        errors.append(
                            "unexpected lidar shape at frame {}: {}".format(
                                stem, points.shape
                            )
                        )
                except Exception as error:
                    errors.append("cannot decode lidar frame {}: {}".format(stem, error))

                for name in ("measurements", "label_raw"):
                    path = route_dir / name / (stem + ".json")
                    try:
                        with path.open() as stream:
                            value = json.load(stream)
                        if name == "measurements" and not isinstance(value, dict):
                            errors.append("measurements frame {} is not an object".format(stem))
                        if name == "label_raw" and not isinstance(value, list):
                            errors.append("label_raw frame {} is not a list".format(stem))
                    except (OSError, ValueError) as error:
                        errors.append(
                            "cannot decode {} frame {}: {}".format(name, stem, error)
                        )

    for filename in ("mining_expert_config.json", "mining_sensor_config.json"):
        if not (route_dir / filename).is_file():
            errors.append("missing file: {}".format(filename))

    return len(all_ids.get("measurements", ())), errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--require-all-completed",
        action="store_true",
        help="also fail when any Leaderboard route status is not Completed",
    )
    args = parser.parse_args()

    summary = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "complete": False,
        "errors": [],
        "warnings": [],
    }

    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size == 0:
        summary["errors"].append("checkpoint does not exist or is empty")
        exit_code = 2
    else:
        try:
            with args.checkpoint.open() as stream:
                checkpoint = json.load(stream)
        except (OSError, ValueError) as error:
            summary["errors"].append("cannot read checkpoint: {}".format(error))
            checkpoint = None
            exit_code = 1

        if checkpoint is not None:
            state = checkpoint.get("_checkpoint", {})
            progress = state.get("progress", [])
            records = state.get("records", [])
            if (
                not isinstance(progress, list)
                or len(progress) != 2
                or not all(isinstance(value, int) for value in progress)
            ):
                summary["errors"].append("invalid checkpoint progress")
                current = total = 0
                exit_code = 1
            else:
                current, total = progress
                summary["progress"] = {"current": current, "total": total}
                summary["record_count"] = len(records)
                if current < total:
                    summary["errors"].append(
                        "collection is incomplete: {}/{} routes".format(current, total)
                    )
                    exit_code = 2
                elif current != total or len(records) != total:
                    summary["errors"].append(
                        "checkpoint count mismatch: progress={}/{} records={}".format(
                            current, total, len(records)
                        )
                    )
                    exit_code = 1
                else:
                    exit_code = 0

            statuses = Counter(record.get("status", "missing") for record in records)
            summary["route_statuses"] = dict(sorted(statuses.items()))
            non_completed = [
                record for record in records if record.get("status") != "Completed"
            ]
            summary["non_completed_routes"] = [
                {
                    "index": record.get("index"),
                    "route_id": record.get("route_id"),
                    "status": record.get("status"),
                }
                for record in non_completed
            ]
            if non_completed:
                summary["warnings"].append(
                    "{} routes did not finish with status Completed".format(
                        len(non_completed)
                    )
                )
                if args.require_all_completed:
                    summary["errors"].append("non-completed routes are not allowed")
                    exit_code = 1

    route_dirs = []
    if args.data_root.is_dir():
        route_dirs = sorted(
            path for path in args.data_root.iterdir()
            if path.is_dir() and (path / "measurements").is_dir()
        )
    summary["route_directory_count"] = len(route_dirs)

    route_errors = {}
    total_frames = 0
    for route_dir in route_dirs:
        frames, errors = inspect_route(route_dir)
        total_frames += frames
        if errors:
            route_errors[route_dir.name] = errors
    summary["saved_frames"] = total_frames
    summary["route_errors"] = route_errors

    expected_routes = summary.get("record_count")
    if exit_code == 0 and len(route_dirs) != expected_routes:
        summary["errors"].append(
            "route directory count mismatch: expected {} found {}".format(
                expected_routes, len(route_dirs)
            )
        )
        exit_code = 1
    if route_errors:
        summary["errors"].append(
            "{} route directories have malformed frame data".format(
                len(route_errors)
            )
        )
        exit_code = 1

    summary["complete"] = exit_code == 0
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
