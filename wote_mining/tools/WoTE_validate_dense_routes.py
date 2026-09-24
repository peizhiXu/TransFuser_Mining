#!/usr/bin/env python3
"""Validate exported dense routes against recorded HD465 ego positions."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _collection_index(name):
    match = re.search(r"_route(\d+)_", name)
    if not match:
        raise ValueError("cannot parse collection route index from %s" % name)
    return int(match.group(1))


def validate(dataset_group, manifest_path, frame_stride=1):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reports = []
    for folder in sorted(Path(dataset_group).iterdir()):
        label_dir = folder / "label_raw"
        if not label_dir.is_dir():
            continue
        index = _collection_index(folder.name)
        entry = manifest["routes"].get(str(index))
        if entry is None:
            raise KeyError("route index %d missing from manifest" % index)
        dense = np.load(manifest_path.parent / entry["file"])["world_xy"]
        ego = []
        files = sorted(label_dir.glob("*.json"))[::frame_stride]
        for path in files:
            labels = json.loads(path.read_text(encoding="utf-8"))
            ego.append(np.asarray(labels[0]["ego_matrix"], dtype=np.float64)[:2, 3])
        ego = np.asarray(ego)
        distance, nearest_index = cKDTree(dense).query(ego)
        reports.append({
            "folder": folder.name,
            "collection_index": index,
            "xml_route_id": entry["xml_route_id"],
            "frames": int(len(ego)),
            "distance_p50_m": float(np.percentile(distance, 50)),
            "distance_p95_m": float(np.percentile(distance, 95)),
            "distance_max_m": float(distance.max()),
            "within_3m_fraction": float((distance <= 3).mean()),
            "nearest_index_start": int(nearest_index[0]),
            "nearest_index_end": int(nearest_index[-1]),
        })
    if not reports:
        raise ValueError("no route folders found")
    p50 = np.asarray([item["distance_p50_m"] for item in reports])
    p95 = np.asarray([item["distance_p95_m"] for item in reports])
    return {
        "dataset_group": str(Path(dataset_group).resolve()),
        "manifest": str(manifest_path.resolve()),
        "route_count": len(reports),
        "route_median_distance_p50_m": float(np.median(p50)),
        "route_median_distance_p95_m": float(np.median(p95)),
        "routes": reports,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-group")
    parser.add_argument("--manifest")
    parser.add_argument("--dataset-root")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--output")
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()
    single = args.dataset_group is not None or args.manifest is not None
    batch = args.dataset_root is not None or args.manifest_dir is not None
    if single == batch:
        parser.error("supply either dataset-group/manifest or dataset-root/manifest-dir")
    if single:
        if not args.dataset_group or not args.manifest:
            parser.error("both --dataset-group and --manifest are required")
        result = validate(args.dataset_group, args.manifest, args.frame_stride)
    else:
        if not args.dataset_root or not args.manifest_dir:
            parser.error("both --dataset-root and --manifest-dir are required")
        groups = []
        root = Path(args.dataset_root)
        manifests = Path(args.manifest_dir)
        for map_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for weather_dir in sorted(path for path in map_dir.iterdir() if path.is_dir()):
                stem = map_dir.name + weather_dir.name
                groups.append(validate(
                    weather_dir, manifests / (stem + "_manifest.json"),
                    args.frame_stride,
                ))
        result = {
            "dataset_root": str(root.resolve()),
            "group_count": len(groups),
            "route_count": sum(item["route_count"] for item in groups),
            "median_group_p50_m": float(np.median([
                item["route_median_distance_p50_m"] for item in groups
            ])),
            "median_group_p95_m": float(np.median([
                item["route_median_distance_p95_m"] for item in groups
            ])),
            "groups": groups,
        }
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
