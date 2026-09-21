#!/usr/bin/env python3
"""Materialize the fixed HD465 train/val/test assignment with symlinks.

The source collection is left untouched.  Every complete route folder under
``raw/<map>/<weather>/`` is matched to the assignment by its collection route
ordinal (the ``routeN`` part of the folder name), then linked into a flat
``train/``, ``val/`` or ``test/`` directory that TransFuser's mining loader
can consume.
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path


REQUIRED_DIRECTORIES = (
    "rgb",
    "lidar",
    "depth",
    "semantics",
    "topdown",
    "label_raw",
    "measurements",
)
EXPECTED_COUNTS = {"train": 120, "val": 20, "test": 20}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path,
                        help="Raw HD465 collection root.")
    parser.add_argument("--assignments", required=True, type=Path,
                        help="split_assignments.csv produced by split v2.")
    parser.add_argument("--output", required=True, type=Path,
                        help="New, empty split directory to create.")
    return parser.parse_args()


def load_assignments(path):
    assignments = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["source_xml_file"], int(row["source_route_ordinal"]))
            if key in assignments:
                raise RuntimeError("Duplicate assignment: %r" % (key,))
            if row["split"] not in EXPECTED_COUNTS:
                raise RuntimeError("Unknown split for %r: %s" %
                                   (key, row["split"]))
            assignments[key] = row
    counts = Counter(row["split"] for row in assignments.values())
    if dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError("Expected %r, found %r" %
                           (EXPECTED_COUNTS, dict(counts)))
    return assignments


def find_route_dirs(source):
    found = []
    for dirpath, dirnames, _ in os.walk(str(source)):
        current = Path(dirpath)
        if all((current / name).is_dir() for name in REQUIRED_DIRECTORIES):
            found.append(current)
            dirnames[:] = []
    return sorted(found)


def collection_key(route_dir):
    match = re.search(r"(?:^|_)route(\d+)(?:_|$)", route_dir.name)
    if match is None:
        raise RuntimeError("Cannot read route ordinal from: %s" % route_dir)
    weather = route_dir.parent.name
    map_name = route_dir.parent.parent.name
    return ("%s%s.xml" % (map_name, weather), int(match.group(1)))


def main():
    args = parse_args()
    source = args.source.resolve()
    assignments = load_assignments(args.assignments.resolve())

    if not source.is_dir():
        raise RuntimeError("Raw source does not exist: %s" % source)
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(
            "Output must be new or empty; refusing to merge with: %s"
            % args.output)

    route_dirs = find_route_dirs(source)
    if len(route_dirs) != 160:
        raise RuntimeError("Expected 160 complete route folders, found %d"
                           % len(route_dirs))

    matched = {}
    for route_dir in route_dirs:
        key = collection_key(route_dir)
        if key not in assignments:
            raise RuntimeError("No assignment for %r (%s)" % (key, route_dir))
        if key in matched:
            raise RuntimeError("Duplicate raw route for %r" % (key,))
        matched[key] = route_dir
    missing = sorted(set(assignments) - set(matched))
    if missing:
        raise RuntimeError("Assignments without raw routes: %r" % missing)

    for split in EXPECTED_COUNTS:
        (args.output / split).mkdir(parents=True, exist_ok=True)

    totals = {split: {"routes": 0, "frames": 0, "estimated_samples": 0}
              for split in EXPECTED_COUNTS}
    records = []
    for key in sorted(matched):
        route_dir = matched[key]
        assignment = assignments[key]
        split = assignment["split"]
        map_name = route_dir.parent.parent.name
        weather = route_dir.parent.name
        link_name = "%s__%s__%s" % (map_name, weather, route_dir.name)
        destination = args.output / split / link_name
        destination.symlink_to(route_dir.resolve(), target_is_directory=True)

        frames = len(list((route_dir / "lidar").glob("*.npy")))
        totals[split]["routes"] += 1
        totals[split]["frames"] += frames
        totals[split]["estimated_samples"] += max(0, frames - 9)
        records.append({
            "split": split,
            "source_xml_file": key[0],
            "source_route_ordinal": key[1],
            "source_route_id": assignment["source_route_id"],
            "route_folder": str(route_dir),
            "link": str(destination),
            "frames": frames,
            "estimated_samples": max(0, frames - 9),
        })

    actual = {split: totals[split]["routes"] for split in EXPECTED_COUNTS}
    if actual != EXPECTED_COUNTS:
        raise RuntimeError("Materialized counts differ: %r" % actual)

    manifest = {
        "source": str(source),
        "assignments": str(args.assignments.resolve()),
        "totals": totals,
        "routes": records,
    }
    manifest_path = args.output / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2))
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
