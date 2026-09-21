#!/usr/bin/env python3
"""Turn a leaderboard checkpoint JSON into one readable route-results CSV.

With --repetitions N, RouteIndexer (leaderboard/utils/route_indexer.py)
assigns each (route, repetition) pair its own unique record index as
``route_number * N + repetition``, so every repetition's result survives
independently in the checkpoint -- nothing gets overwritten. This script
splits that back out into a route_number/repetition pair per row.

Produces route_results.csv, with one row per (route, repetition) run.  A
standard 32-route evaluation with one repetition therefore has exactly 32
data rows.  Frame/telemetry locations and compact diagnostics are included in
the same table when --artifacts-root is supplied.

Optionally enriches rows with the source map and route identity when pointed
at the matching ``*.xml.provenance.json`` file.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

INFRACTION_KEYS = [
    "collisions_pedestrian",
    "collisions_vehicle",
    "collisions_layout",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "route_dev",
    "route_timeout",
    "vehicle_blocked",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path,
                         help="Checkpoint JSON written by --checkpoint.")
    parser.add_argument("--repetitions", required=True, type=int,
                         help="The --repetitions value the eval run used.")
    parser.add_argument("--provenance", type=Path, default=None,
                         help="Optional matching *.xml.provenance.json used "
                              "to attach source map/route columns.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--artifacts-root", type=Path, default=None,
                         help="Optional per-route artifact root created by "
                              "evaluate_hd465_transfuser.sh.")
    return parser.parse_args()


def load_provenance(path):
    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["output_id"]: entry for entry in entries}


def write_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_or_none(values):
    return round(statistics.mean(values), 3) if values else None


def stdev_or_zero(values):
    return round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0


def route_artifact_dir(root, route_id, repetition):
    if root is None:
        return None
    token = str(route_id or "unknown").rsplit("_", 1)[-1]
    if token.isdigit():
        token = "%02d" % int(token)
    token = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in token
    )
    return root / ("route_%s_rep%d" % (token, repetition))


def summarize_telemetry(path):
    summary = {
        "telemetry_frames": 0,
        "speed_mean_mps": None,
        "speed_max_mps": None,
        "learned_desired_speed_mean_mps": None,
        "low_speed_fraction": None,
        "safety_stop_frames": 0,
        "max_recovery_attempt": 0,
        "recovery_exhausted": False,
    }
    if path is None or not path.is_file():
        return summary

    speed_sum = desired_sum = 0.0
    low_speed_frames = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            speed = float(item.get("speed", 0.0))
            desired = float(item.get("learned_desired_speed", 0.0))
            summary["telemetry_frames"] += 1
            speed_sum += speed
            desired_sum += desired
            summary["speed_max_mps"] = max(
                speed,
                summary["speed_max_mps"]
                if summary["speed_max_mps"] is not None else speed,
            )
            low_speed_frames += int(speed < 0.1)
            summary["safety_stop_frames"] += int(
                bool(item.get("safety_stop", False)))
            summary["max_recovery_attempt"] = max(
                summary["max_recovery_attempt"],
                int(item.get("recovery_attempt", 0)),
            )
            summary["recovery_exhausted"] = (
                summary["recovery_exhausted"]
                or bool(item.get("recovery_exhausted", False))
            )

    count = summary["telemetry_frames"]
    if count:
        summary["speed_mean_mps"] = round(speed_sum / count, 3)
        summary["speed_max_mps"] = round(summary["speed_max_mps"], 3)
        summary["learned_desired_speed_mean_mps"] = round(
            desired_sum / count, 3)
        summary["low_speed_fraction"] = round(low_speed_frames / count, 4)
    return summary


def main():
    args = parse_args()
    with args.results.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["_checkpoint"]["records"]
    if not records:
        raise RuntimeError("No route records found in: %s" % args.results)

    provenance_by_route = load_provenance(args.provenance) if args.provenance else {}

    detail_rows = []
    for rec in records:
        index = rec["index"]
        route_number = index // args.repetitions
        repetition = index % args.repetitions
        route_id_text = str(rec.get("route_id") or "")
        output_route_id_text = route_id_text.rsplit("_", 1)[-1]
        output_route_id = (
            int(output_route_id_text)
            if output_route_id_text.isdigit() else route_number
        )
        scores = rec.get("scores", {})
        infractions = rec.get("infractions", {})
        meta = rec.get("meta", {})

        row = {
            "index": index,
            "route_number": route_number,
            "output_route_id": output_route_id,
            "repetition": repetition,
            "route_id": rec.get("route_id"),
            "status": rec.get("status"),
            "score_route": scores.get("score_route"),
            "score_penalty": scores.get("score_penalty"),
            "score_composed": scores.get("score_composed"),
            "route_length_m": meta.get("route_length"),
            "duration_game_s": meta.get("duration_game"),
            "duration_system_s": meta.get("duration_system"),
        }
        for key in INFRACTION_KEYS:
            row[key] = len(infractions.get(key, []))

        prov = provenance_by_route.get(output_route_id)
        if prov:
            row["map"] = prov.get("map")
            row["source_xml_file"] = prov.get("source_xml_file")
            row["source_route_id"] = prov.get("source_route_id")
            row["dataset_route"] = prov.get("dataset_route")

        artifact_dir = route_artifact_dir(
            args.artifacts_root, rec.get("route_id"), repetition)
        frames = []
        telemetry_path = None
        if artifact_dir is not None:
            frames = sorted(
                (artifact_dir / "frames").glob("*.png"),
                key=lambda path: (
                    (0, int(path.stem)) if path.stem.isdigit()
                    else (1, path.stem)
                ),
            ) if (artifact_dir / "frames").is_dir() else []
            telemetry_path = artifact_dir / "control_telemetry.jsonl"
        row["artifact_dir"] = (
            str(artifact_dir.resolve())
            if artifact_dir is not None and artifact_dir.exists() else ""
        )
        row["visualization_frames"] = len(frames)
        row.update(summarize_telemetry(telemetry_path))
        row["artifact_size_mb"] = round(
            sum(path.stat().st_size for path in frames) / (1024.0 * 1024.0), 1)
        row["last_visualization_frame"] = (
            str(frames[-1].resolve()) if frames else "")

        detail_rows.append(row)

    detail_rows.sort(key=lambda r: r["index"])

    by_route = {}
    for row in detail_rows:
        by_route.setdefault(row["route_number"], []).append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "route_results.csv"
    write_csv(detail_path, detail_rows)

    all_composed = [r["score_composed"] for r in detail_rows if r["score_composed"] is not None]
    all_route = [r["score_route"] for r in detail_rows if r["score_route"] is not None]
    all_penalty = [r["score_penalty"] for r in detail_rows if r["score_penalty"] is not None]
    overall = {
        "num_routes": len(by_route),
        "num_runs": len(detail_rows),
        "DS_mean": mean_or_none(all_composed),
        "DS_std": stdev_or_zero(all_composed),
        "RC_mean": mean_or_none(all_route),
        "IS_mean": mean_or_none(all_penalty),
    }
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print("Route result table:", detail_path)


if __name__ == "__main__":
    main()
