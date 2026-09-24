#!/usr/bin/env python3
"""Rebuild collection-time dense routes offline from XML and OpenDRIVE.

CARLA server/Unreal is not required. The script constructs ``carla.Map``
directly from each local .xodr and calls the same mining interpolation helper
used by the expert autopilot. Output coordinates are CARLA world x/y metres.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _prepare_carla(carla_root):
    root = Path(carla_root).resolve()
    eggs = sorted((root / "PythonAPI/carla/dist").glob("carla-*-py3.7-*.egg"))
    if not eggs:
        raise FileNotFoundError("CARLA Python 3.7 egg not found under %s" % root)
    sys.path.insert(0, str(eggs[-1]))
    sys.path.insert(0, str(root / "PythonAPI/carla"))
    return root


def _route_number(folder_name):
    match = re.search(r"_route(\d+)_", folder_name)
    return int(match.group(1)) if match else None


def _parse_routes(xml_path, carla):
    result = []
    for route in ET.parse(str(xml_path)).getroot().iter("route"):
        points = []
        # Mining route files keep waypoints inside the weather element.
        for node in route.iter("waypoint"):
            points.append(carla.Location(
                x=float(node.attrib["x"]), y=float(node.attrib["y"]),
                z=float(node.attrib.get("z", 0.0)),
            ))
        if len(points) < 2:
            raise ValueError("route %s in %s has fewer than two waypoints" % (
                route.attrib.get("id"), xml_path,
            ))
        result.append((int(route.attrib["id"]), route.attrib["town"], points))
    return result


def _resample_polyline(points, spacing):
    """Remove join duplicates and fill route-graph gaps at <= spacing."""
    points = np.asarray(points, dtype=np.float64)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-4]
    points = points[keep]
    output = [points[0]]
    for start, stop in zip(points[:-1], points[1:]):
        distance = np.linalg.norm(stop - start)
        steps = max(1, int(np.ceil(distance / spacing)))
        output.extend(start + (stop - start) * (i / steps)
                      for i in range(1, steps + 1))
    return np.asarray(output, dtype=np.float32)


def export_routes(route_xml, xodr, output_dir, carla_root, hop_resolution=1.0):
    root = _prepare_carla(carla_root)
    import carla

    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

    route_xml, xodr = Path(route_xml), Path(xodr)
    world_map = carla.Map(xodr.stem, xodr.read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "wote-mining-dense-routes-v1",
        "route_xml": str(route_xml.resolve()),
        "xodr": str(xodr.resolve()),
        "carla_root": str(root),
        "hop_resolution_m": float(hop_resolution),
        "construction": "leaderboard sparse XML to GlobalRoutePlanner trace",
        "routes": {},
    }
    # This is the first interpolation in RouteScenario._update_route. Its
    # dense result is what set_global_plan gives to the expert. Calling the
    # expert's second interpolation directly on XML endpoints is wrong: that
    # helper expects already-dense points and rejects long traces (>100).
    dao = GlobalRoutePlannerDAO(world_map, hop_resolution)
    planner = GlobalRoutePlanner(dao)
    planner.setup()
    for collection_index, (route_id, town, sparse) in enumerate(
            _parse_routes(route_xml, carla)):
        dense = []
        for start, stop in zip(sparse[:-1], sparse[1:]):
            dense.extend(
                (waypoint.transform, option)
                for waypoint, option in planner.trace_route(start, stop)
            )
        if len(dense) < 2:
            raise RuntimeError("dense route %s is empty" % route_id)
        source_xy = np.asarray([
            [item[0].location.x, item[0].location.y] for item in dense
        ], dtype=np.float32)
        xy = _resample_polyline(source_xy, hop_resolution)
        z = np.asarray([item[0].location.z for item in dense], dtype=np.float32)
        option = np.asarray([int(item[1].value) for item in dense], dtype=np.int16)
        # AutoPilot names collection folders with RouteIndexer.index (zero
        # based), not the XML route id. Preserve both to prevent silent
        # off-by-one joins (some mining XML ids start at 1).
        filename = "%s_index%d_id%d.npz" % (
            route_xml.stem, collection_index, route_id
        )
        np.savez_compressed(output / filename, world_xy=xy,
                            source_world_xy=source_xy, source_world_z=z,
                            source_road_option=option)
        length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        manifest["routes"][str(collection_index)] = {
            "collection_index": collection_index, "xml_route_id": route_id,
            "town": town, "file": filename, "points": int(len(xy)),
            "length_m": length,
        }
    manifest_path = output / (route_xml.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--route-xml")
    source.add_argument("--route-dir")
    parser.add_argument("--xodr")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--carla-root", required=True)
    parser.add_argument("--hop-resolution", type=float, default=1.0)
    args = parser.parse_args()
    if args.route_xml:
        if not args.xodr:
            parser.error("--xodr is required with --route-xml")
        paths = [export_routes(
            args.route_xml, args.xodr, args.output_dir, args.carla_root,
            args.hop_resolution,
        )]
    else:
        if args.xodr:
            parser.error("--xodr is inferred per town with --route-dir")
        paths = []
        for route_xml in sorted(Path(args.route_dir).glob("*.xml")):
            first = next(ET.parse(str(route_xml)).getroot().iter("route"))
            town = first.attrib["town"]
            xodr = (
                Path(args.carla_root) / "CarlaUE4/Content" / town / "Maps"
                / town / "OpenDrive" / (town + ".xodr")
            )
            paths.append(export_routes(
                route_xml, xodr, args.output_dir, args.carla_root,
                args.hop_resolution,
            ))
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
