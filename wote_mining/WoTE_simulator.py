"""HD465 candidate-road compliance targets from the full current topdown map.

This is a mining-specific, static-map analogue of WoTE's offline DAC label.
It checks the swept *truck footprint*, not just the eight trajectory centers.
It does not predict other agents or claim to provide collision/TTC labels.
"""

import cv2
import numpy as np


MAP_SIDE = 500
PIXELS_PER_METER = 5.0
MAP_CENTER = 250.0
METRIC_COUNT = 5
DAC_INDEX = 1


def evaluate_road_compliance(
    trajectories,
    road_map,
    augmentation_degrees=0.0,
    vehicle_length=9.3969,
    vehicle_width=5.3645,
    margin_m=0.0,
    footprint_spacing_m=0.5,
    samples_per_interval=4,
):
    """Score K 4-second trajectories against the current 100 m x 100 m map.

    ``trajectories`` are augmented current-vehicle-frame [K,8,3] poses.
    The road map is the unaugmented 500x500 road bit from the saved topdown.
    A candidate gets DAC=0 if *any* sampled swept-footprint point is
    explicitly non-road. Map-outside samples make that candidate invalid,
    rather than silently counting unknown terrain as drivable/non-drivable.
    The grid/temporal spacing are configurable approximations to continuous
    footprint sweep; a zero score is evidence of map overlap, not a CARLA
    physics collision event.
    """
    poses = np.asarray(trajectories, dtype=np.float32)
    road = np.asarray(road_map)
    if poses.ndim != 3 or poses.shape[1:] != (8, 3):
        raise ValueError("trajectories must have shape [K,8,3]")
    if poses.shape[0] < 1 or not np.isfinite(poses).all():
        raise ValueError("trajectories must be nonempty and finite")
    if road.shape != (MAP_SIDE, MAP_SIDE) or road.dtype != np.uint8:
        raise ValueError("road_map must be uint8 [500,500]")
    if not np.isin(road, (0, 1)).all():
        raise ValueError("road_map must contain only binary road labels")
    if not np.isfinite(augmentation_degrees):
        raise ValueError("augmentation_degrees must be finite")
    if vehicle_length <= 0 or vehicle_width <= 0 or margin_m < 0:
        raise ValueError("vehicle dimensions must be positive and margin nonnegative")
    if footprint_spacing_m <= 0 or samples_per_interval < 1:
        raise ValueError("footprint spacing and temporal sampling must be positive")

    # Include the current pose and interpolate between all eight 0.5 s poses.
    start = np.concatenate(
        (np.zeros((poses.shape[0], 1, 3), dtype=np.float32), poses), axis=1
    )
    xy0, xy1 = start[:, :-1, :2], start[:, 1:, :2]
    yaw0, yaw1 = start[:, :-1, 2], start[:, 1:, 2]
    delta_yaw = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))
    fraction = np.linspace(
        1.0 / samples_per_interval, 1.0, samples_per_interval,
        dtype=np.float32,
    )
    sampled_xy = (
        xy0[:, :, None] + fraction[None, None, :, None]
        * (xy1 - xy0)[:, :, None]
    ).reshape(poses.shape[0], -1, 2)
    sampled_yaw = (
        yaw0[:, :, None] + fraction[None, None, :] * delta_yaw[:, :, None]
    ).reshape(poses.shape[0], -1)
    sampled_xy = np.concatenate(
        (np.zeros((poses.shape[0], 1, 2), dtype=np.float32), sampled_xy), axis=1
    )
    sampled_yaw = np.concatenate(
        (np.zeros((poses.shape[0], 1), dtype=np.float32), sampled_yaw), axis=1
    )

    half_length = vehicle_length / 2 + margin_m
    half_width = vehicle_width / 2 + margin_m
    longitudinal = np.linspace(
        -half_length, half_length,
        int(np.ceil(2 * half_length / footprint_spacing_m)) + 1,
        dtype=np.float32,
    )
    lateral = np.linspace(
        -half_width, half_width,
        int(np.ceil(2 * half_width / footprint_spacing_m)) + 1,
        dtype=np.float32,
    )
    local_x, local_y = np.meshgrid(longitudinal, lateral, indexing="ij")
    local_x, local_y = local_x.ravel(), local_y.ravel()
    cosine = np.cos(sampled_yaw)[..., None]
    sine = np.sin(sampled_yaw)[..., None]
    x_aug = sampled_xy[..., 0, None] + cosine * local_x - sine * local_y
    y_aug = sampled_xy[..., 1, None] + sine * local_x + cosine * local_y

    # Reverse the existing TransFuser data augmentation before querying the
    # unrotated topdown raster. Its native axes are forward x, right y.
    angle = np.deg2rad(float(augmentation_degrees))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x_raw = cos_a * x_aug - sin_a * y_aug
    y_raw = sin_a * x_aug + cos_a * y_aug
    map_row = (MAP_CENTER - PIXELS_PER_METER * x_raw).reshape(-1, local_x.size)
    map_col = (MAP_CENTER + PIXELS_PER_METER * y_raw).reshape(-1, local_x.size)
    sampled = cv2.remap(
        road, map_col.astype(np.float32), map_row.astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=2,
    ).reshape(poses.shape[0], -1)
    clearance_map = cv2.distanceTransform(
        road, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    ) / PIXELS_PER_METER
    clearance = cv2.remap(
        clearance_map, map_col.astype(np.float32), map_row.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).reshape(poses.shape[0], -1)

    valid = (sampled != 2).all(axis=1)
    compliant = (sampled == 1).all(axis=1)
    return {
        "drivable_compliance": compliant.astype(np.float32),
        "drivable_valid": valid.astype(np.uint8),
        # Diagnostic distance of the nearest sampled truck point to non-road;
        # zero if overlapping it. Do not substitute this for CARLA collision.
        "min_road_clearance_m": np.where(
            valid, clearance.min(axis=1), np.nan
        ).astype(np.float32),
        "sample_count": int(sampled.shape[1]),
    }


def road_metric_targets(evaluation):
    """Put only observed DAC labels into WoTE's five-metric target layout.

    The other four scores are zero *with mask zero*. They are unavailable,
    not negative examples; ``source_style_core_losses`` respects this mask.
    """
    dac = np.asarray(evaluation["drivable_compliance"], dtype=np.float32)
    valid = np.asarray(evaluation["drivable_valid"], dtype=np.uint8)
    if dac.ndim != 1 or valid.shape != dac.shape:
        raise ValueError("evaluation must contain matching [K] DAC and validity")
    targets = np.zeros((dac.size, METRIC_COUNT), dtype=np.float32)
    mask = np.zeros((dac.size, METRIC_COUNT), dtype=np.uint8)
    targets[:, DAC_INDEX] = dac
    mask[:, DAC_INDEX] = valid
    return targets, mask

"""Route-relative diagnostics for 4-second HD465 candidate trajectories.

The reference must be an ordered, dense *navigation route* in the same
augmented current-vehicle frame as the candidates. Expert future waypoints
are deliberately not accepted as a substitute for this reference: doing so
would turn route progress into another imitation metric.
"""

import numpy as np
import json
import re
from pathlib import Path


METRIC_COUNT = 5
EP_INDEX = 2


def world_route_to_ego(route_world_xy, current_ego_matrix,
                       augmentation_degrees=0.0):
    """Convert a saved CARLA-world dense route to today's augmented ego frame."""
    route = np.asarray(route_world_xy, dtype=np.float64)
    matrix = np.asarray(current_ego_matrix, dtype=np.float64)
    if route.ndim != 2 or route.shape[1] != 2 or route.shape[0] < 2:
        raise ValueError("route_world_xy must have shape [N,2] with N >= 2")
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("current_ego_matrix must be finite [4,4]")
    if not np.isfinite(route).all() or not np.isfinite(augmentation_degrees):
        raise ValueError("route and augmentation must be finite")
    homogeneous = np.column_stack((
        route, np.zeros(route.shape[0]), np.ones(route.shape[0])
    ))
    local = (np.linalg.inv(matrix) @ homogeneous.T).T[:, :2]
    angle = np.deg2rad(float(augmentation_degrees))
    ca, sa = np.cos(angle), np.sin(angle)
    rotation = np.array([[ca, sa], [-sa, ca]])
    return (local @ rotation.T).astype(np.float32)


def dense_route_file_for_collection(route_dir, routes_dir):
    """Resolve a raw collection folder to its exported dense-route file."""
    name = Path(route_dir).resolve().name
    match = re.match(r"(.+)_route(\d+)_", name)
    if not match:
        raise ValueError("unrecognized mining collection folder: %s" % name)
    prefix, index = match.group(1), int(match.group(2))
    manifest_path = Path(routes_dir) / (prefix + "_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["routes"].get(str(index))
    if entry is None:
        raise KeyError("collection index %d missing from %s" % (index, manifest_path))
    return manifest_path.parent / entry["file"]


def local_route_window(route_world_xy, current_ego_matrix,
                       augmentation_degrees=0.0, point_count=128,
                       points_behind=8):
    """Return a fixed-size local navigation polyline and a validity mask."""
    if point_count < 2 or points_behind < 0 or points_behind >= point_count:
        raise ValueError("invalid route window dimensions")
    local = world_route_to_ego(
        route_world_xy, current_ego_matrix, augmentation_degrees
    )
    # GlobalRoutePlanner can emit two numerically different points at a road
    # junction whose separation is far below the route resolution. Removing
    # sub-millimetre duplicates keeps projection well conditioned without
    # changing route geometry.
    keep = np.r_[True, np.linalg.norm(np.diff(local, axis=0), axis=1) >= 1e-3]
    local = local[keep]
    if len(local) < 2:
        raise ValueError("dense route contains fewer than two distinct points")
    nearest = int(np.linalg.norm(local, axis=1).argmin())
    start = max(0, nearest - points_behind)
    selected = local[start:start + point_count]
    result = np.empty((point_count, 2), dtype=np.float32)
    result[:len(selected)] = selected
    result[len(selected):] = selected[-1]
    mask = np.zeros(point_count, dtype=np.uint8)
    mask[:len(selected)] = 1
    return {"route_xy": result, "route_mask": mask,
            "nearest_route_index": nearest}


def evaluate_route_progress(trajectories, route_xy, max_segment_m=5.0,
                            required_lookahead_m=0.0,
                            max_start_distance_m=6.0):
    """Return route arc progress, lateral deviation and coverage diagnostics.

    ``trajectories`` are [K,8,3] in forward/right/yaw coordinates; ``route_xy``
    is [N,2] in the same frame. This function intentionally does not return
    an EP training label: target speed, safe stopping, and deviation thresholds
    need mining-specific calibration once an authoritative route is available.
    """
    poses = np.asarray(trajectories, dtype=np.float64)
    route = np.asarray(route_xy, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (8, 3) or poses.shape[0] < 1:
        raise ValueError("trajectories must have shape [K,8,3]")
    if route.ndim != 2 or route.shape[1] != 2 or route.shape[0] < 2:
        raise ValueError("route_xy must have shape [N,2] with N >= 2")
    if not np.isfinite(poses).all() or not np.isfinite(route).all():
        raise ValueError("trajectory and route coordinates must be finite")
    if max_segment_m <= 0 or required_lookahead_m < 0 or max_start_distance_m <= 0:
        raise ValueError("route limits must be positive, except zero lookahead")

    segments = np.diff(route, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if (lengths < 1e-3).any() or (lengths > max_segment_m).any():
        raise ValueError("route must be dense and have no duplicate points")
    arc_start = np.r_[0.0, np.cumsum(lengths[:-1])]
    points = np.concatenate((
        np.zeros((poses.shape[0], 1, 2), dtype=np.float64), poses[:, :, :2]
    ), axis=1)
    delta = points[:, :, None, :] - route[None, None, :-1, :]
    fraction = np.clip(
        np.sum(delta * segments[None, None], axis=-1) / lengths[None, None] ** 2,
        0.0, 1.0,
    )
    projection = route[None, None, :-1, :] + fraction[..., None] * segments[None, None]
    distance = np.linalg.norm(points[:, :, None, :] - projection, axis=-1)
    segment_index = distance.argmin(axis=-1)
    picked_fraction = np.take_along_axis(
        fraction, segment_index[..., None], axis=-1
    )[..., 0]
    picked_distance = np.take_along_axis(
        distance, segment_index[..., None], axis=-1
    )[..., 0]
    arc = arc_start[segment_index] + picked_fraction * lengths[segment_index]
    delta_arc = np.diff(arc, axis=1)
    start_ok = picked_distance[:, 0] <= max_start_distance_m
    coverage_ok = (lengths.sum() - arc[:, 0]) >= required_lookahead_m
    # A clamped projection at the far route end cannot establish further
    # progress; mark this diagnostic invalid instead of crediting a shortcut.
    far_end = (segment_index[:, -1] == len(lengths) - 1) & (
        picked_fraction[:, -1] >= 1.0 - 1e-6
    )
    end_direction = segments[-1] / lengths[-1]
    beyond_end = (points[:, -1] - route[-1]) @ end_direction > 0.5
    valid = start_ok & coverage_ok & ~(far_end & beyond_end)
    return {
        "route_progress_m": (arc[:, -1] - arc[:, 0]).astype(np.float32),
        "route_max_deviation_m": picked_distance.max(axis=1).astype(np.float32),
        "route_endpoint_deviation_m": picked_distance[:, -1].astype(np.float32),
        "route_backtrack_m": np.maximum(-delta_arc, 0.0).sum(axis=1).astype(np.float32),
        "route_valid": valid.astype(np.uint8),
        "route_remaining_m": (lengths.sum() - arc[:, 0]).astype(np.float32),
    }


def route_progress_metric_targets(evaluation, candidate_eligible=None,
                                  progress_threshold_m=0.1):
    """Convert raw along-route progress into WoTE's normalized EP target.

    This follows PDMScorer's per-scene normalization: negative progress is
    clipped, optional multiplicative eligibility (e.g. valid DAC/NC) is
    applied, then candidates are divided by the best remaining progress.
    Only the EP mask is enabled; this helper cannot label other metrics.
    """
    progress = np.asarray(evaluation["route_progress_m"], dtype=np.float32)
    valid = np.asarray(evaluation["route_valid"], dtype=bool)
    if progress.ndim != 1 or valid.shape != progress.shape:
        raise ValueError("route evaluation arrays must have shape [K]")
    eligible = valid.copy()
    if candidate_eligible is not None:
        supplied = np.asarray(candidate_eligible, dtype=bool)
        if supplied.shape != progress.shape:
            raise ValueError("candidate_eligible must have shape [K]")
        eligible &= supplied
    raw = np.clip(progress, 0.0, None) * eligible
    score = np.zeros_like(raw)
    if eligible.any():
        maximum = raw.max()
        if maximum > progress_threshold_m:
            score[eligible] = raw[eligible] / maximum
        else:
            score[eligible] = 1.0
    targets = np.zeros((len(progress), METRIC_COUNT), dtype=np.float32)
    mask = np.zeros_like(targets, dtype=np.uint8)
    targets[:, EP_INDEX] = score
    mask[:, EP_INDEX] = valid.astype(np.uint8)
    return {"metric_targets": targets, "metric_valid": mask}

"""Coverage-gated NC/TTC labels against recorded mining vehicles.

Other actors follow their recorded paths and do not react to a candidate ego
trajectory. Safe labels are enabled only while the candidate remains inside
the region where collection recorded every vehicle; observed conflicts remain
valid positive evidence even outside that region.
"""

import numpy as np


METRIC_COUNT = 5
NC_INDEX = 0
TTC_INDEX = 3


def _interpolate_ego(trajectories, subdivisions):
    poses = np.asarray(trajectories, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (8, 3):
        raise ValueError("trajectories must have shape [K,8,3]")
    if poses.shape[0] < 1 or not np.isfinite(poses).all():
        raise ValueError("trajectories must be nonempty and finite")
    frames = np.concatenate((np.zeros((poses.shape[0], 1, 3), np.float32), poses), axis=1)
    parts = [frames[:, 0]]
    for step in range(8):
        before, after = frames[:, step], frames[:, step + 1]
        dyaw = np.arctan2(np.sin(after[:, 2] - before[:, 2]),
                          np.cos(after[:, 2] - before[:, 2]))
        for substep in range(1, subdivisions + 1):
            fraction = substep / subdivisions
            item = before + fraction * (after - before)
            item[:, 2] = before[:, 2] + fraction * dyaw
            parts.append(item)
    return np.stack(parts, axis=1)


def _interpolate_agents(boxes, mask, subdivisions):
    rows, valid_rows = [boxes[0]], [mask[0]]
    for step in range(8):
        before, after = boxes[step], boxes[step + 1]
        dyaw = np.arctan2(np.sin(after[:, 2] - before[:, 2]),
                          np.cos(after[:, 2] - before[:, 2]))
        for substep in range(1, subdivisions + 1):
            if substep == subdivisions:
                rows.append(after)
                valid_rows.append(mask[step + 1])
            else:
                fraction = substep / subdivisions
                item = before + fraction * (after - before)
                item[:, 2] = before[:, 2] + fraction * dyaw
                rows.append(item)
                valid_rows.append(mask[step] & mask[step + 1])
    return np.stack(rows, axis=0), np.stack(valid_rows, axis=0).astype(bool)


def _sat_axis_gap(ego, agents, ego_length, ego_width):
    """Largest separating-axis gap for [K,T,3] ego and [T,M,5] actors."""
    delta = agents[None, :, :, :2] - ego[:, :, None, :2]
    ego_cos = np.cos(ego[..., 2])[:, :, None]
    ego_sin = np.sin(ego[..., 2])[:, :, None]
    agent_cos = np.cos(agents[..., 2])[None]
    agent_sin = np.sin(agents[..., 2])[None]
    ego_forward = np.stack((ego_cos, ego_sin), axis=-1)
    ego_right = np.stack((-ego_sin, ego_cos), axis=-1)
    actor_forward = np.stack((agent_cos, agent_sin), axis=-1)
    actor_right = np.stack((-agent_sin, agent_cos), axis=-1)
    actor_half_l = agents[None, :, :, 3] / 2
    actor_half_w = agents[None, :, :, 4] / 2

    def dot(left, right):
        return (left * right).sum(axis=-1)

    largest_gap = None
    for axis in (ego_forward, ego_right, actor_forward, actor_right):
        ego_radius = (ego_length / 2 * np.abs(dot(ego_forward, axis))
                      + ego_width / 2 * np.abs(dot(ego_right, axis)))
        actor_radius = (actor_half_l * np.abs(dot(actor_forward, axis))
                        + actor_half_w * np.abs(dot(actor_right, axis)))
        gap = np.abs(dot(delta, axis)) - ego_radius - actor_radius
        largest_gap = gap if largest_gap is None else np.maximum(largest_gap, gap)
    return largest_gap


def _recording_coverage(ego, tracks, subdivisions, safety_margin_m):
    if "recorded_ego" not in tracks or bool(tracks.get("truncated", False)):
        return np.zeros(ego.shape[0], dtype=bool), None
    recorded = np.asarray(tracks["recorded_ego"], dtype=np.float32)
    if recorded.shape != (9, 3) or not np.isfinite(recorded).all():
        raise ValueError("recorded_ego must have shape [9,3]")
    radius = float(tracks.get("recording_radius_m", 50.0))
    if radius <= safety_margin_m:
        raise ValueError("recording radius must exceed safety margin")
    recorded_sampled = _interpolate_ego(recorded[None, 1:], subdivisions)[0]
    distance = np.linalg.norm(ego[..., :2] - recorded_sampled[None, :, :2], axis=-1)
    return (distance <= radius - safety_margin_m).all(axis=1), recorded_sampled


def evaluate_recorded_agent_overlap(
    trajectories, tracks, ego_length=9.3969, ego_width=5.3645,
    samples_per_interval=4, recording_safety_margin_m=15.0,
    ttc_horizon_s=1.0,
):
    """Check oriented-rectangle overlap at 0.125 s by default.

    Result includes binary NC/TTC labels and their per-candidate validity. A
    clean candidate merely avoids the *recorded* vehicles; it is not a
    reactive counterfactual rollout.
    """
    if not isinstance(samples_per_interval, int) or samples_per_interval < 1:
        raise ValueError("samples_per_interval must be a positive integer")
    if not np.isfinite((ego_length, ego_width)).all() or ego_length <= 0 or ego_width <= 0:
        raise ValueError("ego dimensions must be positive and finite")
    boxes = np.asarray(tracks["boxes"], dtype=np.float32)
    mask = np.asarray(tracks["mask"], dtype=bool)
    if boxes.ndim != 3 or boxes.shape[0] != 9 or boxes.shape[2] != 5:
        raise ValueError("track boxes must have shape [9,M,5]")
    if mask.shape != boxes.shape[:2]:
        raise ValueError("track mask must have shape [9,M]")
    if not np.isfinite(boxes[mask]).all() or (boxes[mask, 3:] <= 0).any():
        raise ValueError("visible agent boxes need finite, positive dimensions")
    # Collection files reserve a fixed number of actor slots (currently 64),
    # although a mining scene normally contains only a few vehicles.  Empty
    # columns cannot affect overlap/TTC and otherwise dominate both runtime and
    # memory, so remove them before constructing the [K,T,M,...] SAT arrays.
    observed_columns = mask.any(axis=0)
    boxes = boxes[:, observed_columns]
    mask = mask[:, observed_columns]
    ego = _interpolate_ego(trajectories, samples_per_interval)
    batch, steps = ego.shape[:2]
    coverage, recorded_ego = _recording_coverage(
        ego, tracks, samples_per_interval, recording_safety_margin_m
    )
    if boxes.shape[1] == 0 or not mask.any():
        return {
            "recorded_overlap": np.zeros(batch, dtype=np.uint8),
            "no_collision": np.ones(batch, dtype=np.float32),
            "no_collision_valid": coverage.astype(np.uint8),
            "ttc_within_bound": np.ones(batch, dtype=np.float32),
            "ttc_valid": coverage.astype(np.uint8),
            "first_ttc_risk_s": np.full(batch, np.nan, dtype=np.float32),
            "first_overlap_s": np.full(batch, np.nan, dtype=np.float32),
            "min_axis_gap_m": np.full(batch, np.nan, dtype=np.float32),
            "observed_pairs": np.zeros(batch, dtype=np.int32),
            "tracks_truncated": bool(tracks.get("truncated", False)),
        }
    agents, visible = _interpolate_agents(boxes, mask, samples_per_interval)
    if agents.shape[0] != steps:
        raise AssertionError("ego and actor sample times differ")
    largest_gap = _sat_axis_gap(ego, agents, ego_length, ego_width)
    largest_gap = np.where(visible[None], largest_gap, np.inf)
    overlap = largest_gap <= 0
    hit_by_time = overlap.any(axis=2)
    any_hit = hit_by_time.any(axis=1)
    first_index = hit_by_time.argmax(axis=1)

    # PDM-style binary TTC: from every planned pose, extrapolate ego at its
    # instantaneous heading/speed for up to one second and query recorded
    # actors at the corresponding future times. This is deliberately
    # non-reactive and uses the same 0.125 s interpolation grid as NC.
    dt = 0.5 / samples_per_interval
    displacement = np.zeros_like(ego[..., :2])
    displacement[:, 1:] = np.diff(ego[..., :2], axis=1)
    displacement[:, 0] = displacement[:, 1]
    speed = np.linalg.norm(displacement, axis=-1) / dt
    heading_direction = np.stack(
        (np.cos(ego[..., 2]), np.sin(ego[..., 2])), axis=-1
    )
    max_offset = int(round(ttc_horizon_s / dt))
    offsets = np.arange(0, max_offset + 1, max(1, samples_per_interval // 2))
    ttc_by_time = np.zeros((batch, steps), dtype=bool)
    ttc_coverage = np.ones(batch, dtype=bool) if recorded_ego is not None else np.zeros(batch, dtype=bool)
    radius = float(tracks.get("recording_radius_m", 50.0))
    for offset in offsets:
        count = steps - offset
        if count <= 0:
            continue
        projected = ego[:, :count].copy()
        projected[..., :2] += (
            heading_direction[:, :count]
            * speed[:, :count, None]
            * (offset * dt)
        )
        gap = _sat_axis_gap(
            projected, agents[offset:offset + count], ego_length, ego_width
        )
        projection_visible = visible[offset:offset + count]
        risk = (gap <= 0) & projection_visible[None]
        moving = speed[:, :count] >= 0.05
        ttc_by_time[:, :count] |= risk.any(axis=2) & moving
        if recorded_ego is not None:
            projection_distance = np.linalg.norm(
                projected[..., :2]
                - recorded_ego[None, offset:offset + count, :2], axis=-1
            )
            ttc_coverage &= (
                projection_distance <= radius - recording_safety_margin_m
            ).all(axis=1)
    any_ttc = ttc_by_time.any(axis=1)
    first_ttc = ttc_by_time.argmax(axis=1)
    return {
        "recorded_overlap": any_hit.astype(np.uint8),
        "no_collision": (~any_hit).astype(np.float32),
        # A detected event is valid evidence even if the rest of the rollout
        # leaves the guaranteed 50 m annotation coverage.
        "no_collision_valid": (coverage | any_hit).astype(np.uint8),
        "ttc_within_bound": (~any_ttc).astype(np.float32),
        "ttc_valid": (ttc_coverage | any_ttc).astype(np.uint8),
        "first_ttc_risk_s": np.where(
            any_ttc, first_ttc * dt, np.nan
        ).astype(np.float32),
        "first_overlap_s": np.where(
            any_hit, first_index * (0.5 / samples_per_interval), np.nan
        ).astype(np.float32),
        # SAT separating-axis gap, not Euclidean polygon distance.
        "min_axis_gap_m": np.minimum(largest_gap.min(axis=(1, 2)),
                                       np.finfo(np.float32).max).astype(np.float32),
        "observed_pairs": np.full(batch, int(visible.sum()), dtype=np.int32),
        "tracks_truncated": bool(tracks.get("truncated", False)),
    }


def recorded_agent_metric_targets(evaluation):
    """Place observed non-reactive NC/TTC labels into WoTE's metric layout."""
    nc = np.asarray(evaluation["no_collision"], dtype=np.float32)
    nc_valid = np.asarray(evaluation["no_collision_valid"], dtype=np.uint8)
    ttc = np.asarray(evaluation["ttc_within_bound"], dtype=np.float32)
    ttc_valid = np.asarray(evaluation["ttc_valid"], dtype=np.uint8)
    if not (nc.ndim == 1 and nc_valid.shape == nc.shape
            and ttc.shape == nc.shape and ttc_valid.shape == nc.shape):
        raise ValueError("NC/TTC evaluation arrays must have matching [K] shapes")
    targets = np.zeros((len(nc), METRIC_COUNT), dtype=np.float32)
    valid = np.zeros_like(targets, dtype=np.uint8)
    targets[:, NC_INDEX], valid[:, NC_INDEX] = nc, nc_valid
    targets[:, TTC_INDEX], valid[:, TTC_INDEX] = ttc, ttc_valid
    return targets, valid

"""WoTE/PDM-style binary comfort target adapted to mining trajectories.

The thresholds and six checks follow
``navsim/.../scoring/pdm_comfort_metrics.py`` in the WoTE repository
(Apache-2.0). Original WoTE evaluates dense states produced by its PDM
simulator. The mining adaptation estimates vehicle-frame acceleration from the
nine available 0.5 s poses and applies the same bounds after Savitzky-Golay
smoothing.
"""

import numpy as np
from scipy.signal import savgol_filter


METRIC_COUNT = 5
COMFORT_INDEX = 4
INTERVAL_SECONDS = 0.5

# Retained for explicit comparison with the source implementation. The default
# mining limits below are calibrated separately from HD465 training routes.
SOURCE_WOTE_COMFORT_LIMITS = {
    "max_abs_magnitude_jerk_mps3": 8.37,
    "max_abs_lateral_acceleration_mps2": 4.89,
    "max_longitudinal_acceleration_mps2": 2.40,
    "min_longitudinal_acceleration_mps2": -4.05,
    "max_abs_yaw_acceleration_radps2": 1.93,
    "max_abs_longitudinal_jerk_mps3": 4.13,
    "max_abs_yaw_rate_radps": 0.95,
}

# 99.5% per-component expert envelope from 29,776 four-second windows in the
# 120 HD465 training routes. See wote_mining/assets/comfort. These are
# behavior-calibrated limits, not manufacturer-certified stability limits.
MINING_COMFORT_LIMITS = {
    "max_abs_magnitude_jerk_mps3": 1.57,
    "max_abs_lateral_acceleration_mps2": 1.12,
    "max_longitudinal_acceleration_mps2": 2.00,
    "min_longitudinal_acceleration_mps2": -1.65,
    "max_abs_yaw_acceleration_radps2": 0.18,
    "max_abs_longitudinal_jerk_mps3": 2.02,
    "max_abs_yaw_rate_radps": 0.27,
}

COMFORT_COMPONENTS = (
    "longitudinal_acceleration",
    "lateral_acceleration",
    "magnitude_jerk",
    "longitudinal_jerk",
    "yaw_acceleration",
    "yaw_rate",
)


def _odd_window(maximum, sample_count, minimum=3):
    window = min(int(maximum), int(sample_count))
    if window % 2 == 0:
        window -= 1
    if window < minimum:
        raise ValueError("trajectory has too few samples for comfort filtering")
    return window


def evaluate_comfort(trajectories, interval_seconds=INTERVAL_SECONDS,
                     limits=None):
    """Return source-style binary comfort and six component diagnostics."""
    poses = np.asarray(trajectories, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (8, 3) or poses.shape[0] < 1:
        raise ValueError("trajectories must have shape [K,8,3]")
    if not np.isfinite(poses).all():
        raise ValueError("trajectories must be finite")
    if not np.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive and finite")
    limits = MINING_COMFORT_LIMITS if limits is None else dict(limits)
    if set(limits) != set(SOURCE_WOTE_COMFORT_LIMITS):
        raise ValueError("comfort limits have missing or unknown keys")
    if not np.isfinite(list(limits.values())).all():
        raise ValueError("comfort limits must be finite")

    frames = np.concatenate(
        (np.zeros((poses.shape[0], 1, 3), dtype=np.float64), poses), axis=1
    )
    position = frames[..., :2]
    heading = np.unwrap(frames[..., 2], axis=1)
    # Original WoTE receives velocity/acceleration from PDM simulation. With
    # only 0.5 s mining poses available, estimate those states using centered
    # finite differences before applying the source metric filters.
    velocity_world = np.gradient(
        position, interval_seconds, axis=1, edge_order=2
    )
    acceleration_world = np.gradient(
        velocity_world, interval_seconds, axis=1, edge_order=2
    )
    cosine, sine = np.cos(heading), np.sin(heading)
    longitudinal_acceleration = (
        cosine * acceleration_world[..., 0]
        + sine * acceleration_world[..., 1]
    )
    lateral_acceleration = (
        -sine * acceleration_world[..., 0]
        + cosine * acceleration_world[..., 1]
    )

    sample_count = frames.shape[1]
    accel_window = _odd_window(8, sample_count)
    derivative_window = _odd_window(15, sample_count)
    yaw_window = _odd_window(5, sample_count)
    longitudinal_acceleration = savgol_filter(
        longitudinal_acceleration, accel_window, 2, axis=1
    )
    lateral_acceleration = savgol_filter(
        lateral_acceleration, accel_window, 2, axis=1
    )
    acceleration_magnitude = savgol_filter(
        np.hypot(
            cosine * acceleration_world[..., 0] + sine * acceleration_world[..., 1],
            -sine * acceleration_world[..., 0] + cosine * acceleration_world[..., 1],
        ),
        accel_window, 2, axis=1,
    )
    magnitude_jerk = savgol_filter(
        acceleration_magnitude, derivative_window, 2, deriv=1,
        delta=interval_seconds, axis=1,
    )
    longitudinal_jerk = savgol_filter(
        longitudinal_acceleration, derivative_window, 2, deriv=1,
        delta=interval_seconds, axis=1,
    )
    yaw_rate = savgol_filter(
        heading, yaw_window, 2, deriv=1, delta=interval_seconds, axis=1
    )
    yaw_acceleration = savgol_filter(
        heading, yaw_window, 3, deriv=2, delta=interval_seconds, axis=1
    )

    component_compliance = np.stack((
        ((longitudinal_acceleration
          > limits["min_longitudinal_acceleration_mps2"])
         & (longitudinal_acceleration
            < limits["max_longitudinal_acceleration_mps2"])).all(axis=1),
        (np.abs(lateral_acceleration)
         < limits["max_abs_lateral_acceleration_mps2"]).all(axis=1),
        (np.abs(magnitude_jerk)
         < limits["max_abs_magnitude_jerk_mps3"]).all(axis=1),
        (np.abs(longitudinal_jerk)
         < limits["max_abs_longitudinal_jerk_mps3"]).all(axis=1),
        (np.abs(yaw_acceleration)
         < limits["max_abs_yaw_acceleration_radps2"]).all(axis=1),
        (np.abs(yaw_rate) < limits["max_abs_yaw_rate_radps"]).all(axis=1),
    ), axis=1)
    return {
        "comfort": component_compliance.all(axis=1).astype(np.float32),
        "comfort_valid": np.ones(poses.shape[0], dtype=np.uint8),
        "component_compliance": component_compliance.astype(np.uint8),
        "component_names": COMFORT_COMPONENTS,
        "limits": limits,
        "min_longitudinal_acceleration_mps2": np.min(
            longitudinal_acceleration, axis=1
        ).astype(np.float32),
        "max_longitudinal_acceleration_mps2": np.max(
            longitudinal_acceleration, axis=1
        ).astype(np.float32),
        "max_abs_longitudinal_acceleration_mps2": np.max(
            np.abs(longitudinal_acceleration), axis=1
        ).astype(np.float32),
        "max_abs_lateral_acceleration_mps2": np.max(
            np.abs(lateral_acceleration), axis=1
        ).astype(np.float32),
        "max_abs_magnitude_jerk_mps3": np.max(
            np.abs(magnitude_jerk), axis=1
        ).astype(np.float32),
        "max_abs_longitudinal_jerk_mps3": np.max(
            np.abs(longitudinal_jerk), axis=1
        ).astype(np.float32),
        "max_abs_yaw_acceleration_radps2": np.max(
            np.abs(yaw_acceleration), axis=1
        ).astype(np.float32),
        "max_abs_yaw_rate_radps": np.max(np.abs(yaw_rate), axis=1).astype(np.float32),
    }


def comfort_metric_targets(evaluation):
    """Put Comfort into index four of WoTE's five-metric target layout."""
    score = np.asarray(evaluation["comfort"], dtype=np.float32)
    valid = np.asarray(evaluation["comfort_valid"], dtype=np.uint8)
    if score.ndim != 1 or valid.shape != score.shape:
        raise ValueError("comfort evaluation arrays must have matching [K] shapes")
    targets = np.zeros((len(score), METRIC_COUNT), dtype=np.float32)
    mask = np.zeros_like(targets, dtype=np.uint8)
    targets[:, COMFORT_INDEX] = score
    mask[:, COMFORT_INDEX] = valid
    return targets, mask

"""Offline HD465 candidate evaluation assembled from trustworthy labels."""

import numpy as np

def evaluate_mining_candidates(trajectories, road_map, route_xy,
                               recorded_tracks=None,
                               augmentation_degrees=0.0):
    """Evaluate candidates without inventing unavailable WoTE labels.

    DAC and EP use map/route labels. NC and TTC are activated only where the
    candidate stays inside guaranteed recorded-agent coverage (or where an
    actual conflict is observed). All dynamic labels remain non-reactive.
    """
    road = evaluate_road_compliance(
        trajectories, road_map,
        augmentation_degrees=augmentation_degrees,
    )
    route = evaluate_route_progress(
        trajectories, route_xy, required_lookahead_m=30.0,
    )
    targets, mask = road_metric_targets(road)
    dynamic = None
    dynamic_targets = None
    dynamic_mask = None
    if recorded_tracks is not None:
        dynamic = evaluate_recorded_agent_overlap(
            trajectories, recorded_tracks
        )
        dynamic_targets, dynamic_mask = recorded_agent_metric_targets(dynamic)
        if np.any(mask & dynamic_mask):
            raise AssertionError("candidate metric target masks overlap")
        targets = targets + dynamic_targets
        mask = mask | dynamic_mask

    eligible = (
        road["drivable_valid"].astype(bool)
        & road["drivable_compliance"].astype(bool)
    )
    progress_label_valid = route["route_valid"].astype(bool)
    if dynamic is not None:
        nc_valid = dynamic["no_collision_valid"].astype(bool)
        eligible &= nc_valid & dynamic["no_collision"].astype(bool)
        # Unknown actor coverage must not be converted into a negative EP
        # example.  With known coverage, unsafe candidates receive EP=0 just
        # like PDM's multiplicative progress eligibility.
        progress_label_valid &= nc_valid
    progress = route_progress_metric_targets(
        route, candidate_eligible=eligible,
    )
    progress_targets = progress["metric_targets"]
    progress_mask = progress["metric_valid"]
    progress_mask[:, 2] &= progress_label_valid.astype(np.uint8)
    if np.any(mask & progress_mask):
        raise AssertionError("candidate metric target masks overlap")
    targets = targets + progress_targets
    mask = mask | progress_mask
    comfort = evaluate_comfort(trajectories)
    comfort_targets, comfort_mask = comfort_metric_targets(comfort)
    if np.any(mask & comfort_mask):
        raise AssertionError("candidate metric target masks overlap")
    targets = targets + comfort_targets
    mask = mask | comfort_mask
    result = {
        "metric_targets": targets,
        "metric_valid": mask,
        "road_evaluation": road,
        "route_evaluation": route,
        "comfort_evaluation": comfort,
    }
    if dynamic is not None:
        result["recorded_agent_evaluation"] = dynamic
    return result
