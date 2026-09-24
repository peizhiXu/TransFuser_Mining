"""Mining-specific WoTE 4-second pose targets for the HD465 dataset.

Unlike the existing TransFuser waypoint target, these poses are relative to
the current *vehicle* origin. The virtual-LiDAR translation is not applied.
The axes are vehicle-forward x and vehicle-right y, with yaw in radians.
"""

import numpy as np


FUTURE_POSES = 8


def future_ego_poses(ego_matrices, augmentation_degrees=0.0):
    """Convert nine consecutive ego matrices (now + 8 future) to [8, 3].

    ``augmentation_degrees`` follows the rotation convention already used by
    ``CARLA_Data`` for its existing waypoint target. It rotates each future
    (x, y) by -degrees and subtracts that angle from relative yaw.
    """
    matrices = np.asarray(ego_matrices, dtype=np.float64)
    if matrices.shape != (FUTURE_POSES + 1, 4, 4):
        raise ValueError("ego_matrices must have shape [9, 4, 4]")
    if not np.isfinite(matrices).all():
        raise ValueError("ego_matrices contain non-finite values")

    relative = np.linalg.inv(matrices[0]) @ matrices[1:]
    xy = relative[:, :2, 3].copy()
    yaw = np.arctan2(relative[:, 1, 0], relative[:, 0, 0])
    yaw = np.unwrap(np.r_[0.0, yaw])[1:]

    angle = np.deg2rad(float(augmentation_degrees))
    if angle:
        rotation = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)],
        ])
        xy = (rotation @ xy.T).T
        yaw = yaw - angle

    poses = np.column_stack((xy, yaw)).astype(np.float32)
    if not np.isfinite(poses).all():
        raise ValueError("future poses contain non-finite values")
    return poses


def vehicle_poses_to_lidar_forward_right(poses, lidar_position, augmentation_degrees=0.0):
    """Express vehicle-relative poses as metres from the LiDAR origin.

    This does not discretize to an 8x8 feature grid. It only subtracts the
    LiDAR mounting offset, retaining the vehicle-forward/right axis meaning.
    If ``poses`` were augmented, the mounting offset must be rotated by the
    same angle before subtraction.
    """
    poses = np.asarray(poses)
    lidar_position = np.asarray(lidar_position)
    if poses.shape[-1] != 3 or lidar_position.shape != (3,):
        raise ValueError("expected poses [..., 3] and lidar_position [3]")
    angle = np.deg2rad(float(augmentation_degrees))
    rotation = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)],
    ])
    offset = rotation @ lidar_position[:2]
    result = poses.copy()
    result[..., :2] -= offset
    return result

"""Future scene labels aligned to the current HD465 LiDAR BEV crop.

Mining-specific replacement for WoTE's NAVSIM future map/agent target builder.
The saved 500x500 topdown PNG packs 15 binary planes into three bytes. The
first stored BGR byte contains road/lane; the second contains agents. Do not
use the legacy ``decode_pil_to_npy`` here: it intentionally returns only the
two static planes used by the existing TransFuser auxiliary head.
"""

import cv2
import numpy as np


PIXELS_PER_METER = 5
SOURCE_CENTER = 250
BEV_SIZE = 160
SCENE_LAYERS = ("road", "lane", "vehicle", "pedestrian")


def decode_topdown_scene(encoded_bgr):
    """Return four binary planes [road, lane, vehicle, pedestrian]."""
    image = np.asarray(encoded_bgr)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("encoded topdown must be uint8 [H,W,3]")
    return np.stack(
        [
            (image[..., 0] >> 7) & 1,
            (image[..., 0] >> 6) & 1,
            (image[..., 1] >> 7) & 1,
            (image[..., 1] >> 6) & 1,
        ],
        axis=0,
    ).astype(np.uint8)


def _matrix(value):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError("expected a finite 4x4 pose matrix")
    return result


def future_scene_targets(
    current_ego_matrix,
    future_ego_matrix,
    future_topdown_bgr,
    future_labels,
    lidar_x=3.5,
    augmentation_degrees=0.0,
    max_agents=20,
):
    """Warp the 4-second scene into today's augmented, LiDAR-centered BEV.

    The result map is [4,160,160] at 5 px/m (forward 32 m, lateral +/-16 m).
    ``valid`` marks pixels for which the future 500x500 crop has coverage;
    invalid pixels must be excluded from map loss, not treated as empty road.
    Agent boxes are [x_forward, y_right, yaw, length, width] in today's
    *augmented vehicle frame*, matching the trajectory anchor convention.
    """
    if not isinstance(max_agents, int) or max_agents < 1:
        raise ValueError("max_agents must be a positive integer")
    current = _matrix(current_ego_matrix)
    future = _matrix(future_ego_matrix)
    if np.asarray(future_topdown_bgr).shape != (500, 500, 3):
        raise ValueError("future topdown must have shape [500,500,3]")
    relative = np.linalg.inv(current) @ future
    translation = relative[:2, 3]
    rotation = relative[:2, :2]

    row, col = np.indices((BEV_SIZE, BEV_SIZE), dtype=np.float32)
    # Match load_crop_bev_npy exactly: it rounds the LiDAR mount shift to an
    # integer number of 5 px/m raster cells before rotating/cropping.
    mount_shift_px = int(np.floor(float(lidar_x) * PIXELS_PER_METER + 0.5))
    x_aug = (BEV_SIZE - row) / PIXELS_PER_METER
    y_aug = (col - BEV_SIZE / 2) / PIXELS_PER_METER
    angle = np.deg2rad(float(augmentation_degrees))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x_current = cos_a * x_aug - sin_a * y_aug + mount_shift_px / PIXELS_PER_METER
    y_current = sin_a * x_aug + cos_a * y_aug

    dx = x_current - translation[0]
    dy = y_current - translation[1]
    x_future = rotation[0, 0] * dx + rotation[1, 0] * dy
    y_future = rotation[0, 1] * dx + rotation[1, 1] * dy
    source_row = (SOURCE_CENTER - PIXELS_PER_METER * x_future).astype(np.float32)
    source_col = (SOURCE_CENTER + PIXELS_PER_METER * y_future).astype(np.float32)
    valid = (
        (source_row >= 0) & (source_row <= 499)
        & (source_col >= 0) & (source_col <= 499)
    ).astype(np.uint8)
    warped = cv2.remap(
        np.asarray(future_topdown_bgr), source_col, source_row,
        interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
    )
    scene = decode_topdown_scene(warped)

    # Ego at the candidate endpoint is intentionally NOT baked into this map.
    # WoTE injects each proposed future ego state separately for evaluation.
    boxes = []
    ego_id = future_labels[0]["id"] if future_labels else None
    for label in future_labels:
        if label.get("id") == ego_id or label.get("class") != "Car":
            continue
        actor = np.linalg.inv(current) @ _matrix(label["ego_matrix"])
        actor_x, actor_y = actor[:2, 3]
        x_box = cos_a * actor_x + sin_a * actor_y
        y_box = -sin_a * actor_x + cos_a * actor_y
        yaw = np.arctan2(actor[1, 0], actor[0, 0]) - angle
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        mount_x = float(lidar_x) * cos_a
        mount_y = -float(lidar_x) * sin_a
        x_lidar = x_box - mount_x
        y_lidar = y_box - mount_y
        if not (0 <= x_lidar < 32 and -16 <= y_lidar < 16):
            continue
        extent = np.asarray(label["extent"], dtype=np.float32)
        if extent.shape != (3,):
            raise ValueError("actor extent must have 3 elements")
        boxes.append((x_box, y_box, yaw, extent[1], extent[2]))

    boxes.sort(key=lambda box: box[0] ** 2 + box[1] ** 2)
    padded_boxes = np.zeros((max_agents, 5), dtype=np.float32)
    agent_mask = np.zeros(max_agents, dtype=np.uint8)
    count = min(len(boxes), max_agents)
    if count:
        padded_boxes[:count] = np.asarray(boxes[:count], dtype=np.float32)
        agent_mask[:count] = 1

    return {
        "scene": scene,
        "valid": valid,
        "agent_boxes": padded_boxes,
        "agent_mask": agent_mask,
    }

"""Recorded other-vehicle tracks for offline mining trajectory diagnostics.

These are privileged expert-log futures. They are targets/diagnostics only,
never model inputs, and are not counterfactual reactions to another ego path.
"""

import numpy as np


def recorded_vehicle_tracks(
    current_ego_matrix, labels_by_frame, augmentation_degrees=0.0,
    max_agents=64, max_range_m=60.0, recording_radius_m=50.0,
):
    """Return 9 x M vehicle boxes in the current augmented ego frame.

    Each box is [forward x, right y, yaw, full length, full width]. Track IDs
    are used to interpolate only actors visible at both adjacent time steps.
    """
    if len(labels_by_frame) != 9 or not labels_by_frame[0]:
        raise ValueError("expected current plus eight future label frames")
    if max_agents < 1 or max_range_m <= 0:
        raise ValueError("max_agents and max_range_m must be positive")
    current = np.asarray(current_ego_matrix, dtype=np.float64)
    if current.shape != (4, 4) or not np.isfinite(current).all():
        raise ValueError("current_ego_matrix must be finite [4,4]")
    ego_id = labels_by_frame[0][0]["id"]
    inv_current = np.linalg.inv(current)
    angle = np.deg2rad(float(augmentation_degrees))
    ca, sa = np.cos(angle), np.sin(angle)
    observed = {}
    recorded_ego = np.zeros((9, 3), dtype=np.float32)
    for frame_index, labels in enumerate(labels_by_frame):
        ego_label = next((item for item in labels if item.get("id") == ego_id), None)
        if ego_label is None:
            raise ValueError("ego vehicle missing from recorded frame")
        ego_relative = inv_current @ np.asarray(
            ego_label["ego_matrix"], dtype=np.float64
        )
        ego_x, ego_y = ego_relative[:2, 3]
        recorded_ego[frame_index, :2] = (
            ca * ego_x + sa * ego_y, -sa * ego_x + ca * ego_y
        )
        ego_yaw = np.arctan2(ego_relative[1, 0], ego_relative[0, 0]) - angle
        recorded_ego[frame_index, 2] = np.arctan2(np.sin(ego_yaw), np.cos(ego_yaw))
        for label in labels:
            if label.get("id") == ego_id or label.get("class") != "Car":
                continue
            actor = inv_current @ np.asarray(label["ego_matrix"], dtype=np.float64)
            x, y = actor[:2, 3]
            x_aug, y_aug = ca * x + sa * y, -sa * x + ca * y
            yaw = np.arctan2(actor[1, 0], actor[0, 0]) - angle
            yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
            extent = np.asarray(label["extent"], dtype=np.float64)
            if extent.shape != (3,) or not np.isfinite(extent).all():
                raise ValueError("actor extent must contain three finite values")
            observed.setdefault(label["id"], {})[frame_index] = (
                x_aug, y_aug, yaw, extent[1], extent[2]
            )
    ranked = []
    for actor_id, track in observed.items():
        closest = min(np.hypot(box[0], box[1]) for box in track.values())
        if closest <= max_range_m:
            ranked.append((closest, actor_id))
    ranked.sort(key=lambda item: item[0])
    selected_ids = [actor_id for _, actor_id in ranked[:max_agents]]
    boxes = np.zeros((9, max_agents, 5), dtype=np.float32)
    mask = np.zeros((9, max_agents), dtype=np.uint8)
    for column, actor_id in enumerate(selected_ids):
        for frame_index, box in observed[actor_id].items():
            boxes[frame_index, column] = box
            mask[frame_index, column] = 1
    return {
        "boxes": boxes,
        "mask": mask,
        "track_ids": selected_ids,
        "truncated": len(ranked) > max_agents,
        "recorded_ego": recorded_ego,
        # get_bev_cars records every CARLA vehicle within 50 m of recorded ego.
        "recording_radius_m": float(recording_radius_m),
    }

"""Route-sharded offline metric cache used like WoTE's sim_reward_dict."""

import json
from pathlib import Path

import numpy as np


METRIC_NAMES = (
    "no_collision",
    "drivable_compliance",
    "ego_progress",
    "time_to_collision",
    "comfort",
)


def metric_cache_file_for_collection(route_dir, cache_dir):
    """Resolve one raw collection folder through the cache manifest."""
    cache_dir = Path(cache_dir)
    route_name = Path(route_dir).resolve().name
    manifest_paths = [cache_dir / "manifest.json"]
    if not manifest_paths[0].is_file():
        manifest_paths = sorted(cache_dir.glob("*/manifest.json"))
    matches = []
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tuple(manifest["metric_names"]) != METRIC_NAMES:
            raise ValueError("metric cache order does not match WoTE metric order")
        entry = manifest["routes"].get(route_name)
        if entry is not None:
            matches.append(manifest_path.parent / entry["file"])
    if not matches:
        raise KeyError("route %s missing from metric cache" % route_name)
    if len(matches) != 1:
        raise ValueError("route %s appears in multiple metric caches" % route_name)
    return matches[0]


def load_metric_cache_shard(path, expected_anchor_count=256):
    """Load and validate a route shard, including a frame-to-row lookup."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        frames = archive["frames"].astype(np.int32)
        targets = archive["metric_targets"].astype(np.float32)
        valid = archive["metric_valid"].astype(np.uint8)
    expected = (len(frames), expected_anchor_count, len(METRIC_NAMES))
    if frames.ndim != 1 or targets.shape != expected or valid.shape != expected:
        raise ValueError("invalid metric cache shapes in %s" % path)
    if len(np.unique(frames)) != len(frames):
        raise ValueError("duplicate frame indices in %s" % path)
    if not np.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
        raise ValueError("metric targets must be finite values in [0,1]")
    if not np.isin(valid, (0, 1)).all():
        raise ValueError("metric validity mask must be binary")
    return {
        "frames": frames,
        "metric_targets": targets,
        "metric_valid": valid,
        "frame_to_row": {int(frame): row for row, frame in enumerate(frames)},
    }
