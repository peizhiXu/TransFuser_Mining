"""Closed-loop inference wrapper and HD465 trajectory controller for WoTE."""

from collections import deque
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch import nn

from team_code_transfuser.transfuser import TransfuserBackbone
from wote_mining.WoTE_model import METRIC_NAMES, WoTEMiningPlanner


class PIDController(object):
    def __init__(self, k_p, k_i, k_d, window):
        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.k_d = float(k_d)
        self.window = deque([0.0] * int(window), maxlen=int(window))

    def step(self, error):
        self.window.append(float(error))
        integral = np.mean(self.window)
        derivative = self.window[-1] - self.window[-2]
        return self.k_p * error + self.k_i * integral + self.k_d * derivative


class HD465TrajectoryController(object):
    """Track vehicle-frame 2 Hz WoTE poses using the mining PID policy."""

    def __init__(self, config):
        self.config = config
        self.turn_controller = PIDController(
            config.turn_KP, config.turn_KI, config.turn_KD, config.turn_n
        )
        self.speed_controller = PIDController(
            config.speed_KP, config.speed_KI, config.speed_KD, config.speed_n
        )

    def control_pid(self, waypoints, velocity, is_stuck, pitch=0.0):
        if waypoints.shape[0] != 1 or waypoints.shape[1] < 2:
            raise ValueError("waypoints must have shape [1,N,2] with N >= 2")
        points = waypoints[0].detach().cpu().numpy()
        speed = float(velocity.reshape(-1)[0].detach().cpu())
        pitch = float(pitch)

        # WoTE samples at 2 Hz, so displacement between adjacent poses times
        # two is the planned speed in m/s. Poses are already vehicle-relative;
        # unlike legacy TransFuser waypoints, no LiDAR x offset is added.
        learned_desired_speed = float(np.linalg.norm(points[1] - points[0]) * 2.0)
        desired_speed = (
            float(self.config.default_speed) if is_stuck else learned_desired_speed
        )
        target_stop = desired_speed < self.config.brake_speed
        if target_stop:
            brake = 1.0 if speed > 0.10 else 0.50
        else:
            overspeed = speed - desired_speed
            speed_brake = np.clip(
                (overspeed - self.config.service_brake_deadband)
                * self.config.service_brake_gain,
                0.0, self.config.max_service_brake,
            )
            downhill_hold = 0.0
            if pitch < 0.0 and speed > desired_speed - 0.30:
                downhill_hold = min(
                    self.config.max_downhill_brake,
                    -np.sin(np.radians(pitch))
                    * self.config.downhill_brake_gain,
                )
            brake = float(max(speed_brake, downhill_hold))

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        if brake > 1e-4:
            throttle = 0.0
        else:
            throttle = self.speed_controller.step(delta)
            uphill = max(
                0.0,
                np.sin(np.radians(pitch))
                * self.config.uphill_feed_forward_gain,
            )
            cruise = self.config.cruise_feed_forward if desired_speed > 0.1 else 0.0
            throttle = np.clip(
                throttle + uphill + cruise, 0.0, self.config.clip_throttle
            )

        aim = (points[1] + points[0]) / 2.0
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if speed < 0.01 or target_stop:
            angle = 0.0
        steer = float(np.clip(self.turn_controller.step(angle), -1.0, 1.0))
        telemetry = {
            "learned_desired_speed": learned_desired_speed,
            "desired_speed": desired_speed,
            "speed_error": desired_speed - speed,
            "pitch": pitch,
            "target_stop": bool(target_stop),
        }
        return steer, float(throttle), float(brake), telemetry


class WoTEMiningInferenceModel(nn.Module):
    """Expose WoTE through the legacy Agent's forward/control interface."""

    def __init__(self, config, anchors_path, image_architecture='regnety_032',
                 lidar_architecture='regnety_032', use_velocity=False,
                 planner=None):
        super().__init__()
        if planner is None:
            backbone = TransfuserBackbone(
                config, image_architecture=image_architecture,
                lidar_architecture=lidar_architecture,
                use_velocity=use_velocity,
            )
            planner = WoTEMiningPlanner(backbone, anchors_path)
        self.planner = planner
        self.config = config
        self.controller = HD465TrajectoryController(config)
        self.last_diagnostics = {}
        self.debug_step = 0
        self._pending_debug = None

    @staticmethod
    def _uint8_image(image):
        image = np.asarray(image)
        if image.size and float(image.max()) <= 1.0:
            image = image * 255.0
        return np.clip(image, 0.0, 255.0).astype(np.uint8)

    def _cache_debug_frame(self, rgb, lidar_bev, target_point, selected,
                           ego_vel, save_path, stuck_detector, forced_move):
        rgb_image = rgb[0].detach().float().cpu().permute(1, 2, 0).numpy()
        self._pending_debug = {
            "rgb": self._uint8_image(rgb_image)[:, :, ::-1],
            "lidar": lidar_bev[0, :2].detach().float().cpu().numpy(),
            "target_point": target_point[0].detach().float().cpu().numpy(),
            "trajectory": selected[0, :, :2].detach().float().cpu().numpy(),
            "speed": float(ego_vel.reshape(-1)[0].detach().cpu()),
            "save_path": str(save_path),
            "frame": self.debug_step // 2,
            "stuck_detector": int(stuck_detector),
            "forced_move": bool(forced_move),
            "diagnostics": dict(self.last_diagnostics),
        }

    def save_debug_frame(self, telemetry):
        data = self._pending_debug
        self._pending_debug = None
        if data is None:
            return

        lidar = np.clip(data["lidar"], 0.0, 1.0)
        bev = np.zeros((256, 256, 3), dtype=np.uint8)
        # Low/ground returns are green; high/obstacle returns are red.
        bev[:, :, 1] = (lidar[1] * 210.0).astype(np.uint8)
        bev[:, :, 2] = (lidar[0] * 255.0).astype(np.uint8)
        for pixel in range(0, 257, 32):
            coordinate = min(pixel, 255)
            cv2.line(bev, (coordinate, 0), (coordinate, 255), (45, 45, 45), 1)
            cv2.line(bev, (0, coordinate), (255, coordinate), (45, 45, 45), 1)

        pixels_per_meter = float(self.config.pixels_per_meter)
        lidar_x = float(self.config.lidar_pos[0])

        def metric_to_pixel(point):
            forward = float(point[0]) - lidar_x
            right = float(point[1])
            return (
                int(round(128.0 + right * pixels_per_meter)),
                int(round(256.0 - forward * pixels_per_meter)),
            )

        previous = None
        for index, point in enumerate(data["trajectory"]):
            pixel = metric_to_pixel(point)
            if 0 <= pixel[0] < 256 and 0 <= pixel[1] < 256:
                if previous is not None:
                    cv2.line(bev, previous, pixel, (255, 255, 255), 2)
                previous = pixel
                cv2.circle(bev, pixel, 4, (255, 255, 255), -1)
                cv2.putText(
                    bev, str(index + 1), (pixel[0] + 4, pixel[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                    cv2.LINE_AA,
                )

        target_pixel = metric_to_pixel(data["target_point"])
        if 0 <= target_pixel[0] < 256 and 0 <= target_pixel[1] < 256:
            cv2.drawMarker(
                bev, target_pixel, (0, 255, 255), cv2.MARKER_CROSS, 12, 2
            )
        cv2.circle(bev, (128, 255), 5, (255, 255, 0), -1)
        cv2.putText(
            bev, "LiDAR BEV / selected 4 s trajectory", (7, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

        bev_panel = cv2.resize(bev, (384, 384), interpolation=cv2.INTER_NEAREST)
        info = np.full((384, 576, 3), 24, dtype=np.uint8)
        diagnostics = data["diagnostics"]
        metrics = diagnostics.get("selected_metric_scores", [])
        lines = [
            "WoTE HD465 closed-loop",
            "frame: %d" % data["frame"],
            "anchor: %s" % diagnostics.get("selected_anchor_index", "n/a"),
            "reward: %.5f" % diagnostics.get("selected_reward", float("nan")),
            "speed: %.2f m/s   desired: %.2f m/s" % (
                data["speed"], telemetry.get("desired_speed", float("nan"))
            ),
            "steer: %+.3f   throttle: %.3f   brake: %.3f" % (
                telemetry.get("steer", float("nan")),
                telemetry.get("throttle", float("nan")),
                telemetry.get("brake", float("nan")),
            ),
            "target: (%.2f, %.2f) m" % tuple(data["target_point"][:2]),
            "stuck: %d   recovery: %s" % (
                data["stuck_detector"], data["forced_move"]
            ),
            "metric rewards:",
        ]
        for name, value in zip(METRIC_NAMES, metrics):
            lines.append("  %-21s %.4f" % (name, value))
        for line_index, line in enumerate(lines):
            color = (90, 220, 255) if line_index == 0 else (235, 235, 235)
            cv2.putText(
                info, line, (18, 26 + line_index * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.57, color, 1, cv2.LINE_AA,
            )

        lower = np.concatenate((bev_panel, info), axis=1)
        rgb_panel = cv2.resize(
            data["rgb"], (960, 218), interpolation=cv2.INTER_AREA
        )
        cv2.putText(
            rgb_panel, "left / front / right camera panorama", (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            cv2.LINE_AA,
        )
        composite = np.concatenate((rgb_panel, lower), axis=0)
        save_path = Path(data["save_path"])
        save_path.mkdir(parents=True, exist_ok=True)
        output = save_path / ("%06d.png" % data["frame"])
        try:
            written = cv2.imwrite(
                str(output), composite, [cv2.IMWRITE_PNG_COMPRESSION, 2]
            )
            if not written:
                print("WARNING: failed to write WoTE debug frame: %s" % output)
        except (OSError, cv2.error) as error:
            print("WARNING: failed to write WoTE debug frame %s: %s" % (
                output, error
            ))

    def forward_ego(self, rgb, lidar_bev, target_point, target_point_image,
                    ego_vel, **kwargs):
        lidar_visual = lidar_bev
        if self.config.use_target_point_image:
            lidar_bev = torch.cat((lidar_bev, target_point_image), dim=1)
        outputs = self.planner(
            rgb, lidar_bev, ego_vel, target_point,
            augmentation_degrees=ego_vel.new_zeros(ego_vel.shape[0]),
            predict_future_map=False,
            predict_auxiliary=False,
        )
        selected = outputs["selected_trajectory"]
        batch_index = torch.arange(selected.shape[0], device=selected.device)
        selected_local = outputs["selected_index"]
        selected_metrics = outputs["metric_scores"][batch_index, selected_local]
        selected_rewards = outputs["final_rewards"][batch_index, selected_local]
        self.last_diagnostics = {
            "selected_anchor_index": int(
                outputs["selected_anchor_index"][0].detach().cpu()
            ),
            "selected_reward": float(selected_rewards[0].detach().cpu()),
            "selected_metric_scores": [
                float(value) for value in selected_metrics[0].detach().cpu()
            ],
        }
        self.debug_step += 1
        if (kwargs.get("debug", False) and kwargs.get("save_path") is not None
                and self.debug_step % 2 == 0):
            self._cache_debug_frame(
                rgb, lidar_visual, target_point, selected, ego_vel,
                kwargs["save_path"], kwargs.get("stuck_detector", 0),
                kwargs.get("forced_move", False),
            )
        else:
            self._pending_debug = None
        return selected[..., :2], []

    def control_pid(self, waypoints, velocity, is_stuck, pitch=0.0):
        return self.controller.control_pid(
            waypoints, velocity, is_stuck, pitch=pitch
        )

"""CARLA Leaderboard entry point for the HD465 WoTE planner."""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from team_code_transfuser.submission_agent import HybridAgent
from wote_mining.WoTE_config import WoTEMiningConfig


def get_entry_point():
    return 'WoTEMiningAgent'


class WoTEMiningAgent(HybridAgent):
    """Reuse mining sensors/recovery while replacing the learned planner."""

    def _make_config(self):
        return WoTEMiningConfig(setting='eval')

    def _configure_after_args(self):
        if self.backbone != 'transFuser':
            raise ValueError('WoTE mining Agent requires the transFuser backbone')
        if self.config.use_point_pillars:
            raise ValueError('WoTE mining Agent currently requires histogram LiDAR BEV')
        self.config.multitask = False

    def setup(self, path_to_conf_file, route_index=None):
        super().setup(path_to_conf_file, route_index=route_index)
        if self.artifact_dir is not None:
            metadata_path = self.artifact_dir / 'run_metadata.json'
            if metadata_path.is_file():
                with metadata_path.open('r', encoding='utf-8') as stream:
                    metadata = json.load(stream)
                metadata['composite_contents'] = [
                    'rgb_panorama', 'lidar_bev', 'selected_trajectory',
                    'selected_anchor', 'reward_metrics', 'vehicle_control',
                ]
                with metadata_path.open('w', encoding='utf-8') as stream:
                    json.dump(metadata, stream, indent=2)

    def _checkpoint_files(self, path_to_conf_file):
        requested = os.environ.get('WOTE_CHECKPOINT')
        if requested:
            path = Path(requested)
            if not path.is_absolute():
                path = Path(path_to_conf_file) / path
            if not path.is_file():
                raise FileNotFoundError(str(path))
            return [str(path)] if path.parent != Path(path_to_conf_file) else [path.name]
        latest = Path(path_to_conf_file) / 'latest.pth'
        if latest.is_file():
            return ['latest.pth']
        numbered = sorted(Path(path_to_conf_file).glob('checkpoint_*.pth'))
        return [numbered[-1].name] if numbered else []

    def _build_network(self, image_architecture, lidar_architecture,
                       use_velocity):
        anchors = os.environ.get('WOTE_ANCHORS', self.args.get('anchors'))
        if anchors is None or not Path(anchors).is_file():
            anchors = (
                ROOT / 'wote_mining/assets/anchors/trajectory_anchors_256.npy'
            )
        if not Path(anchors).is_file():
            raise FileNotFoundError('WoTE anchor file not found: %s' % anchors)
        return WoTEMiningInferenceModel(
            self.config, anchors,
            image_architecture=image_architecture,
            lidar_architecture=lidar_architecture,
            use_velocity=use_velocity,
        )

    def _checkpoint_state_dict(self, checkpoint):
        if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
            raise ValueError('WoTE checkpoint must contain a model state_dict')
        return checkpoint['model']

    def _strict_checkpoint_loading(self):
        return True

    def _write_telemetry(self, telemetry):
        if self.nets:
            telemetry.update(self.nets[0].last_diagnostics)
            self.nets[0].save_debug_frame(telemetry)
        super()._write_telemetry(telemetry)
