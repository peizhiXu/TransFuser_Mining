"""Data-collection entry point for the HD465 mining expert."""

from __future__ import print_function

import json
import math

import carla
import numpy as np

from autopilot_mine import MiningExpertMixin
from data_agent import DataAgent
from config import GlobalConfig


def get_entry_point():
    return "MiningDataAgent"


class MiningDataAgent(MiningExpertMixin, DataAgent):
    """Collect TransFuser-compatible data from the Komatsu HD465-7E0."""

    # The HD465 values below were checked with RGB and LiDAR on 0325_5: no
    # self-occlusion was visible and there were no LiDAR returns within five
    # metres.
    #
    # They are read from GlobalConfig rather than written twice.  submission_agent
    # mounts its sensors from the same fields and data.py derives the LiDAR BEV
    # ground split from them, so a second copy here would let the collected data
    # and the deployed model drift apart without anything failing loudly.
    # Read off the class, not an instance: GlobalConfig.__init__ wants a dataset
    # root that does not exist during collection.
    camera_x, camera_y, camera_z = [float(v) for v in GlobalConfig.camera_pos]
    lidar_x, lidar_y, lidar_z = [float(v) for v in GlobalConfig.lidar_pos]

    def setup(self, path_to_conf_file, route_index=None):
        super().setup(path_to_conf_file, route_index)
        self._vehicle_lights_applied = False
        # Three 320x160 images are concatenated into the existing raw 960x160
        # format.  The training loader then crops it to 704x160.
        self.cam_config = {
            "width": int(GlobalConfig.camera_crop_width),
            "height": int(GlobalConfig.camera_crop_height),
            "fov": float(GlobalConfig.camera_crop_fov),
        }
        if self.save_path is not None:
            sensor_config = {
                "version": 1,
                "camera_position_m": [
                    self.camera_x, self.camera_y, self.camera_z
                ],
                "camera_yaws_deg": [-60.0, 0.0, 60.0],
                "camera_each_resolution": [
                    self.cam_config["width"], self.cam_config["height"]
                ],
                "camera_fov_deg": self.cam_config["fov"],
                "lidar_position_m": [
                    self.lidar_x, self.lidar_y, self.lidar_z
                ],
                "lidar_rotation_deg": [0.0, 0.0, -90.0],
                # leaderboard/autoagents/agent_wrapper.py overwrites every LiDAR
                # attribute unconditionally, so whatever sensors() asks for is
                # ignored.  Record what actually gets used.
                "lidar_range_m": 85,
                "lidar_channels": 64,
                "lidar_upper_fov_deg": 10,
                "lidar_lower_fov_deg": -30,
                "lidar_rotation_frequency_hz": 10,
                "lidar_points_per_second": 600000,
                "lidar_attributes_source": (
                    "leaderboard/leaderboard/autoagents/agent_wrapper.py"
                ),
                # A -30 degree lower FOV at a 4.8 m mount leaves the road
                # invisible within 8.3 m of the sensor; the sedan mount at 2.5 m
                # gave 4.3 m.  Objects taller than the ground are still seen.
                "ground_blind_radius_m": round(
                    self.lidar_z / math.tan(math.radians(30.0)), 2
                ),
                "save_frequency_hz": 2,
                "weather_policy": "fixed_from_route_xml",
                "bev_render_device": str(self.datagen_device),
            }
            with open(str(self.save_path / "mining_sensor_config.json"), "w") as stream:
                json.dump(sensor_config, stream, indent=2)

    def shuffle_weather(self):
        """Keep one physically coherent weather condition for the full route.

        DataAgent calls this after every saved frame. The upstream behaviour is
        useful as image augmentation, but changes weather every 0.5 seconds and
        makes a route's XML weather label meaningless. Mining routes already
        provide explicit weather, so leave the evaluator-selected value intact.

        Upstream also drives the vehicle lights from this method. The evaluator
        only switches the ego's lights on at night, so dropping it wholesale
        would leave all background trucks unlit on a night route.
        """
        self._apply_vehicle_lights()

    def _apply_vehicle_lights(self):
        """Light the background trucks to match the route's fixed weather.

        The weather does not change during a route, so this only has to run
        once, after the background traffic has spawned.
        """
        if self._vehicle_lights_applied:
            return
        night = self._world.get_weather().sun_altitude_angle < 0.0
        state = carla.VehicleLightState(
            carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        ) if night else carla.VehicleLightState.NONE
        for vehicle in self._world.get_actors().filter('*vehicle*'):
            vehicle.set_light_state(state)
        print('Vehicle lights: {} ({} vehicles)'.format(
            'on' if night else 'off',
            len(self._world.get_actors().filter('*vehicle*'))), flush=True)
        self._vehicle_lights_applied = True

    def sensors(self):
        sensors = super().sensors()
        camera_ids = {
            "rgb_front", "rgb_left", "rgb_right",
            "semantics_front", "semantics_left", "semantics_right",
            "depth_front", "depth_left", "depth_right",
        }
        for sensor in sensors:
            if sensor.get("id") in camera_ids:
                sensor.update({
                    "x": self.camera_x,
                    "y": self.camera_y,
                    "z": self.camera_z,
                })
            elif sensor.get("id") == "lidar":
                sensor.update({
                    "x": self.lidar_x,
                    "y": self.lidar_y,
                    "z": self.lidar_z,
                })
        return sensors

    def get_relative_transform(self, ego_matrix, vehicle_matrix):
        relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
        relative_pos = ego_matrix[:3, :3].T @ relative_pos
        relative_pos[1] = -relative_pos[1]
        return relative_pos - np.array([
            self.lidar_x, self.lidar_y, self.lidar_z
        ])

    def get_lidar_to_vehicle_transform(self):
        rotation = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[0, 3] = self.lidar_x
        transform[1, 3] = self.lidar_y
        transform[2, 3] = self.lidar_z
        return transform

    def get_image_to_vehicle_transform(self):
        transform = np.eye(4)
        transform[0, 3] = self.camera_x
        transform[1, 3] = self.camera_y
        transform[2, 3] = self.camera_z
        vehicle_to_image_rotation = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
        ], dtype=np.float32)
        transform[:3, :3] = vehicle_to_image_rotation.T
        return transform
