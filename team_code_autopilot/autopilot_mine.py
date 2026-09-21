"""Mining-truck expert for the Komatsu HD465-7E0 CARLA asset.

This module reuses the official TransFuser privileged hazard detector and route
planners, and replaces the passenger-car motion model and the fixed 3/4 m/s
speed choice with a deterministic mining-truck policy.

``autopilot.py`` keeps its original behaviour with one exception: the half-boxes
that ``_get_brake`` builds for the ego collision hull offset their y component by
the half-width instead of the half-length.  ``_get_brake`` is 339 lines and is
inherited unchanged here, so overriding it just to correct four characters would
mean duplicating all of it; the fix lives in ``autopilot.py`` and is commented
there.  The error is heading dependent and left a 0.67 m unchecked gap in the
middle of the 9.4 m truck when it travelled north or south.
"""

from __future__ import print_function

import json
import math
import os
import numpy as np
import carla

from autopilot import AutoPilot
from mining_vehicle_physics import HD465_BLUEPRINT_ID
from nav_planner import PIDController


def get_entry_point():
    return "MiningAutoPilot"


def _wrap_angle_radians(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def estimate_route_curvature(route, sample_spacing=8.0, lookahead=32.0):
    """Estimate the largest absolute curvature ahead, in 1/metres.

    The dense route is sampled by travelled distance before fitting a circle to
    each consecutive triple.  Distance sampling avoids reacting to duplicate or
    unusually dense OpenDRIVE points.
    """
    points = []
    for item in route:
        point = np.asarray(item[0], dtype=np.float64).reshape(-1)[:2]
        if not points or np.linalg.norm(point - points[-1]) > 1e-3:
            points.append(point)

    if len(points) < 3:
        return 0.0

    sampled = [points[0]]
    distance_goal = float(sample_spacing)
    distance_sum = 0.0
    previous = points[0]
    last_considered = points[0]
    for point in points[1:]:
        segment = float(np.linalg.norm(point - previous))
        distance_sum += segment
        previous = point
        last_considered = point
        if distance_sum + 1e-6 >= distance_goal:
            sampled.append(point)
            distance_goal += float(sample_spacing)
        if distance_sum >= float(lookahead):
            break

    if np.linalg.norm(sampled[-1] - last_considered) > sample_spacing * 0.5:
        sampled.append(last_considered)

    curvatures = []
    for p0, p1, p2 in zip(sampled[:-2], sampled[1:-1], sampled[2:]):
        a = float(np.linalg.norm(p1 - p0))
        b = float(np.linalg.norm(p2 - p1))
        c = float(np.linalg.norm(p2 - p0))
        denominator = a * b * c
        if denominator < 1e-6:
            continue
        twice_area = abs(float(np.cross(p1 - p0, p2 - p0)))
        curvatures.append(2.0 * twice_area / denominator)

    return max(curvatures) if curvatures else 0.0


def steering_limit_for_speed(speed_mps):
    """Return the normalized steering limit used to prevent truck rollover."""
    speed_kmh = max(0.0, float(speed_mps)) * 3.6
    speed_axis = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    steer_axis = np.array([1.0, 0.90, 0.70, 0.50, 0.35, 0.25])
    return float(np.interp(speed_kmh, speed_axis, steer_axis))


class MiningEgoModel(object):
    """Effective bicycle model identified from the cooked HD465 asset.

    The wheelbase is the measured 4.298 m.  A full steering command is mapped
    to 0.507 rad so that the model reproduces the measured 8.02 m low-speed
    turning radius.  Longitudinal constants come from the same acceptance run:
    1.75 m/s2 launch acceleration, 3.10 m/s2 full-brake deceleration and
    0.19 m/s2 coast-down.
    """

    def __init__(self, dt=1.0 / 20.0):
        self.dt = float(dt)
        self.front_wb = 2.149
        self.rear_wb = 2.149
        self.steer_gain = 0.507
        self.throttle_accel = 1.75
        self.brake_decel = 3.10
        self.coast_decel = 0.19

    @staticmethod
    def _scalar(value):
        return float(np.asarray(value).reshape(-1)[0])

    def forward(self, locs, yaws, spds, acts):
        action = np.asarray(acts).reshape(-1)
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))

        if brake > 1e-4:
            acceleration = -self.brake_decel * brake
        elif throttle > 1e-4:
            acceleration = self.throttle_accel * throttle
        else:
            acceleration = -self.coast_decel

        wheel_angle = self.steer_gain * steer
        beta = math.atan(
            self.rear_wb / (self.front_wb + self.rear_wb)
            * math.tan(wheel_angle)
        )

        loc = np.asarray(locs, dtype=np.float64).reshape(-1)
        yaw = self._scalar(yaws)
        speed = max(0.0, self._scalar(spds))
        next_loc = np.array([
            loc[0] + speed * math.cos(yaw + beta) * self.dt,
            loc[1] + speed * math.sin(yaw + beta) * self.dt,
        ])
        next_yaw = yaw + speed / self.rear_wb * math.sin(beta) * self.dt
        next_speed = max(0.0, speed + acceleration * self.dt)
        return next_loc, np.array(next_yaw), np.array(next_speed)


class MiningExpertMixin(object):
    """Mining-specific planning and control shared by driving and collection."""

    def setup(self, path_to_conf_file, route_index=None):
        super().setup(path_to_conf_file, route_index)

        self.target_speed_slow = 3.0       # 10.8 km/h: junction/sharp turn
        self.target_speed_normal = 5.0     # 18.0 km/h: normal mine road
        self.target_speed_straight = 6.5   # 23.4 km/h: long straight
        self.max_target_speed = 8.33       # hard ceiling: 30 km/h
        self.lateral_acceleration_limit = 1.10

        # A mining truck needs a longer response horizon than the stock car.
        self.detection_radius = 50.0
        self.extrapolation_seconds_no_junction = 3.0
        self.extrapolation_seconds = 5.0
        self.waypoint_seconds = 5.0
        self.stuck_buffer_size = 80

        # Gentler speed PID.  Grade feed-forward is added in _get_throttle.
        self.clip_delta = 1.5
        self.clip_throttle = 1.0
        self._speed_controller = PIDController(
            K_P=0.28, K_I=0.12, K_D=0.04, n=40
        )
        self._speed_controller_extrapolation = PIDController(
            K_P=0.28, K_I=0.12, K_D=0.04, n=40
        )

        self.ego_model = MiningEgoModel(dt=1.0 / self.frame_rate)
        self.ego_model_gps = MiningEgoModel(dt=1.0 / self.frame_rate_sim)
        # All mine traffic is expected to use the same HD465 blueprint.
        self.vehicle_model = MiningEgoModel(dt=1.0 / self.frame_rate)

        self._mining_target_components = {}
        self._mining_curvature = 0.0
        self._mining_lead_vehicle_id = None
        self._mining_lead_gap_m = None
        self._mining_prev_emergency_brake = False

        if self.save_path is not None:
            expert_config = {
                "version": 1,
                "vehicle_blueprint": HD465_BLUEPRINT_ID,
                "target_speed_slow_mps": self.target_speed_slow,
                "target_speed_normal_mps": self.target_speed_normal,
                "target_speed_straight_mps": self.target_speed_straight,
                "max_target_speed_mps": self.max_target_speed,
                "lateral_acceleration_limit_mps2": (
                    self.lateral_acceleration_limit
                ),
                "detection_radius_m": self.detection_radius,
                "prediction_seconds_no_junction": (
                    self.extrapolation_seconds_no_junction
                ),
                "prediction_seconds_junction": self.extrapolation_seconds,
                "background_traffic": {
                    "vehicle_blueprint": os.environ.get(
                        "BACKGROUND_VEHICLE_MODEL", HD465_BLUEPRINT_ID
                    ),
                    "count_requested": int(os.environ.get(
                        "BACKGROUND_VEHICLE_COUNT", "0"
                    )),
                    "speed_difference_percent": float(os.environ.get(
                        "BACKGROUND_SPEED_DIFFERENCE_PERCENT", "0"
                    )),
                    "minimum_follow_distance_m": float(os.environ.get(
                        "BACKGROUND_MIN_FOLLOW_DISTANCE", "0"
                    )),
                    "automatic_lane_change": os.environ.get(
                        "BACKGROUND_AUTO_LANE_CHANGE", "1"
                    ).strip().lower() in ("1", "true", "yes", "on"),
                    "traffic_manager_seed": int(os.environ.get(
                        "TRAFFIC_MANAGER_SEED", "0"
                    )),
                },
            }
            with open(str(self.save_path / "mining_expert_config.json"), "w") as stream:
                json.dump(expert_config, stream, indent=2)

    def _init(self, hd_map):
        super()._init(hd_map)
        blueprint_id = self._vehicle.type_id
        bbox = self._vehicle.bounding_box
        print(
            "HD465 actor/bounding-box geometry: center=({:.3f}, {:.3f}, {:.3f}) "
            "extent=({:.3f}, {:.3f}, {:.3f})".format(
                bbox.location.x, bbox.location.y, bbox.location.z,
                bbox.extent.x, bbox.extent.y, bbox.extent.z,
            )
        )
        if blueprint_id != HD465_BLUEPRINT_ID:
            print(
                "WARNING: mining expert was calibrated for {}, hero is {}".format(
                    HD465_BLUEPRINT_ID, blueprint_id
                )
            )

    def _grade_speed_cap(self, pitch_deg):
        grade_pct = abs(math.tan(math.radians(float(pitch_deg)))) * 100.0
        uphill = pitch_deg > 0.0
        if uphill:
            if grade_pct >= 15.0:
                return 3.0
            if grade_pct >= 10.0:
                return 3.8
            if grade_pct >= 5.0:
                return 4.5
        else:
            if grade_pct >= 15.0:
                return 2.5
            if grade_pct >= 10.0:
                return 3.3
            if grade_pct >= 5.0:
                return 4.2
        return self.max_target_speed

    def _lead_vehicle_speed_cap(self, ego_speed):
        """Use privileged actor state to keep a deterministic truck headway."""
        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        yaw = math.radians(ego_transform.rotation.yaw)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        ego_half_length = float(self._vehicle.bounding_box.extent.x)

        best = None
        vehicles = self._world.get_actors().filter("*vehicle*")
        for vehicle in vehicles:
            if vehicle.id == self._vehicle.id:
                continue
            location = vehicle.get_location()
            dx = location.x - ego_location.x
            dy = location.y - ego_location.y
            forward = cos_yaw * dx + sin_yaw * dy
            lateral = -sin_yaw * dx + cos_yaw * dy
            if forward <= 0.0 or forward > self.detection_radius:
                continue

            heading_error = abs(_wrap_angle_radians(
                math.radians(vehicle.get_transform().rotation.yaw)
                - yaw
            ))
            lane_gate = max(
                4.0,
                float(self._vehicle.bounding_box.extent.y)
                + float(vehicle.bounding_box.extent.y) + 1.0,
            )
            if heading_error > math.radians(35.0) or abs(lateral) > lane_gate:
                continue

            gap = forward - ego_half_length - float(vehicle.bounding_box.extent.x)
            if best is None or gap < best[0]:
                lead_speed = max(0.0, self._get_forward_speed(
                    transform=vehicle.get_transform(),
                    velocity=vehicle.get_velocity(),
                ))
                best = (gap, vehicle.id, lead_speed)

        if best is None:
            self._mining_lead_vehicle_id = None
            self._mining_lead_gap_m = None
            return self.max_target_speed

        gap, vehicle_id, lead_speed = best
        desired_gap = 8.0 + 2.0 * max(0.0, float(ego_speed))
        cap = lead_speed + 0.25 * (gap - desired_gap)
        self._mining_lead_vehicle_id = int(vehicle_id)
        self._mining_lead_gap_m = float(gap)
        return float(np.clip(cap, 0.0, self.max_target_speed))

    def _get_mining_target_speed(self, waypoint_route, speed):
        curvature = estimate_route_curvature(waypoint_route)
        self._mining_curvature = float(curvature)

        base = (
            self.target_speed_straight
            if curvature < 0.008
            else self.target_speed_normal
        )
        if self.junction:
            base = min(base, self.target_speed_slow)

        curve_cap = self.max_target_speed
        if curvature > 1e-4:
            curve_cap = math.sqrt(
                self.lateral_acceleration_limit / curvature
            )

        pitch = float(self._vehicle.get_transform().rotation.pitch)
        grade_cap = self._grade_speed_cap(pitch)
        lead_cap = self._lead_vehicle_speed_cap(speed)
        end_cap = 0.0 if self._waypoint_planner.is_last else self.max_target_speed

        target = max(0.0, min(
            base, curve_cap, grade_cap, lead_cap, end_cap,
            self.max_target_speed,
        ))
        self._mining_target_components = {
            "base_mps": float(base),
            "curve_cap_mps": float(curve_cap),
            "grade_cap_mps": float(grade_cap),
            "lead_cap_mps": float(lead_cap),
            "end_cap_mps": float(end_cap),
            "selected_mps": float(target),
        }
        return float(target)

    def _controller_throttle(self, target_speed, speed, extrapolation, restore):
        planner = (
            self._waypoint_planner_extrapolation
            if extrapolation else self._waypoint_planner
        )
        if planner.is_last:
            return 0.0

        error = float(target_speed) - float(np.asarray(speed).reshape(-1)[0])
        if error < -0.20:
            return 0.0
        delta = float(np.clip(error, 0.0, self.clip_delta))
        controller = (
            self._speed_controller_extrapolation
            if extrapolation else self._speed_controller
        )
        if restore:
            controller.load()
        throttle = controller.step(delta)
        if restore:
            controller.save()

        pitch = float(self._vehicle.get_transform().rotation.pitch)
        uphill_feed_forward = max(
            0.0, math.sin(math.radians(pitch)) * 5.8
        )
        cruise_feed_forward = 0.12 if target_speed > 0.1 else 0.0
        return float(np.clip(
            throttle + uphill_feed_forward + cruise_feed_forward,
            0.0, self.clip_throttle,
        ))

    def _get_throttle(self, brake, target_speed, speed, restore=True):
        if brake:
            return 0.0
        return self._controller_throttle(
            target_speed, speed, extrapolation=False, restore=restore
        )

    def _get_throttle_extrapolation(self, target_speed, speed, restore=True):
        return self._controller_throttle(
            target_speed, speed, extrapolation=True, restore=restore
        )

    def _service_brake(self, emergency_brake, target_speed, speed):
        if emergency_brake:
            return 1.0
        speed = float(speed)
        if target_speed <= 0.05:
            return 1.0 if speed > 0.10 else 0.50

        overspeed = speed - float(target_speed)
        speed_brake = np.clip((overspeed - 0.25) * 0.22, 0.0, 0.65)
        pitch = float(self._vehicle.get_transform().rotation.pitch)
        downhill_hold = 0.0
        if pitch < 0.0 and speed > target_speed - 0.30:
            downhill_hold = min(
                0.60, -math.sin(math.radians(pitch)) * 3.2
            )
        return float(max(speed_brake, downhill_hold))

    def _get_steer(self, brake, route, pos, theta, speed, restore=True):
        steer = super()._get_steer(
            brake, route, pos, theta, speed, restore=restore
        )
        limit = steering_limit_for_speed(float(np.asarray(speed).reshape(-1)[0]))
        return float(np.clip(steer, -limit, limit))

    def _get_steer_extrapolation(self, route, pos, theta, speed, restore=True):
        steer = super()._get_steer_extrapolation(
            route, pos, theta, speed, restore=restore
        )
        limit = steering_limit_for_speed(float(np.asarray(speed).reshape(-1)[0]))
        return float(np.clip(steer, -limit, limit))

    def _get_control(self, input_data, steer=None, throttle=None,
                     vehicle_hazard=None, light_hazard=None,
                     walker_hazard=None, stop_sign_hazard=None):
        ego_waypoint = self.world_map.get_waypoint(self._vehicle.get_location())
        self.junction = ego_waypoint.is_junction
        speed = float(input_data["speed"][1]["speed"])

        pos = self._get_position(input_data["gps"][1][:2])
        self.gps_buffer.append(pos)
        pos = np.average(self.gps_buffer, axis=0)
        # This override replaces AutoPilot._get_control, so the planner-frame
        # offset the ego forecast needs has to be recorded here too. Without it
        # the forecast keeps the zero initialiser and steers at a target 4.7e6 m
        # away, because the mining maps carry no OpenDRIVE georeference.
        self._update_planner_frame_offset(pos)

        self._waypoint_planner.load()
        waypoint_route = self._waypoint_planner.run_step(pos)
        self._waypoint_planner.save()

        self._waypoint_planner_extrapolation.load()
        self.waypoint_route_extrapolation = (
            self._waypoint_planner_extrapolation.run_step(pos)
        )
        self._waypoint_planner_extrapolation.save()

        target_speed = self._get_mining_target_speed(waypoint_route, speed)
        # _get_brake predicts the ego trajectory at self.target_speed.
        self.target_speed = target_speed

        if (vehicle_hazard is None or light_hazard is None
                or walker_hazard is None or stop_sign_hazard is None):
            emergency_brake = self._get_brake(
                vehicle_hazard, light_hazard, walker_hazard, stop_sign_hazard
            )
        else:
            emergency_brake = bool(
                vehicle_hazard or light_hazard
                or walker_hazard or stop_sign_hazard
            )

        brake = self._service_brake(emergency_brake, target_speed, speed)
        if throttle is None:
            throttle = self._get_throttle(brake, target_speed, speed)

        if steer is None:
            theta = float(input_data["imu"][1][-1])
            if math.isnan(theta):
                theta = 0.0
            # Mild downhill speed control must not halve steering authority.
            steer = self._get_steer(
                emergency_brake, waypoint_route, pos, theta, speed
            )
            self._get_steer_extrapolation(
                waypoint_route, pos, theta, speed
            )

        self.steer_buffer.append(steer)
        control = carla.VehicleControl()
        control.steer = float(np.clip(
            np.mean(self.steer_buffer) + self.steer_noise * np.random.randn(),
            -steering_limit_for_speed(speed),
            steering_limit_for_speed(speed),
        ))
        control.throttle = float(np.clip(throttle, 0.0, 1.0))
        control.brake = float(np.clip(brake, 0.0, 1.0))

        self.steer = control.steer
        self.throttle = control.throttle
        self.brake = control.brake
        self._save_waypoints()

        if self.visualize == 1 and (
                emergency_brake != self._mining_prev_emergency_brake
                or self.step % (2 * self.frame_rate_sim) == 0):
            print(
                "HD465 expert action: speed={:.1f} km/h target={:.1f} km/h "
                "lead_id={} gap={} m steer={:.3f} throttle={:.3f} "
                "brake={:.3f} emergency={}".format(
                    speed * 3.6,
                    target_speed * 3.6,
                    self._mining_lead_vehicle_id,
                    ("-" if self._mining_lead_gap_m is None else
                     "{:.1f}".format(self._mining_lead_gap_m)),
                    control.steer,
                    control.throttle,
                    control.brake,
                    bool(emergency_brake),
                )
            )
        self._mining_prev_emergency_brake = bool(emergency_brake)

        if self.step % self.save_freq == 0 and self.save_path is not None:
            command_route = self._command_planner.run_step(pos)
            far_node, far_command = (
                command_route[1] if len(command_route) > 1
                else command_route[0]
            )
            if (far_node != self.far_node_prev).all():
                self.far_node_prev = far_node
                self.commands.append(far_command.value)

            if not self.render_bev:
                tick_data = self.tick(input_data)
            else:
                tick_data = self.tick(input_data, self.future_states)
            self.save(
                far_node, steer, control.throttle, control.brake,
                target_speed, tick_data,
            )

        return control

    def save(self, far_node, steer, throttle, brake, target_speed, tick_data):
        super().save(
            far_node, steer, throttle, brake, target_speed, tick_data
        )
        frame = self.step // self.save_freq
        path = self.save_path / "measurements" / ("%04d.json" % frame)
        with open(str(path), "r") as stream:
            measurement = json.load(stream)
        measurement.update({
            "mining_route_curvature_1pm": self._mining_curvature,
            "mining_road_pitch_deg": float(
                self._vehicle.get_transform().rotation.pitch
            ),
            "mining_target_components": self._mining_target_components,
            "mining_lead_vehicle_id": self._mining_lead_vehicle_id,
            "mining_lead_gap_m": self._mining_lead_gap_m,
            "mining_expert_version": 1,
        })
        with open(str(path), "w") as stream:
            json.dump(measurement, stream, indent=4)


class MiningAutoPilot(MiningExpertMixin, AutoPilot):
    """Leaderboard entry point for driving without saving camera/LiDAR data."""

    pass
