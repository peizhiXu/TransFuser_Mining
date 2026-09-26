#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime, timedelta
from distutils.version import LooseVersion
import importlib
import os
import pkg_resources
import sys
import carla
import signal
import ctypes
import gc

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager_local import ScenarioManager
from leaderboard.scenarios.route_scenario_local import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper_local import  AgentWrapper, AgentError
from leaderboard.utils.statistics_manager_local import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer


sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer',
    'sensor.stitch_camera.rgb': 'carla_camera',  # for local World on Rails evaluation
    'sensor.camera.semantic_segmentation': 'carla_camera', # for datagen
    'sensor.camera.depth':      'carla_camera', # for datagen
}


class LeaderboardEvaluator(object):

    """
    TODO: document me!
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    @staticmethod
    def _trace_memory(label):
        """Emit stage-level RSS diagnostics when MEMORY_TRACE=1 is set."""
        if os.environ.get('MEMORY_TRACE', '0') != '1':
            return
        values = {}
        with open('/proc/self/status') as status_file:
            for line in status_file:
                if line.startswith(('VmRSS:', 'VmHWM:')):
                    key, value = line.split(':', 1)
                    values[key] = int(value.split()[0]) // 1024
        message = '[memory] {} rss={}MiB hwm={}MiB'.format(
            label, values.get('VmRSS', 0), values.get('VmHWM', 0))
        print(message, flush=True)
        output_root = os.environ.get('OUTPUT_ROOT')
        run_name = os.environ.get('RUN_NAME')
        if output_root and run_name:
            with open(os.path.join(output_root, run_name + '.memory_trace.log'),
                      'a') as trace_file:
                trace_file.write('{} {}\n'.format(
                    datetime.now().isoformat(timespec='seconds'), message))

    def __init__(self, args, statistics_manager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.statistics_manager = statistics_manager
        self.sensors = None
        self.sensor_icons = []
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(args.host, int(args.port))
        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        # Do not load Town01 here.  The first route below loads its requested
        # town.  Loading an unused world is expensive for large mining maps.
        self.world = self.client.get_world()
        # Keep the route-file town name locally.  On these custom mining maps,
        # world.get_map() serializes a very large map object to the Python
        # client, so querying it again between two routes can exhaust RAM.
        self._loaded_town = None
        self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))

        dist = pkg_resources.get_distribution("carla")
        if dist.version != 'leaderboard':
            if LooseVersion(dist.version) < LooseVersion('0.9.10'):
                raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, args.debug > 1)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Create the agent timer
        self._agent_watchdog = Watchdog(int(float(args.timeout)))
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took too long to setup")
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _cleanup(self, config=None):
        """
        Remove and destroy all actors
        """

        # Keep the simulator in synchronous mode between routes.  Switching
        # this very large custom map back to async and then synchronizing it
        # again on the next route forces CARLA 0.9.10 to allocate a large
        # transient world snapshot in the Python client.  The collector owns
        # this CARLA instance and shuts it down after the batch, so leaving it
        # synchronous between routes is safe.
        if self.manager and self.manager.get_running_status() \
                and hasattr(self, 'world') and self.world:
            pass

        if self.manager:
            self.manager.cleanup()

        # ``CarlaDataProvider.set_world`` calls ``world.get_map()`` and
        # rebuilds all spawn points.  For the very large custom mining maps,
        # doing that again for every route makes the CARLA Python client retain
        # several GiB during the next-route transition.  Keep only immutable
        # map state when the CARLA world itself is retained; dynamic actor
        # registries are still cleared by the normal cleanup below.
        provider_static_state = None
        if (self._loaded_town is not None and
                CarlaDataProvider._world is self.world and
                CarlaDataProvider._map is not None):
            provider_static_state = {
                'client': CarlaDataProvider._client,
                'world': CarlaDataProvider._world,
                'map': CarlaDataProvider._map,
                'blueprints': CarlaDataProvider._blueprint_library,
                'spawn_points': CarlaDataProvider._spawn_points,
                'traffic_lights': CarlaDataProvider._traffic_light_map.copy(),
            }

        CarlaDataProvider.cleanup()

        if provider_static_state is not None:
            CarlaDataProvider._client = provider_static_state['client']
            CarlaDataProvider._world = provider_static_state['world']
            CarlaDataProvider._map = provider_static_state['map']
            CarlaDataProvider._blueprint_library = provider_static_state['blueprints']
            CarlaDataProvider._spawn_points = provider_static_state['spawn_points']
            CarlaDataProvider._traffic_light_map = provider_static_state['traffic_lights']

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

        # RouteIndexer retains every RouteConfiguration for the full batch.
        # Leaving the agent here keeps its planners, sensor interface and map
        # data alive after a route has finished.
        if config is not None:
            config.agent = None

        if hasattr(self, 'statistics_manager') and self.statistics_manager:
            self.statistics_manager.scenario = None

        gc.collect()
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError):
            pass

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicles
        """

        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                             vehicle.transform,
                                                                             vehicle.rolename,
                                                                             color=vehicle.color,
                                                                             vehicle_category=vehicle.category))

        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """

        # Never call world.get_map() here.  Besides returning a UE asset path
        # rather than the short route name, CARLA 0.9.10 serializes this large
        # custom OpenDRIVE map to the evaluator on each call.  We already know
        # which town was loaded because this method performs every load.
        loaded_new_world = self._loaded_town != town
        if loaded_new_world:
            print("> Loading map {}".format(town))
            self.world = self.client.load_world(town)
            self._loaded_town = town
        else:
            # Route actors and sensors are destroyed in _cleanup().  Keeping
            # the already-loaded map avoids CARLA 0.9.10 retaining another
            # copy of a large mining world for every route in the same batch.
            print("> Reusing loaded map {}".format(town))
        self._trace_memory('{}: world selected'.format(town))
        settings = self.world.get_settings()
        self._trace_memory('{}: settings fetched'.format(town))
        desired_delta = 1.0 / self.frame_rate
        if (not settings.synchronous_mode or
                settings.fixed_delta_seconds != desired_delta):
            settings.fixed_delta_seconds = desired_delta
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            self._trace_memory('{}: settings applied'.format(town))
        else:
            self._trace_memory('{}: settings retained'.format(town))

        if loaded_new_world:
            self.world.reset_all_traffic_lights()
            self._trace_memory('{}: lights reset'.format(town))
        else:
            self._trace_memory('{}: lights retained'.format(town))

        CarlaDataProvider.set_client(self.client)
        if (CarlaDataProvider._world is not self.world or
                CarlaDataProvider._map is None):
            CarlaDataProvider.set_world(self.world)
            self._trace_memory('{}: provider rebuilt'.format(town))
        else:
            # Static map, spawn points and traffic-light data were retained
            # across a route on this same map.  Only the dynamic sync flag
            # changes with every route.
            CarlaDataProvider._sync_flag = True
            self._trace_memory('{}: provider reused'.format(town))
        CarlaDataProvider.set_traffic_manager_port(int(args.trafficManagerPort))

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(args.trafficManagerSeed))
        self._trace_memory('{}: traffic manager configured'.format(town))

        # Optional background-traffic tuning. The mining launcher sets these
        # because its OpenDRIVE roads advertise 35-40 mph limits and the HD465
        # needs a much larger following gap than a passenger car. Other
        # launchers keep the upstream Traffic Manager defaults.
        speed_difference = os.environ.get(
            'BACKGROUND_SPEED_DIFFERENCE_PERCENT')
        if speed_difference is not None:
            self.traffic_manager.global_percentage_speed_difference(
                float(speed_difference))

        follow_distance = os.environ.get('BACKGROUND_MIN_FOLLOW_DISTANCE')
        if follow_distance is not None:
            self.traffic_manager.set_global_distance_to_leading_vehicle(
                float(follow_distance))

        if speed_difference is not None or follow_distance is not None:
            print(
                'Traffic Manager: speed difference={}%, minimum gap={} m'.format(
                    speed_difference if speed_difference is not None else 'default',
                    follow_distance if follow_distance is not None else 'default',
                )
            )

        # Wait for the world to be ready
        if loaded_new_world and CarlaDataProvider.is_sync_mode():
            self.world.tick()
        elif loaded_new_world:
            self.world.wait_for_tick()
        self._trace_memory('{}: world ready for route'.format(town))

        if self._loaded_town != town:
            raise Exception("The CARLA server uses the wrong map!"
                            "This scenario requires to use map {}".format(town))

    @staticmethod
    def _trajectory_length(config):
        points = getattr(config, 'trajectory', None) or []
        return sum(points[i].distance(points[i + 1])
                   for i in range(len(points) - 1))

    def _print_route_progress(self, route_indexer, config):
        """Report what a finished route cost and extrapolate the batch ETA.

        Every finished route records its own length and wall-clock duration, so
        the remaining time follows from the measured seconds-per-metre instead
        of a fixed guess.
        """
        records = self.statistics_manager._registry_route_records
        done_metres = done_seconds = 0.0
        for record in records:
            if record.status != 'Completed':
                continue
            meta = getattr(record, 'meta', None) or {}
            done_metres += float(meta.get('route_length', 0.0))
            done_seconds += float(meta.get('duration_system', 0.0))

        current = records[config.index]
        current_meta = getattr(current, 'meta', None) or {}
        print('\033[1m[progress] {} {} - {:.0f} m in {:.1f} min\033[0m'.format(
            config.name, current.status,
            float(current_meta.get('route_length', 0.0)),
            float(current_meta.get('duration_system', 0.0)) / 60.0))

        pending = route_indexer._configs_list[route_indexer._index:]
        remaining_metres = sum(self._trajectory_length(item[1]) for item in pending)
        line = '[progress] batch route {}/{}, {:.1f} km collected'.format(
            route_indexer._index, route_indexer.total, done_metres / 1000.0)
        if done_metres > 0 and done_seconds > 0 and remaining_metres > 0:
            eta = remaining_metres * (done_seconds / done_metres)
            finish = datetime.now() + timedelta(seconds=eta)
            line += ', {:.1f} km left, ~{:.0f} min to go (batch ends ~{})'.format(
                remaining_metres / 1000.0, eta / 60.0, finish.strftime('%H:%M'))
        elif not pending:
            line += ', batch complete'
        print(line, flush=True)

    def _register_statistics(self, config, checkpoint, entry_status, crash_message=""):
        """
        Computes and saved the simulation statistics
        """
        # register statistics
        current_stats_record = self.statistics_manager.compute_route_statistics(
            config,
            self.manager.scenario_duration_system,
            self.manager.scenario_duration_game,
            crash_message
        )

        print("\033[1m> Registering the route statistics\033[0m")
        self.statistics_manager.save_record(current_stats_record, config.index, checkpoint)
        self.statistics_manager.save_entry_status(entry_status, False, checkpoint)

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"

        self._trace_memory('{}: begin'.format(config.name))

        print("\n\033[1m========= Preparing {} (repetition {}) =========".format(config.name, config.repetition_index))
        print("> Setting up the agent\033[0m")

        # Prepare the statistics of the route
        self.statistics_manager.set_route(config.name, config.index)
        # Expose stable per-route identity to the agent so debug artifacts from
        # a multi-route batch are isolated instead of overwriting one another.
        os.environ['LEADERBOARD_ROUTE_INDEX'] = str(config.index)
        os.environ['LEADERBOARD_ROUTE_ID'] = config.name.rsplit('_', 1)[-1]
        os.environ['LEADERBOARD_ROUTE_NAME'] = config.name
        os.environ['LEADERBOARD_REPETITION'] = str(config.repetition_index)
        if int(os.environ['DATAGEN'])==1:
            # A single-route retry is re-indexed to zero, which used to force
            # the exact same background spawn layout on every retry.  Allow a
            # collection launcher to vary spawn locations independently from
            # the Traffic Manager behaviour seed.  Full-batch collection keeps
            # the original per-route index behaviour when the variable is not
            # supplied.
            spawn_seed = int(os.environ.get(
                'BACKGROUND_SPAWN_SEED', config.index
            ))
            CarlaDataProvider._rng = random.RandomState(spawn_seed)
            print('Background spawn-point seed: {}'.format(spawn_seed))

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            if int(os.environ['DATAGEN'])==1:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config, config.index)
            else:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config)
            config.agent = self.agent_instance
            self._trace_memory('{}: agent ready'.format(config.name))

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons, args.checkpoint)

            self._agent_watchdog.stop()

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent's sensors were invalid"
            entry_status = "Rejected"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup(config)
            sys.exit(-1)

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent couldn't be set up"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup(config)
            return

        print("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        try:
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
            self._trace_memory('{}: world ready'.format(config.name))
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            self._trace_memory('{}: ego spawned'.format(config.name))
            scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)
            self._trace_memory('{}: scenario built'.format(config.name))
            self.statistics_manager.set_scenario(scenario.scenario)

            # Night mode
            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(scenario, self.agent_instance, config.repetition_index)
            self._trace_memory('{}: sensors ready'.format(config.name))

        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup(config)
            sys.exit(-1)

        print("\033[1m> Running the route\033[0m")

        # Run the scenario
        try:
            self.manager.run_scenario()

        except AgentError as e:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent crashed"

        except Exception as e:
            print("\n\033[91mError during the simulation:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            # Remove all actors
            scenario.remove_all_actors()
            self._trace_memory('{}: actors removed'.format(config.name))

            self._cleanup(config)
            self._trace_memory('{}: cleanup complete'.format(config.name))

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"

        if crash_message == "Simulation crashed":
            sys.exit(-1)

    def run(self, args):
        """
        Run the challenge mode
        """
        route_ids = [route_id.strip() for route_id in args.route_ids.split(',')
                     if route_id.strip()]
        route_indexer = RouteIndexer(
            args.routes, args.scenarios, args.repetitions,
            args.route_id or None, route_ids or None)

        if args.resume:
            route_indexer.resume(args.checkpoint)
            self.statistics_manager.resume(args.checkpoint)
        else:
            self.statistics_manager.clear_record(args.checkpoint)
            route_indexer.save_state(args.checkpoint)

        while route_indexer.peek():
            # setup
            config = route_indexer.next()

            # run
            self._load_and_run_scenario(args, config)

            # Formal data generation must not silently advance past a failed
            # expert route. Leave RouteIndexer progress at the previous route;
            # the batch wrapper will quarantine the partial directory and retry
            # this same index after the cause has been fixed.
            if int(os.environ.get('DATAGEN', '0')) == 1:
                route_record = self.statistics_manager._registry_route_records[config.index]
                if route_record.status != 'Completed':
                    raise RuntimeError(
                        'Data-generation route {} failed: {}'.format(
                            config.index, route_record.status
                        )
                    )

            route_indexer.save_state(args.checkpoint)
            self._print_route_progress(route_indexer, config)

        # save global statistics
        print("\033[1m> Registering the global statistics\033[0m")
        global_stats_record = self.statistics_manager.compute_global_statistics(route_indexer.total)
        StatisticsManager.save_global_record(global_stats_record, self.sensor_icons, route_indexer.total, args.checkpoint)


def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='8000',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="60.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        required=True)
    parser.add_argument('--route-id', default='',
                        help='Run one XML route ID only; intended for guarded smoke tests.')
    parser.add_argument('--route-ids', default='',
                        help='Run comma-separated XML route IDs in order; for guarded multi-route tests.')
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.',
                        required=True)
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")

    arguments = parser.parse_args()

    statistics_manager = StatisticsManager()

    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
        leaderboard_evaluator.run(arguments)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


if __name__ == '__main__':
    main()
