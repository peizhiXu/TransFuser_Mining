import os
import math
from copy import deepcopy
from collections import deque
import xml.etree.ElementTree as ET
import numpy as np

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

DEBUG = False

# Building GlobalRoutePlanner for a large mining OpenDRIVE map creates a very
# large topology graph.  A Leaderboard batch has many routes on the same map,
# so construct that immutable graph once and reuse it for route tracing.
_GLOBAL_ROUTE_PLANNER_CACHE = {}


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._saved_window = deque([0 for _ in range(n)], maxlen=n)
        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        if len(self._window) >= 2:
            integral = sum(self._window)/len(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        if DEBUG:
            self._max = max(self._max, abs(error))
            self._min = -abs(self._max)

            import cv2

            canvas = np.ones((100, 100, 3), dtype=np.uint8)
            w = int(canvas.shape[1] / len(self._window))
            h = 99

            for i in range(1, len(self._window)):
                y1 = (self._max - self._window[i-1]) / (self._max - self._min + 1e-8)
                y2 = (self._max - self._window[i]) / (self._max - self._min + 1e-8)

                cv2.line(
                        canvas,
                        ((i-1) * w, int(y1 * h)),
                        ((i) * w, int(y2 * h)),
                        (255, 255, 255), 2)

            canvas = np.pad(canvas, ((5, 5), (5, 5), (0, 0)))

            cv2.imshow('%.3f %.3f %.3f' % (self._K_P, self._K_I, self._K_D), canvas)
            cv2.waitKey(1)

        return self._K_P * error + self._K_I * integral + self._K_D * derivative

    def save(self):
        self._saved_window = deepcopy(self._window)

    def load(self):
        self._window = self._saved_window

class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x-r, y-r, x+r, y+r), color)

    def show(self):
        if not DEBUG:
            return

        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256):
        self.saved_route = deque()
        self.route = deque()
        self.saved_route_distances = deque()
        self.route_distances = deque()


        self.min_distance = min_distance
        self.max_distance = max_distance
        self.is_last = False

        # self.mean = np.array([49.0, 8.0]) # for carla 9.9
        # self.scale = np.array([111324.60662786, 73032.1570362]) # for carla 9.9
        self.mean = np.array([0.0, 0.0]) # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10


        #if DEBUG:
        #    self.debug = Plotter(debug_size)

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))

        # We do the calculations in the beginning once so that we don't have to do them every time in run_step
        self.route_distances.append(0.0)
        for i in range(1, len(self.route)):
            diff = self.route[i][0] - self.route[i - 1][0]
            distance = (diff[0]**2 + diff[1]**2)**0.5
            self.route_distances.append(distance)

    def run_step(self, gps):
        #if DEBUG:
        #    self.debug.clear()

        if len(self.route) <= 2:
            self.is_last = True
            return self.route

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += self.route_distances[i]

            diff = self.route[i][0] - gps
            distance = (diff[0]**2 + diff[1]**2)**0.5

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

            #if DEBUG:
            #    r = 255 * int(distance > self.min_distance)
            #    g = 255 * int(self.route[i][1].value == 4)
            #    b = 255
            #    self.debug.dot(gps, self.route[i][0], (r, g, b))

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
                self.route_distances.popleft()

        #if DEBUG:
        #    self.debug.dot(gps, self.route[0][0], (0, 255, 0))
        #    self.debug.dot(gps, self.route[1][0], (255, 0, 0))
        #    self.debug.dot(gps, gps, (0, 0, 255))
        #    self.debug.show()

        return self.route

    def save(self):
        self.saved_route = deepcopy(self.route)
        self.saved_route_distances = deepcopy(self.route_distances)

    def load(self):
        self.route = self.saved_route
        self.route_distances = self.saved_route_distances
        self.is_last = False


def interpolate_trajectory(world_map, waypoints_trajectory, hop_resolution=1.0,
                           max_len=100, planner_cache_key=None):
    """
    Given some raw keypoints interpolate a full dense trajectory to be used by the user.
    returns the full interpolated route both in GPS coordinates and also in its original form.
    
    Args:
        - world: an reference to the CARLA world so we can use the planner
        - waypoints_trajectory: the current coarse trajectory
        - hop_resolution: is the resolution, how dense is the provided trajectory going to be made
    """

    grp = None
    if planner_cache_key is not None:
        grp = _GLOBAL_ROUTE_PLANNER_CACHE.get(planner_cache_key)
    if grp is None:
        if planner_cache_key is not None:
            print("Route-graph cache miss: building {} once".format(
                planner_cache_key), flush=True)
        dao = GlobalRoutePlannerDAO(world_map, hop_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        if planner_cache_key is not None:
            # Retain only the active map's graph so a sequence of different
            # maps cannot accumulate multiple full mining-road networks.
            _GLOBAL_ROUTE_PLANNER_CACHE.clear()
            _GLOBAL_ROUTE_PLANNER_CACHE[planner_cache_key] = grp
    elif planner_cache_key is not None:
        print("Route-graph cache hit: reusing {}".format(
            planner_cache_key), flush=True)
    # Obtain route plan
    # Where the mining road graph has no forward connection, GlobalRoutePlanner
    # answers a ~50 m step with a detour of several kilometres, which the
    # max_len cap rejects. Upstream then retries the next step from the same
    # anchor, which usually routes around the break. But when that retry keeps
    # failing, upstream never advances the anchor and the whole remainder of the
    # route is silently lost: the expert reaches the truncation point, believes
    # it has arrived, and brakes until the route times out. Keep the retry, and
    # give up on an anchor after the second failure so the rest survives.
    route = []
    anchor = waypoints_trajectory[0]
    failures = 0
    for waypoint_next in waypoints_trajectory[1:]:
        if anchor.x == waypoint_next.x and anchor.y == waypoint_next.y:
            continue
        interpolated_trace = grp.trace_route(anchor, waypoint_next)
        if len(interpolated_trace) > max_len:
            failures += 1
            if failures >= 2:
                anchor = waypoint_next
                failures = 0
            continue
        for wp_tuple in interpolated_trace:
            route.append((wp_tuple[0].transform, wp_tuple[1]))
        anchor = waypoint_next
        failures = 0

    lat_ref, lon_ref = _get_latlon_ref(world_map)

    return location_route_to_gps(route, lat_ref, lon_ref), route


def location_route_to_gps(route, lat_ref, lon_ref):
    """
        Locate each waypoint of the route into gps, (lat long ) representations.
    :param route:
    :param lat_ref:
    :param lon_ref:
    :return:
    """
    gps_route = []

    for transform, connection in route:
        gps_point = _location_to_gps(lat_ref, lon_ref, transform.location)
        gps_route.append((gps_point, connection))

    return gps_route


def _get_latlon_ref(world_map):
    """
    Convert from waypoints world coordinates to CARLA GPS coordinates
    :return: tuple with lat and lon coordinates
    """
    xodr = world_map.to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    # default reference
    lat_ref = 42.0
    lon_ref = 2.0

    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if georef.text:
                    str_list = georef.text.split(' ')
                    for item in str_list:
                        if '+lat_0' in item:
                            lat_ref = float(item.split('=')[1])
                        if '+lon_0' in item:
                            lon_ref = float(item.split('=')[1])
    return lat_ref, lon_ref


def _location_to_gps(lat_ref, lon_ref, location):
    """
    Convert from world coordinates to GPS coordinates
    :param lat_ref: latitude reference for the current map
    :param lon_ref: longitude reference for the current map
    :param location: location to translate
    :return: dictionary with lat, lon and height
    """

    EARTH_RADIUS_EQUA = 6378137.0   # pylint: disable=invalid-name
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += location.x
    my -= location.y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    z = location.z

    return {'lat': lat, 'lon': lon, 'z': z}
