"""Runtime CARLA physics profiles for the supported mining trucks.

The CAT 797F profile preserves the earlier load-dependent calibration.  The
Komatsu HD465-7E0 profile maps manufacturer specifications to the local
``vehicle.xiaosong55t.xiaosong55t`` asset.
"""

from __future__ import print_function

import math

from collections import namedtuple

import carla


CAT797_BLUEPRINT_ID = "vehicle.cat_ceshi.car_ceshi"
HD465_BLUEPRINT_ID = "vehicle.xiaosong55t.xiaosong55t"

# Caterpillar 797F published specifications.
CAT797_RATED_GROSS_MASS_KG = 623690.0
CAT797_NOMINAL_PAYLOAD_KG = 363000.0
CAT797_EMPTY_OPERATING_MASS_KG = (
    CAT797_RATED_GROSS_MASS_KG - CAT797_NOMINAL_PAYLOAD_KG
)

LoadProfile = namedtuple(
    "LoadProfile",
    [
        "name",
        "payload_fraction",
        "mass_kg",
        "center_of_mass_x_m",
        "center_of_mass_z_m",
    ],
)

# The local asset has a 7.195 m wheelbase. Its wheel centres are 1.961 m below
# the actor origin and the wheel radius is 1.925 m, so the actor origin is
# approximately 3.886 m above the contact plane. CARLA expects center_of_mass
# in actor-local coordinates, not as a height above the road.
ACTOR_ORIGIN_HEIGHT_ABOVE_GROUND_M = 3.886

# Longitudinal CG positions are
# derived from Caterpillar's published front-axle shares (47.2% empty and
# 33.3% loaded). CG heights are explicit simulation assumptions because Cat
# does not publish them; they imply a payload CG height of about 4.86 m.
REAR_AXLE_X_M = -3.620
WHEELBASE_M = 7.195
EMPTY_FRONT_AXLE_FRACTION = 0.472
FULL_FRONT_AXLE_FRACTION = 0.333
EMPTY_CENTER_OF_MASS_X_M = (
    REAR_AXLE_X_M + EMPTY_FRONT_AXLE_FRACTION * WHEELBASE_M
)
FULL_CENTER_OF_MASS_X_M = (
    REAR_AXLE_X_M + FULL_FRONT_AXLE_FRACTION * WHEELBASE_M
)
EMPTY_CENTER_OF_MASS_HEIGHT_ABOVE_GROUND_M = 2.80
FULL_CENTER_OF_MASS_HEIGHT_ABOVE_GROUND_M = 4.00
EMPTY_CENTER_OF_MASS_Z_M = (
    EMPTY_CENTER_OF_MASS_HEIGHT_ABOVE_GROUND_M
    - ACTOR_ORIGIN_HEIGHT_ABOVE_GROUND_M
)
FULL_CENTER_OF_MASS_Z_M = (
    FULL_CENTER_OF_MASS_HEIGHT_ABOVE_GROUND_M
    - ACTOR_ORIGIN_HEIGHT_ABOVE_GROUND_M
)

# ISO 3450:2011 Annex A recommends 0.28 g (2.75 m/s^2) service-brake
# efficiency for machines used on grades up to 20%. CARLA's wheel-brake torque
# does not reproduce that target quantitatively for this custom asset, so keep
# the blueprint's original values for predictable manual-driving behaviour.
# Planner/CBF limits must use the analytic target rather than treating CARLA's
# stopping distance as a calibrated CAT 797F measurement.
SERVICE_BRAKE_TARGET_DECELERATION_MPS2 = 2.75
FRONT_SERVICE_BRAKE_TORQUE_NM = 800000.0
REAR_SERVICE_BRAKE_TORQUE_NM = 1000000.0

# Trend-model fallback for this custom asset. Its native wheel brake behaves
# almost like an on/off switch for every nonzero torque tested. A fixed force
# capacity gives the intended ordering: the same brake hardware decelerates an
# empty truck more strongly than a full truck. It is deliberately a simple
# simulator control model, not a measured CAT 797F brake map.
SMOOTH_SERVICE_BRAKE_FORCE_CAPACITY_N = (
    CAT797_RATED_GROSS_MASS_KG * 2.0
)

# CARLA's simple clutch model does not reproduce the real truck's low-gear
# torque-converter rimpull. The empirical API-force supplement below provides
# a small flat-road base and additional uphill low-gear effort. It stays
# constant through the mine's working-speed region, then fades out smoothly so
# it cannot determine higher-speed behaviour.
LOW_GEAR_TRACTION_BASE_FORCE_N = 300000.0
LOW_GEAR_TRACTION_FORCE_PER_GRADE_PERCENT_N = 30000.0
LOW_GEAR_TRACTION_MAX_FORCE_N = 700000.0
MID_GEAR_TRACTION_FORCE_N = 450000.0
LOW_GEAR_TRACTION_FULL_SPEED_MPS = 6.0
LOW_GEAR_TRACTION_ZERO_SPEED_MPS = 8.0
LOW_GEAR_TRACTION_MULTIPLIER = {1: 1.0, 2: 0.9, 3: 0.75}


def _load_profile(name, fraction):
    fraction = float(fraction)
    return LoadProfile(
        name=name,
        payload_fraction=fraction,
        mass_kg=CAT797_EMPTY_OPERATING_MASS_KG
        + fraction * CAT797_NOMINAL_PAYLOAD_KG,
        center_of_mass_x_m=EMPTY_CENTER_OF_MASS_X_M
        + fraction * (FULL_CENTER_OF_MASS_X_M - EMPTY_CENTER_OF_MASS_X_M),
        center_of_mass_z_m=EMPTY_CENTER_OF_MASS_Z_M
        + fraction * (FULL_CENTER_OF_MASS_Z_M - EMPTY_CENTER_OF_MASS_Z_M),
    )


LOAD_PROFILES = {
    "empty": _load_profile("empty", 0.0),
    "half": _load_profile("half", 0.5),
    "full": _load_profile("full", 1.0),
}

# At low speed the real 797F uses torque-converter drive. CARLA 0.9.10 has no
# equivalent heavy-truck torque-converter model, so the low-RPM section
# approximates its launch assistance. The asset's original transmission ratios
# are retained to avoid changing acceleration and engine braking at the same
# time. Rated-speed torque remains near the 2,983 kW at 1,750 rpm power
# specification.
TORQUE_CURVE_RPM_NM = (
    (0.0, 24000.0),
    (250.0, 26000.0),
    (500.0, 26000.0),
    (800.0, 25000.0),
    (1100.0, 22000.0),
    (1350.0, 19500.0),
    (1500.0, 18500.0),
    (1750.0, 16285.0),
    (1900.0, 12000.0),
)

STEERING_CURVE_SPEED_SCALE = (
    (0.0, 1.0),
    (10.0, 0.80),
    (20.0, 0.60),
    (32.0, 0.45),
    (48.0, 0.32),
    (64.0, 0.22),
    (80.0, 0.16),
    (100.0, 0.12),
)

FORWARD_GEARS = (
    (4.77, 0.00, 0.90),
    (3.55, 0.18, 0.42),
    (2.63, 0.25, 0.55),
    (1.95, 0.35, 0.68),
    (1.45, 0.48, 0.78),
    (1.07, 0.60, 0.88),
    (0.80, 0.72, 1.00),
)

# Deterministic speed-based scheduler used instead of CARLA's engine-RPM
# autobox. The upper thresholds approximately follow the seven original gear
# ratios to reach the asset's 67 km/h top-speed region. Lower thresholds add
# hysteresis and force an early downshift when climbing.
MINING_UPSHIFT_SPEED_KMH = {
    1: 11.0,
    2: 16.0,
    3: 22.0,
    4: 30.0,
    5: 40.0,
    6: 52.0,
}
MINING_DOWNSHIFT_SPEED_KMH = {
    2: 9.5,
    3: 12.0,
    4: 17.0,
    5: 24.0,
    6: 32.0,
    7: 44.0,
}


def select_mining_gear(current_gear, speed_kmh):
    """Select one automatic haul-truck gear with up/downshift hysteresis."""
    gear = max(1, min(7, int(current_gear)))
    speed_kmh = max(0.0, float(speed_kmh))

    while gear > 1 and speed_kmh < MINING_DOWNSHIFT_SPEED_KMH[gear]:
        gear -= 1
    while gear < 7 and speed_kmh >= MINING_UPSHIFT_SPEED_KMH[gear]:
        gear += 1
    return gear


def get_load_profile(name):
    try:
        return LOAD_PROFILES[name]
    except KeyError:
        raise ValueError(
            "Unknown CAT 797 load profile {!r}; expected one of {}".format(
                name, sorted(LOAD_PROFILES)
            )
        )


def _make_curve(points):
    return [carla.Vector2D(x=float(x), y=float(y)) for x, y in points]


def _make_gears():
    gears = []
    for ratio, down_ratio, up_ratio in FORWARD_GEARS:
        gear = carla.GearPhysicsControl()
        gear.ratio = float(ratio)
        gear.down_ratio = float(down_ratio)
        gear.up_ratio = float(up_ratio)
        gears.append(gear)
    return gears


def apply_cat797_physics(
    vehicle,
    load="full",
    center_of_mass=None,
    tire_friction=3.5,
    service_brake_scale=1.0,
):
    """Apply the provisional CAT 797F profile to an already spawned actor.

    Args:
        vehicle: A spawned ``carla.Vehicle`` actor.
        load: One of ``empty``, ``half`` or ``full``.
        center_of_mass: Optional asset-relative ``carla.Vector3D``. The load
            profile's calibrated/assumed CG is used when omitted.
        tire_friction: CARLA/PhysX tire friction scale, not a measured road
            friction coefficient.
        service_brake_scale: Multiplier used only for empirical brake
            calibration. Keep at 1.0 for the selected profile.

    Returns:
        The selected ``LoadProfile``.
    """
    profile = get_load_profile(load)
    physics = vehicle.get_physics_control()

    physics.mass = profile.mass_kg
    physics.max_rpm = 1750.0
    physics.moi = 15.0
    physics.damping_rate_full_throttle = 0.15
    physics.damping_rate_zero_throttle_clutch_engaged = 2.0
    physics.damping_rate_zero_throttle_clutch_disengaged = 0.35
    # CARLA's autobox races through this custom asset's gears at low road
    # speed. Callers use select_mining_gear() and send manual gear commands,
    # which is automatic from the driver's perspective.
    physics.use_gear_autobox = False
    physics.gear_switch_time = 0.5
    physics.clutch_strength = 80.0
    physics.final_ratio = 21.26
    physics.drag_coefficient = 0.10
    physics.torque_curve = _make_curve(TORQUE_CURVE_RPM_NM)
    physics.steering_curve = _make_curve(STEERING_CURVE_SPEED_SCALE)
    physics.forward_gears = _make_gears()

    if center_of_mass is None:
        center_of_mass = carla.Vector3D(
            x=profile.center_of_mass_x_m,
            y=0.0,
            z=profile.center_of_mass_z_m,
        )
    physics.center_of_mass = center_of_mass

    wheels = physics.wheels
    if len(wheels) != 4:
        raise ValueError(
            "CAT 797 profile expects CARLA's four-wheel model, got {} wheels".format(
                len(wheels)
            )
        )

    for index, wheel in enumerate(wheels):
        wheel.radius = 192.5
        wheel.tire_friction = float(tire_friction)
        wheel.damping_rate = 0.25
        wheel.max_steer_angle = 29.67 if index < 2 else 0.0
        wheel.max_brake_torque = (
            FRONT_SERVICE_BRAKE_TORQUE_NM
            if index < 2
            else REAR_SERVICE_BRAKE_TORQUE_NM
        ) * float(service_brake_scale)
        wheel.max_handbrake_torque = 0.0 if index < 2 else 1200000.0
    physics.wheels = wheels

    vehicle.apply_physics_control(physics)
    return profile


# ---------------------------------------------------------------------------
# Komatsu HD465-7E0
# ---------------------------------------------------------------------------
# These values describe what the cooked ``vehicle.xiaosong55t.xiaosong55t``
# asset already carries.  They are NOT applied at runtime and must not be:
# every ``apply_physics_control()`` call on this actor segfaults CARLA 0.9.10,
# scalar-only writes included.  ``tools/mining/configure_hd465_unreal.py`` is
# the authority; it writes them into the Unreal Blueprint, which is then cooked
# and copied into the package.  Keep the two files in step.

HD465_EMPTY_MASS_KG = 43100.0
HD465_NOMINAL_PAYLOAD_KG = 55000.0
HD465_MAX_GROSS_MASS_KG = 99680.0
HD465_RATED_RPM = 2000.0
HD465_MAX_RPM = 2300.0
HD465_WHEEL_RADIUS_CM = 115.0
HD465_MAX_STEER_ANGLE_DEG = 39.0
HD465_TARGET_TOP_SPEED_KMH = 70.0

# CARLA 0.9.10's PhysX vehicle produces no drive torque at all once a gear's
# maximum wheel angular velocity drops below about 4.5 rad/s.  Measured on a
# stock actor at 4600 kg and 43100 kg and at 46, 60 and 75 cm wheel radii, the
# cutoff sits between 4.22 rad/s (dead) and 4.53 rad/s (normal): it depends on
# neither mass nor radius.  The asset's original 16.7595 final ratio with a
# 4.00 first gear put first gear and reverse at 3.59 rad/s, so the truck could
# not pull away or climb at all.
HD465_MIN_SAFE_WHEEL_OMEGA_RAD_S = 4.5

# Peak torque stays at the published 3324 N m and peak power at the published
# 533 kW (2545 N m at 2000 rpm).  Only the shape below 1400 rpm changed: the
# imported curve fell to 400 N m at zero rpm, so on a grade the truck lugged
# down to near-zero engine speed, made 16 kN m at the wheels and slid backwards
# under full throttle.  The real machine drives through a torque converter,
# which delivers peak torque at zero output speed; CARLA's clutch model has no
# converter and ties engine speed to road speed, so holding peak torque flat
# below 1400 rpm is the trend-level stand-in.  It is conservative: a real
# converter multiplies stall torque by roughly 2.5 to 3, this multiplies by 1.
#
# Unreal 4.24 does not expose RuntimeFloatCurve key data to Python, so this is
# shipped as a CurveFloat asset (/Game/HD465_TorqueCurve, imported from CSV)
# attached through RuntimeFloatCurve.external_curve.  CARLA's GetPhysicsControl
# reads EditorCurveData and therefore still reports the old embedded keys at
# runtime: the Python API is not a way to check this curve.
HD465_TORQUE_CURVE_RPM_NM = (
    (0.0, 3324.0),
    (750.0, 3324.0),
    (1200.0, 3324.0),
    (1400.0, 3324.0),
    (1800.0, 2800.0),
    (2000.0, 2545.0),
    (2300.0, 1800.0),
)
HD465_TORQUE_CURVE_ASSET = "/Game/HD465_TorqueCurve"

# Seventh gear keeps the brochure's 70 km/h, so its total reduction is pinned at
# 14.245 for a 1.15 m radius and 2300 rpm.  The 4.5 rad/s floor then caps the
# usable ratio span at 2.80:1, which is narrower than a real seven-range
# powershift box.  These are simulator ratios, not Komatsu's unpublished ones.
HD465_FINAL_RATIO = 14.245
HD465_FORWARD_GEARS = (
    (2.8043, 0.00, 0.82),
    (2.3615, 0.50, 0.82),
    (1.9886, 0.50, 0.82),
    (1.6746, 0.50, 0.82),
    (1.4102, 0.50, 0.82),
    (1.1875, 0.50, 0.82),
    (1.0000, 0.50, 1.00),
)
HD465_REVERSE_GEAR_RATIO = -2.8043

# Kinematic ceiling of each forward gear at 2300 rpm, in km/h.  Useful for
# sanity-checking a speed target against the gear the truck will be in.
HD465_GEAR_TOP_SPEED_KMH = (24.96, 29.64, 35.20, 41.80, 49.64, 58.95, 70.00)

# The clutch passes clutch_strength * slip [N m per rad/s].  At the asset's
# original 10, moving the 3324 N m peak needed 332 rad/s of slip against an
# engine that only reaches 240.9 rad/s, so peak torque never crossed it.
HD465_CLUTCH_STRENGTH = 400.0
HD465_GEAR_SWITCH_TIME_S = 0.2
HD465_ENGINE_MOI = 5.0

# ISO 3450 asks for roughly 2.75 m/s^2 of service braking.  148 kN m of total
# wheel torque over a 1.15 m radius is 128.7 kN, or 2.99 m/s^2 empty.
HD465_FRONT_SERVICE_BRAKE_TORQUE_NM = 40000.0
HD465_REAR_SERVICE_BRAKE_TORQUE_NM = 34000.0
HD465_REAR_HANDBRAKE_TORQUE_NM = 120000.0
HD465_SERVICE_BRAKE_TARGET_DECELERATION_MPS2 = 2.75

# A 24.00-35 tyre and rim is about 1.2 t.  Unreal takes wheel inertia as
# 0.5 * mass * radius^2, so the imported 20 kg gave 13 kg m^2 where the real
# assembly is near 700; wheels spun up and locked almost instantly.
HD465_WHEEL_MASS_KG = 600.0

# PhysX takes the tyre's longitudinal force as
#     force = long_stiff_value * 9.81 * longitudinal_slip   [N, per wheel]
# which is absolute and does not scale with load.  The asset carried Unreal's
# car-sized 2000, capping each tyre at 19.6 kN: 39.2 kN from two driven wheels
# is 0.91 m/s^2 and less than a 9.8 percent grade needs, and 78.5 kN across
# four wheels was the 1.8 m/s^2 ceiling that made the service brake look broken
# whatever torque it was given.
HD465_WHEEL_LONG_STIFF_VALUE = 28000.0

# BodyInstance.COMNudge, in metres as the Python API reports it (Unreal stores
# centimetres; carla::geom::Location divides by 100).  The asset pushed the
# centre of mass 0.5 m below where the body geometry puts it, so the truck held
# 0.95 g of lateral acceleration with under 5 degrees of roll and never tipped.
# At (0.6, 0, 0) it approaches rollover near 0.7 g, which is what a 43 t hauler
# on a 3.25 m mean track does.  The longitudinal 0.6 m is kept: an empty rigid
# hauler is genuinely nose-heavy.
HD465_CENTER_OF_MASS_M = (0.6, 0.0, 0.0)

# Measured on 0325_5 after the fix; see
# docs/mining/README.zh-CN.md.
HD465_MEASURED = {
    "top_speed_kmh": 69.1,
    "time_0_to_30_kmh_s": 5.75,
    "service_brake_decel_mps2": 3.04,
    "service_brake_distance_30kmh_m": 13.2,
    "min_turning_radius_m": 7.8,
    "max_climbable_grade_pct": 15.0,   # steepest road on 0325_5, not a limit
    "rollover_threshold_g": 0.7,
}


def hd465_wheel_omega_rad_s(gear_ratio, final_ratio=HD465_FINAL_RATIO,
                            max_rpm=HD465_MAX_RPM):
    """Maximum wheel angular velocity a gear can reach.

    Any gear below HD465_MIN_SAFE_WHEEL_OMEGA_RAD_S delivers no drive torque in
    CARLA 0.9.10.  Check new ratios with this before cooking them.
    """
    return (float(max_rpm) * 2.0 * math.pi / 60.0) / (
        float(final_ratio) * float(gear_ratio)
    )


def apply_hd465_physics(vehicle, *args, **kwargs):
    """Refuse to mutate the HD465 asset's runtime physics.

    ``apply_physics_control()`` segfaults CARLA 0.9.10 for this cooked actor,
    scalar-only writes included, taking the whole server with it.  Change the
    Unreal Blueprint with tools/mining/configure_hd465_unreal.py and re-cook.
    """
    raise RuntimeError(
        "apply_physics_control() segfaults CARLA 0.9.10 for {}; edit the "
        "Unreal Blueprint with tools/mining/configure_hd465_unreal.py and "
        "re-cook instead".format(HD465_BLUEPRINT_ID)
    )


def physics_summary(vehicle):
    physics = vehicle.get_physics_control()
    return {
        "mass_kg": float(physics.mass),
        "center_of_mass": (
            float(physics.center_of_mass.x),
            float(physics.center_of_mass.y),
            float(physics.center_of_mass.z),
        ),
        "max_rpm": float(physics.max_rpm),
        "final_ratio": float(physics.final_ratio),
        "forward_gears": [
            {
                "ratio": float(gear.ratio),
                "down_ratio": float(gear.down_ratio),
                "up_ratio": float(gear.up_ratio),
            }
            for gear in physics.forward_gears
        ],
        "torque_curve_rpm_nm": [
            (float(point.x), float(point.y)) for point in physics.torque_curve
        ],
        "wheel_count": len(physics.wheels),
        "wheel_radius_cm": [float(wheel.radius) for wheel in physics.wheels],
        "tire_friction": [float(wheel.tire_friction) for wheel in physics.wheels],
        "max_brake_torque_nm": [
            float(wheel.max_brake_torque) for wheel in physics.wheels
        ],
        "max_steer_angle_deg": [
            float(wheel.max_steer_angle) for wheel in physics.wheels
        ],
    }


def signed_forward_speed(vehicle):
    """Return velocity projected onto the vehicle's longitudinal axis."""
    velocity = vehicle.get_velocity()
    forward = vehicle.get_transform().get_forward_vector()
    return (
        velocity.x * forward.x
        + velocity.y * forward.y
        + velocity.z * forward.z
    )


def apply_smooth_service_brake(
    vehicle,
    brake_command,
    delta_seconds,
    force_capacity_n=SMOOTH_SERVICE_BRAKE_FORCE_CAPACITY_N,
):
    """Apply a continuous API-force brake for the custom truck asset.

    Native wheel brake torque must be disabled when this helper is used. The
    force opposes longitudinal motion, is capped so it does not reverse the
    vehicle in one simulation step, and supplies static grade holding near
    zero speed when the requested force is sufficient.
    """
    command = max(0.0, min(1.0, float(brake_command)))
    delta_seconds = max(1e-3, float(delta_seconds))
    if command <= 0.0:
        return 0.0

    physics = vehicle.get_physics_control()
    mass_kg = float(physics.mass)
    force_limit_n = command * float(force_capacity_n)
    transform = vehicle.get_transform()
    forward = transform.get_forward_vector()
    speed_mps = signed_forward_speed(vehicle)

    if abs(speed_mps) > 0.05:
        force_n = min(force_limit_n, mass_kg * abs(speed_mps) / delta_seconds)
        direction = -1.0 if speed_mps > 0.0 else 1.0
    else:
        # Gravity projected onto the pitched longitudinal axis. Static brake
        # force cancels it up to the available command-dependent capacity.
        gravity_along_mps2 = -9.81 * forward.z
        force_n = min(force_limit_n, mass_kg * abs(gravity_along_mps2))
        direction = -1.0 if gravity_along_mps2 > 0.0 else 1.0

    vehicle.add_force(
        carla.Vector3D(
            x=direction * force_n * forward.x,
            y=direction * force_n * forward.y,
            z=direction * force_n * forward.z,
        )
    )
    return force_n


def apply_low_speed_traction_assist(
    vehicle,
    throttle_command,
    gear=1,
    base_force_n=LOW_GEAR_TRACTION_BASE_FORCE_N,
    force_per_grade_percent_n=LOW_GEAR_TRACTION_FORCE_PER_GRADE_PERCENT_N,
    max_force_n=LOW_GEAR_TRACTION_MAX_FORCE_N,
    full_speed_mps=LOW_GEAR_TRACTION_FULL_SPEED_MPS,
    zero_speed_mps=LOW_GEAR_TRACTION_ZERO_SPEED_MPS,
):
    """Approximate torque-converter rimpull missing from CARLA's clutch.

    Use only in the first three forward gears. Uphill grade raises the
    available supplement, approximating demanded low-gear rimpull; the effect
    decreases by gear and is removed before higher-speed operation.
    """
    throttle = max(0.0, min(1.0, float(throttle_command)))
    gear_multiplier = LOW_GEAR_TRACTION_MULTIPLIER.get(int(gear), 0.0)
    if throttle <= 0.0 or gear_multiplier <= 0.0:
        return 0.0

    speed_mps = max(0.0, signed_forward_speed(vehicle))
    forward = vehicle.get_transform().get_forward_vector()
    horizontal = max(1e-3, (forward.x ** 2 + forward.y ** 2) ** 0.5)
    uphill_grade_percent = max(0.0, 100.0 * forward.z / horizontal)
    if int(gear) == 1:
        force_capacity_n = min(
            float(max_force_n),
            float(base_force_n)
            + uphill_grade_percent * float(force_per_grade_percent_n),
        )
    else:
        force_capacity_n = MID_GEAR_TRACTION_FORCE_N

    full_speed_mps = max(0.1, float(full_speed_mps))
    zero_speed_mps = max(full_speed_mps + 0.1, float(zero_speed_mps))
    if speed_mps <= full_speed_mps:
        fade = 1.0
    elif speed_mps < zero_speed_mps:
        fade = (zero_speed_mps - speed_mps) / (
            zero_speed_mps - full_speed_mps
        )
    else:
        fade = 0.0

    force_n = throttle * force_capacity_n * gear_multiplier * fade
    if force_n <= 0.0:
        return 0.0

    vehicle.add_force(
        carla.Vector3D(
            x=force_n * forward.x,
            y=force_n * forward.y,
            z=force_n * forward.z,
        )
    )
    return force_n
