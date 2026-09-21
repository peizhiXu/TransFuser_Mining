"""Write the HD465-7E0 drivetrain and brake profile into its Unreal Blueprint.

Run with UE4Editor-Cmd's PythonScriptCommandlet, not CPython.  The file targets
Unreal Engine 4.24's Python 2.7.

    cd <carla-source>/Unreal/CarlaUE4
    $UE4_ROOT/Engine/Binaries/Linux/UE4Editor-Cmd \
        <carla-source>/Unreal/CarlaUE4/CarlaUE4.uproject \
        -run=pythonscript -script=configure_hd465_unreal.py \
        -EnablePlugins=PythonScriptPlugin -unattended -nopause -nosplash -nullrhi

``-EnablePlugins=PythonScriptPlugin`` is required: the plugin is not listed in
CarlaUE4.uproject and is not enabled by default, so without it the commandlet
class is not found.

This file is the authority for the cook-time values.  The cooked actor cannot be
tuned from the Python API at all: any ``apply_physics_control()`` call on
``vehicle.xiaosong55t.xiaosong55t`` segfaults CARLA 0.9.10, including
scalar-only writes that never touch the wheel array.

WHEEL ANGULAR VELOCITY LIMIT
----------------------------
CARLA 0.9.10's PhysX vehicle delivers no drive torque at all when a gear's
maximum wheel angular velocity falls below roughly 4.5 rad/s:

    omega_max = (max_rpm * 2*pi/60) / (final_ratio * gear_ratio)

Measured on a stock actor at 4600 kg and 43100 kg and at wheel radii of 46, 60
and 75 cm, the cutoff sits between 4.22 rad/s (dead) and 4.53 rad/s (normal),
independent of mass and radius.  The previous configuration used final_ratio
16.7595 with a 4.00 first gear, so first gear ran at 3.59 rad/s and reverse at
3.59 rad/s: both produced zero tractive effort.  Every ratio below keeps
omega_max at or above 6.0 rad/s.

Holding seventh gear at the brochure's 70 km/h fixes the top-gear reduction at
14.245, so the 6.0 rad/s floor caps the usable ratio span at 2.80:1.  The seven
ranges below are a geometric series across that span; they are a simulator
gearbox, not Komatsu's unpublished ratios.
"""

import json
import sys

import unreal


ASSET_PATH = "/Game/xiaosong_blueprint"
CLASS_PATH = "/Game/xiaosong_blueprint.xiaosong_blueprint_C"
RESULT_PATH = "/tmp/hd465_unreal_configure.txt"
MANIFEST_PATH = "/tmp/hd465_profile_manifest.json"
TORQUE_CURVE_CSV = "/tmp/hd465_torque_curve.csv"
TORQUE_CURVE_ASSET = "/Game/HD465_TorqueCurve"
TORQUE_CURVE_NAME = "HD465_TorqueCurve"

WHEEL_RADIUS_M = 1.15
MAX_RPM = 2300.0
TOP_GEAR_TOTAL_REDUCTION = 14.245   # 2300 rpm -> 70.0 km/h at 1.15 m radius
MIN_SAFE_WHEEL_OMEGA_RAD_S = 4.5    # measured cutoff; ratios below target 6.0

FINAL_RATIO = 14.245

# (ratio, down_ratio, up_ratio).  Geometric span 2.804:1, step 1.1875.
FORWARD_GEARS = (
    (2.8043, 0.00, 0.82),
    (2.3615, 0.50, 0.82),
    (1.9886, 0.50, 0.82),
    (1.6746, 0.50, 0.82),
    (1.4102, 0.50, 0.82),
    (1.1875, 0.50, 0.82),
    (1.0000, 0.50, 1.00),
)
REVERSE_GEAR_RATIO = -2.8043

# The clutch transmits clutch_strength * slip [N m per rad/s].  At the previous
# value of 10 the engine's 3324 N m peak needed 332 rad/s of slip, more than the
# 240.9 rad/s the engine can reach, so peak torque could never cross the clutch.
# 400 passes it at 8.3 rad/s.
CLUTCH_STRENGTH = 400.0
GEAR_SWITCH_TIME = 0.2              # powershift transmission, was 0.5
ENGINE_MOI = 5.0                    # large diesel plus flywheel, was 1.0

# ISO 3450 asks for about 2.75 m/s^2 of service-brake retardation.  Total wheel
# torque of 148 kN m over a 1.15 m radius gives 128.7 kN, or 2.99 m/s^2 at the
# 43100 kg empty mass, with a mild front bias that follows load transfer.
FRONT_SERVICE_BRAKE_TORQUE_NM = 40000.0
REAR_SERVICE_BRAKE_TORQUE_NM = 34000.0
# Parking brake sized to hold the empty truck on a 20 percent grade:
# 43100 * 9.81 * 0.196 * 1.15 / 2 rear wheels = 47.6 kN m.
REAR_HANDBRAKE_TORQUE_NM = 120000.0

# A 24.00-35 tyre and rim assembly is roughly 1.2 t.  Unreal derives the wheel
# rotational inertia as 0.5 * mass * radius^2, so the imported 20 kg gave
# 13 kg m^2 against a realistic 700 kg m^2; wheels span up and locked almost
# instantly.  600 kg is a deliberate compromise that keeps the suspension solve
# well conditioned.
WHEEL_MASS_KG = 600.0

# PhysX takes the tyre's longitudinal force as
#     force = long_stiff_value * 9.81 * longitudinal_slip   [N, per wheel]
# an absolute figure that does not scale with load, so it has to be sized for
# the vehicle.  Unreal's 1000 default suits a ~1500 kg car (3.7 kN per wheel);
# the asset carried 2000, which capped each tyre at 19.6 kN.  That single number
# explained the truck's whole behaviour: 39.2 kN from the two driven wheels is
# 0.91 m/s^2 and less than the 41.4 kN a 9.8 percent grade needs, so it rolled
# backwards, and 78.5 kN across four wheels is the 1.8 m/s^2 ceiling that made
# the service brake look broken however much torque it was given.  28000 keeps
# the same stiffness-per-unit-axle-load as a stock Unreal car at 43100 kg.
WHEEL_LONG_STIFF_VALUE = 28000.0

# Torque curve, in (rpm, N m).  Peak torque stays at the published 3324 N m and
# peak power at the published 533 kW (2545 N m at 2000 rpm).  What changes is
# the shape below 1400 rpm: the asset's imported curve fell away to 400 N m at
# zero rpm, so on a grade the truck stalled down to near-zero engine speed,
# produced 16 kN m at the wheels and slid backwards under full throttle.
#
# The real machine drives through a torque converter, which delivers peak
# torque at zero output speed; CARLA's clutch model has no converter, and
# engine speed is tied to road speed.  Holding peak torque flat below 1400 rpm
# is the trend-level stand-in.  It is conservative: a real converter multiplies
# stall torque by roughly 2.5 to 3, this multiplies by 1.
#
# Unreal 4.24 does not expose RuntimeFloatCurve key data to Python, so the curve
# is imported from CSV as a CurveFloat asset and attached through
# RuntimeFloatCurve.external_curve.  That asset has to be cooked and shipped
# alongside the Blueprint.
TORQUE_CURVE_RPM_NM = (
    (0, 3324),
    (750, 3324),
    (1200, 3324),
    (1400, 3324),
    (1800, 2800),
    (2000, 2545),
    (2300, 1800),
)

# BodyInstance.COMNudge offsets the centre of mass from the one Unreal computes
# from the PhysicsAsset bodies.  Unreal stores it in CENTIMETRES; CARLA's
# geom::Location divides by 100, so the Python API reports the same figure in
# metres.  The asset carried (60, 0, -50) cm, i.e. the centre of mass pushed
# half a metre below where the body geometry puts it.  That is why the truck
# sustained 0.95 g of lateral acceleration with under 5 degrees of roll and
# never tipped, where a 43 t hauler on a 3.25 m mean track tips near 0.7 g.
# Dropping the -50 cm restores the geometric height; the longitudinal 60 cm is
# kept because an empty rigid hauler is genuinely nose-heavy.
CENTER_OF_MASS_NUDGE_CM = (60.0, 0.0, 0.0)

WHEEL_ASSETS = (
    ("xiaosong55_LF", 39.0, FRONT_SERVICE_BRAKE_TORQUE_NM, 0.0, False),
    ("xiaosong55_RF", 39.0, FRONT_SERVICE_BRAKE_TORQUE_NM, 0.0, False),
    ("xiaosong55_RL", 0.0, REAR_SERVICE_BRAKE_TORQUE_NM,
     REAR_HANDBRAKE_TORQUE_NM, True),
    ("xiaosong55_RR", 0.0, REAR_SERVICE_BRAKE_TORQUE_NM,
     REAR_HANDBRAKE_TORQUE_NM, True),
)


def wheel_omega(total_reduction):
    return (MAX_RPM * 2.0 * 3.14159265 / 60.0) / total_reduction


def check_ratios():
    """Refuse to write ratios that fall in the zero-torque region."""
    problems = []
    for index, (ratio, _, _) in enumerate(FORWARD_GEARS, start=1):
        omega = wheel_omega(FINAL_RATIO * ratio)
        if omega < MIN_SAFE_WHEEL_OMEGA_RAD_S:
            problems.append("forward gear %d: %.2f rad/s" % (index, omega))
    omega = wheel_omega(FINAL_RATIO * abs(REVERSE_GEAR_RATIO))
    if omega < MIN_SAFE_WHEEL_OMEGA_RAD_S:
        problems.append("reverse: %.2f rad/s" % omega)
    return problems


sys.stdout = open(RESULT_PATH, "w")

problems = check_ratios()
if problems:
    print("ABORT unsafe wheel angular velocity: %s" % "; ".join(problems))
    raise SystemExit(1)

def build_torque_curve():
    """Import the torque curve as a CurveFloat asset and return it."""
    handle = open(TORQUE_CURVE_CSV, "w")
    try:
        for rpm, torque in TORQUE_CURVE_RPM_NM:
            handle.write("%d,%d\n" % (rpm, torque))
    finally:
        handle.close()

    settings = unreal.CSVImportSettings()
    settings.set_editor_property("import_type", unreal.CSVImportType.ECSV_CURVE_FLOAT)
    factory = unreal.CSVImportFactory()
    factory.set_editor_property("automated_import_settings", settings)

    task = unreal.AssetImportTask()
    task.filename = TORQUE_CURVE_CSV
    task.destination_path = "/Game"
    task.destination_name = TORQUE_CURVE_NAME
    task.automated = True
    task.replace_existing = True
    task.save = True
    task.factory = factory
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])

    curve = unreal.load_asset(TORQUE_CURVE_ASSET)
    if curve is None:
        raise RuntimeError("torque curve asset was not created")
    return curve


asset = unreal.load_asset(ASSET_PATH)
generated_class = unreal.load_object(None, CLASS_PATH)
default_object = unreal.get_default_object(generated_class)
movement = default_object.get_component_by_class(
    unreal.WheeledVehicleMovementComponent4W
)

before = {
    "mass": movement.get_editor_property("mass"),
    "engine_moi": movement.get_editor_property("engine_setup").get_editor_property("moi"),
}
transmission = movement.get_editor_property("transmission_setup")
before["final_ratio"] = transmission.get_editor_property("final_ratio")
before["clutch_strength"] = transmission.get_editor_property("clutch_strength")
before["gear_switch_time"] = transmission.get_editor_property("gear_switch_time")
before["reverse_gear_ratio"] = transmission.get_editor_property("reverse_gear_ratio")
before["forward_gears"] = [
    (g.get_editor_property("ratio"),
     g.get_editor_property("down_ratio"),
     g.get_editor_property("up_ratio"))
    for g in transmission.get_editor_property("forward_gears")
]
print("before %s" % json.dumps(before, indent=1, sort_keys=True))

mesh = default_object.get_component_by_class(unreal.SkeletalMeshComponent)
body_instance = mesh.get_editor_property("body_instance")
before["com_nudge_cm"] = [
    body_instance.get_editor_property("com_nudge").x,
    body_instance.get_editor_property("com_nudge").y,
    body_instance.get_editor_property("com_nudge").z,
]
body_instance.set_editor_property(
    "com_nudge", unreal.Vector(*CENTER_OF_MASS_NUDGE_CM)
)
mesh.modify()
mesh.set_editor_property("body_instance", body_instance)

torque_curve_asset = build_torque_curve()
print("torque curve %s" % [
    (rpm, torque_curve_asset.get_float_value(rpm))
    for rpm, _ in TORQUE_CURVE_RPM_NM
])

engine = movement.get_editor_property("engine_setup")
engine.set_editor_property("moi", ENGINE_MOI)
runtime_curve = engine.get_editor_property("torque_curve")
runtime_curve.set_editor_property("external_curve", torque_curve_asset)
engine.set_editor_property("torque_curve", runtime_curve)
movement.modify()
movement.set_editor_property("engine_setup", engine)

gears = []
for ratio, down_ratio, up_ratio in FORWARD_GEARS:
    gear = unreal.VehicleGearData()
    gear.set_editor_property("ratio", ratio)
    gear.set_editor_property("down_ratio", down_ratio)
    gear.set_editor_property("up_ratio", up_ratio)
    gears.append(gear)

transmission.set_editor_property("forward_gears", gears)
transmission.set_editor_property("reverse_gear_ratio", REVERSE_GEAR_RATIO)
transmission.set_editor_property("final_ratio", FINAL_RATIO)
transmission.set_editor_property("gear_switch_time", GEAR_SWITCH_TIME)
transmission.set_editor_property("clutch_strength", CLUTCH_STRENGTH)
transmission.set_editor_property("use_gear_auto_box", True)
movement.set_editor_property("transmission_setup", transmission)
asset.modify()

packages_to_save = [asset.get_outermost(), torque_curve_asset.get_outermost()]
wheel_report = {}
for name, steer_angle, brake_torque, handbrake_torque, handbrake_flag in WHEEL_ASSETS:
    wheel_asset = unreal.load_asset("/Game/{0}".format(name))
    wheel_class = unreal.load_object(None, "/Game/{0}.{0}_C".format(name))
    wheel = unreal.get_default_object(wheel_class)
    wheel_report[name] = {
        "before": {
            "max_brake_torque": wheel.get_editor_property("max_brake_torque"),
            "max_hand_brake_torque": wheel.get_editor_property("max_hand_brake_torque"),
            "mass": wheel.get_editor_property("mass"),
            "long_stiff_value": wheel.get_editor_property("long_stiff_value"),
            "steer_angle": wheel.get_editor_property("steer_angle"),
        }
    }
    wheel.modify()
    wheel.set_editor_property("max_brake_torque", brake_torque)
    wheel.set_editor_property("max_hand_brake_torque", handbrake_torque)
    wheel.set_editor_property("affected_by_handbrake", handbrake_flag)
    wheel.set_editor_property("mass", WHEEL_MASS_KG)
    wheel.set_editor_property("long_stiff_value", WHEEL_LONG_STIFF_VALUE)
    wheel.set_editor_property("steer_angle", steer_angle)
    wheel_asset.modify()
    packages_to_save.append(wheel_asset.get_outermost())
    try:
        unreal.BlueprintEditorLibrary.compile_blueprint(wheel_asset)
    except Exception as error:
        print("wheel_compile_warning %s %r" % (name, error))
    wheel_report[name]["after"] = {
        "max_brake_torque": wheel.get_editor_property("max_brake_torque"),
        "max_hand_brake_torque": wheel.get_editor_property("max_hand_brake_torque"),
        "mass": wheel.get_editor_property("mass"),
        "long_stiff_value": wheel.get_editor_property("long_stiff_value"),
        "steer_angle": wheel.get_editor_property("steer_angle"),
    }

try:
    unreal.BlueprintEditorLibrary.compile_blueprint(asset)
except Exception as error:
    print("compile_warning %r" % error)

saved = unreal.EditorLoadingAndSavingUtils.save_packages(packages_to_save, False)
print("saved %s" % saved)

after_transmission = movement.get_editor_property("transmission_setup")
after = {
    "com_nudge_cm": list(CENTER_OF_MASS_NUDGE_CM),
    "engine_moi": movement.get_editor_property("engine_setup").get_editor_property("moi"),
    "final_ratio": after_transmission.get_editor_property("final_ratio"),
    "clutch_strength": after_transmission.get_editor_property("clutch_strength"),
    "gear_switch_time": after_transmission.get_editor_property("gear_switch_time"),
    "reverse_gear_ratio": after_transmission.get_editor_property("reverse_gear_ratio"),
    "forward_gears": [
        (g.get_editor_property("ratio"),
         g.get_editor_property("down_ratio"),
         g.get_editor_property("up_ratio"))
        for g in after_transmission.get_editor_property("forward_gears")
    ],
}
print("after %s" % json.dumps(after, indent=1, sort_keys=True))

manifest = {
    "before": before,
    "after": after,
    "torque_curve_rpm_nm": [list(point) for point in TORQUE_CURVE_RPM_NM],
    "torque_curve_asset": TORQUE_CURVE_ASSET,
    "wheels": wheel_report,
    "saved": bool(saved),
    "gear_wheel_omega_rad_s": [
        round(wheel_omega(FINAL_RATIO * ratio), 3)
        for ratio, _, _ in FORWARD_GEARS
    ],
    "reverse_wheel_omega_rad_s": round(
        wheel_omega(FINAL_RATIO * abs(REVERSE_GEAR_RATIO)), 3
    ),
    "gear_top_speed_kmh": [
        round(wheel_omega(FINAL_RATIO * ratio) * WHEEL_RADIUS_M * 3.6, 2)
        for ratio, _, _ in FORWARD_GEARS
    ],
}
with open(MANIFEST_PATH, "w") as handle:
    json.dump(manifest, handle, indent=1, sort_keys=True)
print("manifest %s" % MANIFEST_PATH)
print("gear omega rad/s %s" % manifest["gear_wheel_omega_rad_s"])
print("gear top speed km/h %s" % manifest["gear_top_speed_kmh"])
