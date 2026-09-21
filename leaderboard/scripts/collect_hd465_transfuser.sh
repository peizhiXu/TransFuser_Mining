#!/usr/bin/env bash
set -euo pipefail

# Formal HD465 TransFuser collection plan: three maps x three route/weather
# groups, with weather fixed for each route and 50 same-model background trucks.
WORK_DIR=${WORK_DIR:-/home/ubuntu/projects/transfuser}
# The carla-0.9.10 egg is built for Python 3.7 and needs pkg_resources; the
# shell's active conda env is not guaranteed to be that one (e.g. plain
# "(base)" is Python 3.14 with no pkg_resources, which makes CARLA look like
# it never comes up when actually the readiness probe itself can't import
# carla). Pin an absolute interpreter so this does not depend on what the
# caller's shell happens to have activated.
PYTHON_BIN=${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/tfuse/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/media/ubuntu/徐培智/dataset_transfuser_hd465/raw}
# The final 160-route set is the exact source set represented by the collected
# dataset and the fixed 120/20/20 experiment split.
ROUTES_DIR=${ROUTES_DIR:-${WORK_DIR}/leaderboard/data/mining/routes_final}
SCENARIOS=${SCENARIOS:-${WORK_DIR}/leaderboard/data/mining/empty_scenarios.json}
CARLA_ROOT=${CARLA_ROOT:-/home/ubuntu/carla_0.9.10_dirty/CARLA_0.9.10-dirty-xiaosong55-test}
CARLA_PORT=${CARLA_PORT:-2000}
TM_PORT=${TM_PORT:-8000}
# Rendering settings change the collected RGB images, so they belong in the
# batch fingerprint below. Epic matches upstream TransFuser, which passes no
# -quality-level and therefore takes the CARLA 0.9.10 default.
CARLA_QUALITY_LEVEL=${CARLA_QUALITY_LEVEL:-Epic}
CARLA_RES_X=${CARLA_RES_X:-1920}
CARLA_RES_Y=${CARLA_RES_Y:-1080}
CARLA_HEADLESS=${CARLA_HEADLESS:-1}
CARLA_READY_TIMEOUT=${CARLA_READY_TIMEOUT:-120}
BACKGROUND_VEHICLE_COUNT=${BACKGROUND_VEHICLE_COUNT:-50}
BACKGROUND_SPEED_DIFFERENCE_PERCENT=${BACKGROUND_SPEED_DIFFERENCE_PERCENT:-50}
BACKGROUND_MIN_FOLLOW_DISTANCE=${BACKGROUND_MIN_FOLLOW_DISTANCE:-12}
BACKGROUND_AUTO_LANE_CHANGE=${BACKGROUND_AUTO_LANE_CHANGE:-0}
DATAGEN_DEVICE=${DATAGEN_DEVICE:-cpu}
DRY_RUN=${DRY_RUN:-0}

# The seed belongs to the batch and stays unchanged after an interruption.
# This makes every fresh full run reproducible at the Traffic Manager level.
batches=(
  "0325_3 qingtian 3101"
  "0325_3 bangwan 3102"
  "0325_3 yutian 3103"
  "0325_4 qingtian 4101"
  "0325_4 bangwan 4102"
  "0325_4 yutian 4103"
  "0325_5 qingtian 5101"
  "0325_5 bangwan 5102"
  "0325_5 yutian 5103"
)

code_files=(
  "${WORK_DIR}/team_code_autopilot/autopilot.py"
  "${WORK_DIR}/team_code_autopilot/autopilot_mine.py"
  "${WORK_DIR}/team_code_autopilot/data_agent.py"
  "${WORK_DIR}/team_code_autopilot/data_agent_mine.py"
  "${WORK_DIR}/team_code_autopilot/mining_vehicle_physics.py"
  # nav_planner decides which dense route the expert actually drives, so a
  # change here changes the collected data as much as the expert itself.
  "${WORK_DIR}/team_code_autopilot/nav_planner.py"
  "${WORK_DIR}/team_code_transfuser/config.py"
  "${WORK_DIR}/leaderboard/leaderboard/leaderboard_evaluator_local.py"
  "${WORK_DIR}/leaderboard/leaderboard/scenarios/route_scenario_local.py"
  "${WORK_DIR}/leaderboard/leaderboard/scenarios/scenario_manager_local.py"
  "${WORK_DIR}/scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria_local.py"
  "${WORK_DIR}/leaderboard/scripts/datagen_mine.sh"
  "${WORK_DIR}/leaderboard/scripts/collect_hd465_transfuser.sh"
  "${SCENARIOS}"
)
code_hash=$(sha256sum "${code_files[@]}" | sha256sum | cut -d' ' -f1)
git_head=$(git -C "${WORK_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)

render_batch_config() {
  local map_name=$1
  local weather_name=$2
  local seed=$3
  local route_file=$4
  local run_name=$5
  local route_hash
  local weather_ids
  local weather_variant_count
  route_hash=$(sha256sum "${route_file}" | cut -d' ' -f1)
  weather_ids=$(python -c \
    'import sys,xml.etree.ElementTree as E; r=E.parse(sys.argv[1]).getroot(); print(",".join(sorted(set(w.get("id", "unknown") for w in r.iter("weather")))))' \
    "${route_file}")
  weather_variant_count=$(python -c \
    'import sys,xml.etree.ElementTree as E; r=E.parse(sys.argv[1]).getroot(); print(len(set(tuple(sorted(w.attrib.items())) for w in r.iter("weather"))))' \
    "${route_file}")
  printf '%s\n' \
    "format=transfuser_2022" \
    "run_name=${run_name}" \
    "map=${map_name}" \
    "weather=${weather_name}" \
    "xml_weather_ids=${weather_ids}" \
    "xml_weather_variant_count=${weather_variant_count}" \
    "routes=${route_file}" \
    "routes_sha256=${route_hash}" \
    "scenarios=${SCENARIOS}" \
    "carla_root=${CARLA_ROOT}" \
    "carla_quality_level=${CARLA_QUALITY_LEVEL}" \
    "carla_resolution=${CARLA_RES_X}x${CARLA_RES_Y}" \
    "carla_render_backend=opengl" \
    "carla_headless=${CARLA_HEADLESS}" \
    "carla_port=${CARLA_PORT}" \
    "traffic_manager_port=${TM_PORT}" \
    "traffic_manager_seed=${seed}" \
    "ego_vehicle=vehicle.xiaosong55t.xiaosong55t" \
    "background_vehicle=vehicle.xiaosong55t.xiaosong55t" \
    "background_vehicle_count=${BACKGROUND_VEHICLE_COUNT}" \
    "background_speed_difference_percent=${BACKGROUND_SPEED_DIFFERENCE_PERCENT}" \
    "background_min_follow_distance_m=${BACKGROUND_MIN_FOLLOW_DISTANCE}" \
    "background_auto_lane_change=${BACKGROUND_AUTO_LANE_CHANGE}" \
    "datagen_device=${DATAGEN_DEVICE}" \
    "debug_challenge=0" \
    "weather_policy=fixed_from_route_xml" \
    "git_head=${git_head}" \
    "collection_code_sha256=${code_hash}"
}

CARLA_PGID=""

# A stale server on this port is dangerous: the readiness probe would connect to
# it while the newly launched server dies on bind(), and the evaluator would
# then drive the wrong world.
carla_port_busy() {
  ss -ltnH "sport = :${CARLA_PORT}" 2>/dev/null | grep -q .
}

start_carla() {
  local log_file=$1
  if carla_port_busy; then
    echo "ERROR: port ${CARLA_PORT} is already in use. Stop that CARLA first." >&2
    return 1
  fi
  local -a launch_env=()
  if [[ "${CARLA_HEADLESS}" == "1" ]]; then
    # 0.9.10 has no -RenderOffScreen; offscreen SDL is the documented way.
    launch_env=(SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0)
  fi
  setsid env "${launch_env[@]}" "${CARLA_ROOT}/CarlaUE4.sh" \
    -world-port="${CARLA_PORT}" \
    -opengl \
    -quality-level="${CARLA_QUALITY_LEVEL}" \
    -windowed \
    -ResX="${CARLA_RES_X}" \
    -ResY="${CARLA_RES_Y}" \
    -nosound >> "${log_file}" 2>&1 &
  CARLA_PGID=$!

  local deadline=$((SECONDS + CARLA_READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg" \
       "${PYTHON_BIN}" -c "import carla, sys; c = carla.Client('127.0.0.1', ${CARLA_PORT}); c.set_timeout(2.0); c.get_world()" \
       >/dev/null 2>&1; then
      echo "CARLA ready (pgid ${CARLA_PGID}, ${CARLA_QUALITY_LEVEL}, headless=${CARLA_HEADLESS})"
      return 0
    fi
    sleep 3
  done
  echo "ERROR: CARLA did not become ready within ${CARLA_READY_TIMEOUT}s; see ${log_file}" >&2
  stop_carla
  return 1
}

stop_carla() {
  [[ -n "${CARLA_PGID}" ]] || return 0
  # The server was started with setsid, so it owns its own process group and
  # this cannot reach unrelated desktop processes.
  kill -TERM -- "-${CARLA_PGID}" 2>/dev/null || true
  local deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )) && carla_port_busy; do sleep 1; done
  kill -KILL -- "-${CARLA_PGID}" 2>/dev/null || true
  CARLA_PGID=""
}

trap 'stop_carla' EXIT INT TERM

fmt_hms() {
  local t=$1
  printf '%dh%02dm%02ds' $((t/3600)) $(((t%3600)/60)) $((t%60))
}

# Progress and ETA from what has actually been collected: every finished route
# records its own length and wall-clock duration, so the remaining time is
# extrapolated from the measured seconds-per-metre rather than a fixed guess.
print_progress() {
  local elapsed=$1
  python - "${OUTPUT_ROOT}" "${ROUTES_DIR}" "${elapsed}" <<'PY'
import datetime, glob, json, math, os, sys
import xml.etree.ElementTree as ET

output_root, routes_dir, elapsed = sys.argv[1], sys.argv[2], int(sys.argv[3])

def route_length(route):
    pts = [(float(w.get('x')), float(w.get('y'))) for w in route.iter('waypoint')]
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1))

planned_m = planned_n = 0
for path in sorted(glob.glob(os.path.join(routes_dir, '*.xml'))):
    for route in ET.parse(path).getroot().iter('route'):
        planned_m += route_length(route)
        planned_n += 1

done_m = done_n = 0
done_s = 0.0
for path in sorted(glob.glob(os.path.join(output_root, '*', '*.json'))):
    try:
        records = json.load(open(path))['_checkpoint']['records']
    except (OSError, ValueError, KeyError):
        continue
    for record in records:
        if record.get('status') != 'Completed':
            continue
        meta = record.get('meta', {})
        done_m += float(meta.get('route_length', 0.0))
        done_s += float(meta.get('duration_system', 0.0))
        done_n += 1

def hms(t):
    t = int(max(0, t))
    return '%dh%02dm' % (t // 3600, (t % 3600) // 60)

print('  progress : %d/%d routes, %.1f/%.1f km (%.1f%%)'
      % (done_n, planned_n, done_m / 1000, planned_m / 1000,
         100 * done_m / planned_m if planned_m else 0))
print('  elapsed  : %s' % hms(elapsed))
if done_m > 0 and done_s > 0:
    rate = done_s / done_m
    remaining = max(0.0, planned_m - done_m) * rate
    print('  rate     : %.3f s/m  (%.1f min per km)' % (rate, rate * 1000 / 60))
    finish = datetime.datetime.now() + datetime.timedelta(seconds=remaining)
    print('  remaining: ~%s   estimated finish %s'
          % (hms(remaining), finish.strftime('%m-%d %H:%M')))
PY
}

echo "HD465 TransFuser formal collection"
echo "output: ${OUTPUT_ROOT}"
echo "batches: ${#batches[@]}"
echo "NPCs per route: ${BACKGROUND_VEHICLE_COUNT}"

total_route_count=0
collection_start=${SECONDS}
batch_index=0

for spec in "${batches[@]}"; do
  read -r map_name weather_name seed <<< "${spec}"
  batch_index=$((batch_index + 1))
  route_file=${ROUTES_DIR}/${map_name}${weather_name}.xml
  run_name=hd465_tf_${map_name}_${weather_name}_npc${BACKGROUND_VEHICLE_COUNT}_seed${seed}
  # One top-level directory per map, so holding a whole map out as the test set
  # is just a matter of not pointing the split tool at it. The three weather
  # batches stay separate inside it: each keeps its own checkpoint, fingerprint
  # and resume state.
  map_root=${OUTPUT_ROOT}/${map_name}
  checkpoint=${map_root}/${weather_name}.json
  data_root=${map_root}/${weather_name}
  log_file=${map_root}/${weather_name}.log
  report_file=${map_root}/${weather_name}.report.json
  config_file=${map_root}/${weather_name}.config.txt
  [[ "${DRY_RUN}" == "1" ]] || mkdir -p "${map_root}"

  if [[ ! -f "${route_file}" ]]; then
    echo "ERROR: missing route file: ${route_file}" >&2
    exit 1
  fi

  route_count=$(grep -c '<route ' "${route_file}")
  total_route_count=$((total_route_count + route_count))
  echo
  echo "===== ${run_name}: ${route_count} routes ====="

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "routes: ${route_file}"
    echo "checkpoint: ${checkpoint}"
    echo "data: ${data_root}"
    continue
  fi

  # RouteIndexer saves progress only after a route finishes. If CARLA was
  # killed during the next route, move that uncommitted timestamped directory
  # aside before validating the completed routes and retrying it.
  if [[ -s "${checkpoint}" ]]; then
    progress_values=$(python -c \
      'import json,sys; p=json.load(open(sys.argv[1]))["_checkpoint"]["progress"]; print(p[0], p[1])' \
      "${checkpoint}" 2>/dev/null || true)
    if [[ "${progress_values}" =~ ^([0-9]+)[[:space:]]+([0-9]+)$ ]]; then
      current_route=${BASH_REMATCH[1]}
      total_routes=${BASH_REMATCH[2]}
      if [[ ${current_route} -lt ${total_routes} ]]; then
        route_stem=$(basename "${route_file}" .xml)
        shopt -s nullglob
        interrupted_dirs=("${data_root}/${route_stem}_route${current_route}_"*)
        shopt -u nullglob
        if [[ ${#interrupted_dirs[@]} -gt 0 ]]; then
          quarantine=${map_root}/incomplete/${weather_name}/route${current_route}_$(date +%Y%m%d_%H%M%S)
          mkdir -p "${quarantine}"
          mv "${interrupted_dirs[@]}" "${quarantine}/"
          echo "moved interrupted route ${current_route} to ${quarantine}"
        fi
      fi
    fi
  fi

  check_status=2
  if [[ -s "${checkpoint}" ]]; then
    set +e
    python "${WORK_DIR}/tools/dataset/check_mining_collection.py" \
      --checkpoint "${checkpoint}" \
      --data-root "${data_root}" \
      --json-out "${report_file}"
    check_status=$?
    set -e
  fi

  if [[ ${check_status} -eq 0 ]]; then
    echo "already complete; skipping"
    continue
  fi
  if [[ ${check_status} -eq 1 ]]; then
    echo "ERROR: existing batch is inconsistent; inspect ${report_file}" >&2
    exit 1
  fi

  # Only batches that still need work are held to the fingerprint. A finished
  # batch keeps its own config.txt as the record of what collected it; checking
  # that against today's code would abort the whole run every time anything
  # upstream of a completed batch changes.
  if [[ -f "${config_file}" ]]; then
    if ! cmp -s "${config_file}" <(render_batch_config \
      "${map_name}" "${weather_name}" "${seed}" "${route_file}" "${run_name}"); then
      echo "ERROR: current code or settings differ from ${config_file}." >&2
      echo "Do not resume one batch with a different expert or traffic configuration." >&2
      exit 1
    fi
  else
    render_batch_config \
      "${map_name}" "${weather_name}" "${seed}" "${route_file}" "${run_name}" \
      > "${config_file}"
  fi

  resume=0
  if [[ -s "${checkpoint}" ]]; then
    resume=1
    echo "resuming from ${checkpoint}"
  else
    if [[ -d "${data_root}" ]] && [[ -n "$(find "${data_root}" -mindepth 1 -print -quit)" ]]; then
      echo "ERROR: ${data_root} contains data but has no usable checkpoint." >&2
      echo "Move it aside before starting a new batch." >&2
      exit 1
    fi
    echo "starting a new batch"
  fi

  # A fresh server per batch. The Python-side transition leak is fixed, but the
  # CARLA server's own memory across a map switch was never verified, and a
  # restart costs about a minute against a multi-hour batch.
  echo "starting CARLA for ${map_name} (${weather_name})"
  start_carla "${map_root}/${weather_name}.carla.log"

  batch_start=${SECONDS}
  set +e
  WORK_DIR="${WORK_DIR}" \
  CARLA_ROOT="${CARLA_ROOT}" \
  ROUTES="${route_file}" \
  SCENARIOS="${SCENARIOS}" \
  RUN_NAME="${weather_name}" \
  OUTPUT_ROOT="${map_root}" \
  MEMORY_TRACE=1 \
  CARLA_PORT="${CARLA_PORT}" \
  TM_PORT="${TM_PORT}" \
  TRAFFIC_MANAGER_SEED="${seed}" \
  BACKGROUND_VEHICLE_COUNT="${BACKGROUND_VEHICLE_COUNT}" \
  BACKGROUND_SPEED_DIFFERENCE_PERCENT="${BACKGROUND_SPEED_DIFFERENCE_PERCENT}" \
  BACKGROUND_MIN_FOLLOW_DISTANCE="${BACKGROUND_MIN_FOLLOW_DISTANCE}" \
  BACKGROUND_AUTO_LANE_CHANGE="${BACKGROUND_AUTO_LANE_CHANGE}" \
  DATAGEN_DEVICE="${DATAGEN_DEVICE}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  PREVIEW_ONLY=0 \
  DEBUG_CHALLENGE=0 \
  FREE_SPECTATOR=0 \
  RESUME="${resume}" \
  bash "${WORK_DIR}/leaderboard/scripts/datagen_mine.sh" \
    2>&1 | tee -a "${log_file}"
  collection_status=${PIPESTATUS[0]}
  set -e
  batch_seconds=$((SECONDS - batch_start))

  echo "stopping CARLA for ${map_name} (${weather_name})"
  stop_carla

  if [[ ${collection_status} -ne 0 ]]; then
    echo "ERROR: collection process exited with ${collection_status}." >&2
    echo "Restart this same script; it will resume from the checkpoint." >&2
    exit "${collection_status}"
  fi

  set +e
  python "${WORK_DIR}/tools/dataset/check_mining_collection.py" \
    --checkpoint "${checkpoint}" \
    --data-root "${data_root}" \
    --require-all-completed \
    --json-out "${report_file}"
  check_status=$?
  set -e
  if [[ ${check_status} -ne 0 ]]; then
    echo "ERROR: batch did not pass validation; inspect ${report_file}" >&2
    exit "${check_status}"
  fi

  echo
  echo "----- batch ${batch_index}/${#batches[@]} done: ${map_name} ${weather_name} -----"
  echo "  routes   : ${route_count}"
  echo "  batch    : $(fmt_hms ${batch_seconds})"
  print_progress "$((SECONDS - collection_start))"
  echo
done

echo
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete: 9 batches, ${total_route_count} routes. No data was written."
else
  echo "All nine collection batches passed structural validation."
  echo "Total wall-clock time: $(fmt_hms $((SECONDS - collection_start)))"
  echo "Data root: ${OUTPUT_ROOT} (one directory per map)"
fi
