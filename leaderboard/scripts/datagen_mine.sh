#!/usr/bin/env bash
set -euo pipefail

# A CARLA server containing 0325_3/4/5 and xiaosong55t must already be running.
WORK_DIR=${WORK_DIR:-/home/ubuntu/projects/transfuser}
# Pin an absolute interpreter so this does not depend on whatever conda env
# happens to be active in the caller's shell (plain "(base)" is Python 3.14
# and lacks pkg_resources/torch, so both the carla egg and data_agent_mine.py
# would fail to import under it).
PYTHON_BIN=${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/tfuse/bin/python}
CARLA_ROOT=${CARLA_ROOT:-/home/ubuntu/carla_0.9.10_dirty/CARLA_0.9.10-dirty-xiaosong55-test}
ROUTES=${ROUTES:-${WORK_DIR}/leaderboard/data/mining/routes_final/0325_5qingtian.xml}
SCENARIOS=${SCENARIOS:-${WORK_DIR}/leaderboard/data/mining/empty_scenarios.json}
RUN_NAME=${RUN_NAME:-hd465_0325_5_qingtian}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_DIR}/results/mining_datagen}
CARLA_PORT=${CARLA_PORT:-2000}
TM_PORT=${TM_PORT:-8000}
TRAFFIC_MANAGER_SEED=${TRAFFIC_MANAGER_SEED:-0}
PREVIEW_ONLY=${PREVIEW_ONLY:-0}
RESUME=${RESUME:-0}
FREE_SPECTATOR=${FREE_SPECTATOR:-0}
DEBUG_DRAW_STRIDE=${DEBUG_DRAW_STRIDE:-5}
ROUTE_ID=${ROUTE_ID:-}
ROUTE_IDS=${ROUTE_IDS:-}

# team_code_transfuser is on the path so data_agent_mine.py can read the
# sensor mounting from GlobalConfig instead of keeping a second copy.
export PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:${WORK_DIR}/scenario_runner:${WORK_DIR}/leaderboard:${WORK_DIR}/team_code_autopilot:${WORK_DIR}/team_code_transfuser:${PYTHONPATH:-}"
export ROUTES
if [[ "${PREVIEW_ONLY}" == "1" ]]; then
  # Drive and visualize only. No RGB/LiDAR/label directories are created, so
  # debug geometry can never contaminate a training set.
  unset SAVE_PATH
  export DEBUG_CHALLENGE=${DEBUG_CHALLENGE:-1}
  export DATAGEN=0
  AGENT=${WORK_DIR}/team_code_autopilot/autopilot_mine.py
  CHECKPOINT=${OUTPUT_ROOT}/${RUN_NAME}_preview.json
else
  export SAVE_PATH=${OUTPUT_ROOT}/${RUN_NAME}
  export DEBUG_CHALLENGE=${DEBUG_CHALLENGE:-0}
  export DATAGEN=1
  AGENT=${WORK_DIR}/team_code_autopilot/data_agent_mine.py
  CHECKPOINT=${OUTPUT_ROOT}/${RUN_NAME}.json
fi
# PyTorch 1.12/CUDA 11.3 cannot execute kernels on the local RTX 5070 Ti.
# CARLA still renders on the GPU; only the 2 Hz BEV label renderer uses CPU.
export DATAGEN_DEVICE=${DATAGEN_DEVICE:-cpu}

export EGO_VEHICLE_MODEL=vehicle.xiaosong55t.xiaosong55t
export BACKGROUND_VEHICLE_MODEL=vehicle.xiaosong55t.xiaosong55t
export SCENARIO_VEHICLE_MODEL=vehicle.xiaosong55t.xiaosong55t
# Ten HD465 trucks remains the convenient default for a single manual run.
# Formal collection overrides this to 50 in collect_hd465_transfuser.sh.
export BACKGROUND_VEHICLE_COUNT=${BACKGROUND_VEHICLE_COUNT:-10}
# The map advertises 35-40 mph limits, which are too high for loaded mine
# traffic. Fifty percent below those limits gives roughly 28-32 km/h. Keep a
# truck-scale 12 m minimum gap between Traffic Manager vehicles.
export BACKGROUND_SPEED_DIFFERENCE_PERCENT=${BACKGROUND_SPEED_DIFFERENCE_PERCENT:-50}
export BACKGROUND_MIN_FOLLOW_DISTANCE=${BACKGROUND_MIN_FOLLOW_DISTANCE:-12}
# HD465 trucks keep their lane. CARLA 0.9.10's passenger-car lane-change
# controller can otherwise attempt a 15 m lateral move on the mining map.
export BACKGROUND_AUTO_LANE_CHANGE=${BACKGROUND_AUTO_LANE_CHANGE:-0}
export TRAFFIC_MANAGER_SEED
export FREE_SPECTATOR
export DEBUG_DRAW_STRIDE

mkdir -p "${OUTPUT_ROOT}"

evaluator_args=(
  --host=127.0.0.1
  --port="${CARLA_PORT}"
  --trafficManagerPort="${TM_PORT}"
  --trafficManagerSeed="${TRAFFIC_MANAGER_SEED}"
  --scenarios="${SCENARIOS}"
  --routes="${ROUTES}"
  --repetitions=1
  --track=MAP
  --checkpoint="${CHECKPOINT}"
  --agent="${AGENT}"
  --agent-config=""
  --debug="${DEBUG_CHALLENGE}"
)

if [[ -n "${ROUTE_ID}" ]]; then
  evaluator_args+=(--route-id="${ROUTE_ID}")
fi
if [[ -n "${ROUTE_IDS}" ]]; then
  evaluator_args+=(--route-ids="${ROUTE_IDS}")
fi

# argparse in this CARLA Leaderboard version uses type=bool.  Supplying
# "--resume=False" would still evaluate to True, so only add the flag when a
# resume is actually requested.
if [[ "${RESUME}" == "1" ]]; then
  evaluator_args+=(--resume=True)
fi

exec "${PYTHON_BIN}" "${WORK_DIR}/leaderboard/leaderboard/leaderboard_evaluator_local.py" \
  "${evaluator_args[@]}"
