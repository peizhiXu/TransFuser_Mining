#!/usr/bin/env bash
set -euo pipefail

# Closed-loop HD465 evaluation.  A compatible CARLA server must already be
# running.  All paths and ports can be overridden through environment values,
# which makes the same wrapper usable on the training server and eval machine.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK_DIR=${WORK_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
CARLA_ROOT=${CARLA_ROOT:?Set CARLA_ROOT to the custom CARLA installation}
TEAM_CONFIG=${TEAM_CONFIG:?Set TEAM_CONFIG to the checkpoint directory}
TEAM_AGENT=${TEAM_AGENT:-${WORK_DIR}/team_code_transfuser/submission_agent.py}
ROUTES=${ROUTES:-${WORK_DIR}/leaderboard/data/mining/split_v2/val_routes.xml}
SCENARIOS=${SCENARIOS:-${WORK_DIR}/leaderboard/data/mining/empty_scenarios.json}
CHECKPOINT_ENDPOINT=${CHECKPOINT_ENDPOINT:-${WORK_DIR}/results/hd465_transfuser_eval.json}
REPETITIONS=${REPETITIONS:-1}
CARLA_HOST=${CARLA_HOST:-127.0.0.1}
CARLA_PORT=${CARLA_PORT:-2000}
TM_PORT=${TM_PORT:-8000}
TRAFFIC_MANAGER_SEED=${TRAFFIC_MANAGER_SEED:-0}
BACKGROUND_VEHICLE_COUNT=${BACKGROUND_VEHICLE_COUNT:-50}
BACKGROUND_SPEED_DIFFERENCE_PERCENT=${BACKGROUND_SPEED_DIFFERENCE_PERCENT:-50}
BACKGROUND_MIN_FOLLOW_DISTANCE=${BACKGROUND_MIN_FOLLOW_DISTANCE:-12}
BACKGROUND_AUTO_LANE_CHANGE=${BACKGROUND_AUTO_LANE_CHANGE:-0}
RESUME=${RESUME:-0}
ROUTE_ID=${ROUTE_ID:-}
ROUTE_IDS=${ROUTE_IDS:-}
EVAL_ARTIFACTS_ROOT=${EVAL_ARTIFACTS_ROOT:-${CHECKPOINT_ENDPOINT%.json}_frames}
SAVE_COMPOSITE_FRAMES=${SAVE_COMPOSITE_FRAMES:-1}
PROVENANCE=${PROVENANCE:-${ROUTES}.provenance.json}
RESULT_TABLE_DIR=${RESULT_TABLE_DIR:-${CHECKPOINT_ENDPOINT%.json}_result_tables}

for required in "${ROUTES}" "${SCENARIOS}" "${TEAM_AGENT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: required path does not exist: ${required}" >&2
    exit 2
  fi
done
if [[ ! -e "${TEAM_CONFIG}/args.txt" && ! -e "${TEAM_CONFIG}/args.json" ]]; then
  echo "ERROR: checkpoint directory has neither args.txt nor args.json: ${TEAM_CONFIG}" >&2
  exit 2
fi

mkdir -p "$(dirname "${CHECKPOINT_ENDPOINT}")"

if [[ -n "${EVAL_ARTIFACTS_ROOT}" ]]; then
  if [[ "${RESUME}" != "1" && -d "${EVAL_ARTIFACTS_ROOT}" ]] \
      && find "${EVAL_ARTIFACTS_ROOT}" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR: artifact directory is not empty: ${EVAL_ARTIFACTS_ROOT}" >&2
    echo "Use a fresh directory, or set RESUME=1 for the matching run." >&2
    exit 2
  fi
  mkdir -p "${EVAL_ARTIFACTS_ROOT}"
  export SAVE_PATH="${EVAL_ARTIFACTS_ROOT}"
  export SAVE_COMPOSITE_FRAMES
fi

export PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:${WORK_DIR}/scenario_runner:${WORK_DIR}/leaderboard:${WORK_DIR}/team_code_transfuser:${PYTHONPATH:-}"
export DATAGEN=0
export DEBUG_CHALLENGE=${DEBUG_CHALLENGE:-0}
export EGO_VEHICLE_MODEL=vehicle.xiaosong55t.xiaosong55t
export EXPECTED_EGO_VEHICLE_MODEL=${EGO_VEHICLE_MODEL}
export BACKGROUND_VEHICLE_MODEL=vehicle.xiaosong55t.xiaosong55t
export SCENARIO_VEHICLE_MODEL=vehicle.xiaosong55t.xiaosong55t
export TRAFFIC_MANAGER_SEED
export BACKGROUND_VEHICLE_COUNT
export BACKGROUND_SPEED_DIFFERENCE_PERCENT
export BACKGROUND_MIN_FOLLOW_DISTANCE
export BACKGROUND_AUTO_LANE_CHANGE

evaluator_args=(
  --host="${CARLA_HOST}"
  --port="${CARLA_PORT}"
  --trafficManagerPort="${TM_PORT}"
  --trafficManagerSeed="${TRAFFIC_MANAGER_SEED}"
  --scenarios="${SCENARIOS}"
  --routes="${ROUTES}"
  --repetitions="${REPETITIONS}"
  --track=SENSORS
  --checkpoint="${CHECKPOINT_ENDPOINT}"
  --agent="${TEAM_AGENT}"
  --agent-config="${TEAM_CONFIG}"
  --debug="${DEBUG_CHALLENGE}"
)

if [[ "${RESUME}" == "1" ]]; then
  evaluator_args+=(--resume=True)
fi
if [[ -n "${ROUTE_ID}" ]]; then
  evaluator_args+=(--route-id="${ROUTE_ID}")
fi
if [[ -n "${ROUTE_IDS}" ]]; then
  evaluator_args+=(--route-ids="${ROUTE_IDS}")
fi

echo "HD465 eval: routes=${ROUTES} checkpoint=${TEAM_CONFIG} seed=${TRAFFIC_MANAGER_SEED}"
set +e
"${PYTHON_BIN}" \
  "${WORK_DIR}/leaderboard/leaderboard/leaderboard_evaluator_local.py" \
  "${evaluator_args[@]}"
evaluator_status=$?
set -e

# Build the route-result CSV even for a partial/crashed batch, provided the
# evaluator managed to write at least one checkpoint record.
if [[ -s "${CHECKPOINT_ENDPOINT}" ]]; then
  table_args=(
    --results "${CHECKPOINT_ENDPOINT}"
    --repetitions "${REPETITIONS}"
    --output-dir "${RESULT_TABLE_DIR}"
  )
  if [[ -f "${PROVENANCE}" ]]; then
    table_args+=(--provenance "${PROVENANCE}")
  fi
  if [[ -n "${EVAL_ARTIFACTS_ROOT}" ]]; then
    table_args+=(--artifacts-root "${EVAL_ARTIFACTS_ROOT}")
  fi
  if ! "${PYTHON_BIN}" "${WORK_DIR}/tools/eval/summarize_closed_loop_results.py" \
      "${table_args[@]}"; then
    echo "WARNING: evaluator output exists, but result-table generation failed." >&2
  fi
fi

exit "${evaluator_status}"
