#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
SOURCE_ROOT=${SOURCE_ROOT:-/home/kemove/xpz/datasets/mining_dataset_hd465/raw}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/kemove/xpz/datasets/mining_dataset_hd465/split_v2}
ASSIGNMENTS=${ASSIGNMENTS:-${PROJECT_ROOT}/leaderboard/data/mining/split_v2/split_assignments.csv}
PYTHON_BIN=${PYTHON_BIN:-python3}

"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/dataset/apply_hd465_split_v2.py" \
  --source "${SOURCE_ROOT}" \
  --assignments "${ASSIGNMENTS}" \
  --output "${OUTPUT_ROOT}"

