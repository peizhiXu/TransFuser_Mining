#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-/home/kemove/xpz/datasets/mining_dataset_hd465/split_v2}
LOG_ROOT=${LOG_ROOT:-/home/kemove/xpz/outputs/transfuser}
EXPERIMENT_ID=${EXPERIMENT_ID:-hd465_transfuser_split_v2}
GPU_IDS=${GPU_IDS:-0,1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
BATCH_SIZE=${BATCH_SIZE:-4}
EPOCHS=${EPOCHS:-41}
VAL_EVERY=${VAL_EVERY:-5}
RDZV_ID=${RDZV_ID:-20260919}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

for split in train val; do
  if [[ ! -d "${DATA_ROOT}/${split}" ]]; then
    echo "ERROR: missing dataset directory: ${DATA_ROOT}/${split}" >&2
    echo "Run tools/dataset/prepare_hd465_split_v2.sh first." >&2
    exit 2
  fi
done

train_count=$(find "${DATA_ROOT}/train" -maxdepth 1 -type l | wc -l)
val_count=$(find "${DATA_ROOT}/val" -maxdepth 1 -type l | wc -l)
if [[ "${train_count}" -ne 120 || "${val_count}" -ne 20 ]]; then
  echo "ERROR: expected 120 train and 20 val route links; found ${train_count} and ${val_count}." >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}/team_code_transfuser"

echo "Training HD465 TransFuser: GPUs=${GPU_IDS}, batch/GPU=${BATCH_SIZE}, epochs=${EPOCHS}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
torchrun \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --max_restarts=0 \
  --rdzv_id="${RDZV_ID}" \
  --rdzv_backend=c10d \
  train.py \
  --id "${EXPERIMENT_ID}" \
  --root_dir "${DATA_ROOT}" \
  --setting mining \
  --logdir "${LOG_ROOT}" \
  --parallel_training 1 \
  --backbone transFuser \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --val_every "${VAL_EVERY}" \
  --no_semantic_loss 1 \
  "$@"
