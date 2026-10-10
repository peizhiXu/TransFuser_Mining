#!/usr/bin/env bash
# Run from the test machine with conda environment tfuse and CARLA ready.
set -euo pipefail
passes=${1:-1}
seed=${2:-0}
if [[ "$passes" != 1 && "$passes" != 2 ]] || [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash wote_mining/run_v1_world_pass_ablation.sh [1|2] [seed]" >&2
  exit 2
fi
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir="/home/xpz/WoTE_future_bev/closed_loop/residual_v1_epoch30_world${passes}_seed${seed}"
checkpoint_dir=/home/xpz/WoTE_future_bev/checkpoints/residual_adalnzero_topk_detach_epoch30
if [[ -e "$output_dir/test_results.json" || -e "$output_dir/evaluate.log" ]]; then
  echo "Output already exists: $output_dir. Preserve it before rerunning." >&2
  exit 1
fi
test -f "$checkpoint_dir/checkpoint_030.pth"
unset WOTE_COARSE_ONLY ROUTE_ID ROUTE_IDS WORK_DIR TEAM_AGENT
mkdir -p "$output_dir"
cd "$project_dir"
export WOTE_WORLD_PASSES="$passes"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export CARLA_ROOT=/home/xpz/carla/CARLA_0.9.10-dirty-xiaosong55-test
export TEAM_CONFIG="$checkpoint_dir" WOTE_CHECKPOINT=checkpoint_030.pth
export ROUTES="$project_dir/leaderboard/data/mining/split_v2/test_routes.xml"
export SCENARIOS="$project_dir/leaderboard/data/mining/empty_scenarios.json"
export CHECKPOINT_ENDPOINT="$output_dir/test_results.json"
export EVAL_ARTIFACTS_ROOT="$output_dir/test_results_frames"
export RESULT_TABLE_DIR="$output_dir/result_tables"
export SAVE_COMPOSITE_FRAMES=1 SAVE_FRAME_STRIDE=4
export TRAFFIC_MANAGER_SEED="$seed"
export BACKGROUND_VEHICLE_COUNT=50 BACKGROUND_SPEED_DIFFERENCE_PERCENT=50
export BACKGROUND_MIN_FOLLOW_DISTANCE=12 BACKGROUND_AUTO_LANE_CHANGE=0
export REPETITIONS=1 RESUME=0
echo "Residual V1 epoch30: world_passes=$passes seed=$seed; execute decoded trajectories"
bash wote_mining/WoTE_evaluate.sh 2>&1 | tee "$output_dir/evaluate.log"
