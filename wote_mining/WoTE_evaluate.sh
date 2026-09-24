#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK_DIR=${WORK_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
export WORK_DIR
export TEAM_AGENT=${TEAM_AGENT:-${SCRIPT_DIR}/WoTE_agent.py}

exec "${WORK_DIR}/leaderboard/scripts/evaluate_hd465_transfuser.sh" "$@"
