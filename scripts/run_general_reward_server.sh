#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAN_PYTHON="${WAN_PYTHON:-python3}"

if ! command -v "${WAN_PYTHON}" >/dev/null 2>&1; then
  echo "python executable not found: ${WAN_PYTHON}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export GENERAL_REWARD_PORT="${GENERAL_REWARD_PORT:-8090}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"

cd "${REPO_DIR}"
"${WAN_PYTHON}" scripts/serve_general_reward.py "$@"
