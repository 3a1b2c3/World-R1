#!/bin/bash
set -euo pipefail

# Multi-node launcher for the public World-R1 release.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_DIR}/scripts/run_name_utils.sh"
WAN_PYTHON="${WAN_PYTHON:-python3}"
WAN_TORCHRUN="${WAN_TORCHRUN:-torchrun}"

if ! command -v "${WAN_PYTHON}" >/dev/null 2>&1; then
  echo "python executable not found: ${WAN_PYTHON}" >&2
  exit 1
fi
if ! command -v "${WAN_TORCHRUN}" >/dev/null 2>&1; then
  echo "torchrun executable not found: ${WAN_TORCHRUN}" >&2
  exit 1
fi

export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"

CACHE_ROOT="${CACHE_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/world-r1}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export WANDB_PROJECT="${WANDB_PROJECT:-world-r1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

MODEL_FAMILY="${MODEL_FAMILY:-wan}"
if [[ "${MODEL_FAMILY}" == "cogvideox" ]]; then
  DEFAULT_MODEL_PATH="THUDM/CogVideoX1.5-5B"
  DEFAULT_TRAIN_CONFIG="config/world_r1.py:world_r1_cogvideox_5b"
  DEFAULT_TEXT_MAX_LENGTH="226"
  DEFAULT_GUIDANCE_SCALE="6.0"
else
  DEFAULT_MODEL_PATH=""
  DEFAULT_TRAIN_CONFIG="config/world_r1.py:world_r1_large"
  DEFAULT_TEXT_MAX_LENGTH="512"
  DEFAULT_GUIDANCE_SCALE="5.0"
fi

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/logs/world_r1}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RUN_NAME="${RUN_NAME:-}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${DEFAULT_TRAIN_CONFIG}}"
export GENERAL_REWARD_SERVER_URL="${GENERAL_REWARD_SERVER_URL:-http://127.0.0.1:${GENERAL_REWARD_PORT:-8090}}"
TRAIN_TEXT_MAX_LENGTH="${TRAIN_TEXT_MAX_LENGTH:-${DEFAULT_TEXT_MAX_LENGTH}}"
TRAIN_GUIDANCE_SCALE="${TRAIN_GUIDANCE_SCALE:-${DEFAULT_GUIDANCE_SCALE}}"
TRAIN_HEIGHT="${TRAIN_HEIGHT:-}"
TRAIN_WIDTH="${TRAIN_WIDTH:-}"
TRAIN_FRAMES="${TRAIN_FRAMES:-}"
TRAIN_NUM_STEPS="${TRAIN_NUM_STEPS:-}"
TRAIN_EVAL_NUM_STEPS="${TRAIN_EVAL_NUM_STEPS:-}"
TRAIN_NOISE_LEVEL="${TRAIN_NOISE_LEVEL:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
TRAIN_TEST_BATCH_SIZE="${TRAIN_TEST_BATCH_SIZE:-}"
TRAIN_NUM_IMAGE_PER_PROMPT="${TRAIN_NUM_IMAGE_PER_PROMPT:-}"
TRAIN_NUM_BATCHES_PER_EPOCH="${TRAIN_NUM_BATCHES_PER_EPOCH:-}"
TRAIN_SAMPLE_TIME_PER_PROMPT="${TRAIN_SAMPLE_TIME_PER_PROMPT:-}"
TRAIN_GRAD_ACCUM_STEPS="${TRAIN_GRAD_ACCUM_STEPS:-}"
TRAIN_WRAP_STRENGTH="${TRAIN_WRAP_STRENGTH:-}"
TRAIN_LORA_PATH="${TRAIN_LORA_PATH:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required for MODEL_FAMILY=${MODEL_FAMILY}. Set it to a local path or a Hugging Face repo id." >&2
  exit 1
fi

if [[ -z "${REWARD_3D_SERVER_URL:-}" ]]; then
  echo "REWARD_3D_SERVER_URL is required. Example: http://10.0.0.12:18089" >&2
  exit 1
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-1}}"
MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-127.0.0.1}}"
ARNOLD_MASTER_PORT="${ARNOLD_WORKER_0_PORT:-29500}"
MASTER_PORT_ARRAY=(${ARNOLD_MASTER_PORT//,/ })
MASTER_PORT="${MASTER_PORT:-${MASTER_PORT_ARRAY[0]}}"
NNODES="${NNODES:-${ARNOLD_WORKER_NUM:-1}}"
NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-0}}"

if [ "${NNODES}" -le 1 ] && command -v python3 >/dev/null 2>&1; then
  MASTER_PORT=$(python3 - <<'PY'
import socket
with socket.socket() as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
PY
  )
fi

if command -v lsof >/dev/null 2>&1; then
  if lsof -Pi:"${MASTER_PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    for candidate in 29500 29501 29502 29503 29504 29505 29511 29521; do
      if ! lsof -Pi:"${candidate}" -sTCP:LISTEN -t >/dev/null 2>&1; then
        MASTER_PORT=${candidate}
        break
      fi
    done
  fi
fi

AUTO_RUN_JOB_TOKEN="${AUTO_RUN_JOB_TOKEN:-${ARNOLD_JOB_ID:-${SLURM_JOB_ID:-${JOB_ID:-${TORCHELASTIC_RUN_ID:-${MASTER_ADDR}_${MASTER_PORT}}}}}}"
AUTO_RUN_KEY="${AUTO_RUN_KEY:-multinode_${AUTO_RUN_JOB_TOKEN}_${MODEL_FAMILY}_$(config_name_from_ref "${TRAIN_CONFIG}")}"
resolve_training_output_identity
trap cleanup_auto_run_reservation EXIT
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

echo "REPO_DIR=${REPO_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "REWARD_3D_SERVER_URL=${REWARD_3D_SERVER_URL}"
echo "GENERAL_REWARD_SERVER_URL=${GENERAL_REWARD_SERVER_URL}"
echo "NNODES=${NNODES}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS}"

CONFIG_OVERRIDES=(
  "--config.model_family=${MODEL_FAMILY}"
  "--config.text_max_length=${TRAIN_TEXT_MAX_LENGTH}"
  "--config.sample.guidance_scale=${TRAIN_GUIDANCE_SCALE}"
)
if [[ -n "${TRAIN_HEIGHT}" ]]; then CONFIG_OVERRIDES+=("--config.height=${TRAIN_HEIGHT}"); fi
if [[ -n "${TRAIN_WIDTH}" ]]; then CONFIG_OVERRIDES+=("--config.width=${TRAIN_WIDTH}"); fi
if [[ -n "${TRAIN_FRAMES}" ]]; then CONFIG_OVERRIDES+=("--config.frames=${TRAIN_FRAMES}"); fi
if [[ -n "${TRAIN_NUM_STEPS}" ]]; then CONFIG_OVERRIDES+=("--config.sample.num_steps=${TRAIN_NUM_STEPS}"); fi
if [[ -n "${TRAIN_EVAL_NUM_STEPS}" ]]; then CONFIG_OVERRIDES+=("--config.sample.eval_num_steps=${TRAIN_EVAL_NUM_STEPS}"); fi
if [[ -n "${TRAIN_NOISE_LEVEL}" ]]; then CONFIG_OVERRIDES+=("--config.sample.noise_level=${TRAIN_NOISE_LEVEL}"); fi
if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
  CONFIG_OVERRIDES+=("--config.sample.train_batch_size=${TRAIN_BATCH_SIZE}")
  CONFIG_OVERRIDES+=("--config.train.batch_size=${TRAIN_BATCH_SIZE}")
fi
if [[ -n "${TRAIN_TEST_BATCH_SIZE}" ]]; then CONFIG_OVERRIDES+=("--config.sample.test_batch_size=${TRAIN_TEST_BATCH_SIZE}"); fi
if [[ -n "${TRAIN_NUM_IMAGE_PER_PROMPT}" ]]; then CONFIG_OVERRIDES+=("--config.sample.num_image_per_prompt=${TRAIN_NUM_IMAGE_PER_PROMPT}"); fi
if [[ -n "${TRAIN_NUM_BATCHES_PER_EPOCH}" ]]; then CONFIG_OVERRIDES+=("--config.sample.num_batches_per_epoch=${TRAIN_NUM_BATCHES_PER_EPOCH}"); fi
if [[ -n "${TRAIN_SAMPLE_TIME_PER_PROMPT}" ]]; then CONFIG_OVERRIDES+=("--config.sample.sample_time_per_prompt=${TRAIN_SAMPLE_TIME_PER_PROMPT}"); fi
if [[ -n "${TRAIN_GRAD_ACCUM_STEPS}" ]]; then CONFIG_OVERRIDES+=("--config.train.gradient_accumulation_steps=${TRAIN_GRAD_ACCUM_STEPS}"); fi
if [[ -n "${TRAIN_WRAP_STRENGTH}" ]]; then CONFIG_OVERRIDES+=("--config.sample.wrap_strength=${TRAIN_WRAP_STRENGTH}"); fi
if [[ -n "${TRAIN_LORA_PATH}" ]]; then CONFIG_OVERRIDES+=("--config.train.lora_path=${TRAIN_LORA_PATH}"); fi

cd "${REPO_DIR}"

TORCHELASTIC_TIMEOUT="${TORCHELASTIC_TIMEOUT:-18000}" \
"${WAN_TORCHRUN}" \
  --nnodes "${NNODES}" \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  scripts/train_world_r1.py \
  --config "${TRAIN_CONFIG}" \
  --config.pretrained.model="${MODEL_PATH}" \
  --config.run_name="${RUN_NAME}" \
  --config.save_dir="${OUTPUT_DIR}" \
  "${CONFIG_OVERRIDES[@]}" \
  ${EXTRA_TRAIN_ARGS}
