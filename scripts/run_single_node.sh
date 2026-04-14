#!/bin/bash
set -euo pipefail

# Single-node launcher for the public World-R1 release.
# It starts a local reward server and then launches multi-GPU training.

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
  DEFAULT_TRAIN_BATCH_SIZE="1"
  DEFAULT_TRAIN_NUM_IMAGE_PER_PROMPT="1"
  DEFAULT_TEXT_MAX_LENGTH="226"
  DEFAULT_GUIDANCE_SCALE="6.0"
else
  DEFAULT_MODEL_PATH=""
  DEFAULT_TRAIN_CONFIG="config/world_r1.py:world_r1_large"
  DEFAULT_TRAIN_BATCH_SIZE="1"
  DEFAULT_TRAIN_NUM_IMAGE_PER_PROMPT="2"
  DEFAULT_TEXT_MAX_LENGTH="512"
  DEFAULT_GUIDANCE_SCALE="5.0"
fi

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/logs/world_r1}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RUN_NAME="${RUN_NAME:-}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${DEFAULT_TRAIN_CONFIG}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
TRAIN_NUM_IMAGE_PER_PROMPT="${TRAIN_NUM_IMAGE_PER_PROMPT:-${DEFAULT_TRAIN_NUM_IMAGE_PER_PROMPT}}"
TRAIN_TEST_BATCH_SIZE="${TRAIN_TEST_BATCH_SIZE:-}"
TRAIN_NUM_BATCHES_PER_EPOCH="${TRAIN_NUM_BATCHES_PER_EPOCH:-}"
TRAIN_SAMPLE_TIME_PER_PROMPT="${TRAIN_SAMPLE_TIME_PER_PROMPT:-}"
TRAIN_GRAD_ACCUM_STEPS="${TRAIN_GRAD_ACCUM_STEPS:-}"
TRAIN_HEIGHT="${TRAIN_HEIGHT:-}"
TRAIN_WIDTH="${TRAIN_WIDTH:-}"
TRAIN_FRAMES="${TRAIN_FRAMES:-}"
TRAIN_NUM_STEPS="${TRAIN_NUM_STEPS:-}"
TRAIN_EVAL_NUM_STEPS="${TRAIN_EVAL_NUM_STEPS:-}"
TRAIN_NOISE_LEVEL="${TRAIN_NOISE_LEVEL:-}"
TRAIN_WRAP_STRENGTH="${TRAIN_WRAP_STRENGTH:-}"
TRAIN_TEXT_MAX_LENGTH="${TRAIN_TEXT_MAX_LENGTH:-${DEFAULT_TEXT_MAX_LENGTH}}"
TRAIN_GUIDANCE_SCALE="${TRAIN_GUIDANCE_SCALE:-${DEFAULT_GUIDANCE_SCALE}}"
TRAIN_LORA_PATH="${TRAIN_LORA_PATH:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES:-2,3,4,5,6,7}"
SERVER_VISIBLE_DEVICES="${SERVER_VISIBLE_DEVICES:-0,1}"
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29511}"
SERVER_PORT="${SERVER_PORT:-18089}"
GENERAL_REWARD_PORT="${GENERAL_REWARD_PORT:-18090}"
SERVER_START_WAIT_SECONDS="${SERVER_START_WAIT_SECONDS:-70}"
GENERAL_REWARD_START_WAIT_SECONDS="${GENERAL_REWARD_START_WAIT_SECONDS:-20}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required for MODEL_FAMILY=${MODEL_FAMILY}. Set it to a local path or a Hugging Face repo id." >&2
  exit 1
fi

resolve_training_output_identity

SERVER_LOG="${SERVER_LOG:-${OUTPUT_DIR}/reward_3d_server.log}"
GENERAL_REWARD_LOG="${GENERAL_REWARD_LOG:-${OUTPUT_DIR}/general_reward_server.log}"

if [[ -z "${REWARD_3D_SERVER_URL:-}" ]]; then
  export REWARD_3D_SERVER_URL="http://127.0.0.1:${SERVER_PORT}"
fi
if [[ -z "${GENERAL_REWARD_SERVER_URL:-}" ]]; then
  export GENERAL_REWARD_SERVER_URL="http://127.0.0.1:${GENERAL_REWARD_PORT}"
fi

if [[ -z "${NUM_PROCESSES:-}" ]]; then
  NUM_PROCESSES=$("${WAN_PYTHON}" - <<PY
devices = "${TRAIN_VISIBLE_DEVICES}".split(",")
devices = [d for d in devices if d.strip()]
print(len(devices))
PY
  )
fi

echo "REPO_DIR=${REPO_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "REWARD_3D_SERVER_URL=${REWARD_3D_SERVER_URL}"
echo "GENERAL_REWARD_SERVER_URL=${GENERAL_REWARD_SERVER_URL}"
echo "SERVER_VISIBLE_DEVICES=${SERVER_VISIBLE_DEVICES}"
echo "TRAIN_VISIBLE_DEVICES=${TRAIN_VISIBLE_DEVICES}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "TRAIN_MASTER_PORT=${TRAIN_MASTER_PORT}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS}"

CONFIG_OVERRIDES=(
  "--config.model_family=${MODEL_FAMILY}"
  "--config.text_max_length=${TRAIN_TEXT_MAX_LENGTH}"
  "--config.sample.guidance_scale=${TRAIN_GUIDANCE_SCALE}"
  "--config.sample.train_batch_size=${TRAIN_BATCH_SIZE}"
  "--config.train.batch_size=${TRAIN_BATCH_SIZE}"
  "--config.sample.num_image_per_prompt=${TRAIN_NUM_IMAGE_PER_PROMPT}"
)
if [[ -n "${TRAIN_TEST_BATCH_SIZE}" ]]; then CONFIG_OVERRIDES+=("--config.sample.test_batch_size=${TRAIN_TEST_BATCH_SIZE}"); fi
if [[ -n "${TRAIN_NUM_BATCHES_PER_EPOCH}" ]]; then CONFIG_OVERRIDES+=("--config.sample.num_batches_per_epoch=${TRAIN_NUM_BATCHES_PER_EPOCH}"); fi
if [[ -n "${TRAIN_SAMPLE_TIME_PER_PROMPT}" ]]; then CONFIG_OVERRIDES+=("--config.sample.sample_time_per_prompt=${TRAIN_SAMPLE_TIME_PER_PROMPT}"); fi
if [[ -n "${TRAIN_GRAD_ACCUM_STEPS}" ]]; then CONFIG_OVERRIDES+=("--config.train.gradient_accumulation_steps=${TRAIN_GRAD_ACCUM_STEPS}"); fi
if [[ -n "${TRAIN_HEIGHT}" ]]; then CONFIG_OVERRIDES+=("--config.height=${TRAIN_HEIGHT}"); fi
if [[ -n "${TRAIN_WIDTH}" ]]; then CONFIG_OVERRIDES+=("--config.width=${TRAIN_WIDTH}"); fi
if [[ -n "${TRAIN_FRAMES}" ]]; then CONFIG_OVERRIDES+=("--config.frames=${TRAIN_FRAMES}"); fi
if [[ -n "${TRAIN_NUM_STEPS}" ]]; then CONFIG_OVERRIDES+=("--config.sample.num_steps=${TRAIN_NUM_STEPS}"); fi
if [[ -n "${TRAIN_EVAL_NUM_STEPS}" ]]; then CONFIG_OVERRIDES+=("--config.sample.eval_num_steps=${TRAIN_EVAL_NUM_STEPS}"); fi
if [[ -n "${TRAIN_NOISE_LEVEL}" ]]; then CONFIG_OVERRIDES+=("--config.sample.noise_level=${TRAIN_NOISE_LEVEL}"); fi
if [[ -n "${TRAIN_WRAP_STRENGTH}" ]]; then CONFIG_OVERRIDES+=("--config.sample.wrap_strength=${TRAIN_WRAP_STRENGTH}"); fi
if [[ -n "${TRAIN_LORA_PATH}" ]]; then CONFIG_OVERRIDES+=("--config.train.lora_path=${TRAIN_LORA_PATH}"); fi

mkdir -p "$(dirname "${SERVER_LOG}")"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"
cd "${REPO_DIR}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${GENERAL_REWARD_PID:-}" ]]; then
    kill "${GENERAL_REWARD_PID}" >/dev/null 2>&1 || true
    wait "${GENERAL_REWARD_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${SERVER_VISIBLE_DEVICES}" \
  "${WAN_PYTHON}" -u scripts/serve_reward_3d.py --port "${SERVER_PORT}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "SERVER_PID=${SERVER_PID}"
sleep "${SERVER_START_WAIT_SECONDS}"
tail -n 30 "${SERVER_LOG}" || true

CUDA_VISIBLE_DEVICES="${SERVER_VISIBLE_DEVICES}" \
  "${WAN_PYTHON}" -u scripts/serve_general_reward.py --port "${GENERAL_REWARD_PORT}" > "${GENERAL_REWARD_LOG}" 2>&1 &
GENERAL_REWARD_PID=$!

echo "GENERAL_REWARD_PID=${GENERAL_REWARD_PID}"
sleep "${GENERAL_REWARD_START_WAIT_SECONDS}"
tail -n 30 "${GENERAL_REWARD_LOG}" || true

CUDA_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES}" \
"${WAN_TORCHRUN}" \
  --standalone \
  --nproc_per_node "${NUM_PROCESSES}" \
  --master_port "${TRAIN_MASTER_PORT}" \
  scripts/train_world_r1.py \
  --config "${TRAIN_CONFIG}" \
  --config.pretrained.model="${MODEL_PATH}" \
  --config.run_name="${RUN_NAME}" \
  --config.save_dir="${OUTPUT_DIR}" \
  "${CONFIG_OVERRIDES[@]}" \
  ${EXTRA_TRAIN_ARGS}
