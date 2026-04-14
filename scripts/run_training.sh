#!/usr/bin/env bash
set -euo pipefail

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

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export REWARD_3D_SERVER_URL="${REWARD_3D_SERVER_URL:-http://127.0.0.1:${REWARD_3D_PORT:-8089}}"
export GENERAL_REWARD_SERVER_URL="${GENERAL_REWARD_SERVER_URL:-http://127.0.0.1:${GENERAL_REWARD_PORT:-8090}}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export WANDB_PROJECT="${WANDB_PROJECT:-world-r1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CACHE_ROOT="${CACHE_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/world-r1}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

MODEL_FAMILY="${MODEL_FAMILY:-wan}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29503}"
if [[ "${MODEL_FAMILY}" == "cogvideox" ]]; then
  DEFAULT_TRAIN_CONFIG="config/world_r1.py:world_r1_cogvideox_5b"
  DEFAULT_MODEL_PATH="THUDM/CogVideoX1.5-5B"
  DEFAULT_TEXT_MAX_LENGTH="226"
  DEFAULT_GUIDANCE_SCALE="6.0"
else
  DEFAULT_TRAIN_CONFIG="config/world_r1.py:world_r1_large"
  DEFAULT_MODEL_PATH=""
  DEFAULT_TEXT_MAX_LENGTH="512"
  DEFAULT_GUIDANCE_SCALE="5.0"
fi

TRAIN_CONFIG="${TRAIN_CONFIG:-${DEFAULT_TRAIN_CONFIG}}"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/logs/world_r1}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RUN_NAME="${RUN_NAME:-}"
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

resolve_training_output_identity
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

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

echo "REPO_DIR=${REPO_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"

cd "${REPO_DIR}"
"${WAN_TORCHRUN}" \
  --standalone \
  --nproc_per_node="${NUM_PROCESSES}" \
  --master_port="${MAIN_PROCESS_PORT}" \
  scripts/train_world_r1.py \
  --config "${TRAIN_CONFIG}" \
  --config.pretrained.model="${MODEL_PATH}" \
  --config.run_name="${RUN_NAME}" \
  --config.save_dir="${OUTPUT_DIR}" \
  "${CONFIG_OVERRIDES[@]}" \
  ${EXTRA_TRAIN_ARGS} \
  "$@"
