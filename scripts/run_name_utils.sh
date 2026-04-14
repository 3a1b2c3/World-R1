#!/usr/bin/env bash

slugify_component() {
  local raw="$1"
  raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  raw="$(printf '%s' "$raw" | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//; s/_+/_/g')"
  if [[ -z "$raw" ]]; then
    raw="na"
  fi
  printf '%s' "$raw"
}

number_component() {
  local raw="$1"
  raw="$(printf '%s' "$raw" | sed -E 's/\.0+$//; s/(\.[0-9]*[1-9])0+$/\1/')"
  raw="${raw//./p}"
  slugify_component "$raw"
}

first_nonempty() {
  local item
  for item in "$@"; do
    if [[ -n "$item" ]]; then
      printf '%s' "$item"
      return 0
    fi
  done
  return 1
}

config_name_from_ref() {
  local ref="$1"
  local config_path="${ref%%:*}"
  local config_entry=""
  local base_name=""

  if [[ "$ref" == *:* ]]; then
    config_entry="${ref#*:}"
  fi
  if [[ -n "$config_entry" && "$config_entry" != "$ref" ]]; then
    printf '%s' "$config_entry"
    return 0
  fi

  base_name="$(basename "$config_path")"
  printf '%s' "${base_name%.*}"
}

load_config_defaults() {
  local repo_dir="$1"
  local python_bin="$2"
  local config_ref="$3"
  local python_output=""

  if ! python_output="$(
    cd "$repo_dir" && "$python_bin" - "$repo_dir" "$config_ref" <<'PY'
import importlib.util
import os
import shlex
import sys

repo_dir, config_ref = sys.argv[1], sys.argv[2]
config_path, has_entry, config_entry = config_ref.partition(":")
if not config_path:
    raise SystemExit("TRAIN_CONFIG must not be empty.")

module_path = config_path if os.path.isabs(config_path) else os.path.join(repo_dir, config_path)
if not os.path.exists(module_path):
    raise SystemExit(f"Config file not found: {module_path}")

spec = importlib.util.spec_from_file_location("world_r1_launcher_config", module_path)
module = importlib.util.module_from_spec(spec)
if spec.loader is None:
    raise SystemExit(f"Failed to load config module: {module_path}")
spec.loader.exec_module(module)

config = None
if has_entry and hasattr(module, config_entry):
    entry = getattr(module, config_entry)
    config = entry() if callable(entry) else entry
elif hasattr(module, "get_config"):
    config = module.get_config(config_entry) if has_entry else module.get_config()
else:
    raise SystemExit(f"Unable to resolve config entry from {config_ref}")

def lookup(path: str) -> str:
    current = config
    for part in path.split("."):
        if not hasattr(current, part):
            return ""
        current = getattr(current, part)
    if current is None:
        return ""
    if isinstance(current, bool):
        return "1" if current else "0"
    return str(current)

fields = {
    "CFG_MODEL_FAMILY": lookup("model_family"),
    "CFG_TEXT_MAX_LENGTH": lookup("text_max_length"),
    "CFG_HEIGHT": lookup("height"),
    "CFG_WIDTH": lookup("width"),
    "CFG_FRAMES": lookup("frames"),
    "CFG_NUM_STEPS": lookup("sample.num_steps"),
    "CFG_EVAL_NUM_STEPS": lookup("sample.eval_num_steps"),
    "CFG_GUIDANCE_SCALE": lookup("sample.guidance_scale"),
    "CFG_TRAIN_BATCH_SIZE": lookup("sample.train_batch_size"),
    "CFG_NUM_IMAGE_PER_PROMPT": lookup("sample.num_image_per_prompt"),
    "CFG_NOISE_LEVEL": lookup("sample.noise_level"),
    "CFG_WRAP_STRENGTH": lookup("sample.wrap_strength"),
}

for key, value in fields.items():
    print(f"{key}={shlex.quote(value)}")
PY
  )"; then
    echo "Failed to load training config defaults from ${config_ref}" >&2
    exit 1
  fi

  eval "$python_output"
}

build_auto_run_stem() {
  local family=""
  local config_name=""
  local height=""
  local width=""
  local frames=""
  local steps=""
  local guidance=""
  local batch_size=""
  local num_image_per_prompt=""
  local text_max_length=""
  local wrap_strength=""
  local noise_level=""
  local parts=()
  local stem=""
  local i=0

  family="$(first_nonempty "${MODEL_FAMILY:-}" "${CFG_MODEL_FAMILY:-}" "wan")"
  config_name="$(config_name_from_ref "${TRAIN_CONFIG:-}")"
  height="$(first_nonempty "${TRAIN_HEIGHT:-}" "${CFG_HEIGHT:-}")"
  width="$(first_nonempty "${TRAIN_WIDTH:-}" "${CFG_WIDTH:-}")"
  frames="$(first_nonempty "${TRAIN_FRAMES:-}" "${CFG_FRAMES:-}")"
  steps="$(first_nonempty "${TRAIN_NUM_STEPS:-}" "${CFG_NUM_STEPS:-}")"
  guidance="$(first_nonempty "${TRAIN_GUIDANCE_SCALE:-}" "${CFG_GUIDANCE_SCALE:-}")"
  batch_size="$(first_nonempty "${TRAIN_BATCH_SIZE:-}" "${CFG_TRAIN_BATCH_SIZE:-}")"
  num_image_per_prompt="$(first_nonempty "${TRAIN_NUM_IMAGE_PER_PROMPT:-}" "${CFG_NUM_IMAGE_PER_PROMPT:-}")"
  text_max_length="$(first_nonempty "${TRAIN_TEXT_MAX_LENGTH:-}" "${CFG_TEXT_MAX_LENGTH:-}")"
  wrap_strength="$(first_nonempty "${TRAIN_WRAP_STRENGTH:-}" "${CFG_WRAP_STRENGTH:-}")"
  noise_level="$(first_nonempty "${TRAIN_NOISE_LEVEL:-}" "${CFG_NOISE_LEVEL:-}")"

  parts+=("$(slugify_component "$family")")
  parts+=("$(slugify_component "$config_name")")
  if [[ -n "$height" && -n "$width" && -n "$frames" ]]; then
    parts+=("$(slugify_component "${height}x${width}x${frames}")")
  fi
  if [[ -n "$steps" ]]; then
    parts+=("s$(number_component "$steps")")
  fi
  if [[ -n "$guidance" ]]; then
    parts+=("g$(number_component "$guidance")")
  fi
  if [[ -n "$batch_size" ]]; then
    parts+=("b$(number_component "$batch_size")")
  fi
  if [[ -n "$num_image_per_prompt" ]]; then
    parts+=("k$(number_component "$num_image_per_prompt")")
  fi
  if [[ -n "$text_max_length" ]]; then
    parts+=("t$(number_component "$text_max_length")")
  fi
  if [[ -n "$wrap_strength" ]]; then
    parts+=("w$(number_component "$wrap_strength")")
  fi
  if [[ "$family" == "cogvideox" && -n "$noise_level" ]]; then
    parts+=("n$(number_component "$noise_level")")
  fi

  stem="${parts[0]}"
  for ((i = 1; i < ${#parts[@]}; i++)); do
    stem+="_${parts[$i]}"
  done
  printf '%s_run' "$stem"
}

reserve_unique_output_dir() {
  local output_root="$1"
  local stem="$2"
  local candidate_name=""
  local candidate_dir=""
  local index=1

  mkdir -p "$output_root"
  while true; do
    candidate_name="${stem}${index}"
    candidate_dir="${output_root}/${candidate_name}"
    if mkdir "$candidate_dir" 2>/dev/null; then
      RUN_NAME="$candidate_name"
      OUTPUT_DIR="$candidate_dir"
      export RUN_NAME OUTPUT_DIR
      return 0
    fi
    index=$((index + 1))
  done
}

reuse_reserved_output_dir() {
  local output_root="$1"
  local marker_path="$2"
  local reserved_name=""

  if [[ ! -f "$marker_path" ]]; then
    return 1
  fi

  reserved_name="$(head -n 1 "$marker_path" | tr -d '\r\n')"
  if [[ -z "$reserved_name" ]]; then
    return 1
  fi

  RUN_NAME="$reserved_name"
  OUTPUT_DIR="${output_root}/${RUN_NAME}"
  mkdir -p "$OUTPUT_DIR"
  export RUN_NAME OUTPUT_DIR
  return 0
}

reserve_shared_output_dir() {
  local output_root="$1"
  local stem="$2"
  local reservation_key="$3"
  local reservation_root="${output_root}/.run_name_reservations"
  local reservation_slug=""
  local marker_path=""
  local lock_dir=""
  local wait_seconds="${AUTO_RUN_WAIT_SECONDS:-60}"
  local waited=0

  reservation_slug="$(slugify_component "$reservation_key")"
  marker_path="${reservation_root}/${reservation_slug}.txt"
  lock_dir="${reservation_root}/${reservation_slug}.lock"

  mkdir -p "$reservation_root"
  if reuse_reserved_output_dir "$output_root" "$marker_path"; then
    return 0
  fi

  if mkdir "$lock_dir" 2>/dev/null; then
    reserve_unique_output_dir "$output_root" "$stem"
    printf '%s\n' "$RUN_NAME" > "$marker_path"
    AUTO_RUN_MARKER_PATH="$marker_path"
    AUTO_RUN_LOCK_OWNER="1"
    export AUTO_RUN_MARKER_PATH AUTO_RUN_LOCK_OWNER
    rmdir "$lock_dir" >/dev/null 2>&1 || true
    return 0
  fi

  while [[ "$waited" -lt "$wait_seconds" ]]; do
    if reuse_reserved_output_dir "$output_root" "$marker_path"; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done

  echo "Timed out waiting for shared run-name reservation: ${reservation_key}" >&2
  exit 1
}

cleanup_auto_run_reservation() {
  if [[ "${AUTO_RUN_LOCK_OWNER:-}" == "1" && -n "${AUTO_RUN_MARKER_PATH:-}" ]]; then
    rm -f "${AUTO_RUN_MARKER_PATH}"
  fi
}

resolve_training_output_identity() {
  local output_root="${OUTPUT_ROOT:-${REPO_DIR}/logs/world_r1}"
  local stem=""

  export OUTPUT_ROOT="$output_root"

  if [[ -n "${OUTPUT_DIR:-}" ]]; then
    mkdir -p "$OUTPUT_DIR"
    if [[ -z "${RUN_NAME:-}" ]]; then
      RUN_NAME="$(basename "$OUTPUT_DIR")"
      export RUN_NAME
    fi
    return 0
  fi

  if [[ -n "${RUN_NAME:-}" ]]; then
    OUTPUT_DIR="${output_root}/${RUN_NAME}"
    mkdir -p "$OUTPUT_DIR"
    export OUTPUT_DIR RUN_NAME
    return 0
  fi

  load_config_defaults "$REPO_DIR" "$WAN_PYTHON" "$TRAIN_CONFIG"
  stem="$(build_auto_run_stem)"
  if [[ -n "${AUTO_RUN_KEY:-}" ]]; then
    reserve_shared_output_dir "$output_root" "$stem" "$AUTO_RUN_KEY"
  else
    reserve_unique_output_dir "$output_root" "$stem"
  fi
}
