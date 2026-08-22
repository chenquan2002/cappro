#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/scripts/resolve_python.sh"

CONFIG_NAME="pi05_aloha_robotwin_cappro_lora"
GPU_IDS="${1:-0,1}"
EXP_NAME="${2:-cappro_place_fan}"
REPO_ID="${REPO_ID:-source_data_hovapi_repo}"

export XDG_CACHE_HOME="${ROBOTWIN_ROOT}/data/lerobot_data"
export REPO_ID
export TRAIN_TASK_NAMES="${TRAIN_TASK_NAMES:-place_fan}"
export SUPPORT_MANIFEST_PATH="${SUPPORT_MANIFEST_PATH:-${ROBOTWIN_ROOT}/data/support_data/manifests/cappro_place_fan_phase_manifest.jsonl}"
export SUPPORT_ROUNDS_PER_CYCLE="${SUPPORT_ROUNDS_PER_CYCLE:-20}"
export SUPPORT_CHUNK_SIZE="${SUPPORT_CHUNK_SIZE:-1}"
export SUPPORT_VIEW_OVERRIDE="${SUPPORT_VIEW_OVERRIDE:-none}"
export CAPTION_LOSS_WEIGHT="${CAPTION_LOSS_WEIGHT:-0.1}"
export NUM_CAPTION_QUERIES="${NUM_CAPTION_QUERIES:-4}"
export CAPTION_ACTION_GATE_INIT="${CAPTION_ACTION_GATE_INIT:-0.1}"
export CAPTION_ROBOT_TOKENS_PER_IMAGE="${CAPTION_ROBOT_TOKENS_PER_IMAGE:-64}"
DEFAULT_SUPPORT_ASSETS_DIR="${SCRIPT_DIR}/assets/pi05_aloha_robotwin_icl_random_lora"
if [[ ! -d "${DEFAULT_SUPPORT_ASSETS_DIR}" ]]; then
    DEFAULT_SUPPORT_ASSETS_DIR="${ROBOTWIN_ROOT}/policy/hovapi/assets/pi05_aloha_robotwin_icl_random_lora"
fi
export SUPPORT_ASSETS_DIR="${SUPPORT_ASSETS_DIR:-${DEFAULT_SUPPORT_ASSETS_DIR}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

test -d "${XDG_CACHE_HOME}/huggingface/lerobot/${REPO_ID}"
test -f "${XDG_CACHE_HOME}/huggingface/lerobot/${REPO_ID}/meta/episode_origin.jsonl"
test -f "${SUPPORT_MANIFEST_PATH}"
test -f "${SUPPORT_ASSETS_DIR}/${REPO_ID}/norm_stats.json"
case "${SUPPORT_VIEW_OVERRIDE}" in
    none|ego|front|left|right) ;;
    *)
        echo "SUPPORT_VIEW_OVERRIDE must be one of none/ego/front/left/right, got: ${SUPPORT_VIEW_OVERRIDE}" >&2
        exit 1
        ;;
esac

train_args=(
    "${CONFIG_NAME}"
    "--exp-name=${EXP_NAME}"
)
if [[ -n "${NUM_TRAIN_STEPS:-}" ]]; then
    train_args+=("--num-train-steps=${NUM_TRAIN_STEPS}")
fi
if [[ "${OVERWRITE:-false}" == "true" ]]; then
    train_args+=("--overwrite")
fi

cd "${SCRIPT_DIR}"
if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q %q scripts/train.py' "${CUDA_VISIBLE_DEVICES}" "${OPENPI_PYTHON}"
    printf ' %q' "${train_args[@]}"
    printf '\n'
    exit 0
fi
exec "${OPENPI_PYTHON}" scripts/train.py "${train_args[@]}"
