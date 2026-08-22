#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
    echo "Usage: bash run_temporal_action_curves.sh <train_config_name> <exp_name> <ckpt_step> <gpu_id> [eval options...]"
    echo "Example: bash run_temporal_action_curves.sh pi05_aloha_robotwin_cappro_lora cappro_source_v1 70000 0 --support-mode enabled"
    exit 1
fi

TRAIN_CONFIG_NAME=$1
EXP_NAME=$2
CKPT_STEP=$3
GPU_ID=$4
shift 4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/scripts/resolve_python.sh"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROBOTWIN_ROOT}/data/lerobot_data}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-caption}}"
export REPO_ID="${REPO_ID:-source_data_hovapi_repo}"
export TRAIN_DATA_SCOPE="${TRAIN_DATA_SCOPE:-source}"
export CAPTION_MAX_LEN="${CAPTION_MAX_LEN:-96}"
export CAPTION_LOSS_WEIGHT="${CAPTION_LOSS_WEIGHT:-0.1}"

NORM_ASSET_ID="${NORM_ASSET_ID:-source_data_iclpi_repo}"
RAW_SUPPORT_MANIFEST="${SUPPORT_MANIFEST_PATH:-${ROBOTWIN_ROOT}/data/support_data/manifests/hovapi_manifest.jsonl}"
EPISODE_ORIGIN="${XDG_CACHE_HOME}/huggingface/lerobot/${REPO_ID}/meta/episode_origin.jsonl"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${SCRIPT_DIR}/checkpoints}"
SUPPORT_MODE="${SUPPORT_MODE:-enabled}"
SUPPORT_SELECTION="${SUPPORT_SELECTION:-random}"
SUPPORT_SEED="${SUPPORT_SEED:-0}"
SUPPORT_VIEW="${SUPPORT_VIEW:-ego}"
CHUNK_PROGRESS_MODE="${CHUNK_PROGRESS_MODE:-eval_step_limit}"
TASKS="${TEMPORAL_TASKS:-place_fan,rotate_qrcode,move_stapler_pad}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${LOSS_OUTPUT_DIR:-${SCRIPT_DIR}/loss_output/temporal_action_curves_${CKPT_STEP}_${SUPPORT_MODE}_${SUPPORT_SELECTION}_view-${SUPPORT_VIEW}_progress-${CHUNK_PROGRESS_MODE}_${TIMESTAMP}}"

test -f "${RAW_SUPPORT_MANIFEST}"
test -f "${EPISODE_ORIGIN}"
test -d "${CHECKPOINT_BASE_DIR}/${TRAIN_CONFIG_NAME}/${EXP_NAME}/${CKPT_STEP}/params"
test -f "${CHECKPOINT_BASE_DIR}/${TRAIN_CONFIG_NAME}/${EXP_NAME}/${CKPT_STEP}/assets/${NORM_ASSET_ID}/norm_stats.json"
mkdir -p "${OUT_DIR}"

cd "${SCRIPT_DIR}"
exec "${OPENPI_PYTHON}" scripts/eval_temporal_action_curves.py \
    "${TRAIN_CONFIG_NAME}" \
    --exp-name "${EXP_NAME}" \
    --step "${CKPT_STEP}" \
    --model-kind support_caption \
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
    --norm-asset-id "${NORM_ASSET_ID}" \
    --episode-origin "${EPISODE_ORIGIN}" \
    --raw-support-manifest "${RAW_SUPPORT_MANIFEST}" \
    --output-dir "${OUT_DIR}" \
    --repo-id "${REPO_ID}" \
    --batch-size "${TEMPORAL_BATCH_SIZE:-8}" \
    --task-name "${TASKS}" \
    --task-config demo_clean \
    --data-scope source \
    --max-episodes-per-task "${MAX_EPISODES_PER_TASK:-50}" \
    --support-mode "${SUPPORT_MODE}" \
    --support-selection "${SUPPORT_SELECTION}" \
    --support-seed "${SUPPORT_SEED}" \
    --support-view "${SUPPORT_VIEW}" \
    --chunk-progress-mode "${CHUNK_PROGRESS_MODE}" \
    --sample-action-num-steps "${SAMPLE_ACTION_NUM_STEPS:-10}" \
    --rng-steps "${RNG_STEPS:-0}" \
    --loss-seed "${LOSS_SEED:-12345}" \
    --preprocess-mode eval \
    "$@"
