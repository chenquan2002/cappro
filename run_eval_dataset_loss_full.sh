#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
    echo "Usage: bash run_eval_dataset_loss_full.sh <train_config_name> <exp_name> <ckpt_step> <gpu_id> [eval options...]"
    exit 1
fi

TRAIN_CONFIG_NAME=$1
EXP_NAME=$2
CKPT_STEP=$3
GPU_ID=$4
shift 4
if [[ "$#" -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    MAX_BATCHES=$1
    shift
    set -- --max-batches "${MAX_BATCHES}" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/scripts/resolve_python.sh"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROBOTWIN_ROOT}/data/lerobot_data}"
export REPO_ID="${REPO_ID:-source_data_hovapi_repo}"
export TRAIN_DATA_SCOPE="source"
export CAPTION_MAX_LEN="${CAPTION_MAX_LEN:-96}"
export CAPTION_LOSS_WEIGHT="${CAPTION_LOSS_WEIGHT:-0.1}"

RAW_SUPPORT_MANIFEST="${SUPPORT_MANIFEST_PATH:-${ROBOTWIN_ROOT}/data/support_data/manifests/hovapi_manifest.jsonl}"
EPISODE_ORIGIN="${XDG_CACHE_HOME}/huggingface/lerobot/${REPO_ID}/meta/episode_origin.jsonl"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${LOSS_OUTPUT_DIR:-${SCRIPT_DIR}/loss_output/caption_${CKPT_STEP}_full_${TIMESTAMP}}"

mkdir -p "${OUT_DIR}"
test -f "${RAW_SUPPORT_MANIFEST}"
test -f "${EPISODE_ORIGIN}"

cd "${SCRIPT_DIR}"
exec "${OPENPI_PYTHON}" scripts/eval_dataset_loss_full.py \
    "${TRAIN_CONFIG_NAME}" \
    --exp-name "${EXP_NAME}" \
    --step "${CKPT_STEP}" \
    --episode-origin "${EPISODE_ORIGIN}" \
    --raw-support-manifest "${RAW_SUPPORT_MANIFEST}" \
    --output-dir "${OUT_DIR}" \
    --repo-id "${REPO_ID}" \
    --batch-size 8 \
    --data-scope source \
    --support-view ego \
    "$@"
