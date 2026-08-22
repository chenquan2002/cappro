#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
    echo "Usage: bash run_compare_source_losses.sh <gpu_id> [uniform|full|trainlike ...]"
    exit 1
fi

GPU_ID=$1
shift
MODES=("$@")
if [[ "${#MODES[@]}" -eq 0 ]]; then
    MODES=(uniform full trainlike)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/scripts/resolve_python.sh"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROBOTWIN_ROOT}/data/lerobot_data}"
export REPO_ID="${REPO_ID:-source_data_hovapi_repo}"
export TRAIN_DATA_SCOPE=source
export CAPTION_MAX_LEN="${CAPTION_MAX_LEN:-96}"
export CAPTION_LOSS_WEIGHT="${CAPTION_LOSS_WEIGHT:-0.1}"

SUPPORT_CONFIG="${SUPPORT_CONFIG:-pi05_aloha_robotwin_cappro_lora}"
SUPPORT_EXP="${SUPPORT_EXP:-cappro_source_v1}"
SUPPORT_STEP="${SUPPORT_STEP:-30000}"
SUPPORT_CHECKPOINT_BASE="${SUPPORT_CHECKPOINT_BASE:-${SCRIPT_DIR}/checkpoints}"

PI05_CONFIG="${PI05_CONFIG:-pi05_aloha_robotwin_lora}"
PI05_EXP="${PI05_EXP:-source_data_20W_h100}"
PI05_STEP="${PI05_STEP:-30000}"
PI05_CHECKPOINT_BASE="${PI05_CHECKPOINT_BASE:-${ROBOTWIN_ROOT}/policy/pi05/checkpoints}"

NORM_ASSET_ID="${NORM_ASSET_ID:-source_data_iclpi_repo}"
RAW_SUPPORT_MANIFEST="${SUPPORT_MANIFEST_PATH:-${ROBOTWIN_ROOT}/data/support_data/manifests/hovapi_manifest.jsonl}"
EPISODE_ORIGIN="${XDG_CACHE_HOME}/huggingface/lerobot/${REPO_ID}/meta/episode_origin.jsonl"
COMPARE_ROOT="${LOSS_COMPARE_ROOT:-${SCRIPT_DIR}/loss_output/source_3w_compare}"
BATCH_SIZE="${LOSS_COMPARE_BATCH_SIZE:-8}"
MAX_BATCHES="${RUN_COMPARE_MAX_BATCHES:-}"
FULL_EPISODE_CHUNK_SIZE="${FULL_EPISODE_CHUNK_SIZE:-25}"
SAMPLE_ACTION_MSE="${SAMPLE_ACTION_MSE:-0}"
SAMPLE_ACTION_MSE_ONLY="${SAMPLE_ACTION_MSE_ONLY:-0}"
SAMPLE_ACTION_NUM_STEPS="${SAMPLE_ACTION_NUM_STEPS:-10}"
SOURCE_TASKS=(
    click_alarmclock
    grab_roller
    handover_mic
    move_stapler_pad
    open_laptop
    pick_dual_bottles
    place_a2b_left
    place_bread_basket
    place_empty_cup
    place_fan
    press_stapler
    rotate_qrcode
    stack_blocks_two
)

test -f "${RAW_SUPPORT_MANIFEST}"
test -f "${EPISODE_ORIGIN}"
test -d "${SUPPORT_CHECKPOINT_BASE}/${SUPPORT_CONFIG}/${SUPPORT_EXP}/${SUPPORT_STEP}/params"
test -f "${SUPPORT_CHECKPOINT_BASE}/${SUPPORT_CONFIG}/${SUPPORT_EXP}/${SUPPORT_STEP}/assets/${NORM_ASSET_ID}/norm_stats.json"
test -d "${PI05_CHECKPOINT_BASE}/${PI05_CONFIG}/${PI05_EXP}/${PI05_STEP}/params"
test -f "${PI05_CHECKPOINT_BASE}/${PI05_CONFIG}/${PI05_EXP}/${PI05_STEP}/assets/${NORM_ASSET_ID}/norm_stats.json"

mkdir -p "${COMPARE_ROOT}/_shared/selections" "${COMPARE_ROOT}/_shared/replay" "${COMPARE_ROOT}/_logs"

count_task_episodes() {
    "${OPENPI_PYTHON}" -c '
import json
import sys

path, task_name = sys.argv[1:3]
count = 0
with open(path, encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue
        if json.loads(line)["task_name"] == task_name:
            count += 1
print(count)
' "${EPISODE_ORIGIN}" "$1"
}

mode_script() {
    case "$1" in
        uniform) echo "scripts/eval_dataset_loss.py" ;;
        full) echo "scripts/eval_dataset_loss_full.py" ;;
        trainlike) echo "scripts/eval_dataset_loss_trainlike.py" ;;
        *) echo "Unknown mode: $1" >&2; return 1 ;;
    esac
}

run_model() {
    local mode=$1
    local model_label=$2
    local model_kind=$3
    local config_name=$4
    local exp_name=$5
    local step=$6
    local checkpoint_base=$7
    local selection_policy=$8
    local task_name=${9:-all}
    local output_name=${10:-${mode}}
    local episode_chunk_size=${11:-}
    local episode_chunk_index=${12:-}
    local script
    script="$(mode_script "${mode}")"
    local output_dir="${COMPARE_ROOT}/_runs/${model_label}/${output_name}"
    local selection_file="${COMPARE_ROOT}/_shared/selections/source_${output_name//\//_}.npz"
    local log_file="${COMPARE_ROOT}/_logs/${model_label}_${output_name//\//_}.log"

    if [[ -f "${output_dir}/summary.json" && "${FORCE_LOSS_COMPARE:-0}" != "1" ]]; then
        echo "[Skip] ${model_label}/${mode} already completed: ${output_dir}"
        return
    fi
    mkdir -p "${output_dir}"

    local args=(
        "${OPENPI_PYTHON}" "${script}"
        "${config_name}"
        --exp-name "${exp_name}"
        --step "${step}"
        --model-kind "${model_kind}"
        --checkpoint-base-dir "${checkpoint_base}"
        --norm-asset-id "${NORM_ASSET_ID}"
        --episode-origin "${EPISODE_ORIGIN}"
        --raw-support-manifest "${RAW_SUPPORT_MANIFEST}"
        --output-dir "${output_dir}"
        --selection-file "${selection_file}"
        --repo-id "${REPO_ID}"
        --batch-size "${BATCH_SIZE}"
        --rng-steps 0
        --loss-seed 12345
        --preprocess-mode eval
        --task-name "${task_name}"
        --task-config all
        --data-scope source
    )
    if [[ "${SAMPLE_ACTION_MSE_ONLY}" == "1" ]]; then
        args+=(--sample-action-mse-only --sample-action-num-steps "${SAMPLE_ACTION_NUM_STEPS}")
    elif [[ "${SAMPLE_ACTION_MSE}" == "1" ]]; then
        args+=(--sample-action-mse --sample-action-num-steps "${SAMPLE_ACTION_NUM_STEPS}")
    fi
    if [[ "${selection_policy}" == "require" ]]; then
        args+=(--require-existing-selection)
    fi
    if [[ -n "${MAX_BATCHES}" ]]; then
        args+=(--max-batches "${MAX_BATCHES}")
    fi
    if [[ -n "${episode_chunk_size}" ]]; then
        args+=(--episode-chunk-size "${episode_chunk_size}")
    fi
    if [[ -n "${episode_chunk_index}" ]]; then
        args+=(--episode-chunk-index "${episode_chunk_index}")
    fi
    case "${mode}" in
        uniform)
            args+=(--samples-per-episode 5 --support-view ego)
            ;;
        full)
            args+=(--support-view ego)
            ;;
        trainlike)
            args+=(
                --support-view none
                --replay-file "${COMPARE_ROOT}/_shared/replay/source_trainlike.npz"
                --train-steps 30000
                --train-batch-size 96
                --support-rounds-per-cycle 20
                --eval-samples 35750
                --replay-sample-method linspace
                --sample-seed 0
            )
            ;;
    esac

    echo "[Run] ${model_label}/${mode} -> ${output_dir}"
    "${args[@]}" 2>&1 | tee "${log_file}"
}

cd "${SCRIPT_DIR}"
for mode in "${MODES[@]}"; do
    if [[ "${mode}" == "full" ]]; then
        for task_name in "${SOURCE_TASKS[@]}"; do
            task_episode_count="$(count_task_episodes "${task_name}")"
            task_chunk_count=$(((task_episode_count + FULL_EPISODE_CHUNK_SIZE - 1) / FULL_EPISODE_CHUNK_SIZE))
            for chunk_index in $(seq 0 $((task_chunk_count - 1))); do
                chunk_name="$(printf 'chunk_%03d' "${chunk_index}")"
                run_model \
                    "${mode}" support_caption support_caption \
                    "${SUPPORT_CONFIG}" "${SUPPORT_EXP}" "${SUPPORT_STEP}" "${SUPPORT_CHECKPOINT_BASE}" create \
                    "${task_name}" "full/${task_name}/${chunk_name}" \
                    "${FULL_EPISODE_CHUNK_SIZE}" "${chunk_index}"
                run_model \
                    "${mode}" pi05_3w pi05 \
                    "${PI05_CONFIG}" "${PI05_EXP}" "${PI05_STEP}" "${PI05_CHECKPOINT_BASE}" require \
                    "${task_name}" "full/${task_name}/${chunk_name}" \
                    "${FULL_EPISODE_CHUNK_SIZE}" "${chunk_index}"
                "${OPENPI_PYTHON}" scripts/organize_source_loss_comparison.py "${COMPARE_ROOT}"
            done
        done
    else
        run_model \
            "${mode}" support_caption support_caption \
            "${SUPPORT_CONFIG}" "${SUPPORT_EXP}" "${SUPPORT_STEP}" "${SUPPORT_CHECKPOINT_BASE}" create
        run_model \
            "${mode}" pi05_3w pi05 \
            "${PI05_CONFIG}" "${PI05_EXP}" "${PI05_STEP}" "${PI05_CHECKPOINT_BASE}" require
    fi
    "${OPENPI_PYTHON}" scripts/organize_source_loss_comparison.py "${COMPARE_ROOT}"
done

echo "[Done] ${COMPARE_ROOT}"
