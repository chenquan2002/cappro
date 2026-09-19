#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NVIDIA_COMPAT_DIR="${NVIDIA_COMPAT_DIR:-${HOME}/.local/nvidia-550-compat}"
if [[ -d "${NVIDIA_COMPAT_DIR}" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ "$#" -eq 0 ]]; then
    task_name="place_fan"
    task_config="demo_clean"
    train_config_name="pi05_aloha_robotwin_cappro_place_fan_clean_lora"
    model_name="cappro_place_fan_clean_phase_v1"
    seed="0"
    gpu_id="0"
    checkpoint_id="7500"
elif [[ "$#" -ge 6 && "$#" -le 7 ]]; then
    task_name="$1"
    task_config="$2"
    train_config_name="$3"
    model_name="$4"
    seed="$5"
    gpu_id="$6"
    checkpoint_id="${7:-7500}"
else
    echo "Usage: ./eval.sh [<task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id> [checkpoint_id]]"
    exit 1
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"

EVAL_PYTHON="${EVAL_PYTHON:-${OPENPI_PYTHON:-}}"
if [[ -z "${EVAL_PYTHON}" ]]; then
    for candidate in \
        "${SCRIPT_DIR}/.venv/bin/python" \
        "${ROBOTWIN_ROOT}/policy/iclpi/.venv/bin/python" \
        "${ROBOTWIN_ROOT}/policy/hovapi/.venv/bin/python" \
        "${ROBOTWIN_ROOT}/policy/pi05/.venv/bin/python" \
        "${ROBOTWIN_ROOT}/policy/pro/.venv/bin/python"; do
        if [[ -x "${candidate}" ]]; then
            EVAL_PYTHON="${candidate}"
            break
        fi
    done
fi
if [[ -z "${EVAL_PYTHON:-}" || ! -x "${EVAL_PYTHON}" ]]; then
    echo "No evaluation Python with RoboTwin/OpenPI dependencies was found. Set EVAL_PYTHON explicitly." >&2
    exit 1
fi

policy_name="cappro"
support_bank_root="${SUPPORT_BANK_ROOT:-${ROBOTWIN_ROOT}/data/support_data/support_bank}"
support_view="${SUPPORT_VIEW:-ego}"
# Fixed seed evaluation is opt-in. When this is empty, eval_policy.py uses
# its normal progressing-seed mode and keeps trying seeds until test_num
# successful episodes have been collected.
seed_list="${EVAL_SEED_LIST:-}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cappro-matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo "evaluation python: ${EVAL_PYTHON}"

eval_pythonpath="${SCRIPT_DIR}/src:${ROBOTWIN_ROOT}/policy:${ROBOTWIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHONPATH="${eval_pythonpath}" "${EVAL_PYTHON}" "${SCRIPT_DIR}/scripts/check_eval_setup.py" \
    --robotwin-root "${ROBOTWIN_ROOT}" \
    --policy-name "${policy_name}" \
    --task-name "${task_name}" \
    --task-config "${task_config}" \
    --train-config-name "${train_config_name}" \
    --model-name "${model_name}" \
    --checkpoint-id "${checkpoint_id}" \
    --support-bank-root "${support_bank_root}" \
    --support-view "${support_view}"

if [[ "${EVAL_PREFLIGHT_ONLY:-false}" == "true" ]]; then
    exit 0
fi

cd "${ROBOTWIN_ROOT}"
eval_args=(
    script/eval_policy.py
    --config "${SCRIPT_DIR}/deploy_policy.yml" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --train_config_name "${train_config_name}" \
    --model_name "${model_name}" \
    --ckpt_setting "${model_name}_${checkpoint_id}" \
    --checkpoint_root "${SCRIPT_DIR}/checkpoints" \
    --support_bank_root "${support_bank_root}" \
    --support_view "${support_view}" \
    --random_support "${RANDOM_SUPPORT:-true}" \
    --mask_support_video "${MASK_SUPPORT_VIDEO:-false}" \
    --action_chunk_size "${ACTION_CHUNK_SIZE:-20}" \
    --seed "${seed}" \
    --policy_name "${policy_name}"
)
eval_args+=(--checkpoint_id "${checkpoint_id}")
if [[ -n "${seed_list}" ]]; then
    eval_args+=(--seed_list "${seed_list}")
fi
if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf 'PYTHONPATH=%q CUDA_VISIBLE_DEVICES=%q %q' "${eval_pythonpath}" "${CUDA_VISIBLE_DEVICES}" "${EVAL_PYTHON}"
    printf ' %q' "${eval_args[@]}"
    printf '\n'
    exit 0
fi

PYTHONPATH="${eval_pythonpath}" \
PYTHONWARNINGS=ignore::UserWarning \
    "${EVAL_PYTHON}" "${eval_args[@]}"
