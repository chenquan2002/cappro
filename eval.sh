#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ "$#" -lt 6 || "$#" -gt 7 ]]; then
    echo "Usage: ./eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id> [checkpoint_id]"
    exit 1
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"

# The training venv intentionally contains only openpi dependencies. RoboTwin
# simulation evaluation dependencies should be installed in CapPro's own venv.
EVAL_PYTHON="${EVAL_PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"
if [[ ! -x "${EVAL_PYTHON}" ]]; then
    echo "Evaluation Python not found or not executable: ${EVAL_PYTHON}" >&2
    exit 1
fi

policy_name="pi05_test"
task_name="$1"
task_config="$2"
train_config_name="$3"
model_name="$4"
seed="$5"
gpu_id="$6"
checkpoint_id="${7:-}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd "${ROBOTWIN_ROOT}"
eval_args=(
    script/eval_policy.py
    --config "${SCRIPT_DIR}/deploy_policy.yml" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --train_config_name "${train_config_name}" \
    --model_name "${model_name}" \
    --ckpt_setting "${model_name}" \
    --seed "${seed}" \
    --policy_name "${policy_name}"
)
if [[ -n "${checkpoint_id}" ]]; then
    eval_args+=(--checkpoint_id "${checkpoint_id}")
fi

PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
PYTHONWARNINGS=ignore::UserWarning \
    "${EVAL_PYTHON}" "${eval_args[@]}"
