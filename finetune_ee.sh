#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/resolve_python.sh"

train_config_name=$1
model_name=$2
gpu_use=$3

export CUDA_VISIBLE_DEVICES=$gpu_use
echo $CUDA_VISIBLE_DEVICES
cd "${SCRIPT_DIR}"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 "${OPENPI_PYTHON}" scripts/train.py "${train_config_name}" "--exp-name=${model_name}" --overwrite
