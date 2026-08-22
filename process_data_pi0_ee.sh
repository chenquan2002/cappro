#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/resolve_python.sh"

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 <task_name> <setting> <expert_data_num>" >&2
    exit 2
fi

task_name=${1}
setting=${2}
expert_data_num=${3}

"${OPENPI_PYTHON}" "${SCRIPT_DIR}/scripts/process_data_ee.py" "${task_name}" "${setting}" "${expert_data_num}"
