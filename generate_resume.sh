#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/scripts/resolve_python.sh"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROBOTWIN_ROOT}/data/lerobot_data}"

if [[ "$#" -lt 2 ]]; then
    echo "Usage: $0 <raw_data_dir> <repo_id> [converter options...]" >&2
    exit 2
fi

data_dir=${1}
repo_id=${2}
shift 2

"${OPENPI_PYTHON}" "${SCRIPT_DIR}/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin_resume.py" \
  --raw_dir "$data_dir" \
  --repo_id "$repo_id" \
  "$@"
