#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

trap 'echo -e "${RED}脚本在第 ${LINENO} 行出错, 退出码: $?${NC}"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/scripts/resolve_python.sh"

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    echo "用法: bash run_icl.sh <train_config_name> <gpu_use> [all|source]"
    echo "place_fan clean 示例: bash run_icl.sh pi05_aloha_robotwin_cappro_place_fan_clean_lora 0,1 source"
    exit 1
fi

train_config_name="$1"
gpu_use="$2"
data_scope="${3:-${TRAIN_DATA_SCOPE:-source}}"

case "${data_scope}" in
    all|source) ;;
    *)
        echo -e "${RED}数据范围只能是 all 或 source，当前值: ${data_scope}${NC}"
        exit 1
        ;;
esac

default_model_name="cappro_place_fan_clean_phase_v1"
if [[ "${train_config_name}" == "pi05_aloha_robotwin_cappro_place_fan_clean_grids_lora" ]]; then
    default_model_name="cappro_place_fan_clean_grids_phase_v1"
fi
model_name="${MODEL_NAME:-${default_model_name}}"
repo_id="${REPO_ID:-source_data_hovapi_repo}"

export XDG_CACHE_HOME="${ROBOTWIN_ROOT}/data/lerobot_data"
export REPO_ID="${repo_id}"
export TRAIN_DATA_SCOPE="${data_scope}"
export TRAIN_TASK_NAMES="${TRAIN_TASK_NAMES:-place_fan}"
export TRAIN_TASK_CONFIGS="${TRAIN_TASK_CONFIGS:-demo_clean}"
export SUPPORT_MANIFEST_PATH="${SUPPORT_MANIFEST_PATH:-${ROBOTWIN_ROOT}/data/support_data/manifests/cappro_place_fan_phase_manifest.jsonl}"
export SUPPORT_ROUNDS_PER_CYCLE="${SUPPORT_ROUNDS_PER_CYCLE:-20}"
export SUPPORT_CHUNK_SIZE="${SUPPORT_CHUNK_SIZE:-1}"
export SUPPORT_VIEW_OVERRIDE="ego"
export USE_SUPPORT_TOKEN_COMPRESSION="${USE_SUPPORT_TOKEN_COMPRESSION:-true}"
export CAPTION_LOSS_WEIGHT="${CAPTION_LOSS_WEIGHT:-0.1}"
export NUM_CAPTION_QUERIES="${NUM_CAPTION_QUERIES:-4}"
export CAPTION_ACTION_GATE_INIT="${CAPTION_ACTION_GATE_INIT:-0.1}"
export CAPTION_ROBOT_TOKENS_PER_IMAGE="${CAPTION_ROBOT_TOKENS_PER_IMAGE:-64}"
default_caption_max_len=160
if [[ "${data_scope}" == "source" ]]; then
    default_caption_max_len=96
fi
export CAPTION_MAX_LEN="${CAPTION_MAX_LEN:-${default_caption_max_len}}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-7500}"
IFS=',' read -r -a visible_gpu_ids <<< "${gpu_use}"
visible_gpu_count="${#visible_gpu_ids[@]}"
default_support_grid_sampling=false
if [[ "${train_config_name}" == "pi05_aloha_robotwin_cappro_place_fan_clean_grids_lora" ]]; then
    default_support_grid_sampling=true
fi

per_gpu_batch_size="invalid"
if [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] && (( BATCH_SIZE % visible_gpu_count == 0 )); then
    per_gpu_batch_size="$((BATCH_SIZE / visible_gpu_count))"
fi
DEFAULT_SUPPORT_ASSETS_DIR="${SCRIPT_DIR}/assets/pi05_aloha_robotwin_icl_random_lora"
if [[ ! -d "${DEFAULT_SUPPORT_ASSETS_DIR}" ]]; then
    DEFAULT_SUPPORT_ASSETS_DIR="${ROBOTWIN_ROOT}/policy/hovapi/assets/pi05_aloha_robotwin_icl_random_lora"
fi
export SUPPORT_ASSETS_DIR="${SUPPORT_ASSETS_DIR:-${DEFAULT_SUPPORT_ASSETS_DIR}}"
norm_stats_dir="${SUPPORT_ASSETS_DIR}"
export USE_SUPPORT_GRID_SAMPLING="${USE_SUPPORT_GRID_SAMPLING:-${default_support_grid_sampling}}"
export SUPPORT_GRID_TOKENS_PER_FRAME="${SUPPORT_GRID_TOKENS_PER_FRAME:-32}"
if [[ "${train_config_name}" == "pi05_aloha_robotwin_cappro_place_fan_clean_grids_lora" ]]; then
    export USE_SUPPORT_TOKEN_COMPRESSION="false"
fi
norm_stats_asset_id="${repo_id}"
if [[ "${data_scope}" == "source" ]]; then
    export SOURCE_NORM_STATS_DIR="${SOURCE_NORM_STATS_DIR:-${SUPPORT_ASSETS_DIR}}"
    export SOURCE_NORM_STATS_ASSET_ID="${SOURCE_NORM_STATS_ASSET_ID:-source_data_iclpi_repo}"
    norm_stats_dir="${SOURCE_NORM_STATS_DIR}"
    norm_stats_asset_id="${SOURCE_NORM_STATS_ASSET_ID}"
fi
export CUDA_VISIBLE_DEVICES="${gpu_use}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}CapPro semantic-caption 训练${NC}"
echo "repo_id: ${repo_id}"
echo "model_name: ${model_name}"
echo "train_config_name: ${train_config_name}"
echo "gpu_use: ${gpu_use}"
echo "data_scope: ${TRAIN_DATA_SCOPE}"
echo "train_tasks: ${TRAIN_TASK_NAMES}"
echo "train_configs: ${TRAIN_TASK_CONFIGS}"
echo "global_batch_size: ${BATCH_SIZE}"
echo "per_gpu_batch_size: ${per_gpu_batch_size} (${visible_gpu_count} visible GPUs)"
echo "fsdp_devices: 1 (data parallel across visible GPUs, configured in config.py)"
echo "num_train_steps: ${NUM_TRAIN_STEPS}"
echo "save_interval: 5000 (configured in config.py)"
echo "keep_period: 5000 (configured in config.py)"
echo "caption_max_len: ${CAPTION_MAX_LEN}"
echo "caption_loss_weight: ${CAPTION_LOSS_WEIGHT}"
echo "num_caption_queries: ${NUM_CAPTION_QUERIES}"
echo "caption_action_gate_init: ${CAPTION_ACTION_GATE_INIT}"
echo "caption_robot_tokens_per_image: ${CAPTION_ROBOT_TOKENS_PER_IMAGE}"
echo "support_token_compression: ${USE_SUPPORT_TOKEN_COMPRESSION}"
echo "support_grid_sampling: ${USE_SUPPORT_GRID_SAMPLING}"
echo "support_grid_tokens_per_frame: ${SUPPORT_GRID_TOKENS_PER_FRAME}"
echo "support_rounds_per_cycle: ${SUPPORT_ROUNDS_PER_CYCLE}"
echo "support_chunk_size: ${SUPPORT_CHUNK_SIZE}"
echo "support_manifest: ${SUPPORT_MANIFEST_PATH}"
echo "support_view_override: ${SUPPORT_VIEW_OVERRIDE}"
echo "norm_stats: ${norm_stats_dir}/${norm_stats_asset_id}/norm_stats.json"
echo -e "${BLUE}======================================================${NC}"

echo -e "${YELLOW}[0/2] 检查数据、manifest 和 norm stats...${NC}"
test -d "${XDG_CACHE_HOME}/huggingface/lerobot/${repo_id}"
test -f "${XDG_CACHE_HOME}/huggingface/lerobot/${repo_id}/meta/episode_origin.jsonl"
test -f "${SUPPORT_MANIFEST_PATH}"
test -f "${norm_stats_dir}/${norm_stats_asset_id}/norm_stats.json"
case "${SUPPORT_VIEW_OVERRIDE}" in
    none|ego|front|left|right) ;;
    *)
        echo -e "${RED}SUPPORT_VIEW_OVERRIDE 只能是 none/ego/front/left/right，当前值: ${SUPPORT_VIEW_OVERRIDE}${NC}"
        exit 1
        ;;
esac
echo -e "${GREEN}路径检查通过${NC}"

train_args=(
    "${train_config_name}"
    "--exp-name=${model_name}"
    "--num-train-steps=${NUM_TRAIN_STEPS}"
)
if [[ -n "${BATCH_SIZE:-}" ]]; then
    if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
        echo -e "${RED}BATCH_SIZE 必须是正整数，当前值: ${BATCH_SIZE}${NC}"
        exit 1
    fi
    train_args+=("--batch-size=${BATCH_SIZE}")
fi
if [[ "${RESUME:-false}" == "true" ]]; then
    train_args+=("--resume")
elif [[ "${OVERWRITE:-false}" == "true" ]]; then
    train_args+=("--overwrite")
fi

echo -e "${YELLOW}[1/2] 开始训练...${NC}"
cd "${SCRIPT_DIR}"
if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf '%q scripts/train.py' "${OPENPI_PYTHON}"
    printf ' %q' "${train_args[@]}"
    printf '\n'
    exit 0
fi
exec "${OPENPI_PYTHON}" scripts/train.py "${train_args[@]}"
