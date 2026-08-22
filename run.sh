#!/bin/bash
set -e
set -u

trap 'echo -e "${RED}❌ 脚本在第 $LINENO 行出错，退出码: $?${NC}"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/scripts/resolve_python.sh"
cd "${SCRIPT_DIR}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ "$#" -ne 6 ]; then
    echo -e "${RED}❌ 错误: 参数数量不对！${NC}"
    echo "用法: bash run.sh <task_name> <task_config> <expert_data_num> <log_name> <train_config_name> <gpu_use>"
    exit 1
fi

task_name=${1}
task_config=${2}
expert_data_num=${3}
log_name=${4}
train_config_name=${5}
gpu_use=${6}

model_name="${task_name}-${task_config}-${log_name}"
repo_id="${task_name}_${log_name}_repo"
source_data_dir="${SCRIPT_DIR}/processed_data/${task_name}-${task_config}-${expert_data_num}"
target_data_dir="${ROBOTWIN_ROOT}/data/training_data/${model_name}"

STATUS_DIR="${SCRIPT_DIR}/.pipeline_status"
mkdir -p "$STATUS_DIR"
STAGE_FILE="${STATUS_DIR}/.status_${model_name}"

CURRENT_STAGE=0
if [ -f "$STAGE_FILE" ]; then
    CURRENT_STAGE=$(cat "$STAGE_FILE")
    echo -e "${YELLOW}检测到之前的运行记录，准备从第 $((CURRENT_STAGE + 1)) 步继续...${NC}"
fi

update_stage() {
    echo "$1" > "$STAGE_FILE"
}

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}🚀 Pipeline 启动: ${task_name} (Stage: $((CURRENT_STAGE + 1))/7)${NC}"
echo -e "${BLUE}======================================================${NC}"

if [ "$CURRENT_STAGE" -lt 1 ]; then
    echo -e "${YELLOW}[1/7] 正在转换 RoboTwin 数据 -> HDF5...${NC}"
    if [ ! -f "${SCRIPT_DIR}/process_data_pi0.sh" ]; then
        echo -e "${RED}❌ 错误: 未找到 process_data_pi0.sh${NC}"
        exit 1
    fi
    bash "${SCRIPT_DIR}/process_data_pi0.sh" "${task_name}" "${task_config}" "${expert_data_num}"
    update_stage 1
    echo -e "${GREEN}✅ 步骤 1 完成。${NC}"
else
    echo -e "${BLUE}⏭️ 跳过步骤 1 (已完成)${NC}"
fi

if [ "$CURRENT_STAGE" -lt 2 ]; then
    echo -e "${YELLOW}[2/7] 正在移动数据到 NAS...${NC}"

    if [ ! -d "${source_data_dir}" ]; then
        echo -e "${RED}❌ 错误: 源数据目录不存在: ${source_data_dir}${NC}"
        exit 1
    fi

    mkdir -p "${target_data_dir}"

    rsync -rhv --progress \
        --inplace \
        --no-perms --no-owner --no-group \
        "${source_data_dir}/" "${target_data_dir}/"

    update_stage 2
    echo -e "${GREEN}✅ 步骤 2 完成。${NC}"
else
    echo -e "${BLUE}⏭️ 跳过步骤 2 (已完成)${NC}"
fi

echo -e "${YELLOW}[3/7] 设置 HuggingFace 缓存...${NC}"
export XDG_CACHE_HOME="${ROBOTWIN_ROOT}/data/lerobot_data"
mkdir -p "${XDG_CACHE_HOME}"
update_stage 3

if [ "$CURRENT_STAGE" -lt 4 ]; then
    echo -e "${YELLOW}[4/7] 转换为 LeRobot 格式...${NC}"
    if [ ! -f "${SCRIPT_DIR}/generate.sh" ]; then
        echo -e "${RED}❌ 错误: 未找到 generate.sh${NC}"
        exit 1
    fi
    bash "${SCRIPT_DIR}/generate.sh" "${target_data_dir}" "${repo_id}"
    update_stage 4
    echo -e "${GREEN}✅ 步骤 4 完成。${NC}"
else
    echo -e "${BLUE}⏭️ 跳过步骤 4 (已完成)${NC}"
fi

export REPO_ID="${repo_id}"
update_stage 5

if [ "$CURRENT_STAGE" -lt 6 ]; then
    echo -e "${YELLOW}[6/7] 开始计算统计量...${NC}"
    "${OPENPI_PYTHON}" scripts/compute_norm_stats.py --config-name "${train_config_name}"
    update_stage 6
    echo -e "${GREEN}✅ 步骤 6 完成。${NC}"
else
    echo -e "${BLUE}⏭️ 跳过步骤 6 (统计量已计算)${NC}"
fi

if [ "$CURRENT_STAGE" -lt 7 ]; then
    echo -e "${YELLOW}[7/7] 开始训练 (Finetune)...${NC}"
    if [ ! -f "${SCRIPT_DIR}/finetune.sh" ]; then
        echo -e "${RED}❌ 错误: 未找到 finetune.sh${NC}"
        exit 1
    fi

    bash "${SCRIPT_DIR}/finetune.sh" "${train_config_name}" "${model_name}" "${gpu_use}"

    rm -rf "${source_data_dir}"
    rm -f "$STAGE_FILE"
    echo -e "${GREEN}✅ 所有流程圆满完成！${NC}"
else
    echo -e "${BLUE}⏭️ 跳过步骤 7 (已完成)${NC}"
fi

echo -e "${BLUE}🚀 测试使用 repo_id: export REPO_ID=${repo_id}${NC}"
echo -e "${BLUE}======================================================${NC}"
