#!/bin/bash

# ==========================================
# 设置: 遇到错误立即停止，使用未定义变量报错
# ==========================================
set -e
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/scripts/resolve_python.sh"
cd "${SCRIPT_DIR}"
#
# bash run.sh pick_place  demo_clean1  1  test_bash  pi05_ur5_robotwin_lora 0,1,2,3,4,5,6,7
# 定义颜色，方便看日志
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==========================================
# 0. 参数检查与打印
# ==========================================
if [ "$#" -ne 6 ]; then
    echo -e "${RED}❌ 错误: 参数数量不对！需要 6 个参数，你只提供了 $# 个。${NC}"
    echo "用法: bash run.sh <task_name> <task_config> <expert_data_num> <log_name> <train_config_name> <gpu_use>"
    exit 1
fi

task_name=${1}
task_config=${2} #对应数据路径下面
expert_data_num=${3}
log_name=${4}
train_config_name=${5}
gpu_use=${6}

# 组合变量
model_name="${task_name}-${task_config}-${log_name}"
repo_id="${task_name}_${log_name}_repo"
source_data_dir="${SCRIPT_DIR}/processed_data/${task_name}-${task_config}-${expert_data_num}"
target_data_dir="${ROBOTWIN_ROOT}/data/training_data/${model_name}"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}🚀 Pipeline 启动: ${task_name} (Config: ${train_config_name})${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "📄 任务信息:"
echo -e "   - Task: ${task_name}"
echo -e "   - Expert Num: ${expert_data_num}"
echo -e "   - Model Name: ${model_name}"
echo -e "   - Target Repo ID: ${repo_id}"
echo -e "   - GPU: ${gpu_use}"
echo -e "${BLUE}------------------------------------------------------${NC}"

# ==========================================
# 1. Process Data (RoboTwin -> HDF5)
# ==========================================
echo -e "${YELLOW}[1/6] 正在转换 RoboTwin 数据 -> HDF5...${NC}"
# 检查脚本是否存在
if [ ! -f "process_data_pi0.sh" ]; then
    echo -e "${RED}❌ 找不到 process_data_pi0.sh 脚本！${NC}"
    exit 1
fi
#note: task_config需要对其数据路径
bash process_data_pi0.sh "${task_name}" "${task_config}" "${expert_data_num}"
echo -e "${GREEN}✅ HDF5 数据转换完成。${NC}"

# ==========================================
# 2. Move Data (整理文件夹)
# ==========================================
echo -e "${YELLOW}[2/6] 正在移动数据...${NC}"

# 检查源数据是否生成成功
if [ ! -d "${source_data_dir}" ]; then
    echo -e "${RED}❌ 错误: 源数据目录不存在: ${source_data_dir}${NC}"
    echo -e "${RED}   请检查上一步 process_data_pi0.sh 是否真的成功生成了文件夹。${NC}"
    exit 1
fi

# 确保父目录存在
mkdir -p ./training_data

# 安全移动逻辑：如果目标已存在，先提示或删除，防止嵌套
if [ -d "${target_data_dir}" ]; then
    echo -e "${YELLOW}⚠️  警告: 目标目录已存在 (${target_data_dir})，正在覆盖...${NC}"
    rm -rf "${target_data_dir}"
fi

# mv "${source_data_dir}" "${target_data_dir}"
rsync -a --no-perms --no-times "${source_data_dir}/" "${target_data_dir}/"
rm -rf "${source_data_dir}"

echo -e "${GREEN}✅ 数据已移动至: ${target_data_dir}${NC}"

# ==========================================
# 3. Setup Cache
# ==========================================
echo -e "${YELLOW}[3/6] 设置 HuggingFace 缓存...${NC}"
export XDG_CACHE_HOME="${ROBOTWIN_ROOT}/data/lerobot_data"
echo -e "   -> Cache set to: $XDG_CACHE_HOME"

# ==========================================
# 4. Convert to LeRobot Format
# ==========================================
echo -e "${YELLOW}[4/6] 转换为 LeRobot 格式 (RepoID: ${repo_id})...${NC}"
if [ ! -f "generate.sh" ]; then
    echo -e "${RED}❌ 找不到 generate.sh 脚本！${NC}"
    exit 1
fi

bash generate.sh "${target_data_dir}" "${repo_id}"
echo -e "${GREEN}✅ LeRobot 格式转换完成。${NC}"

# ==========================================
# 5. Export Config
# ==========================================
echo -e "${YELLOW}[5/6] 注入环境变量 REPO_ID...${NC}"
export REPO_ID="${repo_id}"
echo -e "   -> REPO_ID = ${REPO_ID}"

# ==========================================
# 6. Finetune & Stats
# ==========================================
echo -e "${YELLOW}[6/6] 开始计算归一化统计量 & 训练...${NC}"

# 计算 Norm Stats
echo -e "   >>> Running compute_norm_stats.py..."
"${OPENPI_PYTHON}" "${SCRIPT_DIR}/scripts/compute_norm_stats.py" --config-name "${train_config_name}"

# Finetune
echo -e "   >>> Starting Finetune..."
if [ ! -f "finetune.sh" ]; then
    echo -e "${RED}❌ 找不到 finetune.sh 脚本！${NC}"
    exit 1
fi

# 这一步时间最长，通常会一直打印训练日志
bash finetune.sh "${train_config_name}" "${model_name}" "${gpu_use}"

# ==========================================
# 结束
# ==========================================
echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}🎉🎉🎉 流程结束 (训练脚本已启动或完成)${NC}"
echo -e "数据repo_id: ${repo_id}"
echo -e "${BLUE}======================================================${NC}"
