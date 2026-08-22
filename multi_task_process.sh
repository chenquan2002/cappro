#!/bin/bash
set -e
set -u

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

trap 'echo -e "${RED}❌ 脚本在第 $LINENO 行出错，退出码: $?${NC}"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -ne 2 ]; then
    echo -e "${RED}❌ 错误: 参数数量不对！${NC}"
    echo "用法: bash run_process_multi.sh <task_config> <expert_data_num>"
    echo "示例: bash run_process_multi.sh demo_clean 422"
    exit 1
fi

task_config=${1}
expert_data_num=${2}

TASKS=(
    click_alarmclock
    press_stapler
    open_laptop
    rotate_qrcode
    handover_mic
    pick_dual_bottles
    grab_roller
    move_stapler_pad
    place_empty_cup
    place_bread_basket
    place_a2b_left
    stack_blocks_two
    place_fan
)

if [ ! -f "${SCRIPT_DIR}/process_data_pi0.sh" ]; then
    echo -e "${RED}❌ 错误: 未找到 process_data_pi0.sh${NC}"
    exit 1
fi

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}🚀 开始批量处理 RoboTwin 数据 -> HDF5${NC}"
echo -e "${BLUE}task_config: ${task_config}${NC}"
echo -e "${BLUE}expert_data_num: ${expert_data_num}${NC}"
echo -e "${BLUE}任务数量: ${#TASKS[@]}${NC}"
echo -e "${BLUE}======================================================${NC}"

for task_name in "${TASKS[@]}"; do
    output_dir="${SCRIPT_DIR}/processed_data/${task_name}-${task_config}-${expert_data_num}"

    echo -e "${YELLOW}------------------------------------------------------${NC}"
    echo -e "${YELLOW}正在处理任务: ${task_name}${NC}"
    echo -e "${YELLOW}输出目录: ${output_dir}${NC}"

    if [ -d "${output_dir}" ]; then
        echo -e "${BLUE}⚠️ 检测到输出目录已存在，跳过: ${output_dir}${NC}"
        continue
    fi

    bash "${SCRIPT_DIR}/process_data_pi0.sh" \
        "${task_name}" \
        "${task_config}" \
        "${expert_data_num}"

    echo -e "${GREEN}✅ 完成任务: ${task_name}${NC}"
done

echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}✅ 所有任务的数据处理完成！${NC}"
echo -e "${BLUE}======================================================${NC}"