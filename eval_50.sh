#!/usr/bin/env bash
set -u
set -o pipefail

# ============================================================
# RoboTwin 50 tasks eval 自动调度脚本
#
# 正确单任务命令格式：
# bash eval.sh click_alarmclock demo_clean pi05_aloha_robotwin_cappro_lora cappro_source_v1 0 0 30000
#
# 参数含义：
# $1 task_name
# $2 task_config
# $3 train_config_name
# $4 model_name
# $5 seed
# $6 gpu_id
# $7 checkpoint_id
# ============================================================

# =========================
# 基本配置
# =========================

TASK_FILE=${1:-tasks_13_source.txt}

TASK_CONFIG=${TASK_CONFIG:-demo_clean}
TRAIN_CONFIG=${TRAIN_CONFIG:-pi05_aloha_robotwin_cappro_lora}
MODEL_NAME=${MODEL_NAME:-cappro_source_v1}
CHECKPOINT_ID=${CHECKPOINT_ID:-30000}
SEED=${SEED:-0}
TEST_ID=${TEST_ID:-test3W-pi05test-caption-targetb-0719}

# 8 张 4090D
GPUS=(0 1 2 3 4 5 6 7)

# 4090D 24GB，单个 eval 大概不到 20GB
# 默认要求至少空闲 18000MB 再启动任务
MIN_FREE_MEM_MB=${MIN_FREE_MEM_MB:-18000}

# 检查间隔，单位是秒，不是分钟
CHECK_INTERVAL=${CHECK_INTERVAL:-15}

# Each evaluated task must have a video-only support bank entry.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SUPPORT_BANK_ROOT=${SUPPORT_BANK_ROOT:-${ROBOTWIN_ROOT}/data/support_data/support_bank_full}
SKIP_UNSUPPORTED_SUPPORT=${SKIP_UNSUPPORTED_SUPPORT:-false}
cd "${SCRIPT_DIR}"

# 日志目录
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_DIR=${LOG_DIR:-eval_logs/${TASK_CONFIG}/${TEST_ID}_${TIME_TAG}}
mkdir -p "${LOG_DIR}"

STATUS_FILE="${LOG_DIR}/status.tsv"
echo -e "time\ttask\tgpu\tpid\tstatus\tlog" > "${STATUS_FILE}"

# =========================
# 检查任务文件
# =========================

if [ ! -f "${TASK_FILE}" ]; then
    echo "❌ 找不到任务列表文件: ${TASK_FILE}"
    exit 1
fi

mapfile -t TASKS < <(grep -vE '^\s*(#|$)' "${TASK_FILE}")

if [ ! -d "${SUPPORT_BANK_ROOT}/human" ]; then
    echo "❌ 找不到 support video 根目录: ${SUPPORT_BANK_ROOT}/human"
    exit 1
fi

SUPPORTED_TASKS=()
UNSUPPORTED_TASKS=()
for task in "${TASKS[@]}"; do
    if compgen -G "${SUPPORT_BANK_ROOT}/human/${task}/${TASK_CONFIG}/*/*/frames.npy" > /dev/null; then
        SUPPORTED_TASKS+=("${task}")
    else
        UNSUPPORTED_TASKS+=("${task}")
    fi
done

if [ "${#UNSUPPORTED_TASKS[@]}" -gt 0 ]; then
    echo "⚠️ 以下任务没有 ${TASK_CONFIG} support video:"
    printf '  %s\n' "${UNSUPPORTED_TASKS[@]}"
    if [ "${SKIP_UNSUPPORTED_SUPPORT}" = "true" ]; then
        echo "⚠️ SKIP_UNSUPPORTED_SUPPORT=true，跳过上述任务。"
        TASKS=("${SUPPORTED_TASKS[@]}")
    else
        echo "❌ 不启动不完整的评测。设置 SKIP_UNSUPPORTED_SUPPORT=true 才会跳过这些任务。"
        exit 1
    fi
fi

TOTAL_TASKS=${#TASKS[@]}

if [ "${TOTAL_TASKS}" -eq 0 ]; then
    echo "❌ 任务列表为空: ${TASK_FILE}"
    exit 1
fi

echo "============================================================"
echo "RoboTwin eval 自动调度启动"
echo "任务列表文件: ${TASK_FILE}"
echo "任务总数: ${TOTAL_TASKS}"
echo "TEST_ID: ${TEST_ID}"
echo "TASK_CONFIG: ${TASK_CONFIG}"
echo "TRAIN_CONFIG: ${TRAIN_CONFIG}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "CHECKPOINT_ID: ${CHECKPOINT_ID}"
echo "SEED: ${SEED}"
echo "GPUS: ${GPUS[*]}"
echo "MIN_FREE_MEM_MB: ${MIN_FREE_MEM_MB}"
echo "CHECK_INTERVAL: ${CHECK_INTERVAL}s"
echo "SUPPORT_BANK_ROOT: ${SUPPORT_BANK_ROOT}"
echo "SKIP_UNSUPPORTED_SUPPORT: ${SKIP_UNSUPPORTED_SUPPORT}"
echo "LOG_DIR: ${LOG_DIR}"
echo "============================================================"

# =========================
# 运行状态
# =========================

declare -A GPU_PID
declare -A GPU_TASK
declare -A GPU_LOG

task_idx=0
finished_count=0
failed_count=0

# =========================
# 判断 GPU 是否有足够空闲显存
# =========================

gpu_has_enough_memory() {
    local gpu=$1
    local free_mem

    free_mem=$(nvidia-smi \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits \
        -i "${gpu}" 2>/dev/null | tr -d ' ')

    if [ -z "${free_mem}" ]; then
        return 1
    fi

    if [ "${free_mem}" -ge "${MIN_FREE_MEM_MB}" ]; then
        return 0
    else
        return 1
    fi
}

# =========================
# 刷新已经结束的任务
# =========================

refresh_jobs() {
    local gpu pid task log rc now

    for gpu in "${GPUS[@]}"; do
        pid=${GPU_PID[$gpu]:-}

        if [ -n "${pid}" ]; then
            if kill -0 "${pid}" 2>/dev/null; then
                continue
            else
                task=${GPU_TASK[$gpu]}
                log=${GPU_LOG[$gpu]}

                wait "${pid}"
                rc=$?

                now=$(date +"%Y-%m-%d %H:%M:%S")

                if [ "${rc}" -eq 0 ]; then
                    echo "✅ [GPU ${gpu}] 完成任务: ${task}"
                    echo -e "${now}\t${task}\t${gpu}\t${pid}\tSUCCESS\t${log}" >> "${STATUS_FILE}"
                else
                    echo "❌ [GPU ${gpu}] 任务失败: ${task}, exit code=${rc}"
                    echo -e "${now}\t${task}\t${gpu}\t${pid}\tFAILED_${rc}\t${log}" >> "${STATUS_FILE}"
                    failed_count=$((failed_count + 1))
                fi

                finished_count=$((finished_count + 1))

                unset GPU_PID[$gpu]
                unset GPU_TASK[$gpu]
                unset GPU_LOG[$gpu]
            fi
        fi
    done
}

# =========================
# 找一个当前可用的 GPU
# =========================

find_available_gpu() {
    local gpu

    for gpu in "${GPUS[@]}"; do
        # 如果这个 GPU 已经被本脚本分配了任务，就跳过
        if [ -n "${GPU_PID[$gpu]:-}" ]; then
            continue
        fi

        # 如果这个 GPU 显存足够，就使用它
        if gpu_has_enough_memory "${gpu}"; then
            echo "${gpu}"
            return 0
        fi
    done

    return 1
}

# =========================
# 启动一个任务
# =========================

launch_task() {
    local task=$1
    local gpu=$2
    local log_file pid now

    log_file="${LOG_DIR}/${task}_gpu${gpu}.log"

    echo "🚀 [GPU ${gpu}] 启动任务: ${task}"
    echo "    命令: bash eval.sh ${task} ${TASK_CONFIG} ${TRAIN_CONFIG} ${MODEL_NAME} ${SEED} ${gpu} ${CHECKPOINT_ID}"
    echo "    日志: ${log_file}"

    bash eval.sh "${task}" "${TASK_CONFIG}" "${TRAIN_CONFIG}" "${MODEL_NAME}" "${SEED}" "${gpu}" "${CHECKPOINT_ID}" \
        > "${log_file}" 2>&1 &

    pid=$!

    GPU_PID[$gpu]=${pid}
    GPU_TASK[$gpu]=${task}
    GPU_LOG[$gpu]=${log_file}

    now=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "${now}\t${task}\t${gpu}\t${pid}\tRUNNING\t${log_file}" >> "${STATUS_FILE}"
}

# =========================
# Ctrl+C 中断时，停止本脚本启动的任务
# =========================

cleanup() {
    echo
    echo "⚠️ 收到中断信号，准备停止本脚本启动的 eval 任务..."

    local gpu pid
    for gpu in "${GPUS[@]}"; do
        pid=${GPU_PID[$gpu]:-}
        if [ -n "${pid}" ]; then
            echo "停止 GPU ${gpu} 上的任务: PID=${pid}, task=${GPU_TASK[$gpu]}"
            kill "${pid}" 2>/dev/null || true
        fi
    done

    exit 130
}

trap cleanup INT TERM

# =========================
# 主调度循环
# =========================

while [ "${finished_count}" -lt "${TOTAL_TASKS}" ]; do
    refresh_jobs

    # 尽可能把还没启动的任务分配到空闲 GPU 上
    while [ "${task_idx}" -lt "${TOTAL_TASKS}" ]; do
        available_gpu=$(find_available_gpu || true)

        if [ -z "${available_gpu:-}" ]; then
            break
        fi

        task=${TASKS[$task_idx]}
        launch_task "${task}" "${available_gpu}"

        task_idx=$((task_idx + 1))

        echo "进度: 已启动 ${task_idx}/${TOTAL_TASKS}, 已完成 ${finished_count}/${TOTAL_TASKS}, 失败 ${failed_count}"
    done

    if [ "${finished_count}" -lt "${TOTAL_TASKS}" ]; then
        sleep "${CHECK_INTERVAL}"
    fi
done

echo "============================================================"
echo "全部 eval 任务结束"
echo "总任务数: ${TOTAL_TASKS}"
echo "失败任务数: ${failed_count}"
echo "日志目录: ${LOG_DIR}"
echo "状态文件: ${STATUS_FILE}"
echo "============================================================"
