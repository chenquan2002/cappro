#!/usr/bin/env bash
set -u
set -o pipefail

# ============================================================
# RoboTwin CapPro source13 eval 自动调度脚本
#
# 单任务命令格式：
# bash eval.sh click_alarmclock demo_clean \
#   pi05_aloha_robotwin_cappro_source13_full_lora \
#   cappro_source13_full_phase_ego_bs128_200k 0 0 25000
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TASK_FILE=${1:-${SCRIPT_DIR}/tasks_13_source.txt}

TASK_CONFIG=${TASK_CONFIG:-demo_clean}
TRAIN_CONFIG=${TRAIN_CONFIG:-pi05_aloha_robotwin_cappro_source13_full_lora}
MODEL_NAME=${MODEL_NAME:-cappro_source13_full_phase_ego_bs128_200k}
CHECKPOINT_ID=${CHECKPOINT_ID:-25000}
SEED=${SEED:-0}
TEST_ID=${TEST_ID:-cappro-source13-full-25000}

# 8 张 4090D
GPUS=(0 1 2 3 4 5 6 7)
MAX_JOBS_PER_GPU=${MAX_JOBS_PER_GPU:-2}
# One 0.40 XLA allocation on a 24GB 4090D is about 9.8GB.  Keep the
# per-launch threshold slightly above that amount.
MIN_FREE_MEM_MB=${MIN_FREE_MEM_MB:-10500}
CHECK_INTERVAL=${CHECK_INTERVAL:-15}

# CapPro inference settings.
NVIDIA_COMPAT_DIR="${NVIDIA_COMPAT_DIR:-${HOME}/.local/nvidia-550-compat}"
if [[ -d "${NVIDIA_COMPAT_DIR}" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export EVAL_PYTHON="${EVAL_PYTHON:-${ROBOTWIN_ROOT}/policy/iclpi/.venv/bin/python}"
export ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-50}"
export SUPPORT_BANK_ROOT="${SUPPORT_BANK_ROOT:-${ROBOTWIN_ROOT}/data/support_data/support_bank}"
export SUPPORT_VIEW="${SUPPORT_VIEW:-ego}"
export RANDOM_SUPPORT="${RANDOM_SUPPORT:-true}"
export MASK_SUPPORT_VIDEO="${MASK_SUPPORT_VIDEO:-false}"
export USE_SUPPORT_TOKEN_COMPRESSION="${USE_SUPPORT_TOKEN_COMPRESSION:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"
unset EVAL_SEED_LIST

cd "${SCRIPT_DIR}"

TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_DIR=${LOG_DIR:-eval_logs/${TASK_CONFIG}/${TEST_ID}_${TIME_TAG}}
mkdir -p "${LOG_DIR}"

STATUS_FILE="${LOG_DIR}/status.tsv"
echo -e "time\ttask\tgpu\tslot\tpid\tstatus\tlog" > "${STATUS_FILE}"

if [ ! -f "${TASK_FILE}" ]; then
    echo "❌ 找不到任务列表文件: ${TASK_FILE}"
    exit 1
fi

mapfile -t TASKS < <(grep -vE '^\s*(#|$)' "${TASK_FILE}")
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
echo "MAX_JOBS_PER_GPU: ${MAX_JOBS_PER_GPU}"
echo "MAX_CONCURRENT_JOBS: $((${#GPUS[@]} * MAX_JOBS_PER_GPU))"
echo "MIN_FREE_MEM_MB: ${MIN_FREE_MEM_MB}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION: ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo "LOG_DIR: ${LOG_DIR}"
echo "============================================================"

declare -A JOB_PID
declare -A JOB_TASK
declare -A JOB_LOG

task_idx=0
finished_count=0
failed_count=0

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

    [ "${free_mem}" -ge "${MIN_FREE_MEM_MB}" ]
}

refresh_jobs() {
    local gpu slot key pid task log rc now

    for gpu in "${GPUS[@]}"; do
        for ((slot = 0; slot < MAX_JOBS_PER_GPU; slot++)); do
            key="${gpu}_${slot}"
            pid=${JOB_PID[$key]:-}

            if [ -z "${pid}" ]; then
                continue
            fi
            if kill -0 "${pid}" 2>/dev/null; then
                continue
            fi

            task=${JOB_TASK[$key]}
            log=${JOB_LOG[$key]}
            wait "${pid}"
            rc=$?
            now=$(date +"%Y-%m-%d %H:%M:%S")

            if [ "${rc}" -eq 0 ]; then
                echo "✅ [GPU ${gpu}:${slot}] 完成任务: ${task}"
                echo -e "${now}\t${task}\t${gpu}\t${slot}\t${pid}\tSUCCESS\t${log}" >> "${STATUS_FILE}"
            else
                echo "❌ [GPU ${gpu}:${slot}] 任务失败: ${task}, exit code=${rc}"
                echo -e "${now}\t${task}\t${gpu}\t${slot}\t${pid}\tFAILED_${rc}\t${log}" >> "${STATUS_FILE}"
                failed_count=$((failed_count + 1))
            fi

            finished_count=$((finished_count + 1))
            unset JOB_PID[$key]
            unset JOB_TASK[$key]
            unset JOB_LOG[$key]
        done
    done
}

find_available_slot() {
    local gpu slot key

    # Fill slot 0 across all GPUs before assigning slot 1.  This spreads the
    # first wave of tasks and gives each first process time to reserve memory.
    for ((slot = 0; slot < MAX_JOBS_PER_GPU; slot++)); do
        for gpu in "${GPUS[@]}"; do
            key="${gpu}_${slot}"
            if [ -n "${JOB_PID[$key]:-}" ]; then
                continue
            fi

            if gpu_has_enough_memory "${gpu}"; then
                echo "${gpu} ${slot}"
                return 0
            fi
        done
    done

    return 1
}

launch_task() {
    local task=$1
    local gpu=$2
    local slot=$3
    local key log_file pid now

    key="${gpu}_${slot}"
    log_file="${LOG_DIR}/${task}_gpu${gpu}_slot${slot}.log"

    echo "🚀 [GPU ${gpu}:${slot}] 启动任务: ${task}"
    echo "    命令: bash eval.sh ${task} ${TASK_CONFIG} ${TRAIN_CONFIG} ${MODEL_NAME} ${SEED} ${gpu} ${CHECKPOINT_ID}"
    echo "    日志: ${log_file}"

    bash "${SCRIPT_DIR}/eval.sh" \
        "${task}" "${TASK_CONFIG}" "${TRAIN_CONFIG}" "${MODEL_NAME}" \
        "${SEED}" "${gpu}" "${CHECKPOINT_ID}" \
        > "${log_file}" 2>&1 &

    pid=$!
    JOB_PID[$key]=${pid}
    JOB_TASK[$key]=${task}
    JOB_LOG[$key]=${log_file}

    now=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "${now}\t${task}\t${gpu}\t${slot}\t${pid}\tRUNNING\t${log_file}" >> "${STATUS_FILE}"
}

cleanup() {
    echo
    echo "⚠️ 收到中断信号，准备停止本脚本启动的 eval 任务..."

    local gpu slot key pid
    for gpu in "${GPUS[@]}"; do
        for ((slot = 0; slot < MAX_JOBS_PER_GPU; slot++)); do
            key="${gpu}_${slot}"
            pid=${JOB_PID[$key]:-}
            if [ -n "${pid}" ]; then
                echo "停止 GPU ${gpu}:${slot} 上的任务: PID=${pid}, task=${JOB_TASK[$key]}"
                kill "${pid}" 2>/dev/null || true
            fi
        done
    done

    exit 130
}

trap cleanup INT TERM

while [ "${finished_count}" -lt "${TOTAL_TASKS}" ]; do
    refresh_jobs

    while [ "${task_idx}" -lt "${TOTAL_TASKS}" ]; do
        available_slot=$(find_available_slot || true)

        if [ -z "${available_slot:-}" ]; then
            break
        fi

        read -r available_gpu available_slot_id <<< "${available_slot}"
        task=${TASKS[$task_idx]}
        launch_task "${task}" "${available_gpu}" "${available_slot_id}"
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
