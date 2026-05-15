#!/usr/bin/env bash
# v27b_chain_runner.sh — wait for current v27b finish, then run 3 experiments sequentially.
# Usage:
#   nohup bash tools/v27b_chain_runner.sh run >> logs/monitor/v27b_chain_runner.nohup.log 2>&1 &
#   bash tools/v27b_chain_runner.sh {status|dry-run}
#   touch logs/monitor/v27b_chain_runner.stop   # graceful abort between experiments

set -u
REPO_DIR="${REPO_DIR:-/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest}"
TRAIN_LOG="${TRAIN_LOG:-${REPO_DIR}/logs/train/training_interleaved_traj_v27b_easy_data_bok_grpo_bok_grpo_20260502_105036.log}"
MONITOR_DIR="${MONITOR_DIR:-${REPO_DIR}/logs/monitor}"
CHAIN_LOG="${CHAIN_LOG:-${MONITOR_DIR}/v27b_chain_runner.log}"
LOCK_FILE="${LOCK_FILE:-${MONITOR_DIR}/v27b_chain_runner.lock}"
PID_FILE="${PID_FILE:-${MONITOR_DIR}/v27b_chain_runner.pid}"
STOP_FILE="${STOP_FILE:-${MONITOR_DIR}/v27b_chain_runner.stop}"
IDLE_MINUTES="${IDLE_MINUTES:-30}"
POLL_SECONDS="${POLL_SECONDS:-300}"
FINISH_REGEX="${FINISH_REGEX:-Training finished successfully\.|Training loop completed|Final validation completed|\[V27b\] Training finished}"

mkdir -p "${MONITOR_DIR}"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
clog() { echo "[$(ts)] $*" | tee -a "${CHAIN_LOG}" ; }

EXPERIMENTS=(
  "exp2_v27b_easy_plus_hard|examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_easy_data.sh|STEPCOUNT_TRAIN_DATA=/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_0_10_easy_plus_hard TRAIN_SAVE_FREQ=60 TRAIN_VAL_FREQ=20 TRAIN_SAVE_LIMIT=6 ACTOR_LR=1.5e-6 BOK_SMART_FILTER_THRESHOLD=0.965 GRAD_SPIKE_THRESHOLD=3.0 V27B_FAILFAST_ENABLE=0"
  "exp3_v27b_hard_oversample|examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_easy_data.sh|STEPCOUNT_TRAIN_DATA=/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_0_10_easy_plus_hard TRAIN_HARD_OVERSAMPLE_FACTOR=2 TRAIN_OVERSAMPLE_NO_MASK_FACTOR=3 TRAIN_SAVE_FREQ=90 TRAIN_VAL_FREQ=30 TRAIN_SAVE_LIMIT=6 ACTOR_LR=1.5e-6 BOK_SMART_FILTER_THRESHOLD=0.965 GRAD_SPIKE_THRESHOLD=3.0 V27B_FAILFAST_ENABLE=0"
)

print_plan() {
    echo "Planned experiment chain (will start AFTER current v27b run finishes):"
    local i=1
    for entry in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r label script env <<< "${entry}"
        echo "  [${i}] ${label}"
        echo "      script: ${script}"
        echo "      env   : ${env}"
        i=$((i+1))
    done
    echo "Stop file: ${STOP_FILE}"
    echo "Chain log: ${CHAIN_LOG}"
}

is_train_alive() {
    local now mtime age_min
    now=$(date +%s)
    [[ -f "${TRAIN_LOG}" ]] || return 1
    mtime=$(stat -c '%Y' "${TRAIN_LOG}" 2>/dev/null || echo 0)
    age_min=$(( (now - mtime) / 60 ))
    if (( age_min < IDLE_MINUTES )); then return 0; fi
    if pgrep -af 'verl\.trainer\.main|ray::|verl_main' >/dev/null 2>&1; then return 0; fi
    return 1
}

is_finished() {
    [[ -f "${TRAIN_LOG}" ]] || return 1
    if tail -n 4000 "${TRAIN_LOG}" 2>/dev/null | grep -Eq "${FINISH_REGEX}"; then return 0; fi
    if ! is_train_alive; then return 0; fi
    return 1
}

wait_for_v27b_finish() {
    clog "wait_for_v27b_finish: poll=${POLL_SECONDS}s idle=${IDLE_MINUTES}min log=${TRAIN_LOG}"
    while true; do
        if [[ -f "${STOP_FILE}" ]]; then clog "STOP file detected; exiting before launch."; exit 0; fi
        if is_finished; then clog "v27b finished signal detected."; return 0; fi
        sleep "${POLL_SECONDS}" & wait $!
    done
}

run_experiment() {
    local label="$1" script="$2" env_str="$3"
    local script_path="${REPO_DIR}/${script}"
    if [[ ! -f "${script_path}" ]]; then clog "[${label}] ERROR: script missing: ${script_path}"; return 99; fi
    clog "[${label}] starting; env=${env_str}"
    local exp_log="${MONITOR_DIR}/chain_${label}_$(date +%Y%m%d_%H%M%S).log"
    clog "[${label}] log=${exp_log}"
    ( cd "${REPO_DIR}"; env ${env_str} bash "${script_path}" ) > >(tee -a "${exp_log}") 2>&1
    local rc=$?
    clog "[${label}] finished rc=${rc}"
    return $rc
}

main_run() {
    echo $$ > "${PID_FILE}"
    clog "chain runner started pid=$$"
    print_plan | tee -a "${CHAIN_LOG}" >/dev/null
    wait_for_v27b_finish
    local idx=1
    for entry in "${EXPERIMENTS[@]}"; do
        if [[ -f "${STOP_FILE}" ]]; then clog "STOP file detected; aborting chain at exp ${idx}."; return 0; fi
        IFS='|' read -r label script env <<< "${entry}"
        run_experiment "${label}" "${script}" "${env}"
        local rc=$?
        if (( rc != 0 )); then clog "[chain] exp ${idx} (${label}) failed rc=${rc}; stopping chain."; return $rc; fi
        sleep 60
        idx=$((idx+1))
    done
    clog "chain runner: all experiments completed."
}

cmd="${1:-run}"
case "${cmd}" in
    run)
        exec 9>"${LOCK_FILE}"
        if ! flock -n 9; then echo "Another instance running (lock: ${LOCK_FILE})." >&2; exit 1; fi
        main_run ;;
    status)
        echo "TRAIN_LOG=${TRAIN_LOG}"
        if [[ -f "${TRAIN_LOG}" ]]; then echo "  size=$(stat -c '%s' "${TRAIN_LOG}") mtime=$(stat -c '%y' "${TRAIN_LOG}")"; fi
        if is_finished; then echo "v27b status: FINISHED"; else echo "v27b status: RUNNING"; fi
        if [[ -f "${PID_FILE}" ]]; then echo "chain pid file: $(cat "${PID_FILE}")"; fi
        echo "stop file present: $([[ -f "${STOP_FILE}" ]] && echo yes || echo no)" ;;
    dry-run|plan) print_plan ;;
    *) echo "Usage: $0 {run|status|dry-run}" >&2; exit 2 ;;
esac
