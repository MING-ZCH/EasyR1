#!/usr/bin/env bash
# v27b_periodic_wandb_monitor.sh — 1h snapshot + 4h wandb sync.
# Usage:
#   nohup bash tools/v27b_periodic_wandb_monitor.sh run >> logs/monitor/v27b_periodic_wandb_monitor.nohup.log 2>&1 &
#   bash tools/v27b_periodic_wandb_monitor.sh {once|snapshot|sync}

set -u
REPO_DIR="${REPO_DIR:-/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest}"
TRAIN_LOG="${TRAIN_LOG:-${REPO_DIR}/logs/train/training_interleaved_traj_v29_bok_grpo_step_easy_20260508_182514.log}"
REPORT_INTERVAL_SECONDS="${REPORT_INTERVAL_SECONDS:-3600}"
WANDB_SYNC_INTERVAL_SECONDS="${WANDB_SYNC_INTERVAL_SECONDS:-14400}"
MONITOR_DIR="${MONITOR_DIR:-${REPO_DIR}/logs/monitor}"
SNAPSHOT_LOG="${SNAPSHOT_LOG:-${MONITOR_DIR}/training_interleaved_traj_v29_bok_grpo_step_easy_20260508_182514_periodic_summary.log}"
WANDB_SYNC_LOG="${WANDB_SYNC_LOG:-${MONITOR_DIR}/training_interleaved_traj_v29_bok_grpo_step_easy_20260508_182514_wandb_sync.log}"
LOCK_FILE="${LOCK_FILE:-${MONITOR_DIR}/v27b_periodic_wandb_monitor.lock}"
PID_FILE="${PID_FILE:-${MONITOR_DIR}/v27b_periodic_wandb_monitor.pid}"

mkdir -p "${MONITOR_DIR}"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log_snap() { echo "[$(ts)] $*" >> "${SNAPSHOT_LOG}"; }
log_sync() { echo "[$(ts)] $*" >> "${WANDB_SYNC_LOG}"; }

write_snapshot() {
    log_snap "===== snapshot start ====="
    if [[ ! -f "${TRAIN_LOG}" ]]; then
        log_snap "TRAIN_LOG not found: ${TRAIN_LOG}"; log_snap "===== snapshot end ====="; return 0
    fi
    log_snap "log=${TRAIN_LOG} size=$(stat -c '%s' "${TRAIN_LOG}" 2>/dev/null) mtime=$(stat -c '%y' "${TRAIN_LOG}" 2>/dev/null)"
    log_snap "--- latest val/answer_acc lines ---"
    grep -E 'val/.*answer.*acc|val_metrics|Validation:' "${TRAIN_LOG}" 2>/dev/null | tail -n 8 >> "${SNAPSHOT_LOG}" || true
    log_snap "--- latest TrainHealth ---"
    grep -E 'TrainHealth' "${TRAIN_LOG}" 2>/dev/null | tail -n 3 >> "${SNAPSHOT_LOG}" || true
    log_snap "--- latest GradSpikeProtect / NaN ---"
    grep -E 'GradSpikeProtect|grad_norm.*spike|nonfinite|nan|NaN' "${TRAIN_LOG}" 2>/dev/null | tail -n 6 >> "${SNAPSHOT_LOG}" || true
    log_snap "--- recent checkpoint saves ---"
    grep -E 'global_step_[0-9]+|Saved.*checkpoint|save_checkpoint' "${TRAIN_LOG}" 2>/dev/null | tail -n 4 >> "${SNAPSHOT_LOG}" || true
    if [[ -f "${REPO_DIR}/scripts/monitor_logs.py" ]]; then
        log_snap "--- monitor_logs.py --last 3 ---"
        ( cd "${REPO_DIR}" && python3 scripts/monitor_logs.py --last 3 2>&1 | tail -n 80 ) >> "${SNAPSHOT_LOG}" || true
    fi
    log_snap "===== snapshot end ====="
}

setup_proxy_for_wandb() {
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
    if command -v curl >/dev/null 2>&1; then
        # shellcheck disable=SC1090
        source <(curl -sSL http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh) 2>/dev/null || true
    fi
}

latest_wandb_run() { ls -1dt "${REPO_DIR}/wandb/"offline-run-* 2>/dev/null | head -n 1; }

sync_wandb() {
    log_sync "===== sync start ====="
    local run_dir; run_dir="$(latest_wandb_run || true)"
    if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
        log_sync "no offline-run-* under ${REPO_DIR}/wandb"; log_sync "===== sync end ====="; return 0
    fi
    log_sync "target=${run_dir}"
    ( setup_proxy_for_wandb; cd "${REPO_DIR}" && wandb sync "${run_dir}" 2>&1 ) | tee -a "${WANDB_SYNC_LOG}" >/dev/null
    log_sync "===== sync end ====="
}

main_loop() {
    echo $$ > "${PID_FILE}"
    log_snap "monitor started pid=$$ snap=${REPORT_INTERVAL_SECONDS}s sync=${WANDB_SYNC_INTERVAL_SECONDS}s"
    log_sync "monitor started pid=$$"
    write_snapshot; sync_wandb
    local last_sync; last_sync=$(date +%s)
    while true; do
        sleep "${REPORT_INTERVAL_SECONDS}" & wait $!
        write_snapshot
        local now; now=$(date +%s)
        if (( now - last_sync >= WANDB_SYNC_INTERVAL_SECONDS )); then
            sync_wandb; last_sync=$(date +%s)
        fi
    done
}

cmd="${1:-run}"
case "${cmd}" in
    run)
        exec 9>"${LOCK_FILE}"
        if ! flock -n 9; then echo "Another instance running (lock: ${LOCK_FILE})." >&2; exit 1; fi
        main_loop ;;
    once) write_snapshot; sync_wandb ;;
    snapshot) write_snapshot ;;
    sync) sync_wandb ;;
    *) echo "Usage: $0 {run|once|snapshot|sync}" >&2; exit 2 ;;
esac
