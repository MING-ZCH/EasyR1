#!/usr/bin/env bash
# Hourly V36 training watchdog: parse the current EasyR1 log and sync the
# corresponding offline W&B run to the easy_r1 project.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_PATH="${V36_WATCH_LOG:-${REPO_DIR}/logs/rl/v32_standalone_20260713_022735.log}"
WANDB_RUN_DIR="${V36_WATCH_WANDB_RUN_DIR:-${REPO_DIR}/logs/wandb/wandb/offline-run-20260713_023029-8z978q27}"
OUT_DIR="${V36_WATCH_OUT_DIR:-${REPO_DIR}/logs/monitor/v36_20260713_0227_monitor}"
PROJECT="${V36_WATCH_PROJECT:-easy_r1}"
INTERVAL_SECONDS="${V36_WATCH_INTERVAL_SECONDS:-3600}"
TOTAL_STEPS="${V36_WATCH_TOTAL_STEPS:-200}"
SYNC_TIMEOUT="${V36_WATCH_SYNC_TIMEOUT:-900}"
STALE_AFTER_SECONDS="${V36_WATCH_STALE_AFTER_SECONDS:-7200}"

mkdir -p "${OUT_DIR}"
LOCK_PATH="${OUT_DIR}/v36_hourly_watchdog.lock"
WATCHDOG_LOG="${OUT_DIR}/v36_hourly_watchdog.log"

exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "[$(date '+%F %T %Z')] another hourly watchdog is already running: ${LOCK_PATH}" | tee -a "${WATCHDOG_LOG}"
  exit 0
fi

echo "[$(date '+%F %T %Z')] hourly watchdog started" | tee -a "${WATCHDOG_LOG}"
echo "log=${LOG_PATH}" | tee -a "${WATCHDOG_LOG}"
echo "wandb_run_dir=${WANDB_RUN_DIR}" | tee -a "${WATCHDOG_LOG}"
echo "out_dir=${OUT_DIR}" | tee -a "${WATCHDOG_LOG}"
echo "project=${PROJECT} interval=${INTERVAL_SECONDS}s total_steps=${TOTAL_STEPS}" | tee -a "${WATCHDOG_LOG}"
echo "stale_after_seconds=${STALE_AFTER_SECONDS}" | tee -a "${WATCHDOG_LOG}"

while true; do
  echo "[$(date '+%F %T %Z')] hourly check begin" | tee -a "${WATCHDOG_LOG}"
  WANDB_MODE=online python3 "${SCRIPT_DIR}/monitor_v36_training.py" \
    --log "${LOG_PATH}" \
    --wandb-run-dir "${WANDB_RUN_DIR}" \
    --out-dir "${OUT_DIR}" \
    --project "${PROJECT}" \
    --interval "${INTERVAL_SECONDS}" \
    --sync-interval "${INTERVAL_SECONDS}" \
    --sync-timeout "${SYNC_TIMEOUT}" \
    --summary-every 10 \
    --total-steps "${TOTAL_STEPS}" \
    --stale-after-seconds "${STALE_AFTER_SECONDS}" \
    --process-pattern "ray::Runner" \
    --process-pattern "ray::WorkerDict" \
    --process-pattern "WorkerDict" \
    --process-pattern "verl.trainer.main" \
    --process-pattern "main_ppo" \
    --process-pattern "vllm" \
    --once >>"${WATCHDOG_LOG}" 2>&1 || true
  echo "[$(date '+%F %T %Z')] hourly check end" | tee -a "${WATCHDOG_LOG}"
  sleep "${INTERVAL_SECONDS}"
done
