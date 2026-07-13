#!/usr/bin/env bash
# Start background V34 live monitor for 32-GPU zw training.
set -euo pipefail

CEPH="/data/workspace/share/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz"
LOGDIR="${CEPH}/logs/rl"
MONDIR="${CEPH}/logs/monitor"
TS=$(date +%Y%m%d_%H%M%S)

# Prefer today's 4-node A run; fall back to newest v34_A_4node log.
TRAIN_LOG="${1:-}"
if [ -z "${TRAIN_LOG}" ]; then
  TRAIN_LOG=$(ls -t "${LOGDIR}"/v34_A_4node_1M_a2_dopt_*.log 2>/dev/null | head -1)
fi
if [ -z "${TRAIN_LOG}" ] || [ ! -f "${TRAIN_LOG}" ]; then
  echo "ERROR: training log not found under ${LOGDIR}" >&2
  exit 1
fi

mkdir -p "${MONDIR}"
MON_LOG="${MONDIR}/v34_live_${TS}.log"
JSON_LOG="${MONDIR}/v34_live_${TS}.jsonl"
PID_FILE="${MONDIR}/v34_live_monitor.pid"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stop previous monitor if ours
if [ -f "${PID_FILE}" ]; then
  OLD_PID=$(cat "${PID_FILE}")
  if kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "Stopping previous monitor PID=${OLD_PID}"
    kill "${OLD_PID}" 2>/dev/null || true
    sleep 1
  fi
fi

nohup python3 "${SCRIPT_DIR}/monitor_v34_live.py" \
  --log "${TRAIN_LOG}" \
  --out "${MON_LOG}" \
  --json-metrics "${JSON_LOG}" \
  --interval 20 \
  --every 5 \
  --summary-every 20 \
  --entropy-alert 1.5 \
  --no-point-warn 0.15 \
  --no-point-critical 0.30 \
  --no-point-rise 0.10 \
  > "${MONDIR}/v34_live_${TS}.stdout" 2>&1 &

echo $! > "${PID_FILE}"
echo "V34 monitor started PID=$(cat "${PID_FILE}")"
echo "  train_log: ${TRAIN_LOG}"
echo "  monitor:   ${MON_LOG}"
echo "  json:      ${JSON_LOG}"
echo "  tail -f ${MON_LOG}"
