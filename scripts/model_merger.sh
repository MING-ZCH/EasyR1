#!/bin/bash
set -Eeuo pipefail

ACTOR_DIR="${1:?Usage: model_merger.sh /path/to/global_step_xxx/actor}"
ACTOR_DIR="${ACTOR_DIR%/}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGE_LOG_DIR="/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/logs/merge"
mkdir -p "${MERGE_LOG_DIR}"

EXP_NAME="$(basename "$(dirname "$(dirname "${ACTOR_DIR}")")")"
STEP_NAME="$(basename "$(dirname "${ACTOR_DIR}")")"
LOG_PATH="${MERGE_LOG_DIR}/merge_${EXP_NAME}_${STEP_NAME}_$(date +%Y%m%d_%H%M%S).log"

ENV_H20="/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/eval_auto/env_h20.sh"
if [[ -f "${ENV_H20}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_H20}"
fi

export PYTHONUNBUFFERED=1
echo "[merge] actor_dir=${ACTOR_DIR}" | tee "${LOG_PATH}"
echo "[merge] log=${LOG_PATH}" | tee -a "${LOG_PATH}"

python3 "${SCRIPT_DIR}/model_merger.py" \
    --local_dir "${ACTOR_DIR}" \
    2>&1 | tee -a "${LOG_PATH}"
