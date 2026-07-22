#!/bin/bash
set -Eeuo pipefail

ACTOR_DIR="${1:?Usage: model_merger.sh /path/to/global_step_xxx/actor}"
ACTOR_DIR="${ACTOR_DIR%/}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${REPO_DIR}/examples/local_path_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/examples/local_path_env.sh"
fi

MERGE_LOG_DIR="${EASYR1_LOG_ROOT:-${REPO_DIR}/logs}/merge"
mkdir -p "${MERGE_LOG_DIR}"

EXP_NAME="$(basename "$(dirname "$(dirname "${ACTOR_DIR}")")")"
STEP_NAME="$(basename "$(dirname "${ACTOR_DIR}")")"
LOG_PATH="${MERGE_LOG_DIR}/merge_${EXP_NAME}_${STEP_NAME}_$(date +%Y%m%d_%H%M%S).log"

export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "[merge] actor_dir=${ACTOR_DIR}" | tee "${LOG_PATH}"
echo "[merge] log=${LOG_PATH}" | tee -a "${LOG_PATH}"
echo "[merge] python=${PYTHON_BIN}" | tee -a "${LOG_PATH}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/model_merger.py" \
    --local_dir "${ACTOR_DIR}" \
    2>&1 | tee -a "${LOG_PATH}"
