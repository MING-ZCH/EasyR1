#!/bin/bash
# Path-only preflight for local v35 training. Does not start Ray or import GPU deps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/examples/local_path_env.sh"

missing=0
check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    printf '[OK] %-28s %s\n' "${label}" "${path}"
  else
    printf '[MISSING] %-23s %s\n' "${label}" "${path}"
    missing=1
  fi
}

check_optional_path() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    printf '[OK] %-28s %s\n' "${label}" "${path}"
  else
    printf '[OPTIONAL-MISSING] %-14s %s\n' "${label}" "${path}"
  fi
}

reward_file="${STEPCOUNT_REWARD_FN_PATH%%:*}"

check_path "repo" "${EASYR1_REPO_ROOT}"
check_path "config" "${EASYR1_REPO_ROOT}/examples/config.yaml"
check_path "reward" "${reward_file}"
check_path "1m_model" "${STEPCOUNT_1M_MODEL_PATH}"
check_path "sft_base_model" "${STEPCOUNT_SFT_BASE_MODEL_PATH}"
check_optional_path "dense_intended" "${STEPCOUNT_DENSE_11_20_PLUS_REPLAY_DATA}"
check_path "dense_default" "${STEPCOUNT_DENSE_TRAIN_DATA_DEFAULT}"
check_path "dense_combined" "${STEPCOUNT_DENSE_COMBINED_DATA}"
check_path "replay_data" "${STEPCOUNT_REPLAY_DATA}"
check_path "val_data" "${STEPCOUNT_VAL_DATA}"
check_path "mask_metadata" "${STEPCOUNT_MASKS_METADATA}"
check_path "mask_dir" "${STEPCOUNT_MASKS_DIR}"

mkdir -p "${EASYR1_LOG_ROOT}" "${EASYR1_SAVE_ROOT}"
check_path "log_root" "${EASYR1_LOG_ROOT}"
check_path "save_root" "${EASYR1_SAVE_ROOT}"

if [[ ! -e "${STEPCOUNT_DENSE_11_20_PLUS_REPLAY_DATA}" ]]; then
  echo "[WARN] Intended v35 train data is missing; v35 currently falls back to ${STEPCOUNT_DENSE_TRAIN_DATA_DEFAULT}."
  echo "[WARN] Build it with: bash examples/prepare_v35_dense_11_20_plus_replay_local.sh"
fi

exit "${missing}"
