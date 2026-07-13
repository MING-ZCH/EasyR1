#!/bin/bash
# Validate the portable V36 runtime paths without starting Ray or importing CUDA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/examples/local_path_env.sh"

missing=0

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    printf '[OK] %-24s %s\n' "${label}" "${path}"
  else
    printf '[MISSING] %-19s %s\n' "${label}" "${path}"
    missing=1
  fi
}

reward_file="${STEPCOUNT_REWARD_FN_PATH%%:*}"

check_path "repo" "${EASYR1_REPO_ROOT}"
check_path "config" "${EASYR1_REPO_ROOT}/examples/config.yaml"
check_path "reward" "${reward_file}"
check_path "model_ckpt476" "${STEPCOUNT_1M_RESUME_CKPT476_MODEL_PATH}"
check_path "train_focused10k" "${STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA}"
check_path "train_maskcomplete" "${STEPCOUNT_DENSE_11_50_MASKCOMPLETE_DATA}"
check_path "val_pixmo" "${STEPCOUNT_VAL_DATA}"
check_path "val_stepcount500" "${STEPCOUNT_STEPCOUNT500_V36_VAL100_DATA}"
check_path "mask_metadata" "${STEPCOUNT_DENSE_11_50_MASKS_METADATA}"
check_path "mask_dir" "${STEPCOUNT_DENSE_11_50_MASKS_DIR}"

mkdir -p "${EASYR1_LOG_ROOT}" "${EASYR1_CHECKPOINT_ROOT}"
check_path "log_root" "${EASYR1_LOG_ROOT}"
check_path "checkpoint_root" "${EASYR1_CHECKPOINT_ROOT}"

active_files=(
  "${EASYR1_REPO_ROOT}/examples/local_path_env.sh"
  "${EASYR1_REPO_ROOT}/examples/v36_dense_11_50_full_from_1m_ckpt476.sh"
  "${EASYR1_REPO_ROOT}/examples/rl_launch/mn_trainer_dense_v36.sh"
  "${EASYR1_REPO_ROOT}/examples/rl_launch/run_v36_8gpu_fast_probe.sh"
  "${EASYR1_REPO_ROOT}/examples/rl_launch/run_v36_8gpu_stable.sh"
  "${EASYR1_REPO_ROOT}/examples/rl_launch/run_v36_8gpu_stable_focused10k.sh"
  "${EASYR1_REPO_ROOT}/examples/rl_launch/run_v36_8gpu_focused10k_ulysses_sp2_probe.sh"
  "${EASYR1_REPO_ROOT}/examples/rl_launch/resume_v36_8gpu_from_ckpt.sh"
  "${EASYR1_REPO_ROOT}/examples/v32_sparse_0_10_stable_drfix.sh"
)

if bash -n "${active_files[@]}"; then
  echo "[OK] shell_syntax             complete active/resume chain"
else
  echo "[ERROR] shell syntax failed in the active/resume chain." >&2
  missing=1
fi

if grep -nE 'hyleochang|/apdcephfs|/data/workspace|share_305110755' "${active_files[@]}"; then
  echo "[ERROR] legacy source-cluster path found in the active V36 launch chain." >&2
  missing=1
else
  echo "[OK] legacy_path_scan         active V36 chain is clean"
fi

if [[ -n "${STEPCOUNT_NCCL_IFNAME:-}" && ! -d "/sys/class/net/${STEPCOUNT_NCCL_IFNAME}" ]]; then
  echo "[ERROR] STEPCOUNT_NCCL_IFNAME does not exist on this host: ${STEPCOUNT_NCCL_IFNAME}" >&2
  missing=1
elif [[ -n "${STEPCOUNT_NCCL_IFNAME:-}" ]]; then
  echo "[OK] nccl_interface           ${STEPCOUNT_NCCL_IFNAME}"
else
  echo "[OK] nccl_interface           cluster/NCCL auto-detect"
fi

echo "[V36 preflight] LOCAL_ROOT=${LOCAL_ROOT}"
echo "[V36 preflight] STEPCOUNT_DATA_ROOT=${STEPCOUNT_DATA_ROOT}"
echo "[V36 preflight] EASYR1_CHECKPOINT_ROOT=${EASYR1_CHECKPOINT_ROOT}"

exit "${missing}"
