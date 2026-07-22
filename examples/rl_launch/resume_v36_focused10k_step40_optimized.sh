#!/bin/bash
# Resume the focused-10k V36 run from global_step_40 with post-Step-40 fixes.
#
# This profile is intentionally checkpoint-specific:
# - Step 40 adaptive-KL next coefficient is restored explicitly because the
#   KL controller itself is not part of the EasyR1 checkpoint.
# - The invalid absolute grad cap is disabled while finite checks, EMA spike
#   detection, and max_grad_norm=1.0 remain active.
# - LR is reduced because removing the cap changes recent applied-update
#   density from roughly one third to nearly every global step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_DIR}/examples/local_path_env.sh"

DEFAULT_RUN_DIR="/mnt/shared-storage-user/puyuan/zhangchenhao/EasyR1-latest/model/StepCount-7B_v36_dense_11_30_quality_focused10k_ckpt476_8gpu_20260713_1406"
CKPT_PATH="${1:-${DEFAULT_RUN_DIR}/global_step_40}"
EXPECTED_CKPT_PATH="${DEFAULT_RUN_DIR}/global_step_40"

if [[ "$(readlink -f "${CKPT_PATH}")" != "$(readlink -f "${EXPECTED_CKPT_PATH}")" \
      && "${V36_ALLOW_NON_STEP40_OPTIMIZED_RESUME:-0}" != "1" ]]; then
  echo "[resume-v36-step40-opt][ERROR] this profile is calibrated for: ${EXPECTED_CKPT_PATH}" >&2
  echo "Got: ${CKPT_PATH}" >&2
  echo "Set V36_ALLOW_NON_STEP40_OPTIMIZED_RESUME=1 only after supplying matching data/LR/KL state." >&2
  exit 1
fi

# Preserve the original run identity, dataloader domain, and 80-step optimizer/
# BoK schedule. After prompt filtering, focused10k has 9,937 rows and 77 full
# batches at batch_size=128. A one-epoch Step-40 resume therefore trains the
# remaining Step 41-77 batches without repeating samples; Step 77 is final-saved.
export STEPCOUNT_TRAIN_DATA="${STEPCOUNT_TRAIN_DATA:-${STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA}}"
export TRAINER_VAL_BEFORE_TRAIN=false
export STEPCOUNT_TOTAL_EPOCHS=1
export V36_MAX_STEPS="${V36_MAX_STEPS:-80}"
export BOK_TOTAL_STEPS="${BOK_TOTAL_STEPS:-80}"
EXPECTED_FINAL_STEP=77

# Post-Step-40 optimization profile.
export ACTOR_LR="${ACTOR_LR:-5e-7}"
export KL_TYPE="${KL_TYPE:-adaptive}"
export KL_COEF="${KL_COEF:-0.07414325533079175}"
export KL_TARGET="${KL_TARGET:-0.10}"
export KL_HORIZON="${KL_HORIZON:-30000}"

# Keep clipping and non-finite protection. Disable only the erroneous raw
# pre-clipping norm cap that discarded valid, already-clipped updates.
export GRAD_SPIKE_PROTECT="${GRAD_SPIKE_PROTECT:-1}"
export GRAD_SPIKE_THRESHOLD="${GRAD_SPIKE_THRESHOLD:-3.0}"
export GRAD_SPIKE_ABSOLUTE_CAP="${GRAD_SPIKE_ABSOLUTE_CAP:-0}"
export GRAD_SPIKE_COOLDOWN="${GRAD_SPIKE_COOLDOWN:-0}"
export GRAD_NONFINITE_COOLDOWN="${GRAD_NONFINITE_COOLDOWN:-0}"
export GRAD_SPIKE_LR_FACTOR="${GRAD_SPIKE_LR_FACTOR:-1.0}"
export GRAD_NONFINITE_LR_FACTOR="${GRAD_NONFINITE_LR_FACTOR:-1.0}"

# The old run reached 112.6/140.1 GB max allocated without a CUDA OOM through
# Step 53. Retain that GPU-side profile; its later external SIGKILL had no CUDA
# traceback and cannot be resolved by pretending it was a GPU-memory failure.
export V31_MICRO_BATCH_UPDATE="${V31_MICRO_BATCH_UPDATE:-4}"
export V31_MICRO_BATCH_EXP="${V31_MICRO_BATCH_EXP:-8}"
export EASYR1_VLLM_NUM_GPU_BLOCKS="${EASYR1_VLLM_NUM_GPU_BLOCKS:-20480}"
export V31_GPU_MEM_UTIL="${V31_GPU_MEM_UTIL:-0.50}"
export V31_MAX_NUM_BATCHED_TOKENS="${V31_MAX_NUM_BATCHED_TOKENS:-49152}"
export V36_ENABLE_ULYSSES_SP=0

# A Step-50 checkpoint and validation provide an early canary after changing
# update density. Rolling retention still keeps only the newest three.
export TRAIN_SAVE_FREQ="${TRAIN_SAVE_FREQ:-10}"
export TRAIN_SAVE_LIMIT="${TRAIN_SAVE_LIMIT:-3}"
export TRAIN_VAL_FREQ="${TRAIN_VAL_FREQ:-10}"
export BASETAG="${BASETAG:-ckpt476_11_30_quality_focused10k_bokgrpo_h200_8gpu_resume_step40_opt}"

cd "${REPO_DIR}"

echo "[resume-v36-step40-opt] checkpoint=${CKPT_PATH}"
echo "[resume-v36-step40-opt] lr=${ACTOR_LR} kl=${KL_COEF} target=${KL_TARGET} horizon=${KL_HORIZON}"
echo "[resume-v36-step40-opt] grad_protect=${GRAD_SPIKE_PROTECT} threshold=${GRAD_SPIKE_THRESHOLD} absolute_cap=${GRAD_SPIKE_ABSOLUTE_CAP}"
echo "[resume-v36-step40-opt] micro=${V31_MICRO_BATCH_UPDATE}/${V31_MICRO_BATCH_EXP} blocks=${EASYR1_VLLM_NUM_GPU_BLOCKS} gpu_mem=${V31_GPU_MEM_UTIL}"
echo "[resume-v36-step40-opt] schedule_steps=${V36_MAX_STEPS} epochs=${STEPCOUNT_TOTAL_EPOCHS} expected_final_step=${EXPECTED_FINAL_STEP}"
echo "[resume-v36-step40-opt] save/val=${TRAIN_SAVE_FREQ}/${TRAIN_VAL_FREQ} val_before=${TRAINER_VAL_BEFORE_TRAIN}"

exec bash "${SCRIPT_DIR}/resume_v36_8gpu_from_ckpt.sh" "${CKPT_PATH}" stable
