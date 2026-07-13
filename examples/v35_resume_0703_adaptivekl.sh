#!/bin/bash
# ================================================================
# V35 0703 Resume Probe (11-20) - Adaptive KL + Drift/OOM Guards
# ================================================================
# Compared with examples/v35_dense_11_20.sh, this 0703 resume script changes:
#   1) Resume from the healthier v35 checkpoint before the drift cliff.
#      Default: global_step_20. global_step_40 is supported via V35_RESUME_STEP=40,
#      but should only be used after Phase0 confirms better point-F1/coverage and
#      no OOD regression.
#   2) Enable AdaptiveKLController by switching from actor-side fixed KL loss to
#      reward-side KL penalty:
#        USE_KL_LOSS=false, KL_TYPE=adaptive, KL_TARGET=0.10, KL_HORIZON=50000.
#      Initial KL_COEF remains 0.06 unless overridden.
#   3) Control format drift/excess exploration:
#        ROLLOUT_TEMPERATURE 1.0 -> 0.8.
#   4) Reduce OOM risk:
#        V31_GPU_MEM_UTIL 0.60 -> 0.50.
#   5) Keep v35's EM-aligned reward and strict canonical format contract:
#        WRONG_CAP=0.05, WITHIN1_CAP=0.12, NEG_ONLY=1,
#        TRAJ_FORMAT_STRICT_KEY=1, TRAJ_FORMAT_TYPO_CREDIT=0.0,
#        TRAJ_FORMAT_REJECTION=structural, TRAJ_POINT_KEY_NORMALIZE=1.
#
# Usage on the trainer pod:
#   cd /mnt/shared-storage-user/zhangchenhao/work/EasyR1-hy-0703
#   bash examples/v35_resume_0703_adaptivekl.sh
#
# Optional:
#   V35_RESUME_STEP=40 bash examples/v35_resume_0703_adaptivekl.sh   # resume from step40 instead
#   MAX_STEPS=60 bash examples/v35_resume_0703_adaptivekl.sh         # optional early hard cap
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_path_env.sh"
SOURCE_RUN="${SOURCE_RUN:-StepCount-7B_v35_dense_11_20_A_emalign_20260701_0432}"
V35_SOURCE_RUN_ROOT=${V35_SOURCE_RUN_ROOT:-${EASYR1_SAVE_ROOT}/${SOURCE_RUN}}

export V35_RESUME_STEP=${V35_RESUME_STEP:-20}
case "${V35_RESUME_STEP}" in
  20|40) ;;
  *) echo "[FATAL] V35_RESUME_STEP must be 20 or 40, got '${V35_RESUME_STEP}'" >&2; exit 2 ;;
esac

export V32_LOAD_CHECKPOINT_PATH=${V32_LOAD_CHECKPOINT_PATH:-${V35_SOURCE_RUN_ROOT}/global_step_${V35_RESUME_STEP}}
if [[ ! -d "${V32_LOAD_CHECKPOINT_PATH}" ]]; then
  echo "[FATAL] resume checkpoint not found: ${V32_LOAD_CHECKPOINT_PATH}" >&2
  exit 2
fi

# Run the FULL epoch following the dataset (~101 steps). Per leo: step 50 (death-zone
# entry) is a KEY analysis point but must NOT stop training. training_steps =
# len(dataloader)*total_epochs (~101); on resume from step S the loop trains S+1..~101.
# The trainer overwrites BOK_TOTAL_STEPS with training_steps (ray_trainer.py:312), so we
# leave BOK_TOTAL_STEPS unset. MAX_STEPS is OPTIONAL: leave empty for the full run, or set
# it explicitly (via env) only if you want an early hard cap. Default empty => full ~101.
export MAX_STEPS=${MAX_STEPS:-}
export V31_EXPERIMENT_NAME=${V31_EXPERIMENT_NAME:-StepCount-7B_v35_0703_resume_s${V35_RESUME_STEP}_adaptivekl_$(date +%Y%m%d_%H%M)}
export V31_SAVE_CHECKPOINT_PATH=${V31_SAVE_CHECKPOINT_PATH:-${EASYR1_SAVE_ROOT}/${V31_EXPERIMENT_NAME}}

# Adaptive KL: must use reward-side KL penalty for AdaptiveKLController to update.
export DISABLE_KL=${DISABLE_KL:-false}
export USE_KL_LOSS=${USE_KL_LOSS:-false}
export KL_TYPE=${KL_TYPE:-adaptive}
export KL_COEF=${KL_COEF:-0.06}
export KL_TARGET=${KL_TARGET:-0.10}
export KL_HORIZON=${KL_HORIZON:-50000}
export KL_PENALTY=${KL_PENALTY:-low_var_kl}

# Drift and OOM guards.
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.8}
export V31_GPU_MEM_UTIL=${V31_GPU_MEM_UTIL:-0.50}
export V31_MICRO_BATCH_UPDATE=${V31_MICRO_BATCH_UPDATE:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Save/eval more frequently for short resume probes.
export TRAIN_SAVE_FREQ=${TRAIN_SAVE_FREQ:-20}
export TRAIN_VAL_FREQ=${TRAIN_VAL_FREQ:-20}
export TRAIN_SAVE_LIMIT=${TRAIN_SAVE_LIMIT:-4}

echo "================================================================"
echo "[V35-0703-resume] resume=${V32_LOAD_CHECKPOINT_PATH}"
echo "[V35-0703-resume] save=${V31_SAVE_CHECKPOINT_PATH}"
echo "[V35-0703-resume] steps: resume_from=${V35_RESUME_STEP} run_to=FULL_EPOCH(~101) max_steps_cap=${MAX_STEPS:-<none>} (step50=key-analysis, NOT a stop)"
echo "[V35-0703-resume] adaptive_kl: disable=${DISABLE_KL} use_kl_loss=${USE_KL_LOSS} type=${KL_TYPE} coef=${KL_COEF} target=${KL_TARGET} horizon=${KL_HORIZON} penalty=${KL_PENALTY}"
echo "[V35-0703-resume] guards: temperature=${ROLLOUT_TEMPERATURE} gpu_mem_util=${V31_GPU_MEM_UTIL} micro_update=${V31_MICRO_BATCH_UPDATE}"
echo "================================================================"

exec bash "${SCRIPT_DIR}/v35_dense_11_20.sh"
