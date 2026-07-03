#!/bin/bash
# V34 Phase1 (secondary): Anti-saturation for the 0-10 ceiling.
#
# WIRED (take effect now): BOK_EASY_THRESHOLD, ROLLOUT_TEMPERATURE.
# NOT-YET-WIRED (require dataloader patch, do nothing if set): TRAIN_VCRL_PASS_RATE_*
#   (VCRL pass-rate resampling). Left here as documented intent only.
#
# Usage:
#   BOK_EASY_THRESHOLD=0.75 ROLLOUT_TEMPERATURE=1.1 bash examples/v34_antisaturation_010_ab.sh
#   V34_DRY_RUN=1 bash examples/v34_antisaturation_010_ab.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

V33_CKPT="/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/checkpoints/StepCount-7B_v33_clean_full_masks_32gpu_20260603_2343/global_step_40/actor/huggingface"
export MODEL_PATH=${MODEL_PATH:-${V33_CKPT}}

export BOK_EASY_THRESHOLD=${BOK_EASY_THRESHOLD:-0.75}        # WIRED (vs 0.50 baseline)
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.1}        # WIRED (vs 1.0 baseline)
export V31_EXPERIMENT_NAME=${V31_EXPERIMENT_NAME:-StepCount-7B_v34_antisat_ez${BOK_EASY_THRESHOLD}_t${ROLLOUT_TEMPERATURE}_$(date +%Y%m%d_%H%M)}

# Keep v32 stable backbone (bok_grpo).
export STEPCOUNT_RL_MODE=${STEPCOUNT_RL_MODE:-bok_grpo}
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-bok_grpo}
export PROCESS_REWARD_ENABLE=${PROCESS_REWARD_ENABLE:-0}

echo "[V34] Anti-saturation: BOK_EASY_THRESHOLD=${BOK_EASY_THRESHOLD} temp=${ROLLOUT_TEMPERATURE}"
echo "[V34] NOTE: VCRL pass_rate resampling is NOT wired (TRAIN_VCRL_PASS_RATE_* ignored)."

if [[ "${V34_DRY_RUN:-0}" == "1" ]]; then
  echo "[V34 DRY RUN] Config printed above"
  exit 0
fi

exec bash "${SCRIPT_DIR}/v32_sparse_0_10_stable_drfix.sh"
