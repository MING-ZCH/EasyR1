#!/bin/bash
# V34 Phase1: Trajectory-level credit A/B (0-10) — all knobs below are WIRED.
#
# Baseline (A): v32 bok_grpo (trajectory-broadcast advantage), current production recipe.
# Variant  (B): bok_grpo_step + mask-based coverage/consistency/stop trajectory rewards.
#
# Scientific question: does trajectory-level credit reduce the length-degradation
# slope (pixmo 7-10 EM) WITHOUT hurting overall pixmo EM? It is NOT expected to move
# the 0-10 headline number much (V31==V33 proved point reward is non-load-bearing).
#
# Usage:
#   bash examples/v34_trajectory_credit_ab.sh                       # variant B
#   V34_AB_VARIANT=baseline bash examples/v34_trajectory_credit_ab.sh
#   V34_DRY_RUN=1 bash examples/v34_trajectory_credit_ab.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

V34_AB_VARIANT=${V34_AB_VARIANT:-variant}

# Resume from V33 best (verified real path on shared CephFS).
V33_CKPT="/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/checkpoints/StepCount-7B_v33_clean_full_masks_32gpu_20260603_2343/global_step_40/actor/huggingface"
export MODEL_PATH=${MODEL_PATH:-${V33_CKPT}}
export V31_EXPERIMENT_NAME=${V31_EXPERIMENT_NAME:-StepCount-7B_v34_traj_credit_${V34_AB_VARIANT}_$(date +%Y%m%d_%H%M)}
export V31_SAVE_CHECKPOINT_PATH=${V31_SAVE_CHECKPOINT_PATH:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/checkpoints/${V31_EXPERIMENT_NAME}}

# V34_AB_VARIANT: baseline(A) | stopshaping(B) | progress(C, alias: variant)
# Mirrors the dense launcher's three arms (revised after 2-reviewer consensus).
case "${V34_AB_VARIANT}" in
  baseline)
    export STEPCOUNT_RL_MODE=bok_grpo; export ADV_ESTIMATOR=bok_grpo; export PROCESS_REWARD_ENABLE=0 ;;
  stopshaping)
    # broadcast + amplified trajectory stop-shaping (NOT per-step stoptiming, which is degenerate)
    export STEPCOUNT_RL_MODE=bok_grpo; export ADV_ESTIMATOR=bok_grpo; export PROCESS_REWARD_ENABLE=0
    export TRAJ_STOP_NONE_ENABLE=1
    export TRAJ_EARLY_STOP_PENALTY=${TRAJ_EARLY_STOP_PENALTY:-0.30}
    export TRAJ_STOP_NONE_BONUS=${TRAJ_STOP_NONE_BONUS:-0.10} ;;
  *)  # progress (C) — the only per-step signal with genuine within-group variance
    export STEPCOUNT_RL_MODE=bok_grpo_step; export ADV_ESTIMATOR=bok_grpo_step; export PROCESS_REWARD_ENABLE=1
    export BOK_STEP_SIGNAL=progress
    export BOK_PROGRESS_DUP_PENALTY=${BOK_PROGRESS_DUP_PENALTY:-1.0}
    export BOK_STEP_WEIGHT=${BOK_STEP_WEIGHT:-0.2}
    export BOK_STEP_GATE=${BOK_STEP_GATE:-answer_soft}
    export BOK_STEP_MIN_GATE=${BOK_STEP_MIN_GATE:-0.2}
    export TRAJ_RETURN_POINT_STEP_SCORES=${TRAJ_RETURN_POINT_STEP_SCORES:-1} ;;
esac
# NOTE: per-step `stoptiming` (BOK_STEP_SIGNAL=stoptiming) is RETAINED in code as an
# optional ablation only: reviewers showed it is within-group degenerate
# (value=f(k,N), std=0 -> always batch-fallback, never credits <answer>). Falsifiable
# smoke check: its `[BoK-GRPO-Step] low_var_steps=` would be ~100%, progress's far lower.

# ---- Trajectory-level rewards (all WIRED in StepCount_mask_reward.py) ----
export TRAJ_COVERAGE_PENALTY_ENABLE=${TRAJ_COVERAGE_PENALTY_ENABLE:-1}
export TRAJ_COVERAGE_PENALTY_MODE=${TRAJ_COVERAGE_PENALTY_MODE:-mask}   # gt_count - matched_unused
export TRAJ_COVERAGE_PENALTY_PER_MISS=${TRAJ_COVERAGE_PENALTY_PER_MISS:-0.3}
export TRAJ_COVERAGE_PENALTY_CAP=${TRAJ_COVERAGE_PENALTY_CAP:-0.6}
export TRAJ_EXTRA_POINT_PENALTY_LAMBDA=${TRAJ_EXTRA_POINT_PENALTY_LAMBDA:-1.0}
export TRAJ_CONSISTENCY_PENALTY=${TRAJ_CONSISTENCY_PENALTY:-0.5}
export TRAJ_STOP_NONE_ENABLE=${TRAJ_STOP_NONE_ENABLE:-1}
export TRAJ_ANSWER_GATE_MODE=${TRAJ_ANSWER_GATE_MODE:-off}

# NOTE: hard_only oversample is OFF by default — StepCountQA-RL-Traj_0_10_hard_only
# is NOT present on the current mount. To use it, first create the dataset and set
# TRAIN_HARD_OVERSAMPLE_FACTOR>1 + TRAIN_HARD_REFERENCE_PATH explicitly.
export TRAIN_HARD_OVERSAMPLE_FACTOR=${TRAIN_HARD_OVERSAMPLE_FACTOR:-1}

echo "[V34] Trajectory credit A/B variant=${V34_AB_VARIANT}"
echo "[V34] ADV_ESTIMATOR=${ADV_ESTIMATOR} PROCESS_REWARD=${PROCESS_REWARD_ENABLE}"
echo "[V34] coverage(mask)=${TRAJ_COVERAGE_PENALTY_ENABLE} stop_none=${TRAJ_STOP_NONE_ENABLE}"
echo "[V34] MODEL_PATH=${MODEL_PATH}"

if [[ "${V34_DRY_RUN:-0}" == "1" ]]; then
  echo "[V34 DRY RUN] Would exec v32_sparse_0_10_stable_drfix.sh with the env above"
  exit 0
fi

exec bash "${SCRIPT_DIR}/v32_sparse_0_10_stable_drfix.sh"
