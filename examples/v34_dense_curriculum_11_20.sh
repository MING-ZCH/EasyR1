#!/bin/bash
# ================================================================
# V34 Dense BoK-GRPO three-arm curriculum (11-20), base = SFT.
# ================================================================
# Plan: StepCountModel/plan/V34_Dense_BoKGRPO_ThreeArm_Plan.md
#
# Three arms (V34_ARM = A | B | C), identical base/data/dense-reward;
# only the per-turn credit signal differs:
#   A Control       : bok_grpo (pure trajectory-broadcast, no per-turn credit)
#   B Stop-timing   : bok_grpo_step + BOK_STEP_SIGNAL=stoptiming (sign(N-count_k))
#   C Coverage-prog : bok_grpo_step + BOK_STEP_SIGNAL=progress  (+1 new / -dup / 0 miss)
#
# Base = SFT (NOT v33): oracle shows 0-10 RL is negative-transfer to dense; SFT->dense-RL
# is the clean attribution. Swap to 1M-SFT later via MODEL_PATH.
#
# Scope = 11-20 only (full 11-50 fails: all-wrong groups >20 + chain > 21 turns +
# perception ceiling ~11). Build the data first:
#   python3 tools/v34_prepare_dense_subset.py \
#     --combined /mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_11_50_Combined/data \
#     --replay   /mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_0_10/data \
#     --count_min 11 --count_max 20 --replay_ratio 0.25 \
#     --out_dir  /mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_11_20_plus_replay/data
#
# Usage:
#   V34_ARM=A bash examples/v34_dense_curriculum_11_20.sh
#   V34_ARM=B bash examples/v34_dense_curriculum_11_20.sh
#   V34_ARM=C bash examples/v34_dense_curriculum_11_20.sh
#   V34_DRY_RUN=1 V34_ARM=B bash examples/v34_dense_curriculum_11_20.sh
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_path_env.sh"

SFT_BASE="${STEPCOUNT_SFT_BASE_MODEL_PATH}"

# ---- Base = SFT (override with 1M-SFT when ready) ----
export MODEL_PATH=${MODEL_PATH:-${SFT_BASE}}

# ---- Arm selection ----
V34_ARM=${V34_ARM:-A}
case "${V34_ARM}" in
  A)  # Control: pure trajectory-broadcast
      export STEPCOUNT_RL_MODE=bok_grpo
      export ADV_ESTIMATOR=bok_grpo
      export PROCESS_REWARD_ENABLE=0
      ARM_TAG="A_control" ;;
  B)  # Stop-timing: broadcast estimator + AMPLIFIED trajectory stop-shaping.
      # (Revised after 2-reviewer consensus: per-step stoptiming sign(N-count_k) is a
      #  pure function of (k,N) -> zero within-group variance -> always batch-fallback ->
      #  never credits the <answer> token. Stop-timing belongs at the TRAJECTORY level,
      #  where Arm A already encodes it; Arm B tests AMPLIFIED stop-shaping vs A's baseline.)
      export STEPCOUNT_RL_MODE=bok_grpo
      export ADV_ESTIMATOR=bok_grpo
      export PROCESS_REWARD_ENABLE=0
      export TRAJ_STOP_NONE_ENABLE=1
      export TRAJ_EARLY_STOP_PENALTY=${TRAJ_EARLY_STOP_PENALTY:-0.30}      # vs A default 0.15 (90% errors=early-stop)
      export TRAJ_STOP_NONE_BONUS=${TRAJ_STOP_NONE_BONUS:-0.10}           # vs A default 0.05
      export TRAJ_COVERAGE_PENALTY_PER_MISS=${TRAJ_COVERAGE_PENALTY_PER_MISS:-0.2}  # vs A 0.1
      export TRAJ_COVERAGE_PENALTY_CAP=${TRAJ_COVERAGE_PENALTY_CAP:-0.5}  # vs A 0.3
      ARM_TAG="B_stopshaping" ;;
  C)  # Coverage-progress per-turn credit
      export STEPCOUNT_RL_MODE=bok_grpo_step
      export ADV_ESTIMATOR=bok_grpo_step
      export PROCESS_REWARD_ENABLE=1
      export BOK_STEP_SIGNAL=progress
      export BOK_STEP_WEIGHT=${BOK_STEP_WEIGHT:-0.2}
      export BOK_PROGRESS_DUP_PENALTY=${BOK_PROGRESS_DUP_PENALTY:-1.0}
      export BOK_STEP_GATE=${BOK_STEP_GATE:-answer_soft}
      export BOK_STEP_MIN_GATE=${BOK_STEP_MIN_GATE:-0.2}
      ARM_TAG="C_progress" ;;
  *) echo "[FATAL] V34_ARM must be A|B|C, got '${V34_ARM}'" >&2; exit 2 ;;
esac

export V31_EXPERIMENT_NAME=${V31_EXPERIMENT_NAME:-StepCount-7B_v34_dense_11_20_${ARM_TAG}_$(date +%Y%m%d_%H%M)}
export V31_SAVE_CHECKPOINT_PATH=${V31_SAVE_CHECKPOINT_PATH:-${EASYR1_SAVE_ROOT}/${V31_EXPERIMENT_NAME}}

# ---- Data: 11-20 + 25% 0-10 replay (build with tools/v34_prepare_dense_subset.py) ----
export STEPCOUNT_TRAIN_DATA=${STEPCOUNT_TRAIN_DATA:-${STEPCOUNT_DENSE_TRAIN_DATA_DEFAULT}}

# ---- Long-chain rollout / length (reviewer-tuned for 11-20) ----
export INTERLEAVED_MAX_TURNS=${INTERLEAVED_MAX_TURNS:-21}
export INTERLEAVED_PER_TURN_MAX_TOKENS=${INTERLEAVED_PER_TURN_MAX_TOKENS:-2048}
export INTERLEAVED_ANSWER_TURN_MAX_TOKENS=${INTERLEAVED_ANSWER_TURN_MAX_TOKENS:-2048}
export V31_MAX_RESPONSE_LENGTH=${V31_MAX_RESPONSE_LENGTH:-6144}   # NEW env (v32 hardcode now overridable)
export V31_MAX_MODEL_LEN=${V31_MAX_MODEL_LEN:-32768}
export V31_MAX_NUM_BATCHED_TOKENS=${V31_MAX_NUM_BATCHED_TOKENS:-49152}
# export V31_MAX_PROMPT_LENGTH=12288   # uncomment if dense-image filter rate > 2-5%

# ---- Optimizer / memory (cold-start safe) ----
export ACTOR_LR=${ACTOR_LR:-1e-6}                # start 1e-6; raise to 1.5e-6 after 20-30 healthy steps
export V31_MICRO_BATCH_UPDATE=${V31_MICRO_BATCH_UPDATE:-4}
export V31_MICRO_BATCH_EXP=${V31_MICRO_BATCH_EXP:-4}
export V31_GPU_MEM_UTIL=${V31_GPU_MEM_UTIL:-0.62}
export ROLLOUT_N=${ROLLOUT_N:-16}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export BOK_TOTAL_STEPS=${BOK_TOTAL_STEPS:-104}    # ~26,613 rows / 256 (trainer overrides via meta if available)

# ---- Reward weights (dense-tuned: answer-primary) ----
export ANSWER_WEIGHT=${ANSWER_WEIGHT:-0.7}
export POINT_WEIGHT=${POINT_WEIGHT:-0.2}
export TRAJECTORY_FORMAT_WEIGHT=${TRAJECTORY_FORMAT_WEIGHT:-0.1}

# ---- Dense trajectory reward shaping (WIRED) ----
export TRAJ_DENSE_CONTINUOUS_REWARD=${TRAJ_DENSE_CONTINUOUS_REWARD:-1}
export TRAJ_DENSE_CONTINUOUS_MIN_GT=${TRAJ_DENSE_CONTINUOUS_MIN_GT:-11}
export TRAJ_DENSE_CONTINUOUS_WRONG_CAP=${TRAJ_DENSE_CONTINUOUS_WRONG_CAP:-0.35}
export TRAJ_DENSE_CONTINUOUS_WITHIN1_CAP=${TRAJ_DENSE_CONTINUOUS_WITHIN1_CAP:-0.45}
export TRAJ_COVERAGE_PENALTY_ENABLE=${TRAJ_COVERAGE_PENALTY_ENABLE:-1}
export TRAJ_COVERAGE_PENALTY_MODE=${TRAJ_COVERAGE_PENALTY_MODE:-mask}
export TRAJ_COVERAGE_PENALTY_PER_MISS=${TRAJ_COVERAGE_PENALTY_PER_MISS:-0.1}
export TRAJ_COVERAGE_PENALTY_CAP=${TRAJ_COVERAGE_PENALTY_CAP:-0.3}
export TRAJ_EXTRA_POINT_PENALTY_LAMBDA=${TRAJ_EXTRA_POINT_PENALTY_LAMBDA:-1.0}
export TRAJ_STOP_NONE_ENABLE=${TRAJ_STOP_NONE_ENABLE:-1}
export TRAJ_RETURN_POINT_STEP_SCORES=${TRAJ_RETURN_POINT_STEP_SCORES:-1}  # needed for progress hit/dup + step rewards

# ---- Format strict-key (v34.1 fix: gradient pressure against benign "point_22d" drift) ----
# Root cause (Arm A run): _normalize_point_keys made "point_22d" earn the SAME format
# reward as "point_2d", so once the policy drifted at step ~45 there was ZERO gradient to
# fix it (drift locked in 45->101, fed the entropy/KL climb). Fix: keep the PARSER
# typo-tolerant (answer/point unaffected, never crash) but make FORMAT credit favor the
# canonical "point_2d" so clean vs drifted trajectories are no longer reward-identical.
#
# Strict default: typo earns ZERO format credit while the parser still normalizes it for
# answer/point reward. Use structural rejection so fully-drifted but parseable trajectories
# lose only the format term instead of hard-zeroing answer/point (which would collapse a
# typo-heavy dense batch). This matches the desired contract:
#   - parsing: tolerant to benign point_22d drift
#   - format scoring: full credit only for the canonical point_2d key
#   - rejection: hard-zero only real structural garbage (no answer / invalid answer / no point tag)
# Conservative attribution alt: TRAJ_FORMAT_TYPO_CREDIT=0.2 + TRAJ_FORMAT_REJECTION=1
# keeps broken-JSON / non-normalizable garbage hard-zeroed, but gives typo partial format
# credit and therefore does NOT strictly require the original key for format score.
# NOTE (reviewer): this gradient PREVENTS drift when canonical still exists in a group; it is
# near-inert at REVERSING an already-locked-in drift (canonical ~0% late -> GRPO mean cancels
# it). To reverse, pair with higher kl_coef and/or a pre-step-45 checkpoint rewind.
export TRAJ_FORMAT_STRICT_KEY=${TRAJ_FORMAT_STRICT_KEY:-1}
export TRAJ_FORMAT_TYPO_CREDIT=${TRAJ_FORMAT_TYPO_CREDIT:-0.0}
export TRAJ_FORMAT_GRADED=${TRAJ_FORMAT_GRADED:-1}
export TRAJ_FORMAT_REJECTION=${TRAJ_FORMAT_REJECTION:-structural}

# ---- BoK routing (cold-start: keep all-wrong directional signal) ----
export BOK_TAU_INIT=${BOK_TAU_INIT:-0.7}
export BOK_TAU_FINAL=${BOK_TAU_FINAL:-0.4}
export BOK_CLIP=${BOK_CLIP:-4.0}
export BOK_ALLWRONG_CAP=${BOK_ALLWRONG_CAP:-0.5}
export BOK_ALLWRONG_NEG_ONLY=${BOK_ALLWRONG_NEG_ONLY:-0}
export BOK_EASY_THRESHOLD=${BOK_EASY_THRESHOLD:-0.50}
export BOK_SMART_FILTER_THRESHOLD=${BOK_SMART_FILTER_THRESHOLD:-0.955}

echo "================================================================"
echo "[V34-dense] ARM=${V34_ARM} (${ARM_TAG})"
echo "[V34-dense] base(SFT)=${MODEL_PATH}"
echo "[V34-dense] data=${STEPCOUNT_TRAIN_DATA}"
echo "[V34-dense] adv=${ADV_ESTIMATOR} process_reward=${PROCESS_REWARD_ENABLE} step_signal=${BOK_STEP_SIGNAL:-none}"
echo "[V34-dense] max_turns=${INTERLEAVED_MAX_TURNS} response_len=${V31_MAX_RESPONSE_LENGTH} lr=${ACTOR_LR} micro=${V31_MICRO_BATCH_UPDATE}/${V31_MICRO_BATCH_EXP}"
echo "[V34-dense] weights answer/point/format=${ANSWER_WEIGHT}/${POINT_WEIGHT}/${TRAJECTORY_FORMAT_WEIGHT}"
echo "[V34-dense] save=${V31_SAVE_CHECKPOINT_PATH}"
echo "================================================================"

if [[ "${V34_DRY_RUN:-0}" == "1" ]]; then
  export V32_DRY_RUN=1
fi

exec bash "${SCRIPT_DIR}/v32_sparse_0_10_stable_drfix.sh"
