#!/bin/bash
# ================================================================
# V35 Dense RL (11-20) = EM-ALIGNED fix of v34A's Goodhart regression.
# ================================================================
# Original-format launch (mirrors v34_dense_curriculum -> v32 exec chain):
# this thin layer sets the v35 params (per leo's final spec, referencing
# docs/v34_reports/V34_Next_Training_Recommendations_20260630.md + v34 reward
# distribution), then exec's the proven v34_dense_curriculum_11_20.sh
# (which exec's v32_sparse_0_10_stable_drfix.sh). Downstream wiring is reused.
#
# WHY v35 (docs/StepCount_Key_Findings F10 + V34_Training_Timing_Analysis):
# v34A RL REGRESSED exact-EM (11-20 13.35%->10.6%) + collapsed OOD (7.44->0.83)
# because partial-credit answer reward (WRONG_CAP 0.35/WITHIN1 0.45) + NEG_ONLY=0
# best-wrong reinforcement optimized CLOSENESS not EXACTNESS (Goodhart).
#
# v35 (smoke-verified exact-vs-close gap 0.55->0.85; BoK gradient toward exact at
# ~13% exact rate, not dead; 2-Opus-review GO):
#   - EM-align reward: WRONG_CAP 0.35->0.05, WITHIN1_CAP 0.45->0.12
#   - NEG_ONLY 0->1 (stop best-wrong); KL 0.03->0.06 (ref-anchor for base/OOD)
#   - GRAD-SAFETY (leo): keep skip-bad-update (params safe via dp_actor:355/441
#     unconditional zero_grad), but NEVER modify LR -> COOLDOWN=0 + LR_FACTOR=1.0
#     + BRAKE_MAX=1e6; absolute_cap=3.5 skips raw-grad>3.5 (no LR touch).
#   - reward distribution per v34: ANSWER/POINT/FORMAT = 0.7/0.2/0.1
#   - DAPO off (low-leverage in dense); GSPO-token IS; 1M-SFT base; BOK_TOTAL_STEPS=101.
# Checkpoints saved every 20 steps for post-hoc EM-gated selection.
#
# Usage (single-node debug):  V34_ARM=A bash examples/v35_dense_11_20.sh
# Multi-node 32-GPU launch:   examples/rl_launch/mn_trainer_dense_v35.sh A 4 256
# ================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_path_env.sh"

# ---- v35 base = 1M SFT (override via MODEL_PATH) ----
export MODEL_PATH=${MODEL_PATH:-${STEPCOUNT_1M_MODEL_PATH}}
export BOK_TOTAL_STEPS=${BOK_TOTAL_STEPS:-101}

# ---- reward distribution (per v34) ----
export ANSWER_WEIGHT=${ANSWER_WEIGHT:-0.7}
export POINT_WEIGHT=${POINT_WEIGHT:-0.2}
export TRAJECTORY_FORMAT_WEIGHT=${TRAJECTORY_FORMAT_WEIGHT:-0.1}

# ---- EM-align answer reward: narrow wrong-answer partial credit ----
export TRAJ_DENSE_CONTINUOUS_REWARD=${TRAJ_DENSE_CONTINUOUS_REWARD:-1}
export TRAJ_DENSE_CONTINUOUS_MIN_GT=${TRAJ_DENSE_CONTINUOUS_MIN_GT:-11}
export TRAJ_DENSE_CONTINUOUS_WRONG_CAP=${TRAJ_DENSE_CONTINUOUS_WRONG_CAP:-0.05}
export TRAJ_DENSE_CONTINUOUS_WITHIN1_CAP=${TRAJ_DENSE_CONTINUOUS_WITHIN1_CAP:-0.12}

# ---- format: strict-key (typo earns 0 format credit; parser still normalizes) ----
export TRAJ_FORMAT_STRICT_KEY=${TRAJ_FORMAT_STRICT_KEY:-1}
export TRAJ_FORMAT_TYPO_CREDIT=${TRAJ_FORMAT_TYPO_CREDIT:-0.0}
export TRAJ_FORMAT_GRADED=${TRAJ_FORMAT_GRADED:-1}
export TRAJ_FORMAT_REJECTION=${TRAJ_FORMAT_REJECTION:-structural}
export TRAJ_POINT_KEY_NORMALIZE=${TRAJ_POINT_KEY_NORMALIZE:-1}

# ---- distribution regularization + BoK all-wrong ----
export KL_COEF=${KL_COEF:-0.06}
export BOK_ALLWRONG_NEG_ONLY=${BOK_ALLWRONG_NEG_ONLY:-1}
export BOK_ALLWRONG_CAP=${BOK_ALLWRONG_CAP:-0.5}
export BOK_DAPO_FILTER=${BOK_DAPO_FILTER:-0}
export POLICY_LOSS_IS_LEVEL=${POLICY_LOSS_IS_LEVEL:-sequence_token}

# ---- soft answer-gate (canary baseline; unlisted by leo -> keep as canary, not v32 default off) ----
export TRAJ_ANSWER_GATE_MODE=${TRAJ_ANSWER_GATE_MODE:-soft}
export TRAJ_SOFT_GATE_BASE=${TRAJ_SOFT_GATE_BASE:-0.7}

# ---- grad-safety: skip bad updates (params safe) but NEVER modify LR (leo spec) ----
export GRAD_SPIKE_PROTECT=${GRAD_SPIKE_PROTECT:-1}
export GRAD_SPIKE_THRESHOLD=${GRAD_SPIKE_THRESHOLD:-3.0}
export GRAD_SPIKE_ABSOLUTE_CAP=${GRAD_SPIKE_ABSOLUTE_CAP:-3.5}
export GRAD_SPIKE_COOLDOWN=${GRAD_SPIKE_COOLDOWN:-0}
export GRAD_NONFINITE_COOLDOWN=${GRAD_NONFINITE_COOLDOWN:-0}
export GRAD_SPIKE_LR_FACTOR=${GRAD_SPIKE_LR_FACTOR:-1.0}
export GRAD_NONFINITE_LR_FACTOR=${GRAD_NONFINITE_LR_FACTOR:-1.0}
export GRAD_SPIKE_BRAKE_MAX=${GRAD_SPIKE_BRAKE_MAX:-1000000}
export GRAD_NONFINITE_BRAKE_MAX=${GRAD_NONFINITE_BRAKE_MAX:-1000000}

export V31_EXPERIMENT_NAME=${V31_EXPERIMENT_NAME:-StepCount-7B_v35_dense_11_20_A_emalign_$(date +%Y%m%d_%H%M)}

echo "================================================================"
echo "[V35-dense] reward w=${ANSWER_WEIGHT}/${POINT_WEIGHT}/${TRAJECTORY_FORMAT_WEIGHT} | WRONG_CAP=${TRAJ_DENSE_CONTINUOUS_WRONG_CAP} WITHIN1=${TRAJ_DENSE_CONTINUOUS_WITHIN1_CAP} NEG_ONLY=${BOK_ALLWRONG_NEG_ONLY} KL=${KL_COEF} steps=${BOK_TOTAL_STEPS}"
echo "[V35-dense] grad: thr=${GRAD_SPIKE_THRESHOLD} abs_cap=${GRAD_SPIKE_ABSOLUTE_CAP} cooldown=${GRAD_SPIKE_COOLDOWN}/${GRAD_NONFINITE_COOLDOWN} lr_factor=${GRAD_SPIKE_LR_FACTOR} brake_max=${GRAD_SPIKE_BRAKE_MAX} | DAPO=${BOK_DAPO_FILTER} IS=${POLICY_LOSS_IS_LEVEL}"
echo "[V35-dense] format: strict_key=${TRAJ_FORMAT_STRICT_KEY} typo_credit=${TRAJ_FORMAT_TYPO_CREDIT} graded=${TRAJ_FORMAT_GRADED} reject=${TRAJ_FORMAT_REJECTION} normalize=${TRAJ_POINT_KEY_NORMALIZE} | base=${MODEL_PATH}"
echo "================================================================"

exec bash "${SCRIPT_DIR}/v34_dense_curriculum_11_20.sh"
