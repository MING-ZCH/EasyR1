#!/bin/bash
# mn_trainer_dense_v35.sh ARM NNODES BATCH [MODEL_PATH] [BASETAG]
# =============================================================================
# v35 = EM-ALIGNED dense RL (fixes the v34A Goodhart: train-reward up but EM down + OOD collapse).
# Root cause (F10 + 2 Opus reviews): partial-credit answer reward (WRONG_CAP 0.35 / WITHIN1 0.45)
# + BOK_ALLWRONG_NEG_ONLY=0 best-wrong reinforcement => optimize CLOSENESS not EXACTNESS =>
# strong 1M base dragged toward "approximately right" => 11-20 EM 13.35->10.6, OOD 7.44->0.83.
#
# v35 deltas vs canary mn_trainer_dense.sh (everything else identical):
#   1) EM-align answer reward:  WRONG_CAP 0.35->0.05 , WITHIN1_CAP 0.45->0.12
#      (wrong answers earn ~0; correct soft_base 0.7 => large exact-vs-close gap; kills closeness-Goodhart.
#       Viable w/o dead-gradient: 1M base ~13% exact on 11-20, N=16 => ~2 exact/group => directional signal.)
#   2) BOK_ALLWRONG_NEG_ONLY 0->1 (stop reinforcing best-wrong; safe now that exact signal exists)
#   3) KL_COEF 0.03->0.06 (stronger ref-anchor to protect base + OOD generalization)
#   4) POLICY_LOSS_IS_LEVEL default sequence_token (GSPO-token, reviewed innocent; matches a2 for clean compare)
#   5) GRAD-SAFETY (leo req): KEEP skip-bad-update (dp_actor.py:355 nonfinite zero_grad+return is
#      UNCONDITIONAL = params always safe), but NEVER modify LR afterward -> LR_FACTOR=1.0 (no cooldown
#      reduction) + BRAKE_MAX=1e6 (no permanent LR halve). v34A's brake pinned LR->1e-7 for ~20 dead
#      steps; that brake treated a reward-Goodhart symptom. With v35's fixed reward, skip-only suffices.
#   DAPO (leo req, analyzed): BOK_DAPO_FILTER=0 (OFF). low_var ~6% in dense (max 10.9%); the low-var
#      FALLBACK (drgrpo centering) already handles homogeneous groups, so DAPO is redundant+low-leverage
#      and cannot fix entropy/nonfinite. v35's narrow wrong-cap may raise low-var (all-wrong cluster at
#      0.05), but the ~13%-exact LEARNABLE groups still drive the gradient -> keep OFF (optional ablation).
# KEEP: point-key normalize + structural format-rejection (no absorbing collapse), coverage/consistency
#       penalties (process integrity), graded format, grad/nonfinite safeguards (from v32).
# CKPTS: save_freq/val_freq=20 inherited (key ckpts at 20/40/60/80/100 for post-hoc EM-gated selection).
# DECISION (per review): monitor 11-20 EM + OOD EM on saved ckpts; if "stable but flat" (~=base, not
#   exceeding) => grounding ceiling (F9) is the binding constraint => next pivot = SFT grounding.
# =============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_DIR}/examples/local_path_env.sh"
ARM="${1:?usage: mn_trainer_dense_v35.sh ARM NNODES BATCH [MODEL_PATH] [BASETAG]}"
NNODES="${2:?nnodes}"
BATCH="${3:?batch}"
MODEL_PATH_ARG="${4:-${STEPCOUNT_1M_MODEL_PATH}}"
BASETAG="${5:-1M_v35}"
TS=$(date +%Y%m%d_%H%M)
LOGDIR=${EASYR1_LOG_ROOT}/rl
LOG="${LOGDIR}/v35_${ARM}_${NNODES}node_${BASETAG}_dopt_${TS}.log"
mkdir -p "$LOGDIR"
MODEL_ENV=""
[ -n "$MODEL_PATH_ARG" ] && MODEL_ENV="MODEL_PATH=${MODEL_PATH_ARG}"

export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/cusparselt/lib:/usr/local/cuda/lib64
EXPECT=$((NNODES*8))
GPU=$(ray status 2>/dev/null | grep -oP '(?<=/)[\d.]+(?=\s+GPU)' | head -1)
echo "[v35-trainer] cluster GPUs=${GPU:-?} expected=${EXPECT}"

export http_proxy=http://star-proxy.oa.com:3128 https_proxy=http://star-proxy.oa.com:3128
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_DIR=${WANDB_DIR:-${EASYR1_LOG_ROOT}/wandb}; mkdir -p "$WANDB_DIR"
cd "${REPO_DIR}"

setsid nohup env \
  ${MODEL_ENV} \
  INDEX=0 HOST_NUM="$NNODES" HOST_GPU_NUM=8 \
  V34_ARM="$ARM" V31_NNODES="$NNODES" V31_N_GPUS_PER_NODE=8 \
  V31_ROLLOUT_BATCH_SIZE="$BATCH" V31_GLOBAL_BATCH_SIZE="$BATCH" V31_VAL_BATCH_SIZE="$BATCH" \
  V31_MICRO_BATCH_UPDATE=4 V31_MICRO_BATCH_EXP=4 V31_GPU_MEM_UTIL=0.60 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  EASYR1_VLLM_NUM_GPU_BLOCKS=${EASYR1_VLLM_NUM_GPU_BLOCKS:-16384} \
  ACTOR_LR=1e-6 KL_COEF=0.06 ROLLOUT_TEMPERATURE=1.0 BOK_TOTAL_STEPS=101 \
  ANSWER_WEIGHT=0.7 POINT_WEIGHT=0.2 TRAJECTORY_FORMAT_WEIGHT=0.1 \
  TRAJ_FORMAT_GRADED=1 TRAJ_FORMAT_REJECTION=structural TRAJ_POINT_KEY_NORMALIZE=1 \
  TRAJ_ANSWER_GATE_MODE=soft TRAJ_SOFT_GATE_BASE=0.7 \
  TRAJ_DENSE_CONTINUOUS_REWARD=1 TRAJ_DENSE_CONTINUOUS_MIN_GT=11 \
  TRAJ_DENSE_CONTINUOUS_WRONG_CAP=0.05 TRAJ_DENSE_CONTINUOUS_WITHIN1_CAP=0.12 \
  TRAJ_COVERAGE_PENALTY_ENABLE=1 TRAJ_COVERAGE_PENALTY_PER_MISS=0.1 TRAJ_COVERAGE_PENALTY_CAP=0.3 \
  TRAJ_STOP_NONE_ENABLE=1 TRAJ_EARLY_STOP_PENALTY=0.2 TRAJ_STOP_NONE_BONUS=0.05 \
  POLICY_LOSS_IS_LEVEL=${POLICY_LOSS_IS_LEVEL:-sequence_token} \
  BOK_CLIP=4.0 BOK_ALLWRONG_NEG_ONLY=1 BOK_ALLWRONG_CAP=0.5 \
  BOK_TAU_INIT=0.7 BOK_TAU_FINAL=0.4 BOK_SMART_FILTER_THRESHOLD=0.955 \
  BOK_STEP_WEIGHT=0.2 BOK_PROGRESS_DUP_PENALTY=1.0 BOK_STEP_GATE=answer_soft BOK_STEP_MIN_GATE=0.2 \
  GRAD_SPIKE_PROTECT=1 GRAD_SPIKE_THRESHOLD=3.0 GRAD_SPIKE_ABSOLUTE_CAP=3.5 \
  GRAD_SPIKE_COOLDOWN=0 GRAD_NONFINITE_COOLDOWN=0 \
  GRAD_SPIKE_LR_FACTOR=1.0 GRAD_NONFINITE_LR_FACTOR=1.0 \
  GRAD_SPIKE_BRAKE_MAX=1000000 GRAD_NONFINITE_BRAKE_MAX=1000000 \
  BOK_DAPO_FILTER=0 \
  WANDB_MODE=online WANDB_API_KEY="$WANDB_API_KEY" WANDB_DIR="$WANDB_DIR" \
  RAY_worker_register_timeout_seconds=120 \
  http_proxy="$http_proxy" https_proxy="$https_proxy" \
  bash examples/v35_dense_11_20.sh > "$LOG" 2>&1 &
sleep 3
echo "[v35-trainer] LAUNCHED ARM=$ARM NNODES=$NNODES BATCH=$BATCH LOG=$LOG"
