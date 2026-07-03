#!/bin/bash
# ================================================================
# V27b Launch Script — StepCount-7B Interleaved Point-to-Count
# ================================================================
# V27b = v26 + targeted fixes based on root-cause analysis
#
# ROOT CAUSE (v26/v25/v12fix all崩溃的统一根因):
#   _parse_pred_point regex兜底允许破损JSON获得与合法JSON相同的point_reward
#   → 模型发现"省token" shortcut → 策略漂移 → format_reward崩溃
#
# Targeted fixes vs v27/v26:
#   1. TRAJ_POINT_STRICT_JSON=1
#   2. GRAD_SPIKE_THRESHOLD=3.0 (v26用3.5)
#      → v22/v23/v24: threshold=3.0, NaN=0, format稳定
#      → v21/v25/v26: threshold=3.5, format漂移更快
#   3. 移除GRAD_SPIKE_ABSOLUTE_CAP (v26用2.5)
#      → v26 abs_cap=2.5过激进，反而加速format崩溃
#      → v22-v24: 无abs_cap且最稳定（NaN=0全程）
#   4. reward: answer = 0.6, point = 0.3, format = 0.1
# UNCHANGED from v26:
#   - ACTOR_LR=1.5e-6  (20260427_071318 run: stable to step 106, no nonfinite gradients)
#   - BOK_CLIP=4.0      (keep the BoK winner signal; GradSpike cooldown catches rare outliers)
#   - BOK_SMART_FILTER_THRESHOLD=0.965
#     (raise from 0.955 to keep answer-correct groups with point_mean < ~0.883 learnable)
#   - ppo_epochs=1, kl=0.03, max_grad_norm=1.0
#
# DATA: easy-only (default) | easy_plus_hard (via env var)
#   V27b探索: easy-only数据 vs v27 easy+hard，对比数据影响
#   Easy+hard 13641样本：v22-v25全部使用，峰值更高(v25 0.7826)，pixmo-test对齐
#   Easy-only 11455样本：更保守，峰值约0.7750(v12)
#   切换: STEPCOUNT_TRAIN_DATA=.../StepCountQA-RL-Traj_0_10 bash v27.sh
#
# EXPECTED (with STRICT JSON fix):
#   - format_reward >= 0.97 全程不崩溃           HIGH confidence
#   - peak val_ans >= 0.7750 with point-quality headroom MEDIUM confidence
#   - NaN = 0                                      HIGH (threshold=3.0x proved by v22-v24)
#
# NO RESUME — SFT Cold Start
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# v25: Resume from V21-R2 step-60 checkpoint (same as V24)
# # v21_RESUME_CKPT=""


LOG_DIR="/data/workspace/hyleochang/EasyR1-latest/logs/train"
mkdir -p ${LOG_DIR}
export output_path='./'
export ckpt_path='./ckpt'

export PYTHONUNBUFFERED=1
# WANDB_API_KEY should be provided by the runtime environment; do not hard-code secrets.
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16
export PYTHONHASHSEED=0
export TRAIN_SAVE_FREQ=${TRAIN_SAVE_FREQ:-20}
export TRAIN_SAVE_LIMIT=${TRAIN_SAVE_LIMIT:-8}
export TRAIN_VAL_FREQ=${TRAIN_VAL_FREQ:-20}

# ── V25 NEW: Weighted sampling for data distribution control ──
# Two independent weight multipliers (multiplicative):
#   1) No-mask oversample: 567 no-mask samples get 3x weight (4.2% → 11.5%)
#   2) Hard oversample: 3809 hard-matched samples get Nx weight
# Set factor=1 to disable either one.
export TRAIN_OVERSAMPLE_NO_MASK_FACTOR=${TRAIN_OVERSAMPLE_NO_MASK_FACTOR:-1}
export TRAIN_HARD_OVERSAMPLE_FACTOR=${TRAIN_HARD_OVERSAMPLE_FACTOR:-1}  # Config A: disabled (easy-only, no hard oversample)
export TRAIN_HARD_REFERENCE_PATH=${TRAIN_HARD_REFERENCE_PATH:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCountQA-RL-Traj_0_10_hard_only/data}

# ── V25 NEW: stat_mask_sim fallback for no-mask samples ──
# Replaces count_iou (inflated reward) with statistical approximation (error <0.005)
export TRAJ_NO_SEQUENCE_FALLBACK=${TRAJ_NO_SEQUENCE_FALLBACK:-stat_mask_sim}

# StepCount mask reward settings
export STEPCOUNT_MASKS_METADATA=${STEPCOUNT_MASKS_METADATA:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCount-RL_Masks-Sharded/extracted/masks_metadata.json}
export STEPCOUNT_MASKS_DIR=${STEPCOUNT_MASKS_DIR:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCount-RL_Masks-Sharded/extracted/masks}
export STEPCOUNT_MASK_REQUIRE=${STEPCOUNT_MASK_REQUIRE:-1}
export STEPCOUNT_MASK_PREFILL_BY_TURN=${STEPCOUNT_MASK_PREFILL_BY_TURN:-1}
export STEPCOUNT_MASK_LOG_CONFIG=${STEPCOUNT_MASK_LOG_CONFIG:-1}
export STEPCOUNT_MASK_DEBUG=${STEPCOUNT_MASK_DEBUG:-1}
export STEPCOUNT_MASK_DEBUG_EVERY=${STEPCOUNT_MASK_DEBUG_EVERY:-10}

export EASYR1_REWARD_DEBUG_EVERY=${EASYR1_REWARD_DEBUG_EVERY:-10}
export EASYR1_REWARD_SAMPLE_DEBUG=${EASYR1_REWARD_SAMPLE_DEBUG:-1}
export EASYR1_REWARD_SAMPLE_DEBUG_MAX=${EASYR1_REWARD_SAMPLE_DEBUG_MAX:-1}
export EASYR1_REWARD_HEALTH_DEBUG=${EASYR1_REWARD_HEALTH_DEBUG:-1}
export EASYR1_REWARD_HEALTH_DEBUG_EVERY=${EASYR1_REWARD_HEALTH_DEBUG_EVERY:-1}
export STEPCOUNT_TRAJ_REASON_DEBUG=${STEPCOUNT_TRAJ_REASON_DEBUG:-1}
export STEPCOUNT_TRAJ_REASON_DEBUG_EVERY=${STEPCOUNT_TRAJ_REASON_DEBUG_EVERY:-10}
export STEPCOUNT_TRAJ_EVENT_LOG=${STEPCOUNT_TRAJ_EVENT_LOG:-1}
export STEPCOUNT_TRAJ_EVENT_LOG_EVERY=${STEPCOUNT_TRAJ_EVENT_LOG_EVERY:-10}
export STEPCOUNT_TRAJ_EVENT_LOG_MAX=${STEPCOUNT_TRAJ_EVENT_LOG_MAX:-50}

export STEPCOUNT_FORCE_TRAJECTORY_FOR_NUMERIC_GT=${STEPCOUNT_FORCE_TRAJECTORY_FOR_NUMERIC_GT:-1}

# Interleaved rollout knobs
export ROLLOUT_N=${ROLLOUT_N:-16}
export INTERLEAVED_MAX_TURNS=${INTERLEAVED_MAX_TURNS:-11}
export INTERLEAVED_PER_TURN_MAX_TOKENS=${INTERLEAVED_PER_TURN_MAX_TOKENS:-2000}
export INTERLEAVED_ANSWER_TURN_MAX_TOKENS=${INTERLEAVED_ANSWER_TURN_MAX_TOKENS:-2000}
export INTERLEAVED_FIRST_TURN_PROMPT_FILE=${INTERLEAVED_FIRST_TURN_PROMPT_FILE:-}
export INTERLEAVED_PROCESS_PROMPT_FILE=${INTERLEAVED_PROCESS_PROMPT_FILE:-${PROJECT_ROOT}/examples/format_prompt/StepCount_interleaved_process_prompt.txt}
export SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-${PROJECT_ROOT}/examples/format_prompt/StepCount_interleaved_system_prompt.txt}
export INTERLEAVED_HISTORY_MODE=${INTERLEAVED_HISTORY_MODE:-0}
export INTERLEAVED_DEBUG=${INTERLEAVED_DEBUG:-1}
export INTERLEAVED_DEBUG_PRINT_CHARS=${INTERLEAVED_DEBUG_PRINT_CHARS:-50}
export TRAJ_RETURN_POINT_STEP_SCORES=${TRAJ_RETURN_POINT_STEP_SCORES:-1}
export TRAJ_MISS_DECAY_ENABLE=${TRAJ_MISS_DECAY_ENABLE:-1}
export TRAJ_CONSISTENCY_PENALTY=${TRAJ_CONSISTENCY_PENALTY:-0.5}
export TRAJ_ANSWER_GATE_THRESHOLD=${TRAJ_ANSWER_GATE_THRESHOLD:-0.4}
export TRAJ_EVAL_ANSWER_ONLY_ON_NO_MASK=${TRAJ_EVAL_ANSWER_ONLY_ON_NO_MASK:-1}
export TRAJ_ANSWER_GATE_MODE=${TRAJ_ANSWER_GATE_MODE:-off}
export TRAJ_FORMAT_REJECTION=${TRAJ_FORMAT_REJECTION:-1}
export TRAJ_POINT_STRICT_JSON=${TRAJ_POINT_STRICT_JSON:-1}  # Strict JSON parser; format rejection zeros malformed trajectories and this keeps point diagnostics aligned
export TRAJ_SOFT_ANSWER_DECAY=${TRAJ_SOFT_ANSWER_DECAY:-1}
export TRAJ_ANSWER_DECAY_ALPHA=${TRAJ_ANSWER_DECAY_ALPHA:-8.0}
export TRAJ_ANSWER_DECAY_CAP=${TRAJ_ANSWER_DECAY_CAP:-0.4}
export TRAJ_EXTRA_POINT_PENALTY_LAMBDA=${TRAJ_EXTRA_POINT_PENALTY_LAMBDA:-1.0}
export TRAJ_UNDER_ALPHA_GT_SCALE=${TRAJ_UNDER_ALPHA_GT_SCALE:-0.5}
export TRAJ_UNDER_ALPHA_GT_THRESHOLD=${TRAJ_UNDER_ALPHA_GT_THRESHOLD:-5}

# ===================== RL mode: bok_grpo =====================
export STEPCOUNT_RL_MODE=${STEPCOUNT_RL_MODE:-bok_grpo}

ADV_ESTIMATOR=${ADV_ESTIMATOR:-}
PROCESS_REWARD_ENABLE=${PROCESS_REWARD_ENABLE:-}
ACTOR_LR=${ACTOR_LR:-}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-}
CLIP_RATIO_DUAL=${CLIP_RATIO_DUAL:-}
DISABLE_KL=${DISABLE_KL:-}

case "${STEPCOUNT_RL_MODE}" in
  bok_grpo)
    [[ -z "${ADV_ESTIMATOR}" ]] && ADV_ESTIMATOR="bok_grpo"
    [[ -z "${PROCESS_REWARD_ENABLE}" ]] && PROCESS_REWARD_ENABLE="0"
    [[ -z "${ACTOR_LR}" ]] && ACTOR_LR="1.5e-6"           # V27b restart: do not raise to 2e-6; step-106 log already had two >4x spikes at 1.5e-6
    [[ -z "${ROLLOUT_TEMPERATURE}" ]] && ROLLOUT_TEMPERATURE="1.0"
    [[ -z "${CLIP_RATIO_LOW}" ]] && CLIP_RATIO_LOW="0.2"
    [[ -z "${CLIP_RATIO_HIGH}" ]] && CLIP_RATIO_HIGH="0.28"
    [[ -z "${CLIP_RATIO_DUAL}" ]] && CLIP_RATIO_DUAL="3.0"
    [[ -z "${DISABLE_KL}" ]] && DISABLE_KL="false"

    # ── τ annealing ──
    export BOK_TAU=${BOK_TAU:-0.5}
    export BOK_CLIP=${BOK_CLIP:-4.0}              # V26: v23_SC anchor (NaN=0 + peak 0.7769)
    export BOK_UNIFORM_MIX=${BOK_UNIFORM_MIX:-0.1}
    export BOK_TAU_INIT=${BOK_TAU_INIT:-0.7}
    export BOK_TAU_FINAL=${BOK_TAU_FINAL:-0.3}
    export BOK_TOTAL_STEPS=${BOK_TOTAL_STEPS:-179}  # V27b: easy-only 11455/32≈358 (batch=32 for 195GB RAM); set 179 for batch=64
    export BOK_TAU_SCHEDULE=${BOK_TAU_SCHEDULE:-cosine}
    export BOK_ADV_NORMALIZE=${BOK_ADV_NORMALIZE:-0}
    export BOK_LOW_VAR_THRESHOLD=${BOK_LOW_VAR_THRESHOLD:-1e-5}
    export BOK_FALLBACK_MODE=${BOK_FALLBACK_MODE:-drgrpo}
    export BOK_DAPO_FILTER=${BOK_DAPO_FILTER:-0}
    export BOK_DAPO_AUTO_DISABLE_THRESHOLD=${BOK_DAPO_AUTO_DISABLE_THRESHOLD:-0.5}
    export BOK_MIN_BATCH_STD=${BOK_MIN_BATCH_STD:-0.1}

    # ── Difficulty-Aware Routing ──
    export BOK_EASY_THRESHOLD=${BOK_EASY_THRESHOLD:-0.50}
    export BOK_EASY_SCORE_THRESHOLD=${BOK_EASY_SCORE_THRESHOLD:-0.5}
    export BOK_FILTER_ALL_CORRECT=${BOK_FILTER_ALL_CORRECT:-1}
    export BOK_SMART_FILTER_THRESHOLD=${BOK_SMART_FILTER_THRESHOLD:-0.965}  # V27b restart: release more AC groups with still-improvable point quality

    # ── AllWrong Cap ──
    export BOK_ALLWRONG_CAP=${BOK_ALLWRONG_CAP:-1.0}
    export BOK_ALLWRONG_ANSWER_THRESHOLD=${BOK_ALLWRONG_ANSWER_THRESHOLD:-0.5}
    # V25 NEW: all-wrong groups only penalize, never reward wrong answers
    export BOK_ALLWRONG_NEG_ONLY=${BOK_ALLWRONG_NEG_ONLY:-0}   # disabled (V25 bug: never took effect)

    # ── V25 NEW: Winner Amplification ──
    # Boost rare-correct advantages by min(boost, sqrt(K/n_correct))
    # 1.5 = moderate (at 2/16 correct: boost=min(1.5, sqrt(8))=1.5; at 8/16: sqrt(2)≈1.41)
    export BOK_WINNER_BOOST=${BOK_WINNER_BOOST:-0}              # disabled (V25 bug: never took effect)

    # ── Easy Gradient Dampening & Quality Bonus ──
    export BOK_EASY_SCALE=${BOK_EASY_SCALE:-1.0}
    export BOK_QUALITY_BONUS=${BOK_QUALITY_BONUS:-0}

    # ── Grad Norm Spike Protection ──
    export GRAD_SPIKE_PROTECT=${GRAD_SPIKE_PROTECT:-1}
    export GRAD_SPIKE_THRESHOLD=${GRAD_SPIKE_THRESHOLD:-3.0}  # V27: 3.5→3.0 (v22-v24 proved 3.0 is most stable, 0 NaN)
    export GRAD_SPIKE_COOLDOWN=${GRAD_SPIKE_COOLDOWN:-3}
    export GRAD_SPIKE_LR_FACTOR=${GRAD_SPIKE_LR_FACTOR:-0.1}
    export GRAD_SPIKE_BRAKE_WINDOW=${GRAD_SPIKE_BRAKE_WINDOW:-40}
    export GRAD_SPIKE_BRAKE_MAX=${GRAD_SPIKE_BRAKE_MAX:-999}
    export GRAD_NONFINITE_COOLDOWN=${GRAD_NONFINITE_COOLDOWN:-10}
    export GRAD_NONFINITE_LR_FACTOR=${GRAD_NONFINITE_LR_FACTOR:-0.1}
    export GRAD_NONFINITE_BRAKE_WINDOW=${GRAD_NONFINITE_BRAKE_WINDOW:-40}
    export GRAD_NONFINITE_BRAKE_MAX=${GRAD_NONFINITE_BRAKE_MAX:-999}
    # GRAD_SPIKE_ABSOLUTE_CAP: REMOVED in V27 (v22-v24无abs_cap且NaN=0最稳定；v26 2.5过激进加速崩溃)
    export FP16_GRAD_UNDERFLOW_MONITOR=${FP16_GRAD_UNDERFLOW_MONITOR:-0}

    # ── Fail-fast monitor thresholds ──
    export V27B_FAILFAST_ENABLE=${V27B_FAILFAST_ENABLE:-0}
    export V27B_FAILFAST_NAN_LIMIT=${V27B_FAILFAST_NAN_LIMIT:-50}
    export V27B_FAILFAST_ENTROPY_CRITICAL=${V27B_FAILFAST_ENTROPY_CRITICAL:-0.85}
    export V27B_FAILFAST_ENTROPY_CONSEC=${V27B_FAILFAST_ENTROPY_CONSEC:-3}
    export V27B_FAILFAST_FORMAT_CRITICAL=${V27B_FAILFAST_FORMAT_CRITICAL:-0.30}
    export V27B_FAILFAST_FORMAT_CONSEC=${V27B_FAILFAST_FORMAT_CONSEC:-3}
    export V27B_FAILFAST_LOWVAR_CRITICAL=${V27B_FAILFAST_LOWVAR_CRITICAL:-45}
    export V27B_FAILFAST_ZEROREWARD_CRITICAL=${V27B_FAILFAST_ZEROREWARD_CRITICAL:-30}
    export V27B_FAILFAST_BOK_CONSEC=${V27B_FAILFAST_BOK_CONSEC:-2}
    ;;
  *)
    echo "[FATAL] V27 only supports bok_grpo mode. Got: ${STEPCOUNT_RL_MODE}" >&2
    exit 1
    ;;
esac

# Reward weights (V12 answer-margin baseline; format safety is enforced by TRAJ_FORMAT_REJECTION=1)
export ANSWER_WEIGHT=${ANSWER_WEIGHT:-0.6}
export POINT_WEIGHT=${POINT_WEIGHT:-0.3}
export TRAJECTORY_FORMAT_WEIGHT=${TRAJECTORY_FORMAT_WEIGHT:-0.1}

export ADV_ESTIMATOR
export PROCESS_REWARD_ENABLE
export ACTOR_LR
export ROLLOUT_TEMPERATURE
export CLIP_RATIO_LOW
export CLIP_RATIO_HIGH
export CLIP_RATIO_DUAL
export DISABLE_KL

MODEL_PATH=/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/models/StepCount-7B-SFT-30k-high/checkpoint-3537
CONFIG_PATH="${PROJECT_ROOT}/examples/config.yaml"
REWARD_FN_PATH="${PROJECT_ROOT}/examples/reward_function/StepCount_mask_reward.py:compute_score"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[FATAL] config not found: ${CONFIG_PATH}" >&2
    exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/examples/reward_function/StepCount_mask_reward.py" ]]; then
    echo "[FATAL] reward function file not found" >&2
    exit 1
fi
if [[ -n "${INTERLEAVED_PROCESS_PROMPT_FILE}" && ! -f "${INTERLEAVED_PROCESS_PROMPT_FILE}" ]]; then
    echo "[FATAL] interleaved process prompt file not found: ${INTERLEAVED_PROCESS_PROMPT_FILE}" >&2
    exit 1
fi
if [[ -n "${SYSTEM_PROMPT_FILE}" && ! -f "${SYSTEM_PROMPT_FILE}" ]]; then
    echo "[FATAL] system prompt file not found: ${SYSTEM_PROMPT_FILE}" >&2
    exit 1
fi
for required_path in \
    "${MODEL_PATH}" \
    "${STEPCOUNT_TRAIN_DATA:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCountQA-RL-Traj_0_10}" \
    "/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/pixmo-test" \
    "${STEPCOUNT_MASKS_METADATA}" \
    "${STEPCOUNT_MASKS_DIR}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[FATAL] required path not found: ${required_path}" >&2
        exit 1
    fi
done

RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE="${LOG_DIR}/training_interleaved_traj_v27b_easy_data_${STEPCOUNT_RL_MODE}_${ADV_ESTIMATOR}_${RUN_TS}.log"

echo "================================================================"
echo "[V27b RunConfig] H200 easy-only lr=1.5e-6 batch=64 gpu_util=0.65 (RAM fix)"
echo "================================================================"
echo "[RunConfig] model=${MODEL_PATH}"
echo "[RunConfig] mode=${STEPCOUNT_RL_MODE} adv=${ADV_ESTIMATOR} lr=${ACTOR_LR} max_grad_norm=1.0"
echo "[RunConfig] clip_low=${CLIP_RATIO_LOW} clip_high=${CLIP_RATIO_HIGH} ppo_epochs=1 kl=0.03"
echo "[RunConfig] no_mask_oversample=${TRAIN_OVERSAMPLE_NO_MASK_FACTOR} hard_oversample=${TRAIN_HARD_OVERSAMPLE_FACTOR} no_seq_fallback=${TRAJ_NO_SEQUENCE_FALLBACK}"
echo "[RunConfig] winner_boost=${BOK_WINNER_BOOST} allwrong_neg_only=${BOK_ALLWRONG_NEG_ONLY}"
echo "[RunConfig] easy_threshold=${BOK_EASY_THRESHOLD} smart_filter=${BOK_SMART_FILTER_THRESHOLD}"
echo "[RunConfig] GradSpike threshold=${GRAD_SPIKE_THRESHOLD} cooldown_lr_factor=${GRAD_SPIKE_LR_FACTOR} spike_brake_max=${GRAD_SPIKE_BRAKE_MAX} nonfinite_brake_max=${GRAD_NONFINITE_BRAKE_MAX}"
echo "[RunConfig] FailFast enable=${V27B_FAILFAST_ENABLE} entropy>${V27B_FAILFAST_ENTROPY_CRITICAL}x${V27B_FAILFAST_ENTROPY_CONSEC} format>${V27B_FAILFAST_FORMAT_CRITICAL}x${V27B_FAILFAST_FORMAT_CONSEC} nan_limit=${V27B_FAILFAST_NAN_LIMIT}"
echo "[RunConfig] answer_w=${ANSWER_WEIGHT} point_w=${POINT_WEIGHT} format_w=${TRAJECTORY_FORMAT_WEIGHT}"
echo "[RunConfig] resume=NONE (SFT cold start)"
echo "[RunConfig] save_freq=${TRAIN_SAVE_FREQ} save_limit=${TRAIN_SAVE_LIMIT} val_freq=${TRAIN_VAL_FREQ}"
echo "[RunConfig] log=${LOG_FILE}"
echo "================================================================"

if [[ "${V27B_DRY_RUN:-0}" == "1" ]]; then
    echo "[DryRun] V27b dry-run requested; validated paths and printed launch env. Exiting before monitor/trainer startup."
    echo "[DryRun] train_data=${STEPCOUNT_TRAIN_DATA:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCountQA-RL-Traj_0_10}"
    echo "[DryRun] process_prompt=${INTERLEAVED_PROCESS_PROMPT_FILE}"
    echo "[DryRun] system_prompt=${SYSTEM_PROMPT_FILE}"
    echo "[DryRun] strict_json=${TRAJ_POINT_STRICT_JSON} format_rejection=${TRAJ_FORMAT_REJECTION}"
    echo "[DryRun] lr=${ACTOR_LR} ppo_epochs=1 kl=0.03 smart_filter=${BOK_SMART_FILTER_THRESHOLD}"
    echo "[DryRun] save_freq=${TRAIN_SAVE_FREQ} save_limit=${TRAIN_SAVE_LIMIT} val_freq=${TRAIN_VAL_FREQ}"
    echo "[DryRun] grad_spike_threshold=${GRAD_SPIKE_THRESHOLD} cooldown_lr_factor=${GRAD_SPIKE_LR_FACTOR} spike_brake_max=${GRAD_SPIKE_BRAKE_MAX}"
    echo "[DryRun] nonfinite_cooldown=${GRAD_NONFINITE_COOLDOWN} nonfinite_lr_factor=${GRAD_NONFINITE_LR_FACTOR} nonfinite_brake_max=${GRAD_NONFINITE_BRAKE_MAX}"
    echo "[DryRun] failfast_enable=${V27B_FAILFAST_ENABLE} entropy>${V27B_FAILFAST_ENTROPY_CRITICAL}x${V27B_FAILFAST_ENTROPY_CONSEC} format>${V27B_FAILFAST_FORMAT_CRITICAL}x${V27B_FAILFAST_FORMAT_CONSEC} nan_limit=${V27B_FAILFAST_NAN_LIMIT}"
    exit 0
fi

# ── Training Monitor + fail-fast guard ──
MONITOR_SCRIPT="${PROJECT_ROOT}/tools/monitor_training_v2.py"
FAILFAST_SCRIPT="${PROJECT_ROOT}/tools/monitor_v28_failfast.py"
MONITOR_DIR="${LOG_DIR}/../monitor"
mkdir -p "${MONITOR_DIR}"
MONITOR_LOG="${MONITOR_DIR}/monitor_v27b_${RUN_TS}.log"
MONITOR_JSON="${MONITOR_DIR}/metrics_v27b_${RUN_TS}.jsonl"
FAILFAST_LOG="${MONITOR_DIR}/failfast_v27b_${RUN_TS}.log"
export STOP_FILE="/tmp/v27b_failfast_stop_${RUN_TS}"
rm -f "${STOP_FILE}"

if [[ -f "${MONITOR_SCRIPT}" ]]; then
    echo "[Monitor] Starting V2 monitor: ${MONITOR_LOG}"
    nohup python3 "${MONITOR_SCRIPT}" \
        --log "${LOG_FILE}" \
        --out "${MONITOR_LOG}" \
        --stop-file "${STOP_FILE}" \
        --interval 60 \
        --every 5 \
        --summary-every 20 \
        --json-metrics "${MONITOR_JSON}" \
        > /dev/null 2>"${MONITOR_DIR}/monitor_v27b_stderr_${RUN_TS}.log" &
    MONITOR_PID=$!
    echo "[Monitor] PID=${MONITOR_PID} stop_file=${STOP_FILE}"
else
    echo "[Monitor] Script not found: ${MONITOR_SCRIPT}, skipping"
    MONITOR_PID=""
fi

if [[ "${V27B_FAILFAST_ENABLE}" == "1" && -f "${FAILFAST_SCRIPT}" ]]; then
    echo "[FailFast] Starting fail-fast monitor: ${FAILFAST_LOG}"
    nohup python3 "${FAILFAST_SCRIPT}" \
        --run-name V27b \
        --log "${LOG_FILE}" \
        --out "${FAILFAST_LOG}" \
        --stop-file "${STOP_FILE}" \
        --nan-limit "${V27B_FAILFAST_NAN_LIMIT}" \
        --entropy-critical "${V27B_FAILFAST_ENTROPY_CRITICAL}" \
        --entropy-consec "${V27B_FAILFAST_ENTROPY_CONSEC}" \
        --format-critical "${V27B_FAILFAST_FORMAT_CRITICAL}" \
        --format-consec "${V27B_FAILFAST_FORMAT_CONSEC}" \
        --lowvar-critical "${V27B_FAILFAST_LOWVAR_CRITICAL}" \
        --zeroreward-critical "${V27B_FAILFAST_ZEROREWARD_CRITICAL}" \
        --bok-consec "${V27B_FAILFAST_BOK_CONSEC}" \
        > /dev/null 2>"${MONITOR_DIR}/failfast_v27b_stderr_${RUN_TS}.log" &
    FAILFAST_PID=$!
    echo "[FailFast] PID=${FAILFAST_PID}"
else
    FAILFAST_PID=""
fi
STOP_WATCH_PID=""
cleanup_monitors() {
    if [[ -n "${MONITOR_PID:-}" ]]; then kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; fi
    if [[ -n "${FAILFAST_PID:-}" ]]; then kill "${FAILFAST_PID}" 2>/dev/null || true; wait "${FAILFAST_PID}" 2>/dev/null || true; fi
    if [[ -n "${STOP_WATCH_PID:-}" ]]; then kill "${STOP_WATCH_PID}" 2>/dev/null || true; wait "${STOP_WATCH_PID}" 2>/dev/null || true; fi
}
trap cleanup_monitors EXIT

python3 -m verl.trainer.main \
    config=${CONFIG_PATH} \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.disable_kl=${DISABLE_KL} \
    algorithm.kl_coef=3e-2 \
    data.train_files=${STEPCOUNT_TRAIN_DATA:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCountQA-RL-Traj_0_10} \
    data.system_prompt_file=${SYSTEM_PROMPT_FILE} \
    data.format_prompt=null \
    data.val_files=/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/pixmo-test \
    data.max_prompt_length=7500 \
    data.max_response_length=3200 \
    data.shuffle=true \
    worker.actor.optim.lr=${ACTOR_LR} \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    worker.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    worker.actor.clip_ratio_dual=${CLIP_RATIO_DUAL} \
    worker.actor.ppo_epochs=1 \
    worker.actor.max_grad_norm=1.0 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.rollout.gpu_memory_utilization=0.65 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    worker.rollout.stop='["</answer>"]' \
    worker.rollout.interleaved_point_to_count=true \
    worker.rollout.interleaved_max_turns=${INTERLEAVED_MAX_TURNS} \
    worker.rollout.interleaved_per_turn_max_tokens=${INTERLEAVED_PER_TURN_MAX_TOKENS} \
    worker.rollout.interleaved_answer_turn_max_tokens=${INTERLEAVED_ANSWER_TURN_MAX_TOKENS} \
    worker.rollout.interleaved_history_mode=${INTERLEAVED_HISTORY_MODE} \
    worker.rollout.interleaved_first_turn_prompt_file=${INTERLEAVED_FIRST_TURN_PROMPT_FILE} \
    worker.rollout.interleaved_process_prompt_file=${INTERLEAVED_PROCESS_PROMPT_FILE} \
    worker.rollout.interleaved_stop_tag='</answer>' \
    worker.rollout.interleaved_debug=${INTERLEAVED_DEBUG} \
    worker.rollout.interleaved_debug_print_chars=${INTERLEAVED_DEBUG_PRINT_CHARS} \
    worker.rollout.auto_protect_min_n=8 \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=${REWARD_FN_PATH} \
    worker.reward.reward_function_kwargs.answer_weight=${ANSWER_WEIGHT} \
    worker.reward.reward_function_kwargs.point_weight=${POINT_WEIGHT} \
    worker.reward.reward_function_kwargs.trajectory_format_weight=${TRAJECTORY_FORMAT_WEIGHT} \
    worker.reward.reward_function_kwargs.max_turns=${INTERLEAVED_MAX_TURNS} \
    trainer.experiment_name=StepCount-7B-SFT-30k_v27b_easy_data_${ADV_ESTIMATOR}_$(date +%Y%m%d_%H%M) \
    trainer.logger=['console','wandb'] \
    trainer.save_checkpoint_path=/data/workspace/hyleochang/EasyR1-latest/save/StepCount-7B-SFT-30k_v27b_easy_data_${ADV_ESTIMATOR}_$(date +%Y%m%d_%H%M) \
    trainer.total_epochs=${STEPCOUNT_TOTAL_EPOCHS:-1} \
    trainer.save_freq=${TRAIN_SAVE_FREQ} \
    trainer.save_limit=${TRAIN_SAVE_LIMIT} \
    trainer.val_freq=${TRAIN_VAL_FREQ} \
    data.rollout_batch_size=64 \
    data.val_batch_size=64 \
    worker.actor.global_batch_size=64 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    trainer.n_gpus_per_node=4 > >(tee "${LOG_FILE}") 2>&1 &


TRAIN_PID=$!
if [[ "${V27B_FAILFAST_ENABLE}" == "1" ]]; then
    (
        while kill -0 "${TRAIN_PID}" 2>/dev/null; do
            if [[ -f "${STOP_FILE}" ]]; then
                echo "[V27b FailFast] STOP file detected. Terminating trainer pid=${TRAIN_PID}." | tee -a "${LOG_FILE}"
                kill -TERM "${TRAIN_PID}" 2>/dev/null || true
                for _ in {1..20}; do
                    kill -0 "${TRAIN_PID}" 2>/dev/null || exit 0
                    sleep 1
                done
                kill -KILL "${TRAIN_PID}" 2>/dev/null || true
                exit 0
            fi
            sleep 15
        done
    ) &
    STOP_WATCH_PID=$!
fi

set +e
wait "${TRAIN_PID}"
trainer_exit=$?
set -e

if [[ ${trainer_exit} -ne 0 ]]; then
    echo "[FATAL] trainer failed with exit code ${trainer_exit}."
    grep -nE "Traceback|Exception|RuntimeError|CUDA out of memory|NCCL|RewardError" "${LOG_FILE}" | tail -n 100 || true
    exit ${trainer_exit}
fi

echo "[V27b] Training finished successfully."
