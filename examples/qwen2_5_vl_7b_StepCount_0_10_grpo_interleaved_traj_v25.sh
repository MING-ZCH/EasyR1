#!/bin/bash
# ================================================================
# v25 Launch Script — StepCount-7B Interleaved Point-to-Count
# ================================================================
# V25 = V24 base + 3 BoK-GRPO optimizations + no-mask oversampling + V12 grad strategy
#
# KEY CHANGES from V24 (6 changes):
#   1) max_grad_norm: 1.0 -> 0.5 (V12 sign-GD strategy: constant step-size, prevents oscillation)
#   2) TRAIN_OVERSAMPLE_NO_MASK_FACTOR: 1 -> 3 (189 no-mask samples see 3x sampling frequency)
#   3) TRAJ_NO_SEQUENCE_FALLBACK: count_iou -> stat_mask_sim (fix inflated reward for no-mask samples)
#   4) BOK_WINNER_BOOST: 0 -> 1.5 (amplify rare correct trajectories in hard groups)
#   5) BOK_ALLWRONG_NEG_ONLY: 1 (new: all-wrong groups only penalize, never reward wrong answers)
#   6) [Code change] answer-based routing in core_algos.py (auto-active, no env var needed)
#
# ANALYSIS BASIS (from V21-V24 comparison and 3 deep-dive questions):
#   - V12 (best=81.47%) used grad_norm=0.5 causing 100% clip → sign GD (constant 0.5 step)
#   - V12 synergy: clip=0.5 × ppo=2 = max 1.0 total update/batch
#   - V24 with clip=1.0 × ppo=1 = max 1.0 but non-sign GD → oscillation risk
#   - V25 strategy: clip=0.5 × ppo=1 = max 0.5 (conservative but stable sign-GD)
#   - If too slow, increase LR to 2e-6 or add ppo_epochs=2
#
# BoK-GRPO optimizations (3 code changes already applied):
#   A) answer-based routing: use answer_scores (binary correct/wrong) for easy/hard routing
#      instead of mixed score (0.6*answer+0.3*point+0.1*format). Prevents point-inflated routing.
#   B) allwrong neg-only: cap all-wrong group advantages to [-cap, 0] instead of [-cap, +cap].
#      Prevents BoK from reinforcing "least wrong" answers.
#   C) winner_boost=1.5: amplify correct-trajectory advantages by sqrt(K/n_correct),
#      bounded by 1.5. Stronger gradient for rare correct paths in hard groups.
#
# No-mask oversampling:
#   - 189 training samples lack SAM mask metadata → use count_iou fallback (inflated reward)
#   - stat_mask_sim fixes the reward computation (error <0.005 vs real mask reward)
#   - 3x oversampling slightly increases their representation (1.38% → 4.0%)
#   - WeightedRandomSampler auto-detects no-mask samples via mask metadata matching
#
# Inherited from V24 (proven effective):
#   FORMAT_REJECTION=1, GradSpikeProtect@3.0x, ppo_epochs=1,
#   BOK_EASY_THRESHOLD=0.50, BOK_CLIP=4.0,
#   tau_annealing 0.7->0.3, AllWrong_Cap=1.0, DAPO dual clip (0.2/0.28)
#   ANSWER_WEIGHT=0.6, POINT_WEIGHT=0.3, FORMAT_WEIGHT=0.1
#   SMART_FILTER_THRESHOLD=0.955
#
# Changed from V24:
#   max_grad_norm: 1.0->0.5, TRAIN_OVERSAMPLE_NO_MASK_FACTOR=3,
#   TRAJ_NO_SEQUENCE_FALLBACK=stat_mask_sim, BOK_WINNER_BOOST=1.5,
#   BOK_ALLWRONG_NEG_ONLY=1, answer-based routing (code)
#
# Resume: V21-R2 step-60 checkpoint (clean ppo=1 run, val=0.7561, 0 NaN, saves ~2.4h)
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# v25: Resume from V21-R2 step-60 checkpoint (same as V24)
v21_RESUME_CKPT="${v21_RESUME_CKPT:-/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest/save/StepCount-7B-SFT-30k_v21_mask_reward_v4bok_grpo_hm0_gateoff_bok_grpo_20260406_0311/global_step_60}"
# v21_RESUME_CKPT=""  # Uncomment to start fresh from SFT

set -x

LOG_DIR="/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest/logs/train"
mkdir -p ${LOG_DIR}
export output_path='./'
export ckpt_path='./ckpt'

export PYTHONUNBUFFERED=1
export WANDB_API_KEY='5c554b68aed1a458465f44705a27102d8579c1cb'
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16
export PYTHONHASHSEED=0

# ── V25 NEW: Weighted sampling for data distribution control ──
# Two independent weight multipliers (multiplicative):
#   1) No-mask oversample: 567 no-mask samples get 3x weight (4.2% → 11.5%)
#   2) Hard oversample: 3809 hard-matched samples get Nx weight
# Set factor=1 to disable either one.
export TRAIN_OVERSAMPLE_NO_MASK_FACTOR=${TRAIN_OVERSAMPLE_NO_MASK_FACTOR:-3}
export TRAIN_HARD_OVERSAMPLE_FACTOR=${TRAIN_HARD_OVERSAMPLE_FACTOR:-1}  # 1=off. Set 2 to double hard sampling
export TRAIN_HARD_REFERENCE_PATH=${TRAIN_HARD_REFERENCE_PATH:-/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_0_10_hard_only/data}

# ── V25 NEW: stat_mask_sim fallback for no-mask samples ──
# Replaces count_iou (inflated reward) with statistical approximation (error <0.005)
export TRAJ_NO_SEQUENCE_FALLBACK=${TRAJ_NO_SEQUENCE_FALLBACK:-stat_mask_sim}

# StepCount mask reward settings
export STEPCOUNT_MASKS_METADATA=${STEPCOUNT_MASKS_METADATA:-/mnt/shared-storage-user/zhangchenhao/StepCount-RL_masks_output/masks_metadata.json}
export STEPCOUNT_MASKS_DIR=${STEPCOUNT_MASKS_DIR:-/mnt/shared-storage-user/zhangchenhao/StepCount-RL_masks_output/masks}
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
export INTERLEAVED_PER_TURN_MAX_TOKENS=${INTERLEAVED_PER_TURN_MAX_TOKENS:-1800}
export INTERLEAVED_ANSWER_TURN_MAX_TOKENS=${INTERLEAVED_ANSWER_TURN_MAX_TOKENS:-1800}
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
    [[ -z "${ACTOR_LR}" ]] && ACTOR_LR="1.5e-6"
    [[ -z "${ROLLOUT_TEMPERATURE}" ]] && ROLLOUT_TEMPERATURE="1.0"
    [[ -z "${CLIP_RATIO_LOW}" ]] && CLIP_RATIO_LOW="0.2"
    [[ -z "${CLIP_RATIO_HIGH}" ]] && CLIP_RATIO_HIGH="0.28"
    [[ -z "${CLIP_RATIO_DUAL}" ]] && CLIP_RATIO_DUAL="3.0"
    [[ -z "${DISABLE_KL}" ]] && DISABLE_KL="false"

    # ── τ annealing ──
    export BOK_TAU=${BOK_TAU:-0.5}
    export BOK_CLIP=${BOK_CLIP:-4.0}
    export BOK_UNIFORM_MIX=${BOK_UNIFORM_MIX:-0.1}
    export BOK_TAU_INIT=${BOK_TAU_INIT:-0.7}
    export BOK_TAU_FINAL=${BOK_TAU_FINAL:-0.3}
    export BOK_TOTAL_STEPS=${BOK_TOTAL_STEPS:-213}  # 13647/64≈213
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
    export BOK_SMART_FILTER_THRESHOLD=${BOK_SMART_FILTER_THRESHOLD:-0.955}  # V24: release low-point AC groups

    # ── AllWrong Cap ──
    export BOK_ALLWRONG_CAP=${BOK_ALLWRONG_CAP:-1.0}
    export BOK_ALLWRONG_ANSWER_THRESHOLD=${BOK_ALLWRONG_ANSWER_THRESHOLD:-0.5}
    # V25 NEW: all-wrong groups only penalize, never reward wrong answers
    export BOK_ALLWRONG_NEG_ONLY=${BOK_ALLWRONG_NEG_ONLY:-1}

    # ── V25 NEW: Winner Amplification ──
    # Boost rare-correct advantages by min(boost, sqrt(K/n_correct))
    # 1.5 = moderate (at 2/16 correct: boost=min(1.5, sqrt(8))=1.5; at 8/16: sqrt(2)≈1.41)
    export BOK_WINNER_BOOST=${BOK_WINNER_BOOST:-1.5}

    # ── Easy Gradient Dampening & Quality Bonus ──
    export BOK_EASY_SCALE=${BOK_EASY_SCALE:-1.0}
    export BOK_QUALITY_BONUS=${BOK_QUALITY_BONUS:-0}

    # ── Grad Norm Spike Protection ──
    export GRAD_SPIKE_PROTECT=${GRAD_SPIKE_PROTECT:-1}
    export GRAD_SPIKE_THRESHOLD=${GRAD_SPIKE_THRESHOLD:-3.0}
    export GRAD_SPIKE_COOLDOWN=${GRAD_SPIKE_COOLDOWN:-3}
    export GRAD_SPIKE_LR_FACTOR=${GRAD_SPIKE_LR_FACTOR:-0.1}
    export GRAD_SPIKE_BRAKE_WINDOW=${GRAD_SPIKE_BRAKE_WINDOW:-40}
    export GRAD_SPIKE_BRAKE_MAX=${GRAD_SPIKE_BRAKE_MAX:-6}
    export FP16_GRAD_UNDERFLOW_MONITOR=${FP16_GRAD_UNDERFLOW_MONITOR:-0}
    ;;
  *)
    echo "[FATAL] V25 only supports bok_grpo mode. Got: ${STEPCOUNT_RL_MODE}" >&2
    exit 1
    ;;
esac

# Reward weights (consistent across modes)
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

MODEL_PATH=/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/model/StepCount-7B-SFT-30k-high/checkpoint-3537
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
    "/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_0_10_easy_plus_hard" \
    "/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/pixmo-test" \
    "${STEPCOUNT_MASKS_METADATA}" \
    "${STEPCOUNT_MASKS_DIR}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[FATAL] required path not found: ${required_path}" >&2
        exit 1
    fi
done

RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE="${LOG_DIR}/training_interleaved_traj_v25_${STEPCOUNT_RL_MODE}_${ADV_ESTIMATOR}_${RUN_TS}.log"

echo "================================================================"
echo "[V25 RunConfig] BoK-GRPO + No-Mask Oversample + Sign-GD"
echo "================================================================"
echo "[RunConfig] model=${MODEL_PATH}"
echo "[RunConfig] mode=${STEPCOUNT_RL_MODE} adv=${ADV_ESTIMATOR} lr=${ACTOR_LR} max_grad_norm=0.5"
echo "[RunConfig] clip_low=${CLIP_RATIO_LOW} clip_high=${CLIP_RATIO_HIGH} ppo_epochs=1"
echo "[RunConfig] no_mask_oversample=${TRAIN_OVERSAMPLE_NO_MASK_FACTOR} hard_oversample=${TRAIN_HARD_OVERSAMPLE_FACTOR} no_seq_fallback=${TRAJ_NO_SEQUENCE_FALLBACK}"
echo "[RunConfig] winner_boost=${BOK_WINNER_BOOST} allwrong_neg_only=${BOK_ALLWRONG_NEG_ONLY}"
echo "[RunConfig] easy_threshold=${BOK_EASY_THRESHOLD} smart_filter=${BOK_SMART_FILTER_THRESHOLD}"
echo "[RunConfig] answer_w=${ANSWER_WEIGHT} point_w=${POINT_WEIGHT} format_w=${TRAJECTORY_FORMAT_WEIGHT}"
echo "[RunConfig] resume=${v21_RESUME_CKPT}"
echo "[RunConfig] log=${LOG_FILE}"
echo "================================================================"

# ── Training Monitor V2 ──
MONITOR_SCRIPT="/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest/tools/monitor_training_v2.py"
MONITOR_DIR="${LOG_DIR}/../monitor"
mkdir -p "${MONITOR_DIR}"
MONITOR_LOG="${MONITOR_DIR}/monitor_v25_${RUN_TS}.log"
MONITOR_JSON="${MONITOR_DIR}/metrics_v25_${RUN_TS}.jsonl"
export STOP_FILE="/tmp/v25_stop_${RUN_TS}"
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
        > /dev/null 2>"${MONITOR_DIR}/monitor_v25_stderr_${RUN_TS}.log" &
    MONITOR_PID=$!
    echo "[Monitor] PID=${MONITOR_PID} stop_file=${STOP_FILE}"
else
    echo "[Monitor] Script not found: ${MONITOR_SCRIPT}, skipping"
    MONITOR_PID=""
fi

python3 -m verl.trainer.main \
    config=${CONFIG_PATH} \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.disable_kl=${DISABLE_KL} \
    algorithm.kl_coef=2e-2 \
    data.train_files=${STEPCOUNT_TRAIN_DATA:-/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_0_10_easy_plus_hard} \
    data.system_prompt_file=${SYSTEM_PROMPT_FILE} \
    data.format_prompt=null \
    data.val_files=/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/pixmo-test \
    data.max_prompt_length=7500 \
    data.max_response_length=3200 \
    data.shuffle=true \
    worker.actor.optim.lr=${ACTOR_LR} \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    worker.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    worker.actor.clip_ratio_dual=${CLIP_RATIO_DUAL} \
    worker.actor.ppo_epochs=1 \
    worker.actor.max_grad_norm=0.5 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.rollout.gpu_memory_utilization=0.4 \
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
    trainer.experiment_name=StepCount-7B-SFT-30k_v25_signGD_winboost_negonly_oversample_${ADV_ESTIMATOR}_$(date +%Y%m%d_%H%M) \
    trainer.logger=['console','wandb'] \
    trainer.save_checkpoint_path=/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest/save/StepCount-7B-SFT-30k_v25_signGD_winboost_negonly_${ADV_ESTIMATOR}_$(date +%Y%m%d_%H%M) \
    trainer.load_checkpoint_path=${v21_RESUME_CKPT} \
    trainer.total_epochs=${STEPCOUNT_TOTAL_EPOCHS:-1} \
    trainer.save_freq=30 \
    trainer.save_limit=-1 \
    trainer.val_freq=15 \
    trainer.n_gpus_per_node=4 2>&1 | tee "${LOG_FILE}"

trainer_exit=${PIPESTATUS[0]}

# ── Cleanup Monitor ──
if [[ -n "${MONITOR_PID:-}" ]]; then
    echo "[Monitor] Training done, stopping monitor PID=${MONITOR_PID}"
    kill ${MONITOR_PID} 2>/dev/null || true
    wait ${MONITOR_PID} 2>/dev/null || true
fi

if [[ ${trainer_exit} -ne 0 ]]; then
    echo "[FATAL] trainer failed with exit code ${trainer_exit}."
    grep -nE "Traceback|Exception|RuntimeError|CUDA out of memory|NCCL|RewardError" "${LOG_FILE}" | tail -n 100 || true
    exit ${trainer_exit}
fi

echo "[V25] Training finished successfully."
