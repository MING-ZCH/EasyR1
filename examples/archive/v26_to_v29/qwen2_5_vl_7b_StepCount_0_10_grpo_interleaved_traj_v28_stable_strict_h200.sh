#!/bin/bash
# ================================================================
# V28 Launch Script — Stable Strict-JSON BoK-GRPO on 4xH200
# ================================================================
# Purpose:
#   Beat the V12 sparse baseline while preventing the three observed
#   failure modes from V12-fix/V26:
#     1. non-finite gradients / NaN cascades,
#     2. entropy drift from overly sharp or corrupted advantages,
#     3. trajectory format collapse / point JSON reward hacking.
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

set -x

LOG_DIR="/data/workspace/hyleochang/EasyR1-latest/logs/train"
mkdir -p "${LOG_DIR}"
export output_path='./'
export ckpt_path='./ckpt'
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

# Data: high-ceiling mixed easy+hard, with no extra oversampling.
export STEPCOUNT_TRAIN_DATA=${STEPCOUNT_TRAIN_DATA:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCountQA-RL-Traj_0_10_easy_plus_hard}
export BOK_TOTAL_STEPS=${BOK_TOTAL_STEPS:-213}
export TRAIN_OVERSAMPLE_NO_MASK_FACTOR=${TRAIN_OVERSAMPLE_NO_MASK_FACTOR:-1}
export TRAIN_HARD_OVERSAMPLE_FACTOR=${TRAIN_HARD_OVERSAMPLE_FACTOR:-1}
export TRAIN_HARD_REFERENCE_PATH=${TRAIN_HARD_REFERENCE_PATH:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCountQA-RL-Traj_0_10_hard_only/data}

# Reward / trajectory configuration.
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
export TRAJ_NO_SEQUENCE_FALLBACK=${TRAJ_NO_SEQUENCE_FALLBACK:-stat_mask_sim}
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
export TRAJ_POINT_STRICT_JSON=1
export ANSWER_WEIGHT=${ANSWER_WEIGHT:-0.6}
export POINT_WEIGHT=${POINT_WEIGHT:-0.3}
export TRAJECTORY_FORMAT_WEIGHT=${TRAJECTORY_FORMAT_WEIGHT:-0.1}

# Interleaved rollout configuration.
export ROLLOUT_N=${ROLLOUT_N:-16}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export INTERLEAVED_MAX_TURNS=${INTERLEAVED_MAX_TURNS:-11}
export INTERLEAVED_PER_TURN_MAX_TOKENS=${INTERLEAVED_PER_TURN_MAX_TOKENS:-1800}
export INTERLEAVED_ANSWER_TURN_MAX_TOKENS=${INTERLEAVED_ANSWER_TURN_MAX_TOKENS:-1800}
export INTERLEAVED_FIRST_TURN_PROMPT_FILE=${INTERLEAVED_FIRST_TURN_PROMPT_FILE:-}
export INTERLEAVED_PROCESS_PROMPT_FILE=${INTERLEAVED_PROCESS_PROMPT_FILE:-${PROJECT_ROOT}/examples/format_prompt/StepCount_interleaved_process_prompt.txt}
export SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-${PROJECT_ROOT}/examples/format_prompt/StepCount_interleaved_system_prompt.txt}
export INTERLEAVED_HISTORY_MODE=${INTERLEAVED_HISTORY_MODE:-0}
export INTERLEAVED_DEBUG=${INTERLEAVED_DEBUG:-1}
export INTERLEAVED_DEBUG_PRINT_CHARS=${INTERLEAVED_DEBUG_PRINT_CHARS:-50}

# BoK-GRPO: V23 stability backbone + strict-JSON reward fix.
export STEPCOUNT_RL_MODE=${STEPCOUNT_RL_MODE:-bok_grpo}
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-bok_grpo}
export PROCESS_REWARD_ENABLE=${PROCESS_REWARD_ENABLE:-0}
export ACTOR_LR=${ACTOR_LR:-1e-6}
export DISABLE_KL=${DISABLE_KL:-false}
export KL_COEF=${KL_COEF:-0.03}
export CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
export CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
export CLIP_RATIO_DUAL=${CLIP_RATIO_DUAL:-3.0}
export BOK_TAU=${BOK_TAU:-0.5}
export BOK_CLIP=${BOK_CLIP:-4.0}
export BOK_UNIFORM_MIX=${BOK_UNIFORM_MIX:-0.1}
export BOK_TAU_INIT=${BOK_TAU_INIT:-0.7}
export BOK_TAU_FINAL=${BOK_TAU_FINAL:-0.3}
export BOK_TAU_SCHEDULE=${BOK_TAU_SCHEDULE:-cosine}
export BOK_ADV_NORMALIZE=${BOK_ADV_NORMALIZE:-0}
export BOK_LOW_VAR_THRESHOLD=${BOK_LOW_VAR_THRESHOLD:-1e-5}
export BOK_FALLBACK_MODE=${BOK_FALLBACK_MODE:-drgrpo}
export BOK_DAPO_FILTER=${BOK_DAPO_FILTER:-0}
export BOK_DAPO_AUTO_DISABLE_THRESHOLD=${BOK_DAPO_AUTO_DISABLE_THRESHOLD:-0.5}
export BOK_MIN_BATCH_STD=${BOK_MIN_BATCH_STD:-0.1}
export BOK_EASY_THRESHOLD=${BOK_EASY_THRESHOLD:-0.50}
export BOK_EASY_SCORE_THRESHOLD=${BOK_EASY_SCORE_THRESHOLD:-0.5}
export BOK_FILTER_ALL_CORRECT=${BOK_FILTER_ALL_CORRECT:-1}
export BOK_SMART_FILTER_THRESHOLD=${BOK_SMART_FILTER_THRESHOLD:-0.955}
export BOK_ALLWRONG_CAP=${BOK_ALLWRONG_CAP:-1.0}
export BOK_ALLWRONG_ANSWER_THRESHOLD=${BOK_ALLWRONG_ANSWER_THRESHOLD:-0.5}
export BOK_ALLWRONG_NEG_ONLY=${BOK_ALLWRONG_NEG_ONLY:-0}
export BOK_WINNER_BOOST=${BOK_WINNER_BOOST:-0}
export BOK_EASY_SCALE=${BOK_EASY_SCALE:-1.0}
export BOK_QUALITY_BONUS=${BOK_QUALITY_BONUS:-0}

# Gradient safety: skip bad updates and use temporary cooldown only.
# Emergency LR brakes are disabled by default because V28 should fail fast
# rather than silently slowing the whole run after a single outlier step.
export GRAD_SPIKE_PROTECT=${GRAD_SPIKE_PROTECT:-1}
export GRAD_SPIKE_THRESHOLD=${GRAD_SPIKE_THRESHOLD:-3.0}
export GRAD_SPIKE_COOLDOWN=${GRAD_SPIKE_COOLDOWN:-3}
export GRAD_SPIKE_LR_FACTOR=${GRAD_SPIKE_LR_FACTOR:-0.1}
export GRAD_SPIKE_BRAKE_WINDOW=${GRAD_SPIKE_BRAKE_WINDOW:-40}
export GRAD_SPIKE_BRAKE_MAX=${GRAD_SPIKE_BRAKE_MAX:-999}
export GRAD_SPIKE_ABSOLUTE_CAP=0
export GRAD_NONFINITE_COOLDOWN=${GRAD_NONFINITE_COOLDOWN:-10}
export GRAD_NONFINITE_LR_FACTOR=${GRAD_NONFINITE_LR_FACTOR:-0.1}
export GRAD_NONFINITE_BRAKE_WINDOW=${GRAD_NONFINITE_BRAKE_WINDOW:-40}
export GRAD_NONFINITE_BRAKE_MAX=${GRAD_NONFINITE_BRAKE_MAX:-999}
export FP16_GRAD_UNDERFLOW_MONITOR=${FP16_GRAD_UNDERFLOW_MONITOR:-0}
export V28_FAILFAST_ENABLE=${V28_FAILFAST_ENABLE:-1}
export V28_FAILFAST_NAN_LIMIT=${V28_FAILFAST_NAN_LIMIT:-3}
export V28_FAILFAST_ENTROPY_CRITICAL=${V28_FAILFAST_ENTROPY_CRITICAL:-0.85}
export V28_FAILFAST_ENTROPY_CONSEC=${V28_FAILFAST_ENTROPY_CONSEC:-3}
export V28_FAILFAST_FORMAT_CRITICAL=${V28_FAILFAST_FORMAT_CRITICAL:-0.30}
export V28_FAILFAST_FORMAT_CONSEC=${V28_FAILFAST_FORMAT_CONSEC:-3}
export V28_FAILFAST_LOWVAR_CRITICAL=${V28_FAILFAST_LOWVAR_CRITICAL:-45}
export V28_FAILFAST_ZEROREWARD_CRITICAL=${V28_FAILFAST_ZEROREWARD_CRITICAL:-30}
export V28_FAILFAST_BOK_CONSEC=${V28_FAILFAST_BOK_CONSEC:-2}

MODEL_PATH=${MODEL_PATH:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/models/StepCount-7B-SFT-30k-high/checkpoint-3537}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/examples/config.yaml}
REWARD_FN_PATH=${REWARD_FN_PATH:-${PROJECT_ROOT}/examples/reward_function/StepCount_mask_reward.py:compute_score}

for required_file in "${CONFIG_PATH}" "${PROJECT_ROOT}/examples/reward_function/StepCount_mask_reward.py"; do
    [[ -f "${required_file}" ]] || { echo "[FATAL] required file not found: ${required_file}" >&2; exit 1; }
done
for required_path in "${MODEL_PATH}" "${STEPCOUNT_TRAIN_DATA}" "/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/pixmo-test" "${STEPCOUNT_MASKS_METADATA}" "${STEPCOUNT_MASKS_DIR}"; do
    [[ -e "${required_path}" ]] || { echo "[FATAL] required path not found: ${required_path}" >&2; exit 1; }
done
[[ -z "${INTERLEAVED_PROCESS_PROMPT_FILE}" || -f "${INTERLEAVED_PROCESS_PROMPT_FILE}" ]] || { echo "[FATAL] interleaved process prompt file not found: ${INTERLEAVED_PROCESS_PROMPT_FILE}" >&2; exit 1; }
[[ -z "${SYSTEM_PROMPT_FILE}" || -f "${SYSTEM_PROMPT_FILE}" ]] || { echo "[FATAL] system prompt file not found: ${SYSTEM_PROMPT_FILE}" >&2; exit 1; }

RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE="${LOG_DIR}/training_interleaved_traj_v28_stable_strict_h200_${STEPCOUNT_RL_MODE}_${ADV_ESTIMATOR}_${RUN_TS}.log"
MONITOR_SCRIPT="${PROJECT_ROOT}/tools/monitor_training_v2.py"
FAILFAST_SCRIPT="${PROJECT_ROOT}/tools/monitor_v28_failfast.py"
MONITOR_DIR="${LOG_DIR}/../monitor"
mkdir -p "${MONITOR_DIR}"
MONITOR_LOG="${MONITOR_DIR}/monitor_v28_stable_strict_h200_${RUN_TS}.log"
MONITOR_JSON="${MONITOR_DIR}/metrics_v28_stable_strict_h200_${RUN_TS}.jsonl"
FAILFAST_LOG="${MONITOR_DIR}/failfast_v28_stable_strict_h200_${RUN_TS}.log"
export STOP_FILE="/tmp/v28_stable_strict_h200_stop_${RUN_TS}"
rm -f "${STOP_FILE}"
if [[ -f "${MONITOR_SCRIPT}" ]]; then
    nohup python3 "${MONITOR_SCRIPT}" --log "${LOG_FILE}" --out "${MONITOR_LOG}" --stop-file "${STOP_FILE}" --interval 60 --every 5 --summary-every 20 --json-metrics "${MONITOR_JSON}" > /dev/null 2>"${MONITOR_DIR}/monitor_v28_stderr_${RUN_TS}.log" &
    MONITOR_PID=$!
else
    MONITOR_PID=""
fi
if [[ "${V28_FAILFAST_ENABLE}" == "1" && -f "${FAILFAST_SCRIPT}" ]]; then
    nohup python3 "${FAILFAST_SCRIPT}" --log "${LOG_FILE}" --out "${FAILFAST_LOG}" --stop-file "${STOP_FILE}" > /dev/null 2>"${MONITOR_DIR}/failfast_v28_stderr_${RUN_TS}.log" &
    FAILFAST_PID=$!
else
    FAILFAST_PID=""
fi
STOP_WATCH_PID=""
cleanup_monitor() {
    if [[ -n "${MONITOR_PID:-}" ]]; then kill "${MONITOR_PID}" 2>/dev/null || true; fi
    if [[ -n "${FAILFAST_PID:-}" ]]; then kill "${FAILFAST_PID}" 2>/dev/null || true; fi
    if [[ -n "${STOP_WATCH_PID:-}" ]]; then kill "${STOP_WATCH_PID}" 2>/dev/null || true; fi
}
trap cleanup_monitor EXIT

echo "================================================================"
echo "[V28 RunConfig] Stable-Strict H200 BoK-GRPO"
echo "[RunConfig] data=${STEPCOUNT_TRAIN_DATA} total_steps=${BOK_TOTAL_STEPS}"
echo "[RunConfig] lr=${ACTOR_LR} ppo_epochs=1 max_grad_norm=1.0 kl=${KL_COEF} (V23-stable backbone)"
echo "[RunConfig] strict_json=${TRAJ_POINT_STRICT_JSON} fmt_rej=${TRAJ_FORMAT_REJECTION} soft_answer_decay=${TRAJ_SOFT_ANSWER_DECAY}"
echo "[RunConfig] BOK_CLIP=${BOK_CLIP} smart_filter=${BOK_SMART_FILTER_THRESHOLD} tau=${BOK_TAU_INIT}->${BOK_TAU_FINAL}"
echo "[RunConfig] GradSpike threshold=${GRAD_SPIKE_THRESHOLD} absolute_cap=${GRAD_SPIKE_ABSOLUTE_CAP} cooldown_lr_factor=${GRAD_SPIKE_LR_FACTOR} spike_brake_max=${GRAD_SPIKE_BRAKE_MAX} nonfinite_brake_max=${GRAD_NONFINITE_BRAKE_MAX}"
echo "[RunConfig] FailFast enable=${V28_FAILFAST_ENABLE} entropy>${V28_FAILFAST_ENTROPY_CRITICAL}x${V28_FAILFAST_ENTROPY_CONSEC} format>${V28_FAILFAST_FORMAT_CRITICAL}x${V28_FAILFAST_FORMAT_CONSEC} nan_limit=${V28_FAILFAST_NAN_LIMIT}"
echo "[RunConfig] H200 gpu_memory_utilization=0.65 n_gpus=4 val_freq=10 save_freq=10"
echo "[RunConfig] log=${LOG_FILE}"
echo "================================================================"

python3 -m verl.trainer.main \
    config=${CONFIG_PATH} \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.disable_kl=${DISABLE_KL} \
    algorithm.kl_coef=${KL_COEF} \
    data.train_files=${STEPCOUNT_TRAIN_DATA} \
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
    trainer.experiment_name=StepCount-7B-SFT-30k_v28_stable_strict_h200_${ADV_ESTIMATOR}_$(date +%Y%m%d_%H%M) \
    trainer.logger=['console','wandb'] \
    trainer.save_checkpoint_path=/data/workspace/hyleochang/EasyR1-latest/save/StepCount-7B-SFT-30k_v28_stable_strict_h200_${ADV_ESTIMATOR}_$(date +%Y%m%d_%H%M) \
    trainer.total_epochs=${STEPCOUNT_TOTAL_EPOCHS:-1} \
    trainer.save_freq=10 \
    trainer.save_limit=-1 \
    trainer.val_freq=10 \
    trainer.n_gpus_per_node=4 > >(tee "${LOG_FILE}") 2>&1 &

TRAIN_PID=$!
if [[ "${V28_FAILFAST_ENABLE}" == "1" ]]; then
    (
        while kill -0 "${TRAIN_PID}" 2>/dev/null; do
            if [[ -f "${STOP_FILE}" ]]; then
                echo "[V28 FailFast] STOP file detected. Terminating trainer pid=${TRAIN_PID}." | tee -a "${LOG_FILE}"
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
if [[ -n "${STOP_WATCH_PID:-}" ]]; then kill "${STOP_WATCH_PID}" 2>/dev/null || true; fi
exit ${trainer_exit}
