#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

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

# StepCount mask reward settings
export STEPCOUNT_MASKS_METADATA=${STEPCOUNT_MASKS_METADATA:-/mnt/shared-storage-user/zhangchenhao/StepCount-RL_masks_output/masks_metadata.json}
export STEPCOUNT_MASKS_DIR=${STEPCOUNT_MASKS_DIR:-/mnt/shared-storage-user/zhangchenhao/StepCount-RL_masks_output/masks}
export STEPCOUNT_MASK_REQUIRE=${STEPCOUNT_MASK_REQUIRE:-1}  # 强制要求 mask 可用（否则报错退出）
export STEPCOUNT_MASK_PREFILL_BY_TURN=${STEPCOUNT_MASK_PREFILL_BY_TURN:-1}  # 按 turn 预填充 used_sample_ids
export STEPCOUNT_MASK_LOG_CONFIG=${STEPCOUNT_MASK_LOG_CONFIG:-1}
export STEPCOUNT_MASK_DEBUG=${STEPCOUNT_MASK_DEBUG:-1}    # 开启 mask debug 统计
export STEPCOUNT_MASK_DEBUG_EVERY=${STEPCOUNT_MASK_DEBUG_EVERY:-10}

export EASYR1_REWARD_DEBUG_EVERY=${EASYR1_REWARD_DEBUG_EVERY:-10} #
export EASYR1_REWARD_SAMPLE_DEBUG=${EASYR1_REWARD_SAMPLE_DEBUG:-1}
export EASYR1_REWARD_SAMPLE_DEBUG_MAX=${EASYR1_REWARD_SAMPLE_DEBUG_MAX:-1}
export EASYR1_REWARD_HEALTH_DEBUG=${EASYR1_REWARD_HEALTH_DEBUG:-1}
export EASYR1_REWARD_HEALTH_DEBUG_EVERY=${EASYR1_REWARD_HEALTH_DEBUG_EVERY:-1}
export STEPCOUNT_TRAJ_REASON_DEBUG=${STEPCOUNT_TRAJ_REASON_DEBUG:-1}
export STEPCOUNT_TRAJ_REASON_DEBUG_EVERY=${STEPCOUNT_TRAJ_REASON_DEBUG_EVERY:-10} #
export STEPCOUNT_TRAJ_EVENT_LOG=${STEPCOUNT_TRAJ_EVENT_LOG:-1}
export STEPCOUNT_TRAJ_EVENT_LOG_EVERY=${STEPCOUNT_TRAJ_EVENT_LOG_EVERY:-10}
export STEPCOUNT_TRAJ_EVENT_LOG_MAX=${STEPCOUNT_TRAJ_EVENT_LOG_MAX:-50}

# Treat numeric-string GT (e.g. "0") as trajectory so interleaved rollout uses trajectory+mask reward.
export STEPCOUNT_FORCE_TRAJECTORY_FOR_NUMERIC_GT=${STEPCOUNT_FORCE_TRAJECTORY_FOR_NUMERIC_GT:-1}

# Interleaved rollout knobs (override via env when needed)
export ROLLOUT_N=${ROLLOUT_N:-16}   # 每个 prompt 生成 16 条 trajectory（GRPO 组大小）
export INTERLEAVED_MAX_TURNS=${INTERLEAVED_MAX_TURNS:-11}   # 最大轮数（10 轮 point + 1 轮 answer）
export INTERLEAVED_PER_TURN_MAX_TOKENS=${INTERLEAVED_PER_TURN_MAX_TOKENS:-1800}  # 中间 point 轮最大 token 数
export INTERLEAVED_ANSWER_TURN_MAX_TOKENS=${INTERLEAVED_ANSWER_TURN_MAX_TOKENS:-1800}   # answer 轮最大 token 数
export INTERLEAVED_FIRST_TURN_PROMPT_FILE=${INTERLEAVED_FIRST_TURN_PROMPT_FILE:-${PROJECT_ROOT}/examples/format_prompt/StepCount_interleaved_first_turn_prompt.txt}
export INTERLEAVED_PROCESS_PROMPT_FILE=${INTERLEAVED_PROCESS_PROMPT_FILE:-${PROJECT_ROOT}/examples/format_prompt/StepCount_interleaved_process_prompt.txt}
export INTERLEAVED_HISTORY_MODE=${INTERLEAVED_HISTORY_MODE:-0} # 1=保留最近1轮历史（与评测对齐），-1=全历史，0=无文本历史，N>0=保留最近N轮
export INTERLEAVED_DEBUG=${INTERLEAVED_DEBUG:-1}    # 开启 rollout debug 日志
export INTERLEAVED_DEBUG_PRINT_CHARS=${INTERLEAVED_DEBUG_PRINT_CHARS:-50}
export TRAJ_RETURN_POINT_STEP_SCORES=${TRAJ_RETURN_POINT_STEP_SCORES:-1}
export TRAJ_MISS_DECAY_ENABLE=${TRAJ_MISS_DECAY_ENABLE:-1}  # dense mask point reward, 设 0 回退到原始0/1二值
export TRAJ_CONSISTENCY_PENALTY=${TRAJ_CONSISTENCY_PENALTY:-0.5} # 对point与answer不一致样本额外乘以惩罚系数
export TRAJ_ANSWER_GATE_THRESHOLD=${TRAJ_ANSWER_GATE_THRESHOLD:-0.4} # point分数低于0.4的样本的answer置于0
export TRAJ_EVAL_ANSWER_ONLY_ON_NO_MASK=${TRAJ_EVAL_ANSWER_ONLY_ON_NO_MASK:-1} # eval时若无mask序列，仅评测answer（禁用point-gate）
export TRAJ_ANSWER_GATE_MODE=${TRAJ_ANSWER_GATE_MODE:-soft}  # soft=soft-gating(默认), hard=hard-gating, off=无gating
export TRAJ_SOFT_GATE_BASE=${TRAJ_SOFT_GATE_BASE:-0.8}  # soft gating: answer_score = base + (1-base)*point_quality
export TRAJ_FORMAT_REJECTION=${TRAJ_FORMAT_REJECTION:-1}  # 1=format不合格的trajectory reward置0

# ===================== RL mode switch =====================
# 可选: grpo_standard | drgrpo | gspo | process_reward | dapo | bok_grpo
# - grpo_standard: 最基础标准GRPO（初始默认配置）
# - drgrpo: 仅启用 Dr.GRPO（不分发 per-turn reward）
# - gspo: 启用 step-level GRPO + per-turn process reward
# - process_reward: 保持标准 GRPO，仅启用 per-turn process reward（做消融）
# - dapo: 标准GRPO + DAPO clipping参数
# - bok_grpo: Best-of-K GRPO: Softmax-temperature advantage for pass@K->pass@1 conversion. Uses reward-weighted soft selection instead of symmetric z-normalization.


# 先跑 Dr.GRPO (ADV_ESTIMATOR=drgrpo)，观察advantage是否从≈0提升至有意义的值(≈0.5-2.0)
# 确认advantage信号健康后，加入 Process Reward (PROCESS_REWARD_ENABLE=1)
# 如果需要更精细的per-turn控制，切换到 GRPO_STEP (ADV_ESTIMATOR=grpo_step)
# 同时监控：ppo_kl 应该从0变为 >0.001，pg_loss 绝对值应增大

export STEPCOUNT_RL_MODE=${STEPCOUNT_RL_MODE:-grpo_standard}

# 允许用户外部显式覆盖；未设置时按 mode 自动给默认值
ADV_ESTIMATOR=${ADV_ESTIMATOR:-}
PROCESS_REWARD_ENABLE=${PROCESS_REWARD_ENABLE:-}
ACTOR_LR=${ACTOR_LR:-}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-}
CLIP_RATIO_DUAL=${CLIP_RATIO_DUAL:-}
DISABLE_KL=${DISABLE_KL:-}


case "${STEPCOUNT_RL_MODE}" in
  grpo_standard)
    [[ -z "${ADV_ESTIMATOR}" ]] && ADV_ESTIMATOR="grpo"
    [[ -z "${PROCESS_REWARD_ENABLE}" ]] && PROCESS_REWARD_ENABLE="0"
    [[ -z "${ACTOR_LR}" ]] && ACTOR_LR="1e-6"
    [[ -z "${ROLLOUT_TEMPERATURE}" ]] && ROLLOUT_TEMPERATURE="1.0"
    [[ -z "${CLIP_RATIO_LOW}" ]] && CLIP_RATIO_LOW="0.2"
    [[ -z "${CLIP_RATIO_HIGH}" ]] && CLIP_RATIO_HIGH="0.3"
    [[ -z "${CLIP_RATIO_DUAL}" ]] && CLIP_RATIO_DUAL="3.0"
    [[ -z "${DISABLE_KL}" ]] && DISABLE_KL="false"
    ;;
  drgrpo)
    [[ -z "${ADV_ESTIMATOR}" ]] && ADV_ESTIMATOR="drgrpo"
    [[ -z "${PROCESS_REWARD_ENABLE}" ]] && PROCESS_REWARD_ENABLE="0"
    [[ -z "${ACTOR_LR}" ]] && ACTOR_LR="1e-6"
    [[ -z "${ROLLOUT_TEMPERATURE}" ]] && ROLLOUT_TEMPERATURE="1.0"
    [[ -z "${CLIP_RATIO_LOW}" ]] && CLIP_RATIO_LOW="0.2"
    [[ -z "${CLIP_RATIO_HIGH}" ]] && CLIP_RATIO_HIGH="0.28" # 0.3
    [[ -z "${CLIP_RATIO_DUAL}" ]] && CLIP_RATIO_DUAL="3.0"
    [[ -z "${DISABLE_KL}" ]] && DISABLE_KL="false"
    ;;
  gspo)
    [[ -z "${ADV_ESTIMATOR}" ]] && ADV_ESTIMATOR="grpo_step"
    [[ -z "${PROCESS_REWARD_ENABLE}" ]] && PROCESS_REWARD_ENABLE="1"
    [[ -z "${ACTOR_LR}" ]] && ACTOR_LR="1e-6"
    [[ -z "${ROLLOUT_TEMPERATURE}" ]] && ROLLOUT_TEMPERATURE="1.0"
    [[ -z "${CLIP_RATIO_LOW}" ]] && CLIP_RATIO_LOW="3e-4"  # 0.2, standard gspo
    [[ -z "${CLIP_RATIO_HIGH}" ]] && CLIP_RATIO_HIGH="4e-4" # 0.3, new-standard gspo
    [[ -z "${CLIP_RATIO_DUAL}" ]] && CLIP_RATIO_DUAL="3.0"
    [[ -z "${DISABLE_KL}" ]] && DISABLE_KL="true" # new-standard gspo
    ;;
  process_reward)
    [[ -z "${ADV_ESTIMATOR}" ]] && ADV_ESTIMATOR="grpo"
    [[ -z "${PROCESS_REWARD_ENABLE}" ]] && PROCESS_REWARD_ENABLE="1"
    [[ -z "${ACTOR_LR}" ]] && ACTOR_LR="1e-6"
    [[ -z "${ROLLOUT_TEMPERATURE}" ]] && ROLLOUT_TEMPERATURE="1.0"
    [[ -z "${CLIP_RATIO_LOW}" ]] && CLIP_RATIO_LOW="0.2"
    [[ -z "${CLIP_RATIO_HIGH}" ]] && CLIP_RATIO_HIGH="0.3"
    [[ -z "${CLIP_RATIO_DUAL}" ]] && CLIP_RATIO_DUAL="3.0"
    [[ -z "${DISABLE_KL}" ]] && DISABLE_KL="false"
    ;;
  dapo)
    [[ -z "${ADV_ESTIMATOR}" ]] && ADV_ESTIMATOR="grpo"
    [[ -z "${PROCESS_REWARD_ENABLE}" ]] && PROCESS_REWARD_ENABLE="0"
    [[ -z "${ACTOR_LR}" ]] && ACTOR_LR="1e-6"
    [[ -z "${ROLLOUT_TEMPERATURE}" ]] && ROLLOUT_TEMPERATURE="1.0"
    [[ -z "${CLIP_RATIO_LOW}" ]] && CLIP_RATIO_LOW="0.2"
    [[ -z "${CLIP_RATIO_HIGH}" ]] && CLIP_RATIO_HIGH="0.28"
    [[ -z "${CLIP_RATIO_DUAL}" ]] && CLIP_RATIO_DUAL="3.0"
    [[ -z "${DISABLE_KL}" ]] && DISABLE_KL="true"  # new-standard dapo
    ;;
  bok_grpo)
    # Best-of-K GRPO: Softmax-temperature advantage for pass@K->pass@1 conversion.
    # Uses reward-weighted soft selection instead of symmetric z-normalization.
    # bok_tau=0.3 balances selectivity vs exploration; lower = more aggressive.
    [[ -z "${ADV_ESTIMATOR}" ]] && ADV_ESTIMATOR="bok_grpo"
    [[ -z "${PROCESS_REWARD_ENABLE}" ]] && PROCESS_REWARD_ENABLE="0"
    [[ -z "${ACTOR_LR}" ]] && ACTOR_LR="1e-6"
    [[ -z "${ROLLOUT_TEMPERATURE}" ]] && ROLLOUT_TEMPERATURE="1.0"
    [[ -z "${CLIP_RATIO_LOW}" ]] && CLIP_RATIO_LOW="0.2"
    [[ -z "${CLIP_RATIO_HIGH}" ]] && CLIP_RATIO_HIGH="0.28"
    [[ -z "${CLIP_RATIO_DUAL}" ]] && CLIP_RATIO_DUAL="3.0"
    [[ -z "${DISABLE_KL}" ]] && DISABLE_KL="false"
    export BOK_TAU=${BOK_TAU:-0.3}
    export BOK_CLIP=${BOK_CLIP:-3.0}
    export BOK_UNIFORM_MIX=${BOK_UNIFORM_MIX:-0.1}
    # τ annealing: cosine schedule from TAU_INIT -> TAU_FINAL over training
    # Set both > 0 to enable; they override BOK_TAU dynamically
    export BOK_TAU_INIT=${BOK_TAU_INIT:-0.8}   # start with mild exploration (τ=0.8)
    export BOK_TAU_FINAL=${BOK_TAU_FINAL:-0.3}  # anneal to selective (low τ)
    # MUST set total_steps > 1 for annealing to activate (core_algos.py check)
    # NOTE: total_steps is auto-computed from meta_info; env var is fallback only
    export BOK_TOTAL_STEPS=${BOK_TOTAL_STEPS:-178}
    # τ schedule: "cosine" (default, slow start/end) or "linear"
    export BOK_TAU_SCHEDULE=${BOK_TAU_SCHEDULE:-cosine}
    # Per-group advantage normalization: 0=disabled (default, per Dr.GRPO finding)
    export BOK_ADV_NORMALIZE=${BOK_ADV_NORMALIZE:-0}
    # Low-var threshold: group std < this → batch baseline fallback
    export BOK_LOW_VAR_THRESHOLD=${BOK_LOW_VAR_THRESHOLD:-1e-5}
    # --- v7 New: Algorithm improvements ---
    # Fallback mode for low-var groups: zscore (default/v6) | drgrpo (no std div) | clip_std
    export BOK_FALLBACK_MODE=${BOK_FALLBACK_MODE:-drgrpo}
    # DAPO dynamic sampling: filter homogeneous groups (0=off default, 1=on)
    export BOK_DAPO_FILTER=${BOK_DAPO_FILTER:-0}
    # Min batch std for clip_std mode (prevents over-amplification)
    export BOK_MIN_BATCH_STD=${BOK_MIN_BATCH_STD:-0.1}
    ;;
  *)
    echo "[FATAL] Unknown STEPCOUNT_RL_MODE=${STEPCOUNT_RL_MODE}. Choose one of: grpo_standard, drgrpo, gspo, process_reward, dapo, bok_grpo" >&2
    exit 1
    ;;
esac

# Reward 权重（可按任务需要覆盖）
# 推荐搭配：
# - grpo_standard: ANSWER_WEIGHT=0.6 POINT_WEIGHT=0.3 TRAJECTORY_FORMAT_WEIGHT=0.1
# - drgrpo:        ANSWER_WEIGHT=0.6 POINT_WEIGHT=0.3 TRAJECTORY_FORMAT_WEIGHT=0.1
# - gspo:          ANSWER_WEIGHT=0.5 POINT_WEIGHT=0.4 TRAJECTORY_FORMAT_WEIGHT=0.1
# - process_reward:ANSWER_WEIGHT=0.6 POINT_WEIGHT=0.3 TRAJECTORY_FORMAT_WEIGHT=0.1
# - dapo:          ANSWER_WEIGHT=0.6 POINT_WEIGHT=0.3 TRAJECTORY_FORMAT_WEIGHT=0.1
# - bok_grpo:      ANSWER_WEIGHT=0.7 POINT_WEIGHT=0.2 TRAJECTORY_FORMAT_WEIGHT=0.1
# NOTE: Use consistent weights across ALL modes for fair comparison!
# With TRAJ_FORMAT_REJECTION=1, format_weight can be lowered (format acts as filter, not reward)
export ANSWER_WEIGHT=${ANSWER_WEIGHT:-0.7}
export POINT_WEIGHT=${POINT_WEIGHT:-0.2}
export TRAJECTORY_FORMAT_WEIGHT=${TRAJECTORY_FORMAT_WEIGHT:-0.1}

# 兼容 process reward / GSPO：需要输出 step_scores
if [[ "${PROCESS_REWARD_ENABLE}" == "1" ]]; then
  export TRAJ_RETURN_POINT_STEP_SCORES=${TRAJ_RETURN_POINT_STEP_SCORES:-1}
fi

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
    echo "[FATAL] reward function file not found: ${PROJECT_ROOT}/examples/reward_function/StepCount_mask_reward.py" >&2
    exit 1
fi
if [[ -n "${INTERLEAVED_FIRST_TURN_PROMPT_FILE}" && ! -f "${INTERLEAVED_FIRST_TURN_PROMPT_FILE}" ]]; then
    echo "[FATAL] interleaved first-turn prompt file not found: ${INTERLEAVED_FIRST_TURN_PROMPT_FILE}" >&2
    exit 1
fi
if [[ -n "${INTERLEAVED_PROCESS_PROMPT_FILE}" && ! -f "${INTERLEAVED_PROCESS_PROMPT_FILE}" ]]; then
    echo "[FATAL] interleaved process prompt file not found: ${INTERLEAVED_PROCESS_PROMPT_FILE}" >&2
    exit 1
fi
for required_path in \
    "${MODEL_PATH}" \
    "/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_0_10" \
    "/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/pixmo-test" \
    "${STEPCOUNT_MASKS_METADATA}" \
    "${STEPCOUNT_MASKS_DIR}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[FATAL] required path not found: ${required_path}" >&2
        exit 1
    fi
done

RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_interleaved_traj_v7_StepCount_mask_reward_v4_${STEPCOUNT_RL_MODE}_hm${INTERLEAVED_HISTORY_MODE}_gate${TRAJ_ANSWER_GATE_MODE}_fmtrej${TRAJ_FORMAT_REJECTION}_${ADV_ESTIMATOR}_${RUN_TS}.log"

echo "[RunConfig] model=${MODEL_PATH}"
echo "[RunConfig] mode=${STEPCOUNT_RL_MODE} adv=${ADV_ESTIMATOR} disable_kl=${DISABLE_KL} process_reward=${PROCESS_REWARD_ENABLE}"
echo "[RunConfig] clip_low=${CLIP_RATIO_LOW} clip_high=${CLIP_RATIO_HIGH} clip_dual=${CLIP_RATIO_DUAL}"
echo "[RunConfig] lr=${ACTOR_LR} temp=${ROLLOUT_TEMPERATURE} history_mode=${INTERLEAVED_HISTORY_MODE} rollout_n=${ROLLOUT_N} max_turns=${INTERLEAVED_MAX_TURNS}"
echo "[RunConfig] first_prompt=${INTERLEAVED_FIRST_TURN_PROMPT_FILE}"
echo "[RunConfig] process_prompt=${INTERLEAVED_PROCESS_PROMPT_FILE}"
echo "[RunConfig] reward_debug_every=${EASYR1_REWARD_DEBUG_EVERY} reward_health_every=${EASYR1_REWARD_HEALTH_DEBUG_EVERY}"
echo "[RunConfig] gate_mode=${TRAJ_ANSWER_GATE_MODE} soft_base=${TRAJ_SOFT_GATE_BASE} fmt_rej=${TRAJ_FORMAT_REJECTION} consistency_pen=${TRAJ_CONSISTENCY_PENALTY}"
echo "[RunConfig] answer_w=${ANSWER_WEIGHT} point_w=${POINT_WEIGHT} format_w=${TRAJECTORY_FORMAT_WEIGHT}"
echo "[RunConfig] log_file=${LOG_FILE}"
echo "[RunConfig] bok_fallback=${BOK_FALLBACK_MODE:-zscore} dapo_filter=${BOK_DAPO_FILTER:-0} min_batch_std=${BOK_MIN_BATCH_STD:-0.1}"

    # lr_warmup prevents early instability
python3 -m verl.trainer.main \
    config=${CONFIG_PATH} \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.disable_kl=${DISABLE_KL} \
    algorithm.kl_coef=5e-2 \
    data.train_files=/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_0_10 \
    data.val_files=/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/pixmo-test \
    data.max_prompt_length=7500 \
    data.max_response_length=3200 \
    data.shuffle=true \
    worker.actor.optim.lr=${ACTOR_LR} \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    worker.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    worker.actor.clip_ratio_dual=${CLIP_RATIO_DUAL} \
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
    trainer.experiment_name=StepCount-7B-SFT-30k_v7_mask_reward_v4${STEPCOUNT_RL_MODE}_hm${INTERLEAVED_HISTORY_MODE}_gate${TRAJ_ANSWER_GATE_MODE}_fmtrej${TRAJ_FORMAT_REJECTION}_${ADV_ESTIMATOR}_KL${DISABLE_KL}_ailab_$(date +%Y%m%d_%H%M) \
    trainer.logger=['console','wandb'] \
    trainer.save_checkpoint_path=/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest/save/StepCount-7B-SFT-30k_v7_mask_reward_v4${STEPCOUNT_RL_MODE}_hm${INTERLEAVED_HISTORY_MODE}_gate${TRAJ_ANSWER_GATE_MODE}_${ADV_ESTIMATOR}_$(date +%Y%m%d_%H%M) \
    trainer.save_freq=66 \
    trainer.n_gpus_per_node=4 2>&1 | tee "${LOG_FILE}"

trainer_exit=${PIPESTATUS[0]}
if [[ ${trainer_exit} -ne 0 ]]; then
    echo "[FATAL] trainer failed with exit code ${trainer_exit}."
    echo "[Diag] extracting key lines for quick debug..."
    grep -nE "Traceback|Exception|RuntimeError|CUDA out of memory|NCCL|RewardError|RewardHealth|RewardDebug|trajectory_reward\[detail\]|trajectory_reward|InterleavedWarning|InterleavedDebug\] response_mask_mode=pad_based|point_step_[0-9]+|point_target_count|prompt_effective_source|question_non_empty|reward_score|overall_reward|format_fail_reward|stop_violation_reward|stopped_by_answer_reward" "${LOG_FILE}" | tail -n 260 || true
    exit ${trainer_exit}
fi

echo "[Diag] training finished successfully; latest key debug lines:"
grep -nE "RewardHealth|RewardDebug|trajectory_reward\[detail\]|trajectory_reward|InterleavedWarning|InterleavedDebug\] response_mask_mode=pad_based|point_step_[0-9]+|point_target_count|prompt_effective_source|question_non_empty|reward_score|overall_reward|format_fail_reward|stop_violation_reward|stopped_by_answer_reward" "${LOG_FILE}" | tail -n 200 || true
      