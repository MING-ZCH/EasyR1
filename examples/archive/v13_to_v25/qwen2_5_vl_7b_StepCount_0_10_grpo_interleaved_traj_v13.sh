#!/bin/bash
# ================================================================
# V13 Launch Script — StepCount-7B Interleaved Point-to-Count
# ================================================================
# Based on V12 with the following NaN-fix and extended training:
#   1. Resume from V12 S132 checkpoint (val=0.771, best ever)
#   2. Keep low-tau regime (tau_final=0.3) but add adaptive numerical guard
#   3. total_epochs kept at 1 per requirement (no schedule stretching)
#   4. BOK_TOTAL_STEPS kept at 178 to match total_epochs=1
#   5. BOK_LOW_VAR_THRESHOLD: 1e-5 -> 3e-4 (stability/recall tradeoff)
#      from near-homogeneous groups, root cause of NaN gradient)
#   6. Softmax logit clamping added in core_algos.py (code-level NaN guard)
#   7. All other params identical to V12 (Fix1+Fix2, KL=0.02, LR=1.5e-6)
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# V13: Resume from V12 S132 checkpoint (val=0.771 best ever)
# Alternative: Resume from V12 S66 (clean, val=0.754)
# V12_RESUME_CKPT="${V12_RESUME_CKPT:-/data/workspace/hyleochang/EasyR1-latest/save/StepCount-7B-SFT-30k_v11_mask_reward_v4bok_grpo_hm0_gateoff_bok_grpo_20260313_1852/global_step_66}"
# V13_RESUME_CKPT="${V13_RESUME_CKPT:-/data/workspace/hyleochang/EasyR1-latest/save/StepCount-7B-SFT-30k_v12_mask_reward_v4bok_grpo_hm0_gateoff_bok_grpo_20260315_0136/global_step_66}"
V13_RESUME_CKPT="${V13_RESUME_CKPT:-/data/workspace/hyleochang/EasyR1-latest/save/StepCount-7B-SFT-30k_v12_mask_reward_v4bok_grpo_hm0_gateoff_bok_grpo_20260315_0136/global_step_178}"

set -x

LOG_DIR="/data/workspace/hyleochang/EasyR1-latest/logs/train"
mkdir -p ${LOG_DIR}
export output_path='./'
export ckpt_path='./ckpt'

export PYTHONUNBUFFERED=1
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16
export PYTHONHASHSEED=0

# StepCount mask reward settings
export STEPCOUNT_MASKS_METADATA=${STEPCOUNT_MASKS_METADATA:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCount-RL_Masks-Sharded/extracted/masks_metadata.json}
export STEPCOUNT_MASKS_DIR=${STEPCOUNT_MASKS_DIR:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCount-RL_Masks-Sharded/extracted/masks}
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
# v7: first_turn_prompt disabled; system prompt is now injected via data.system_prompt_file
export INTERLEAVED_FIRST_TURN_PROMPT_FILE=${INTERLEAVED_FIRST_TURN_PROMPT_FILE:-}
export INTERLEAVED_PROCESS_PROMPT_FILE=${INTERLEAVED_PROCESS_PROMPT_FILE:-${PROJECT_ROOT}/examples/format_prompt/StepCount_interleaved_process_prompt.txt}
export SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-${PROJECT_ROOT}/examples/format_prompt/StepCount_interleaved_system_prompt.txt}
export INTERLEAVED_HISTORY_MODE=${INTERLEAVED_HISTORY_MODE:-0} # 1=保留最近1轮历史（与评测对齐），-1=全历史，0=无文本历史，N>0=保留最近N轮
export INTERLEAVED_DEBUG=${INTERLEAVED_DEBUG:-1}    # 开启 rollout debug 日志
export INTERLEAVED_DEBUG_PRINT_CHARS=${INTERLEAVED_DEBUG_PRINT_CHARS:-50}
export TRAJ_RETURN_POINT_STEP_SCORES=${TRAJ_RETURN_POINT_STEP_SCORES:-1}
export TRAJ_MISS_DECAY_ENABLE=${TRAJ_MISS_DECAY_ENABLE:-1}  # dense mask point reward, 设 0 回退到原始0/1二值
export TRAJ_CONSISTENCY_PENALTY=${TRAJ_CONSISTENCY_PENALTY:-0.5} # 对point与answer不一致样本额外乘以惩罚系数
export TRAJ_ANSWER_GATE_THRESHOLD=${TRAJ_ANSWER_GATE_THRESHOLD:-0.4} # point分数低于0.4的样本的answer置于0
export TRAJ_EVAL_ANSWER_ONLY_ON_NO_MASK=${TRAJ_EVAL_ANSWER_ONLY_ON_NO_MASK:-1} # eval时若无mask序列，仅评测answer（禁用point-gate）
export TRAJ_ANSWER_GATE_MODE=${TRAJ_ANSWER_GATE_MODE:-off}  # off=无gating, soft=soft-gating, hard=hard-gating (v9 Opt-C: off to maximize correct-wrong gap)
export TRAJ_SOFT_GATE_BASE=${TRAJ_SOFT_GATE_BASE:-0.8}  # soft gating: answer_score = base + (1-base)*point_quality
export TRAJ_FORMAT_REJECTION=${TRAJ_FORMAT_REJECTION:-0}  # V11: OFF (with z-norm, rejection has negligible effect)  # 1=format不合格的trajectory reward置0
export TRAJ_SOFT_ANSWER_DECAY=${TRAJ_SOFT_ANSWER_DECAY:-1}  # v9: wrong answer gets distance-decay partial reward instead of 0
export TRAJ_ANSWER_DECAY_ALPHA=${TRAJ_ANSWER_DECAY_ALPHA:-8.0}  # v9 Opt-C: 8.0 (was 5.0), reduces high-GT wrong answer reward
export TRAJ_ANSWER_DECAY_CAP=${TRAJ_ANSWER_DECAY_CAP:-0.4}  # v9 Opt-C: cap wrong answer reward ≤ 0.4 (correct=1.0, gap≥0.48)
export TRAJ_EXTRA_POINT_PENALTY_LAMBDA=${TRAJ_EXTRA_POINT_PENALTY_LAMBDA:-1.0}  # V11-fix: penalize extra points beyond GT to eliminate overcounting bias
export TRAJ_UNDER_ALPHA_GT_SCALE=${TRAJ_UNDER_ALPHA_GT_SCALE:-0.5}  # V11-fix: GT-scaled alpha for undercounting at high GT
export TRAJ_UNDER_ALPHA_GT_THRESHOLD=${TRAJ_UNDER_ALPHA_GT_THRESHOLD:-5}  # V11-fix: GT threshold for scaled undercounting alpha

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

export STEPCOUNT_RL_MODE=${STEPCOUNT_RL_MODE:-bok_grpo}

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
    [[ -z "${ACTOR_LR}" ]] && ACTOR_LR="1.5e-6"  # V13: keep 1.5e-6 (proven in V11/V12)
    [[ -z "${ROLLOUT_TEMPERATURE}" ]] && ROLLOUT_TEMPERATURE="1.0"  # v8: keep 1.0 for controlled experiment (seed fix provides diversity)
    [[ -z "${CLIP_RATIO_LOW}" ]] && CLIP_RATIO_LOW="0.2"
    [[ -z "${CLIP_RATIO_HIGH}" ]] && CLIP_RATIO_HIGH="0.28"
    [[ -z "${CLIP_RATIO_DUAL}" ]] && CLIP_RATIO_DUAL="3.0"
    [[ -z "${DISABLE_KL}" ]] && DISABLE_KL="false"
    export BOK_TAU=${BOK_TAU:-0.3}
    export BOK_CLIP=${BOK_CLIP:-3.0}   # v8: raised from 3.0→4.0, Hard gradient +13.5%, Easy unchanged; V9: 4.0 -> 3.0
    export BOK_UNIFORM_MIX=${BOK_UNIFORM_MIX:-0.1}
    # τ annealing: cosine schedule from TAU_INIT -> TAU_FINAL over training
    # Set both > 0 to enable; they override BOK_TAU dynamically
    export BOK_TAU_INIT=${BOK_TAU_INIT:-0.7}  # keep high start tau for stable early updates
    export BOK_TAU_FINAL=${BOK_TAU_FINAL:-0.3}  # keep low-tau advantage for BoK selectivity
    # MUST set total_steps > 1 for annealing to activate (core_algos.py check)
    # NOTE: total_steps is auto-computed from meta_info; env var is fallback only
    export BOK_TOTAL_STEPS=${BOK_TOTAL_STEPS:-356} # total_epoch 1 =178
    # τ schedule: "cosine" (default, slow start/end) or "linear"
    export BOK_TAU_SCHEDULE=${BOK_TAU_SCHEDULE:-cosine}
    # Per-group advantage normalization: 0=disabled (default, per Dr.GRPO finding)
    export BOK_ADV_NORMALIZE=${BOK_ADV_NORMALIZE:-0}
    # Low-var threshold: group std < this → batch baseline fallback
    export BOK_LOW_VAR_THRESHOLD=${BOK_LOW_VAR_THRESHOLD:-3e-4}
    export BOK_LOGIT_CAP=${BOK_LOGIT_CAP:-12.0}
    export BOK_TAU_ADAPTIVE=${BOK_TAU_ADAPTIVE:-1}
    export BOK_TAU_MIN=${BOK_TAU_MIN:-0.08}
    # --- v7 New: Algorithm improvements ---
    # Fallback mode for low-var groups: zscore (default/v6) | drgrpo (no std div) | clip_std
    export BOK_FALLBACK_MODE=${BOK_FALLBACK_MODE:-drgrpo}
    # DAPO dynamic sampling: filter homogeneous groups (0=off default, 1=on)
    # WARNING: Setting to 1 with FORMAT_REJECTION=1 causes death spiral!
    #   All-fail groups get zero gradient → format collapses → more all-fail → cascading failure.
    #   Auto-disabled at runtime when low_var_rate > BOK_DAPO_AUTO_DISABLE_THRESHOLD (default 50%).
    export BOK_DAPO_FILTER=${BOK_DAPO_FILTER:-0}  # V11: OFF (V10 had 1, caused death spiral risk)
    export BOK_DAPO_AUTO_DISABLE_THRESHOLD=${BOK_DAPO_AUTO_DISABLE_THRESHOLD:-0.5}
    # Min batch std for clip_std mode (prevents over-amplification)
    export BOK_MIN_BATCH_STD=${BOK_MIN_BATCH_STD:-0.1}
    # V11 NEW: Conditional-Advantage (Difficulty-Aware Routing)
    # Easy groups (pass_rate > threshold) -> DrGRPO advantage (2.6x stronger gradient)
    # Hard groups -> BOK softmax (concentrate on rare correct trajectories)
    export BOK_EASY_THRESHOLD=${BOK_EASY_THRESHOLD:-0.75}
    export BOK_EASY_SCORE_THRESHOLD=${BOK_EASY_SCORE_THRESHOLD:-0.5}
    # P2: Zero gradient for groups where ALL K trajectories are correct (trivially solved)
    export BOK_FILTER_ALL_CORRECT=${BOK_FILTER_ALL_CORRECT:-1}
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
# - bok_grpo:      ANSWER_WEIGHT=0.8 POINT_WEIGHT=0.1 (v9: with soft_answer_decay) TRAJECTORY_FORMAT_WEIGHT=0.1
# NOTE: Use consistent weights across ALL modes for fair comparison!
# With TRAJ_FORMAT_REJECTION=1, format_weight can be lowered (format acts as filter, not reward)
export ANSWER_WEIGHT=${ANSWER_WEIGHT:-0.6}  # V11: 0.7->0.6 (V10 point_mean declined 0.94->0.85, increase point gradient)
export POINT_WEIGHT=${POINT_WEIGHT:-0.3}  # V11: 0.2->0.3 (compensate for point degradation in V10)
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

MODEL_PATH=/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/models/StepCount-7B-SFT-30k-high/checkpoint-3537
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
if [[ -n "${SYSTEM_PROMPT_FILE}" && ! -f "${SYSTEM_PROMPT_FILE}" ]]; then
    echo "[FATAL] system prompt file not found: ${SYSTEM_PROMPT_FILE}" >&2
    exit 1
fi
for required_path in \
    "${MODEL_PATH}" \
    "/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCountQA-RL-Traj_0_10" \
    "/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/pixmo-test" \
    "${STEPCOUNT_MASKS_METADATA}" \
    "${STEPCOUNT_MASKS_DIR}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[FATAL] required path not found: ${required_path}" >&2
        exit 1
    fi
done

RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE="${LOG_DIR}/training_interleaved_traj_v13_StepCount_mask_reward_v4_${STEPCOUNT_RL_MODE}_hm${INTERLEAVED_HISTORY_MODE}_gate${TRAJ_ANSWER_GATE_MODE}_fmtrej${TRAJ_FORMAT_REJECTION}_${ADV_ESTIMATOR}_${RUN_TS}.log"

echo "[RunConfig] model=${MODEL_PATH}"
echo "[RunConfig] mode=${STEPCOUNT_RL_MODE} adv=${ADV_ESTIMATOR} disable_kl=${DISABLE_KL} process_reward=${PROCESS_REWARD_ENABLE}"
echo "[RunConfig] clip_low=${CLIP_RATIO_LOW} clip_high=${CLIP_RATIO_HIGH} clip_dual=${CLIP_RATIO_DUAL}"
echo "[RunConfig] lr=${ACTOR_LR} temp=${ROLLOUT_TEMPERATURE} history_mode=${INTERLEAVED_HISTORY_MODE} rollout_n=${ROLLOUT_N} max_turns=${INTERLEAVED_MAX_TURNS}"
echo "[RunConfig] first_prompt=${INTERLEAVED_FIRST_TURN_PROMPT_FILE}"
echo "[RunConfig] process_prompt=${INTERLEAVED_PROCESS_PROMPT_FILE}"
echo "[RunConfig] system_prompt=${SYSTEM_PROMPT_FILE}"
echo "[RunConfig] reward_debug_every=${EASYR1_REWARD_DEBUG_EVERY} reward_health_every=${EASYR1_REWARD_HEALTH_DEBUG_EVERY}"
echo "[RunConfig] gate_mode=${TRAJ_ANSWER_GATE_MODE} soft_base=${TRAJ_SOFT_GATE_BASE} fmt_rej=${TRAJ_FORMAT_REJECTION} consistency_pen=${TRAJ_CONSISTENCY_PENALTY}"
echo "[RunConfig] answer_w=${ANSWER_WEIGHT} point_w=${POINT_WEIGHT} format_w=${TRAJECTORY_FORMAT_WEIGHT}"
echo "[RunConfig] log_file=${LOG_FILE}"
echo "[RunConfig] bok_fallback=${BOK_FALLBACK_MODE:-zscore} dapo_filter=${BOK_DAPO_FILTER:-0} min_batch_std=${BOK_MIN_BATCH_STD:-0.1}"
echo "[RunConfig] soft_answer_decay=${TRAJ_SOFT_ANSWER_DECAY:-0} decay_alpha=${TRAJ_ANSWER_DECAY_ALPHA:-8.0} decay_cap=${TRAJ_ANSWER_DECAY_CAP:-0.4}"
echo "[RunConfig] V13: resume_ckpt=${V13_RESUME_CKPT:-none} ppo_epochs=2 max_grad_norm=0.5 easy_threshold=${BOK_EASY_THRESHOLD:-0.75} easy_score_threshold=${BOK_EASY_SCORE_THRESHOLD:-0.5} filter_all_correct=${BOK_FILTER_ALL_CORRECT:-1}"


# ── V11 Watchdog: 自动监控 + 异常预警 ──
WATCHDOG_SCRIPT="/data/workspace/hyleochang/EasyR1-latest/tools/monitor_v11_watchdog.py"
MONITOR_DIR="${LOG_DIR}/../monitor"
mkdir -p "${MONITOR_DIR}"
WATCHDOG_LOG="${MONITOR_DIR}/watchdog_v13_${RUN_TS}.log"
WATCHDOG_JSON="${MONITOR_DIR}/metrics_v13_${RUN_TS}.jsonl"
STOP_FILE="/tmp/v13_stop_${RUN_TS}"
rm -f "${STOP_FILE}"  # clean stale stop file

if [[ -f "${WATCHDOG_SCRIPT}" ]]; then
    echo "[Watchdog] 启动后台监控: ${WATCHDOG_LOG}"
    nohup python3 "${WATCHDOG_SCRIPT}" \
        --log "${LOG_FILE}" \
        --out "${WATCHDOG_LOG}" \
        --stop-file "${STOP_FILE}" \
        --auto-stop \
        --interval 120 \
        --every 5 \
        --json-metrics "${WATCHDOG_JSON}" \
        > /dev/null 2>"${MONITOR_DIR}/watchdog_v13_stderr_${RUN_TS}.log" &
    WATCHDOG_PID=$!
    echo "[Watchdog] PID=${WATCHDOG_PID} stop_file=${STOP_FILE}"
else
    echo "[Watchdog] 脚本未找到: ${WATCHDOG_SCRIPT}, 跳过监控"
    WATCHDOG_PID=""
fi

    # lr_warmup prevents early instability
    # V13: keep epoch count at 1; stabilize numerics via adaptive BoK guards
python3 -m verl.trainer.main \
    config=${CONFIG_PATH} \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.disable_kl=${DISABLE_KL} \
    algorithm.kl_coef=2e-2 \
    data.train_files=/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCountQA-RL-Traj_0_10 \
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
    worker.actor.ppo_epochs=2 \
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
    trainer.experiment_name=StepCount-7B-SFT-30k_v13_mask_reward_v4${STEPCOUNT_RL_MODE}_hm${INTERLEAVED_HISTORY_MODE}_gate${TRAJ_ANSWER_GATE_MODE}_fmtrej${TRAJ_FORMAT_REJECTION}_${ADV_ESTIMATOR}_KL${DISABLE_KL}_ailab_$(date +%Y%m%d_%H%M) \
    trainer.logger=['console','wandb'] \
    trainer.save_checkpoint_path=/data/workspace/hyleochang/EasyR1-latest/save/StepCount-7B-SFT-30k_v13_mask_reward_v4${STEPCOUNT_RL_MODE}_hm${INTERLEAVED_HISTORY_MODE}_gate${TRAJ_ANSWER_GATE_MODE}_${ADV_ESTIMATOR}_$(date +%Y%m%d_%H%M) \
    trainer.load_checkpoint_path=${V13_RESUME_CKPT:-null} \
    trainer.total_epochs=2 \
    trainer.save_freq=20 \
    trainer.n_gpus_per_node=4 2>&1 | tee "${LOG_FILE}"

trainer_exit=${PIPESTATUS[0]}

# ── 清理 Watchdog 进程 ──
if [[ -n "${WATCHDOG_PID}" ]]; then
    echo "[Watchdog] 训练结束，终止监控 PID=${WATCHDOG_PID}"
    kill ${WATCHDOG_PID} 2>/dev/null || true
    wait ${WATCHDOG_PID} 2>/dev/null || true
    echo "[Watchdog] 监控日志: ${WATCHDOG_LOG}"
    echo "[Watchdog] 指标JSON: ${WATCHDOG_JSON}"
fi

if [[ ${trainer_exit} -ne 0 ]]; then
    echo "[FATAL] trainer failed with exit code ${trainer_exit}."
    echo "[Diag] extracting key lines for quick debug..."
    grep -nE "Traceback|Exception|RuntimeError|CUDA out of memory|NCCL|RewardError|RewardHealth|RewardDebug|trajectory_reward\[detail\]|trajectory_reward|InterleavedWarning|InterleavedDebug\] response_mask_mode=pad_based|point_step_[0-9]+|point_target_count|prompt_effective_source|question_non_empty|reward_score|overall_reward|format_fail_reward|stop_violation_reward|stopped_by_answer_reward" "${LOG_FILE}" | tail -n 260 || true
    exit ${trainer_exit}
fi

echo "[Diag] training finished successfully; latest key debug lines:"
grep -nE "RewardHealth|RewardDebug|trajectory_reward\[detail\]|trajectory_reward|InterleavedWarning|InterleavedDebug\] response_mask_mode=pad_based|point_step_[0-9]+|point_target_count|prompt_effective_source|question_non_empty|reward_score|overall_reward|format_fail_reward|stop_violation_reward|stopped_by_answer_reward" "${LOG_FILE}" | tail -n 200 || true
      