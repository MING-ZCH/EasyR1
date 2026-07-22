#!/bin/bash
# ================================================================
# V32 Self-Contained Training Script - 0-10 Sparse Stable + Dr.GRPO Easy Fix
# ================================================================
# This is a SELF-CONTAINED script that does NOT exec v31_dr_grpo_lr_uplift.sh.
# It now supports hardware profiles:
#   - h200 (default): keep normal NVIDIA driver paths, no H20 cuBLAS/GCS workaround.
#   - h20: enable the older cuBLAS/SIGFPE and Ray GCS workaround.
#
# Design:
# - V23-stable backbone (lr=1e-6, ppo=1, kl=0.03)
# - Strict JSON + format rejection
# - BoK with Dr.GRPO fallback
# - smart_filter=0.955 (per Phase0 diagnosis)
# - Fail-fast monitor enabled
# - Hardware-profile-specific CUDA/Ray setup
#
# Usage:
#   bash examples/v32_sparse_0_10_stable_drfix.sh
#
# Dry run:
#   V32_DRY_RUN=1 bash examples/v32_sparse_0_10_stable_drfix.sh
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/local_path_env.sh"
cd "${PROJECT_ROOT}"

# ==================== CUDA / Hardware Profile ====================
# H200 does not need the H20 cuBLAS/SIGFPE workaround. Keep the NVIDIA driver
# paths visible and enable the old workaround only when explicitly requested.
STEPCOUNT_HARDWARE_PROFILE=${STEPCOUNT_HARDWARE_PROFILE:-h200}
CUBLAS126=${CUBLAS126:-${LOCAL_ROOT}/eval_auto/cublas126/nvidia/cublas/lib}
CUSPARSELT_CEPH=${CUSPARSELT_CEPH:-${LOCAL_ROOT}/eval_auto/cusparselt/lib}
_ld_parts=()
if [[ "${STEPCOUNT_HARDWARE_PROFILE}" == "h20" ]]; then
    for _p in "${CUBLAS126}" "${CUSPARSELT_CEPH}" \
              /usr/local/lib/python3.12/site-packages/cusparselt/lib \
              /usr/local/lib/python3.12/site-packages/nvidia/cublas/lib; do
        [[ -d "${_p}" ]] && _ld_parts+=("${_p}")
    done
fi
for _p in /usr/local/nvidia/lib /usr/local/nvidia/lib64 /usr/local/cuda/lib64; do
    [[ -d "${_p}" ]] && _ld_parts+=("${_p}")
done
if [[ ${#_ld_parts[@]} -gt 0 ]]; then
    _ld_prefix="$(IFS=:; echo "${_ld_parts[*]}")"
    export LD_LIBRARY_PATH="${_ld_prefix}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ "${STEPCOUNT_HARDWARE_PROFILE}" == "h20" && -d "${CUBLAS126}" ]]; then
    echo "[V32] hardware=${STEPCOUNT_HARDWARE_PROFILE}; LD_LIBRARY_PATH set with cuBLAS override first: ${CUBLAS126}"
else
    echo "[V32] hardware=${STEPCOUNT_HARDWARE_PROFILE}; LD_LIBRARY_PATH prepared with available NVIDIA/CUDA library paths"
fi

# H20 + CUDA cuBLAS SIGFPE multi-layer defense, disabled by default on H200.
if [[ "${STEPCOUNT_HARDWARE_PROFILE}" == "h20" ]]; then
    export VLLM_USE_V1=${VLLM_USE_V1:-0}
    export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
    export DISABLE_ADDMM_CUDA_LT=${DISABLE_ADDMM_CUDA_LT:-1}
    export NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-1}
    export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}
elif [[ "${STEPCOUNT_H200_CLEAR_H20_WORKAROUNDS:-1}" == "1" ]]; then
    unset VLLM_USE_V1
    unset VLLM_ATTENTION_BACKEND
    unset DISABLE_ADDMM_CUDA_LT
    unset NVIDIA_TF32_OVERRIDE
    unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
fi

# ==================== Ray Configuration ====================
export RAY_DISABLE_USAGE_STATS=1
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
if [[ "${STEPCOUNT_HARDWARE_PROFILE}" == "h20" ]]; then
    # H20 GCS timeout workaround.
    export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/ray}
    export RAY_gcs_server_request_timeout_seconds=${RAY_gcs_server_request_timeout_seconds:-300}
    export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-300}
    export RAY_GCS_RPC_TIMEOUT_MS=${RAY_GCS_RPC_TIMEOUT_MS:-120000}
    export RAY_DASHBOARD_HOST=${RAY_DASHBOARD_HOST:-127.0.0.1}
    export RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-10000000000}
else
    export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray}
fi

# On the current H200 Ray cluster, leaving NCCL on the default IB path can fail
# during early FSDP broadcasts with NET/IB vendor errors. The known-good path is
# the brainpf0 socket interface with IB disabled; callers can override or disable
# this with NCCL_SOCKET_IFNAME / STEPCOUNT_H200_SOCKET_NCCL=0.
if [[ "${STEPCOUNT_HARDWARE_PROFILE}" == "h200" && "${STEPCOUNT_H200_SOCKET_NCCL:-1}" == "1" ]]; then
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${STEPCOUNT_NCCL_IFNAME:-brainpf0}}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}
    export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
fi
if [[ "${STEPCOUNT_CLEAR_PROXY:-1}" == "1" ]]; then
    unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY
    echo "[V32] proxy env cleared for offline training"
fi

# ==================== Multi-node Guard ====================
if [[ "${V32_DRY_RUN:-0}" == "1" ]]; then
    echo "[V32] Dry-run mode: skip Ray start/wait and validate launcher config only."
else
if ! command -v ray >/dev/null 2>&1; then
    if python3 -c "import ray" >/dev/null 2>&1; then
        RAY_CMD=(python3 -m ray)
        echo "[V32] ray CLI not found; using python3 -m ray"
    else
        echo "[V32][ERROR] Ray is not available in this shell. Activate the EasyR1 RL environment with ray installed before launching training." >&2
        exit 127
    fi
else
    RAY_CMD=(ray)
fi
if [[ "${INDEX:-0}" != "0" ]]; then
    echo "[V32] Worker node INDEX=${INDEX}, starting Ray worker → head=${CHIEF_IP}:6379"
    sleep 20
    mkdir -p "${RAY_TMPDIR}" 2>/dev/null || true
    "${RAY_CMD[@]}" start --address="${CHIEF_IP}:6379" --num-gpus=${HOST_GPU_NUM:-8} \
        --temp-dir="${RAY_TMPDIR}" --block
    exit 0
fi

echo "[V32] Launcher node (INDEX=${INDEX:-0}), starting Ray head + trainer..."

# Ensure Ray tmpdir exists
mkdir -p "${RAY_TMPDIR}" 2>/dev/null || true

# A hung or unhealthy GCS must not make the launcher wait forever inside one
# `ray status` call. The outer release wait has its own, longer deadline.
export RAY_STATUS_TIMEOUT_SECONDS=${RAY_STATUS_TIMEOUT_SECONDS:-10}
if ! [[ "${RAY_STATUS_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[V32][ERROR] RAY_STATUS_TIMEOUT_SECONDS must be a positive integer, got: ${RAY_STATUS_TIMEOUT_SECONDS}" >&2
    exit 1
fi
command -v timeout >/dev/null 2>&1 || {
    echo "[V32][ERROR] coreutils timeout is required for bounded Ray status probes." >&2
    exit 1
}
if [[ "${V37_REQUIRE_EXACT_RAY_GPUS:-0}" == "1" ]]; then
    export RAY_START_TIMEOUT_SECONDS=${RAY_START_TIMEOUT_SECONDS:-60}
    export RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS=${RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS:-900}
    for _ray_timeout_name in RAY_START_TIMEOUT_SECONDS RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS; do
        _ray_timeout_value="${!_ray_timeout_name}"
        if ! [[ "${_ray_timeout_value}" =~ ^[1-9][0-9]*$ ]]; then
            echo "[V32][ERROR] ${_ray_timeout_name} must be a positive integer, got: ${_ray_timeout_value}" >&2
            exit 1
        fi
    done
fi

# Start Ray head if not already running
if ! timeout --signal=KILL "${RAY_STATUS_TIMEOUT_SECONDS}" "${RAY_CMD[@]}" status >/dev/null 2>&1; then
    echo "[V32] Starting Ray HEAD with extended timeouts..."
    _ray_start_args=(
        start --head --port=6379 --num-gpus=${HOST_GPU_NUM:-8}
        --disable-usage-stats --include-dashboard=false
        --temp-dir="${RAY_TMPDIR}"
    )
    if [[ "${V37_REQUIRE_EXACT_RAY_GPUS:-0}" == "1" ]]; then
        if ! timeout --signal=KILL "${RAY_START_TIMEOUT_SECONDS}" \
                "${RAY_CMD[@]}" "${_ray_start_args[@]}"; then
            echo "[V32][ERROR] Ray head failed to start within ${RAY_START_TIMEOUT_SECONDS}s." >&2
            exit 1
        fi
    else
        "${RAY_CMD[@]}" "${_ray_start_args[@]}"
    fi
    sleep 15
fi

# Wait for all worker nodes to join
EXPECTED_GPUS=$(( ${HOST_NUM:-4} * ${HOST_GPU_NUM:-8} ))
echo "[V32] Waiting for ${EXPECTED_GPUS} GPUs in Ray cluster..."
export RAY_GPU_WAIT_TIMEOUT_SECONDS=${RAY_GPU_WAIT_TIMEOUT_SECONDS:-300}
if ! [[ "${RAY_GPU_WAIT_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[V32][ERROR] RAY_GPU_WAIT_TIMEOUT_SECONDS must be a positive integer, got: ${RAY_GPU_WAIT_TIMEOUT_SECONDS}" >&2
    exit 1
fi
MAX_WAIT=${RAY_GPU_WAIT_TIMEOUT_SECONDS}
WAIT_INTERVAL=15
WAITED=0
WAIT_STARTED=${SECONDS}
RAY_STATUS_OUTPUT=""
while true; do
    WALL_WAITED=$((SECONDS - WAIT_STARTED))
    if [ "${WALL_WAITED}" -gt "${WAITED}" ]; then
        WAITED=${WALL_WAITED}
    fi
    if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
        if [[ "${V37_REQUIRE_EXACT_RAY_GPUS:-0}" == "1" ]]; then
            if [[ "${TOTAL_GPUS_INT:-}" =~ ^[0-9]+$ ]] \
                  && [ "${TOTAL_GPUS_INT}" -eq "${EXPECTED_GPUS}" ]; then
                echo "[V32][ERROR] V37 formal requires all ${EXPECTED_GPUS} Ray GPUs to be free; timed out after ${MAX_WAIT}s with used=${USED_GPUS:-unknown}, free=${FREE_GPUS:-unknown}, total=${TOTAL_GPUS_INT}." >&2
            else
                echo "[V32][ERROR] V37 formal requires ${EXPECTED_GPUS} Ray GPUs; got ${TOTAL_GPUS_INT:-0} after ${MAX_WAIT}s." >&2
            fi
            exit 1
        fi
        echo "[V32] WARNING: Timeout waiting for GPUs. Got ${TOTAL_GPUS_INT:-0}/${EXPECTED_GPUS}."
        echo "[V32] Proceeding with available GPUs..."
        break
    fi

    STATUS_TIMEOUT=${RAY_STATUS_TIMEOUT_SECONDS}
    REMAINING=$((MAX_WAIT - WAITED))
    if [ "${STATUS_TIMEOUT}" -gt "${REMAINING}" ]; then
        STATUS_TIMEOUT=${REMAINING}
    fi
    STATUS_STARTED=${SECONDS}
    if RAY_STATUS_OUTPUT=$(timeout --signal=KILL "${STATUS_TIMEOUT}" "${RAY_CMD[@]}" status 2>/dev/null); then
        RAY_STATUS_RC=0
    else
        RAY_STATUS_RC=$?
        RAY_STATUS_OUTPUT=""
    fi
    STATUS_ELAPSED=$((SECONDS - STATUS_STARTED))
    if [ "${RAY_STATUS_RC}" -eq 124 ] && [ "${STATUS_ELAPSED}" -lt "${STATUS_TIMEOUT}" ]; then
        STATUS_ELAPSED=${STATUS_TIMEOUT}
    fi
    WAITED=$((WAITED + STATUS_ELAPSED))
    WALL_WAITED=$((SECONDS - WAIT_STARTED))
    if [ "${WALL_WAITED}" -gt "${WAITED}" ]; then
        WAITED=${WALL_WAITED}
    fi

    GPU_USAGE=$(printf '%s\n' "${RAY_STATUS_OUTPUT}" | awk '
        / GPU/ {
            for (i = 1; i < NF; i++) {
                if ($i ~ /^[0-9]+([.][0-9]+)?\/[0-9]+([.][0-9]+)?$/ && $(i + 1) == "GPU") {
                    print $i
                    exit
                }
            }
        }
    ')
    USED_GPUS=${GPU_USAGE%%/*}
    TOTAL_GPUS=${GPU_USAGE##*/}
    [[ "${GPU_USAGE}" == */* ]] || { USED_GPUS=""; TOTAL_GPUS=""; }
    TOTAL_GPUS_INT=""
    if [[ "${TOTAL_GPUS}" =~ ^([0-9]+)([.]0+)?$ ]]; then
        TOTAL_GPUS_INT=${BASH_REMATCH[1]}
    elif [[ "${TOTAL_GPUS}" =~ ^[0-9]+[.][0-9]+$ ]]; then
        if [[ "${V37_REQUIRE_EXACT_RAY_GPUS:-0}" == "1" ]]; then
            echo "[V32][ERROR] V37 formal requires an integral Ray GPU total, got ${TOTAL_GPUS}." >&2
            exit 1
        fi
        TOTAL_GPUS_INT=${TOTAL_GPUS%%.*}
    fi
    FREE_GPUS=$(awk -v total="${TOTAL_GPUS:-0}" -v used="${USED_GPUS:-0}" \
        'BEGIN { printf "%.3f", total - used }')
    if [[ "${TOTAL_GPUS_INT}" =~ ^[0-9]+$ ]] \
          && [ "${TOTAL_GPUS_INT}" -ge "${EXPECTED_GPUS}" ]; then
        if [[ "${V37_REQUIRE_EXACT_RAY_GPUS:-0}" == "1" \
              && "${TOTAL_GPUS_INT}" -ne "${EXPECTED_GPUS}" ]]; then
            echo "[V32][ERROR] V37 formal requires exactly ${EXPECTED_GPUS} Ray GPUs, got ${TOTAL_GPUS_INT}." >&2
            exit 1
        fi
        if [[ "${V37_REQUIRE_EXACT_RAY_GPUS:-0}" != "1" \
              || "${USED_GPUS:-}" =~ ^0+([.]0+)?$ ]]; then
            echo "[V32] Ray GPUs ready: used=${USED_GPUS:-unknown}, free=${FREE_GPUS}, total=${TOTAL_GPUS_INT}."
            break
        fi
    fi
    if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
        if [[ "${V37_REQUIRE_EXACT_RAY_GPUS:-0}" == "1" ]]; then
            if [[ "${TOTAL_GPUS_INT:-0}" -eq "${EXPECTED_GPUS}" ]]; then
                echo "[V32][ERROR] V37 formal requires all ${EXPECTED_GPUS} Ray GPUs to be free; timed out after ${MAX_WAIT}s with used=${USED_GPUS:-unknown}, free=${FREE_GPUS}, total=${TOTAL_GPUS_INT}." >&2
            else
                echo "[V32][ERROR] V37 formal requires ${EXPECTED_GPUS} Ray GPUs; got ${TOTAL_GPUS_INT:-0} after ${MAX_WAIT}s." >&2
            fi
            exit 1
        fi
        echo "[V32] WARNING: Timeout waiting for GPUs. Got ${TOTAL_GPUS_INT:-0}/${EXPECTED_GPUS}."
        echo "[V32] Proceeding with available GPUs..."
        break
    fi
    SLEEP_FOR=${WAIT_INTERVAL}
    REMAINING=$((MAX_WAIT - WAITED))
    if [ "${SLEEP_FOR}" -gt "${REMAINING}" ]; then
        SLEEP_FOR=${REMAINING}
    fi
    echo "[V32] Waiting... (used=${USED_GPUS:-unknown}, free=${FREE_GPUS}, total=${TOTAL_GPUS_INT:-0}, expected=${EXPECTED_GPUS}, ${WAITED}s/${MAX_WAIT}s elapsed)"
    sleep "${SLEEP_FOR}"
    WAITED=$((WAITED + SLEEP_FOR))
done

echo "[V32] Ray cluster status:"
if [[ -n "${RAY_STATUS_OUTPUT}" ]]; then
    printf '%s\n' "${RAY_STATUS_OUTPUT}"
else
    echo "[V32] Ray status output unavailable from the final bounded probe."
fi

export RAY_ADDRESS="auto"
echo "[V32] RAY_ADDRESS=${RAY_ADDRESS}"
fi

# ==================== Environment ====================
LOG_DIR="${EASYR1_LOG_ROOT}/rl"
DEBUG_LOG_DIR="${EASYR1_LOG_ROOT}/debug"
mkdir -p "${LOG_DIR}" "${DEBUG_LOG_DIR}"
export TRAINING_DEBUG_LOG_DIR="${DEBUG_LOG_DIR}"
export output_path='./'
export ckpt_path='./ckpt'
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-online}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

# ==================== Core Recipe (V23-stable backbone + Dr fallback) ====================
export STEPCOUNT_TRAIN_DATA=${STEPCOUNT_TRAIN_DATA:-${STEPCOUNT_REPLAY_DATA}}
export BOK_TOTAL_STEPS=${BOK_TOTAL_STEPS:-49}
export TRAIN_OVERSAMPLE_NO_MASK_FACTOR=${TRAIN_OVERSAMPLE_NO_MASK_FACTOR:-1}
export TRAIN_HARD_OVERSAMPLE_FACTOR=${TRAIN_HARD_OVERSAMPLE_FACTOR:-1}
export TRAIN_SAVE_FREQ=${TRAIN_SAVE_FREQ:-20}
export TRAIN_SAVE_LIMIT=${TRAIN_SAVE_LIMIT:-6}
export TRAIN_VAL_FREQ=${TRAIN_VAL_FREQ:-20}

export ACTOR_LR=${ACTOR_LR:-1e-6}
export DISABLE_KL=${DISABLE_KL:-false}
export KL_COEF=${KL_COEF:-0.03}
export USE_KL_LOSS=${USE_KL_LOSS:-true}
export ADAPTIVE_ACTOR_KL=${ADAPTIVE_ACTOR_KL:-false}
export KL_TYPE=${KL_TYPE:-fixed}
export KL_TARGET=${KL_TARGET:-0.0}
export KL_HORIZON=${KL_HORIZON:-0.0}
export KL_PENALTY=${KL_PENALTY:-low_var_kl}
export TORCH_LOGPROB_FALLBACK_MODE=${TORCH_LOGPROB_FALLBACK_MODE:-legacy}
export CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
export CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
export CLIP_RATIO_DUAL=${CLIP_RATIO_DUAL:-3.0}

export STEPCOUNT_RL_MODE=${STEPCOUNT_RL_MODE:-bok_grpo}
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-bok_grpo}
export PROCESS_REWARD_ENABLE=${PROCESS_REWARD_ENABLE:-0}
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

# ==================== Reward & Trajectory ====================
# masks_metadata.json: use StepCount-RL_Masks-Full (282,454 entries, complete).
# masks dir: use StepCount-RL_Masks-Sharded which has all 282,455 .npz files.
# (StepCount-RL_Masks-Full/extracted/masks only has 9,384 files — shards truncated.)
export STEPCOUNT_MASKS_METADATA=${STEPCOUNT_MASKS_METADATA:-${LOCAL_ROOT}/StepCount-RL_masks_output/masks_metadata.json}
export STEPCOUNT_MASKS_DIR=${STEPCOUNT_MASKS_DIR:-${LOCAL_ROOT}/StepCount-RL_masks_output/masks}
export STEPCOUNT_MASK_REQUIRE=${STEPCOUNT_MASK_REQUIRE:-1}
export STEPCOUNT_MASK_PREFILL_BY_TURN=${STEPCOUNT_MASK_PREFILL_BY_TURN:-1}
export STEPCOUNT_MASK_LOG_CONFIG=${STEPCOUNT_MASK_LOG_CONFIG:-1}
# --- Mask LRU cache (Opt 2): eliminates repeated CephFS np.load across rollouts/steps.
# Pod has ~1881 GB RAM; all ~282K masks fit easily. Cap at 300K to be safe.
# STEPCOUNT_MASK_CACHE=0 disables (legacy behaviour, every call re-loads from disk).
export STEPCOUNT_MASK_CACHE=${STEPCOUNT_MASK_CACHE:-1}
export STEPCOUNT_MASK_CACHE_MAX=${STEPCOUNT_MASK_CACHE_MAX:-300000}
# --- Reward worker parallelism: more CephFS I/O threads (default was 4).
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}
export STEPCOUNT_MASK_DEBUG=${STEPCOUNT_MASK_DEBUG:-1}
export STEPCOUNT_MASK_DEBUG_EVERY=${STEPCOUNT_MASK_DEBUG_EVERY:-200}
export EASYR1_REWARD_DEBUG_EVERY=${EASYR1_REWARD_DEBUG_EVERY:-200}
export EASYR1_REWARD_SAMPLE_DEBUG=${EASYR1_REWARD_SAMPLE_DEBUG:-1}
export EASYR1_REWARD_SAMPLE_DEBUG_MAX=${EASYR1_REWARD_SAMPLE_DEBUG_MAX:-1}
export EASYR1_REWARD_HEALTH_DEBUG=${EASYR1_REWARD_HEALTH_DEBUG:-1}
export EASYR1_REWARD_HEALTH_DEBUG_EVERY=${EASYR1_REWARD_HEALTH_DEBUG_EVERY:-1}
export STEPCOUNT_TRAJ_REASON_DEBUG=${STEPCOUNT_TRAJ_REASON_DEBUG:-1}
export STEPCOUNT_TRAJ_REASON_DEBUG_EVERY=${STEPCOUNT_TRAJ_REASON_DEBUG_EVERY:-200}
export STEPCOUNT_TRAJ_EVENT_LOG=${STEPCOUNT_TRAJ_EVENT_LOG:-1}
export STEPCOUNT_TRAJ_EVENT_LOG_EVERY=${STEPCOUNT_TRAJ_EVENT_LOG_EVERY:-200}
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
export TRAJ_POINT_STRICT_JSON=${TRAJ_POINT_STRICT_JSON:-1}
export TRAJ_SOFT_ANSWER_DECAY=${TRAJ_SOFT_ANSWER_DECAY:-1}
export TRAJ_ANSWER_DECAY_ALPHA=${TRAJ_ANSWER_DECAY_ALPHA:-8.0}
export TRAJ_ANSWER_DECAY_CAP=${TRAJ_ANSWER_DECAY_CAP:-0.4}
export TRAJ_EXTRA_POINT_PENALTY_LAMBDA=${TRAJ_EXTRA_POINT_PENALTY_LAMBDA:-1.0}
export TRAJ_UNDER_ALPHA_GT_SCALE=${TRAJ_UNDER_ALPHA_GT_SCALE:-0.5}
export TRAJ_UNDER_ALPHA_GT_THRESHOLD=${TRAJ_UNDER_ALPHA_GT_THRESHOLD:-5}
export ANSWER_WEIGHT=${ANSWER_WEIGHT:-0.6}
export POINT_WEIGHT=${POINT_WEIGHT:-0.3}
export TRAJECTORY_FORMAT_WEIGHT=${TRAJECTORY_FORMAT_WEIGHT:-0.1}

# ==================== Interleaved Rollout ====================
export ROLLOUT_N=${ROLLOUT_N:-16}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export INTERLEAVED_MAX_TURNS=${INTERLEAVED_MAX_TURNS:-11}
export INTERLEAVED_PER_TURN_MAX_TOKENS=${INTERLEAVED_PER_TURN_MAX_TOKENS:-1800}
export INTERLEAVED_ANSWER_TURN_MAX_TOKENS=${INTERLEAVED_ANSWER_TURN_MAX_TOKENS:-1800}
export INTERLEAVED_ADAPTIVE_MAX_TURNS=${INTERLEAVED_ADAPTIVE_MAX_TURNS:-false}
export INTERLEAVED_ADAPTIVE_MAX_TURNS_MARGIN=${INTERLEAVED_ADAPTIVE_MAX_TURNS_MARGIN:-2}
export INTERLEAVED_POINT_TURN_USE_ANSWER_BUDGET=${INTERLEAVED_POINT_TURN_USE_ANSWER_BUDGET:-false}
export INTERLEAVED_FIRST_TURN_PROMPT_FILE=${INTERLEAVED_FIRST_TURN_PROMPT_FILE:-}
export INTERLEAVED_PROCESS_PROMPT_FILE=${INTERLEAVED_PROCESS_PROMPT_FILE:-${PROJECT_ROOT}/examples/format_prompt/StepCount_interleaved_process_prompt.txt}
export SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-${PROJECT_ROOT}/examples/format_prompt/StepCount_interleaved_system_prompt.txt}
export INTERLEAVED_HISTORY_MODE=${INTERLEAVED_HISTORY_MODE:-0}
export INTERLEAVED_DEBUG=${INTERLEAVED_DEBUG:-1}
export INTERLEAVED_DEBUG_PRINT_CHARS=${INTERLEAVED_DEBUG_PRINT_CHARS:-50}

# ==================== Gradient Safety ====================
export GRAD_SPIKE_PROTECT=${GRAD_SPIKE_PROTECT:-1}
export GRAD_SPIKE_THRESHOLD=${GRAD_SPIKE_THRESHOLD:-3.0}
export GRAD_SPIKE_COOLDOWN=${GRAD_SPIKE_COOLDOWN:-3}
export GRAD_SPIKE_LR_FACTOR=${GRAD_SPIKE_LR_FACTOR:-0.1}
export GRAD_SPIKE_BRAKE_WINDOW=${GRAD_SPIKE_BRAKE_WINDOW:-50}
export GRAD_SPIKE_BRAKE_MAX=${GRAD_SPIKE_BRAKE_MAX:-4}
export GRAD_SPIKE_ABSOLUTE_CAP=${GRAD_SPIKE_ABSOLUTE_CAP:-0}
export GRAD_NONFINITE_COOLDOWN=${GRAD_NONFINITE_COOLDOWN:-12}
export GRAD_NONFINITE_LR_FACTOR=${GRAD_NONFINITE_LR_FACTOR:-0.05}
export GRAD_NONFINITE_BRAKE_WINDOW=${GRAD_NONFINITE_BRAKE_WINDOW:-40}
export GRAD_NONFINITE_BRAKE_MAX=${GRAD_NONFINITE_BRAKE_MAX:-2}
export FP16_GRAD_UNDERFLOW_MONITOR=${FP16_GRAD_UNDERFLOW_MONITOR:-0}

# ==================== Fail-fast ====================
export V30_FAILFAST_ENABLE=${V30_FAILFAST_ENABLE:-1}
export V30_FAILFAST_NAN_LIMIT=${V30_FAILFAST_NAN_LIMIT:-6}
export V30_FAILFAST_ENTROPY_CRITICAL=${V30_FAILFAST_ENTROPY_CRITICAL:-0.85}
export V30_FAILFAST_ENTROPY_CONSEC=${V30_FAILFAST_ENTROPY_CONSEC:-3}
export V30_FAILFAST_FORMAT_CRITICAL=${V30_FAILFAST_FORMAT_CRITICAL:-0.30}
export V30_FAILFAST_FORMAT_CONSEC=${V30_FAILFAST_FORMAT_CONSEC:-3}
export V30_FAILFAST_LOWVAR_CRITICAL=${V30_FAILFAST_LOWVAR_CRITICAL:-45}
export V30_FAILFAST_ZEROREWARD_CRITICAL=${V30_FAILFAST_ZEROREWARD_CRITICAL:-30}
export V30_FAILFAST_BOK_CONSEC=${V30_FAILFAST_BOK_CONSEC:-2}

# ==================== 32-GPU Topology ====================
V31_NNODES=${V31_NNODES:-${HOST_NUM:-4}}
V31_N_GPUS_PER_NODE=${V31_N_GPUS_PER_NODE:-${HOST_GPU_NUM:-8}}
V31_ROLLOUT_BATCH_SIZE=${V31_ROLLOUT_BATCH_SIZE:-256}
V31_VAL_BATCH_SIZE=${V31_VAL_BATCH_SIZE:-256}
V31_GLOBAL_BATCH_SIZE=${V31_GLOBAL_BATCH_SIZE:-256}
# OOM fix: reduce from 8 to 4 — halves backward activation peak memory.
# Root cause: interleaved rollouts with max-length (10-turn) sequences create
# large activation tensors during backward; 8 micro-batches * long seq overflows 95GB.
V31_MICRO_BATCH_UPDATE=${V31_MICRO_BATCH_UPDATE:-4}
V31_MICRO_BATCH_EXP=${V31_MICRO_BATCH_EXP:-8}
V31_GPU_MEM_UTIL=${V31_GPU_MEM_UTIL:-0.65}
V31_MAX_MODEL_LEN=${V31_MAX_MODEL_LEN:-16384}
V31_MAX_NUM_BATCHED_TOKENS=${V31_MAX_NUM_BATCHED_TOKENS:-32768}
V31_ENFORCE_EAGER=${V31_ENFORCE_EAGER:-true}
V31_FILTER_OVERLONG_NUM_PROC=${V31_FILTER_OVERLONG_NUM_PROC:-${EASYR1_FILTER_OVERLONG_NUM_PROC:-64}}
export EASYR1_FILTER_OVERLONG_NUM_PROC=${EASYR1_FILTER_OVERLONG_NUM_PROC:-${V31_FILTER_OVERLONG_NUM_PROC}}
TRAINER_PROJECT_NAME=${TRAINER_PROJECT_NAME:-easy_r1}
TRAINER_VAL_BEFORE_TRAIN=${TRAINER_VAL_BEFORE_TRAIN:-true}
TRAINER_VAL_ONLY=${TRAINER_VAL_ONLY:-false}
TRAINER_VAL_GENERATIONS_TO_LOG=${TRAINER_VAL_GENERATIONS_TO_LOG:-3}

# ==================== Paths ====================
MODEL_PATH=${MODEL_PATH:-${STEPCOUNT_SFT_BASE_MODEL_PATH}}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/examples/config.yaml}
REWARD_FN_PATH=${REWARD_FN_PATH:-${STEPCOUNT_REWARD_FN_PATH}}

export V31_EXPERIMENT_NAME=${V31_EXPERIMENT_NAME:-StepCount-7B_v32_sparse_0_10_stable_drfix_32gpu_$(date +%Y%m%d_%H%M)}
export V31_SAVE_CHECKPOINT_PATH=${V31_SAVE_CHECKPOINT_PATH:-${EASYR1_SAVE_ROOT}/${V31_EXPERIMENT_NAME}}

# ==================== Validation ====================
for required_file in "${CONFIG_PATH}" "${PROJECT_ROOT}/examples/reward_function/StepCount_mask_reward.py"; do
    [[ -f "${required_file}" ]] || { echo "[FATAL] required file not found: ${required_file}" >&2; exit 1; }
done
for required_path in "${MODEL_PATH}" "${STEPCOUNT_TRAIN_DATA}" "${STEPCOUNT_MASKS_METADATA}" "${STEPCOUNT_MASKS_DIR}"; do
    [[ -e "${required_path}" ]] || { echo "[FATAL] required path not found: ${required_path}" >&2; exit 1; }
done
IFS=',' read -r -a _val_specs <<< "${STEPCOUNT_VAL_DATA}"
for _val_spec in "${_val_specs[@]}"; do
    _val_path="${_val_spec#*::}"
    [[ -e "${_val_path}" ]] || { echo "[FATAL] val path not found: ${_val_path} (from ${_val_spec})" >&2; exit 1; }
done
[[ -z "${INTERLEAVED_PROCESS_PROMPT_FILE}" || -f "${INTERLEAVED_PROCESS_PROMPT_FILE}" ]] || { echo "[FATAL] process prompt not found: ${INTERLEAVED_PROCESS_PROMPT_FILE}" >&2; exit 1; }
[[ -z "${SYSTEM_PROMPT_FILE}" || -f "${SYSTEM_PROMPT_FILE}" ]] || { echo "[FATAL] system prompt not found: ${SYSTEM_PROMPT_FILE}" >&2; exit 1; }

# ==================== Print Config ====================
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE="${LOG_DIR}/v32_standalone_${RUN_TS}.log"
MONITOR_SCRIPT="${PROJECT_ROOT}/tools/monitor_training_v2.py"
FAILFAST_SCRIPT="${PROJECT_ROOT}/tools/monitor_v28_failfast.py"
MONITOR_DIR="${LOG_DIR}/../monitor"
mkdir -p "${MONITOR_DIR}"
MONITOR_LOG="${MONITOR_DIR}/monitor_v32_${RUN_TS}.log"
MONITOR_JSON="${MONITOR_DIR}/metrics_v32_${RUN_TS}.jsonl"
FAILFAST_LOG="${MONITOR_DIR}/failfast_v32_${RUN_TS}.log"
export STOP_FILE="/tmp/v32_stop_${RUN_TS}"
rm -f "${STOP_FILE}"

if [[ -n "${V37_EFFECTIVE_ENVIRONMENT_PATH:-}" ]]; then
  python3 - "${V37_EFFECTIVE_ENVIRONMENT_PATH}" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

target = Path(sys.argv[1])
expected = Path(os.environ["V31_SAVE_CHECKPOINT_PATH"]) / "v37_effective_environment.json"
target = Path(os.path.abspath(os.path.expanduser(str(target))))
expected = Path(os.path.abspath(os.path.expanduser(str(expected))))
current = Path(target.anchor)
for part in target.parts[1:-1]:
    current /= part
    metadata = os.lstat(current)
    if stat.S_ISLNK(metadata.st_mode):
        raise SystemExit(f"V37 effective environment path contains a symlink: {current}")
if target != expected or target.exists() or target.is_symlink():
    raise SystemExit("invalid or pre-existing V37 effective environment target")
if os.environ.get("BASH_ENV") or os.environ.get("ENV") or any(
    key.startswith("BASH_FUNC_") for key in os.environ
):
    raise SystemExit("forbidden shell startup/function environment reached V37 training entry")
audited_prefixes = (
    "ACTION_", "ACTOR_", "ADAPTIVE_", "ANSWER_", "BOK_", "CLIP_", "EASYR1_", "GRAD_", "INTERLEAVED_",
    "KL_", "POINT_", "POLICY_", "PROCESS_", "REWARD_", "ROLLOUT_", "STEPCOUNT_",
    "TRAIN_", "TRAINER_", "TRAJECTORY_", "TRAJ_", "V31_", "V32_", "V36_", "V37_", "VCRL_",
    "CUBLAS_", "CUDA_", "FLASH_", "FSDP_", "MKL_", "NCCL_", "OMP_", "PYTORCH_",
    "RAY_", "TOKENIZERS_", "TORCH_", "TRANSFORMERS_", "VLLM_", "XFORMERS_",
    "FI_", "GLOO_", "MASTER_", "NVIDIA_", "OMPI_", "PMI_", "PMIX_", "TRITON_", "UCX_",
)
allowlist = {
    "CC", "CHIEF_IP", "CONFIG_PATH", "CONDA_PREFIX", "CUDA_HOME", "CXX", "DISABLE_KL",
    "HOST_GPU_NUM", "HOST_NUM", "INDEX", "LD_LIBRARY_PATH", "LD_PRELOAD", "MAX_STEPS", "MODEL_PATH",
    "HF_DATASETS_OFFLINE", "HF_HOME", "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_OFFLINE",
    "HOME", "HOSTNAME", "LANG", "LC_ALL", "LOCAL_RANK", "LOGNAME", "PATH", "PWD", "PYTHONHASHSEED",
    "PYTHONIOENCODING", "PYTHONNOUSERSITE", "PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONWARNINGS",
    "RANK", "SHELL", "SYSTEM_PROMPT_FILE", "TERM", "TMP", "TEMP", "TMPDIR", "TZ", "USE_KL_LOSS",
    "USER", "VIRTUAL_ENV", "WANDB_DIR", "WANDB_ENTITY", "WANDB_MODE", "WANDB_PROJECT", "WORLD_SIZE",
    "XDG_CACHE_HOME",
}
environment = {
    key: value for key, value in sorted(os.environ.items())
    if key.startswith(audited_prefixes) or key in allowlist
}
encoded = json.dumps(environment, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
payload = {
    "schema_version": 1,
    "contract": "v37_effective_pre_trainer_environment_v1",
    "environment_sha256": hashlib.sha256(encoded).hexdigest(),
    "environment": environment,
}
temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(temporary, flags, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
if not stat.S_ISREG(os.lstat(temporary).st_mode):
    raise SystemExit("V37 effective environment temporary is not a regular file")
try:
    os.link(temporary, target, follow_symlinks=False)
finally:
    temporary.unlink(missing_ok=True)
PY
fi

echo "================================================================"
echo "[V32] Self-Contained — $((V31_NNODES * V31_N_GPUS_PER_NODE)) GPU (${V31_NNODES} nodes × ${V31_N_GPUS_PER_NODE} GPUs)"
echo "[V32] CUDA library path prepared; profile=${STEPCOUNT_HARDWARE_PROFILE} CUBLAS126=${CUBLAS126}"
echo "[V32] Ray tmpdir=${RAY_TMPDIR}"
echo "[V32] network: NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<cluster-default>} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-<cluster-default>} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-<cluster-default>}"
echo "[V32] model=${MODEL_PATH}"
echo "[V32] data=${STEPCOUNT_TRAIN_DATA}"
echo "[V32] lr=${ACTOR_LR} kl=${KL_COEF} total_steps=${BOK_TOTAL_STEPS}"
echo "[V32] batch: rollout=${V31_ROLLOUT_BATCH_SIZE} global=${V31_GLOBAL_BATCH_SIZE}"
echo "[V32] actor: micro=${V31_MICRO_BATCH_UPDATE}/${V31_MICRO_BATCH_EXP} padding_free=true ulysses_sp=${V31_ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
echo "[V32] BOK: smart_filter=${BOK_SMART_FILTER_THRESHOLD} fallback=${BOK_FALLBACK_MODE}"
echo "[V32] trainer: project=${TRAINER_PROJECT_NAME} val_before_train=${TRAINER_VAL_BEFORE_TRAIN} val_only=${TRAINER_VAL_ONLY} val_log=${TRAINER_VAL_GENERATIONS_TO_LOG}"
echo "[V32] data: filter_overlong_num_proc=${V31_FILTER_OVERLONG_NUM_PROC}"
echo "[V32] save: freq=${TRAIN_SAVE_FREQ} path=${V31_SAVE_CHECKPOINT_PATH}"
echo "[V32] log=${LOG_FILE}"
echo "================================================================"

if [[ "${V32_DRY_RUN:-0}" == "1" ]]; then
    echo "[DryRun] Validation passed."
    echo "  LD_LIBRARY_PATH starts with: $(echo $LD_LIBRARY_PATH | cut -d: -f1)"
    exit 0
fi

# ==================== Monitor & Failfast ====================
if [[ -f "${MONITOR_SCRIPT}" ]]; then
    nohup python3 "${MONITOR_SCRIPT}" \
        --log "${LOG_FILE}" \
        --out "${MONITOR_LOG}" \
        --stop-file "${STOP_FILE}" \
        --interval 60 --every 5 --summary-every 20 \
        --json-metrics "${MONITOR_JSON}" \
        > /dev/null 2>"${MONITOR_DIR}/monitor_v32_stderr_${RUN_TS}.log" &
    MONITOR_PID=$!
else
    MONITOR_PID=""
fi

if [[ "${V30_FAILFAST_ENABLE}" == "1" && -f "${FAILFAST_SCRIPT}" ]]; then
    nohup python3 "${FAILFAST_SCRIPT}" \
        --run-name V32 \
        --log "${LOG_FILE}" \
        --out "${FAILFAST_LOG}" \
        --stop-file "${STOP_FILE}" \
        --nan-limit "${V30_FAILFAST_NAN_LIMIT}" \
        --entropy-critical "${V30_FAILFAST_ENTROPY_CRITICAL}" \
        --entropy-consec "${V30_FAILFAST_ENTROPY_CONSEC}" \
        --format-critical "${V30_FAILFAST_FORMAT_CRITICAL}" \
        --format-consec "${V30_FAILFAST_FORMAT_CONSEC}" \
        --lowvar-critical "${V30_FAILFAST_LOWVAR_CRITICAL}" \
        --zeroreward-critical "${V30_FAILFAST_ZEROREWARD_CRITICAL}" \
        --bok-consec "${V30_FAILFAST_BOK_CONSEC}" \
        > /dev/null 2>"${MONITOR_DIR}/failfast_v32_stderr_${RUN_TS}.log" &
    FAILFAST_PID=$!
else
    FAILFAST_PID=""
fi

cleanup_monitor() {
    if [[ -n "${MONITOR_PID:-}" ]]; then kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; fi
    if [[ -n "${FAILFAST_PID:-}" ]]; then kill "${FAILFAST_PID}" 2>/dev/null || true; wait "${FAILFAST_PID}" 2>/dev/null || true; fi
}
trap cleanup_monitor EXIT

# ==================== Launch Training ====================
python3 -m verl.trainer.main \
    config=${CONFIG_PATH} \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.disable_kl=${DISABLE_KL} \
    algorithm.use_kl_loss=${USE_KL_LOSS} \
    algorithm.adaptive_actor_kl=${ADAPTIVE_ACTOR_KL} \
    algorithm.kl_type=${KL_TYPE} \
    algorithm.kl_target=${KL_TARGET} \
    algorithm.kl_horizon=${KL_HORIZON} \
    algorithm.kl_penalty=${KL_PENALTY} \
    algorithm.kl_coef=${KL_COEF} \
    data.train_files=${STEPCOUNT_TRAIN_DATA} \
    data.system_prompt_file=${SYSTEM_PROMPT_FILE} \
    data.format_prompt=null \
    data.val_files="'${STEPCOUNT_VAL_DATA}'" \
    data.max_prompt_length=${V31_MAX_PROMPT_LENGTH:-7500} \
    data.max_response_length=${V31_MAX_RESPONSE_LENGTH:-3200} \
    data.max_pixels=${V31_MAX_PIXELS:-12845056} \
    data.min_pixels=${V31_MIN_PIXELS:-262144} \
    data.filter_overlong_num_proc=${V31_FILTER_OVERLONG_NUM_PROC} \
    data.shuffle=true \
    data.seed=${V31_DATA_SEED:-42} \
    worker.actor.optim.lr=${ACTOR_LR} \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    worker.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    worker.actor.clip_ratio_dual=${CLIP_RATIO_DUAL} \
    worker.actor.ppo_epochs=1 \
    worker.actor.max_grad_norm=1.0 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.padding_free=true \
    worker.actor.torch_logprob_fallback_mode=${TORCH_LOGPROB_FALLBACK_MODE} \
    worker.actor.ulysses_sequence_parallel_size=${V31_ULYSSES_SEQUENCE_PARALLEL_SIZE:-1} \
    worker.actor.micro_batch_size_per_device_for_update=${V31_MICRO_BATCH_UPDATE} \
    worker.actor.micro_batch_size_per_device_for_experience=${V31_MICRO_BATCH_EXP} \
    worker.rollout.gpu_memory_utilization=${V31_GPU_MEM_UTIL} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_model_len=${V31_MAX_MODEL_LEN} \
    worker.rollout.max_num_batched_tokens=${V31_MAX_NUM_BATCHED_TOKENS} \
    worker.rollout.enforce_eager=${V31_ENFORCE_EAGER} \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    worker.rollout.top_p=${ROLLOUT_TOP_P:-1.0} \
    worker.rollout.seed=${V31_ROLLOUT_SEED:-1} \
    worker.rollout.stop='["</answer>"]' \
    worker.rollout.interleaved_point_to_count=true \
    worker.rollout.interleaved_max_turns=${INTERLEAVED_MAX_TURNS} \
    worker.rollout.interleaved_adaptive_max_turns=${INTERLEAVED_ADAPTIVE_MAX_TURNS} \
    worker.rollout.interleaved_adaptive_max_turns_margin=${INTERLEAVED_ADAPTIVE_MAX_TURNS_MARGIN} \
    worker.rollout.interleaved_point_turn_use_answer_budget=${INTERLEAVED_POINT_TURN_USE_ANSWER_BUDGET} \
    worker.rollout.interleaved_per_turn_max_tokens=${INTERLEAVED_PER_TURN_MAX_TOKENS} \
    worker.rollout.interleaved_answer_turn_max_tokens=${INTERLEAVED_ANSWER_TURN_MAX_TOKENS} \
    worker.rollout.interleaved_history_mode=${INTERLEAVED_HISTORY_MODE} \
    worker.rollout.interleaved_first_turn_prompt_file=${INTERLEAVED_FIRST_TURN_PROMPT_FILE} \
    worker.rollout.interleaved_process_prompt_file=${INTERLEAVED_PROCESS_PROMPT_FILE} \
    worker.rollout.interleaved_process_prompt_sha256=${INTERLEAVED_PROCESS_PROMPT_SHA256:-null} \
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
    worker.reward.reward_function_kwargs.adaptive_max_turns=${INTERLEAVED_ADAPTIVE_MAX_TURNS} \
    worker.reward.reward_function_kwargs.adaptive_max_turns_margin=${INTERLEAVED_ADAPTIVE_MAX_TURNS_MARGIN} \
    trainer.project_name=${TRAINER_PROJECT_NAME} \
    trainer.experiment_name=${V31_EXPERIMENT_NAME} \
    trainer.logger=['console','wandb'] \
    trainer.save_checkpoint_path=${V31_SAVE_CHECKPOINT_PATH} \
    trainer.total_epochs=${STEPCOUNT_TOTAL_EPOCHS:-1} \
    ${MAX_STEPS:+trainer.max_steps=${MAX_STEPS}} \
    trainer.save_freq=${TRAIN_SAVE_FREQ} \
    trainer.save_limit=${TRAIN_SAVE_LIMIT} \
    trainer.val_freq=${TRAIN_VAL_FREQ} \
    trainer.val_before_train=${TRAINER_VAL_BEFORE_TRAIN} \
    trainer.val_only=${TRAINER_VAL_ONLY} \
    trainer.val_generations_to_log=${TRAINER_VAL_GENERATIONS_TO_LOG} \
    ${V32_LOAD_CHECKPOINT_PATH:+trainer.load_checkpoint_path=${V32_LOAD_CHECKPOINT_PATH}} \
    ${V37_RUN_CLASS:+trainer.v37_run_class=${V37_RUN_CLASS}} \
    ${V37_RESUME_MODE:+trainer.v37_resume_mode=${V37_RESUME_MODE}} \
    ${V37_EXPECTED_RESUME_CHECKPOINT_PATH:+trainer.v37_expected_resume_checkpoint_path=${V37_EXPECTED_RESUME_CHECKPOINT_PATH}} \
    ${V37_EXPECTED_RESUME_CHECKPOINT_SHA256:+trainer.v37_expected_resume_checkpoint_sha256=${V37_EXPECTED_RESUME_CHECKPOINT_SHA256}} \
    data.rollout_batch_size=${V31_ROLLOUT_BATCH_SIZE} \
    data.val_batch_size=${V31_VAL_BATCH_SIZE} \
    worker.actor.global_batch_size=${V31_GLOBAL_BATCH_SIZE} \
    trainer.nnodes=${V31_NNODES} \
    trainer.n_gpus_per_node=${V31_N_GPUS_PER_NODE} 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
echo "[V32] Training finished with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
