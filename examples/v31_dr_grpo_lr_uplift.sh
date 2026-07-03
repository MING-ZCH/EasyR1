#!/bin/bash
# ================================================================
# V31 Launch Script - DR-GRPO LR Uplift (32 GPU / 4 Nodes)
# ================================================================
# Base: V30 V27-Lifted Stable, with changes:
#   - 32 GPU: 4 nodes × 8 H20 GPUs (Taiji gz cluster)
#   - DR-GRPO fallback as primary (more stable than BoK for multi-node)
#   - LR uplifted to 1.5e-6 (from 1.25e-6) to compensate larger batch
#   - Adjusted batch sizes for 32-GPU topology
#   - Multi-node guard: only launcher (INDEX=0) runs trainer; workers sleep
#
# Usage (在 Pod launcher 节点执行):
#   cd /apdcephfs/private_hyleochang/EasyR1-latest
#   bash examples/v31_dr_grpo_lr_uplift.sh
#
# Dry run:
#   V30_DRY_RUN=1 bash examples/v31_dr_grpo_lr_uplift.sh
# ================================================================
set -euo pipefail

# ========== Multi-node guard ==========
# verl uses Ray internally. Launcher (INDEX=0) starts Ray head + trainer.
# Workers (INDEX>0) start Ray worker and join the head node.

# LD_LIBRARY_PATH: pip-installed nvidia-cublas (12.2.5.6) MUST come BEFORE /usr/local/cuda/lib64
# (system cuda may have cuBLAS 12.4 which has SIGFPE bug on H20).
# Order matters: nvidia cublas pip → cusparselt → system cuda → existing
NVIDIA_CUBLAS_LIB=$(find /usr/local/lib*/python3.*/site-packages/nvidia/cublas/lib -maxdepth 0 2>/dev/null | head -1)
CUSPARSELT_LIB=$(find /usr/local/lib*/python3.*/site-packages -path "*/cusparselt/lib" -type d 2>/dev/null | head -1)
export LD_LIBRARY_PATH="${NVIDIA_CUBLAS_LIB:-/usr/local/lib64/python3.12/site-packages/nvidia/cublas/lib}:${CUSPARSELT_LIB:-/usr/local/lib/python3.12/site-packages/cusparselt/lib}:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# H20 + CUDA 12.2 cuBLAS SIGFPE fix (multi-layer defense):
# 1. VLLM_USE_V1=0: avoid torch.compile's gemm_and_bias (calls cublasLt directly)
# 2. enforce_eager: skip torch.compile entirely
# 3. DISABLE_ADDMM_CUDA_LT=1: prevent addmm from using cublasLt
# 4. NVIDIA_TF32_OVERRIDE=1: force TF32 mode which uses different cuBLAS algorithm
#    selection, avoiding the buggy cublasLtTSTMatmulAlgoGetHeuristic code path
# 5. TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1: ensure PyTorch respects TF32 override
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export DISABLE_ADDMM_CUDA_LT=1
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

if [[ "${INDEX:-0}" != "0" ]]; then
    echo "[V31] Worker node INDEX=${INDEX}, starting Ray worker → head=${CHIEF_IP}:6379"
    # Wait for head node to be ready
    sleep 15
    ray start --address="${CHIEF_IP}:6379" --block
    exit 0
fi

echo "[V31] Launcher node (INDEX=${INDEX:-0}), starting Ray head + trainer..."
# Only start Ray head if not already running
if ! ray status >/dev/null 2>&1; then
    ray start --head --port=6379 --num-gpus=${HOST_GPU_NUM:-8}
    sleep 10
fi

# Wait for all worker nodes to join the cluster
EXPECTED_GPUS=$(( ${V31_NNODES:-${HOST_NUM:-4}} * ${V31_N_GPUS_PER_NODE:-${HOST_GPU_NUM:-8}} ))
echo "[V31] Waiting for ${EXPECTED_GPUS} GPUs to be available in Ray cluster..."
MAX_WAIT=180  # 3 minutes max
WAITED=0
while true; do
    AVAILABLE_GPUS=$(ray status 2>/dev/null | grep -oP '[\d.]+(?=/[\d.]+\s+GPU)' | head -1)
    TOTAL_GPUS=$(ray status 2>/dev/null | grep -oP '(?<=/)[\d.]+(?=\s+GPU)' | head -1)
    TOTAL_GPUS_INT=${TOTAL_GPUS%.*}  # Remove decimal
    if [ "${TOTAL_GPUS_INT:-0}" -ge "${EXPECTED_GPUS}" ]; then
        echo "[V31] All ${TOTAL_GPUS_INT} GPUs available!"
        break
    fi
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[V31] WARNING: Timeout waiting for GPUs. Got ${TOTAL_GPUS_INT:-0}/${EXPECTED_GPUS}."
        echo "[V31] Proceeding with available GPUs (may fail if insufficient)..."
        break
    fi
    echo "[V31] Waiting... (${TOTAL_GPUS_INT:-0}/${EXPECTED_GPUS} GPUs, ${WAITED}s elapsed)"
    sleep 10
    WAITED=$((WAITED + 10))
done

echo "[V31] Ray cluster status:"
ray status

# CRITICAL: Tell Python trainer to connect to existing Ray cluster (not start a new one)
export RAY_ADDRESS="auto"
echo "[V31] RAY_ADDRESS=${RAY_ADDRESS} (will connect to existing cluster)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/logs/rl"
DEBUG_LOG_DIR="/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/logs/debug"
mkdir -p "${LOG_DIR}" "${DEBUG_LOG_DIR}"
export TRAINING_DEBUG_LOG_DIR="${DEBUG_LOG_DIR}"
export output_path='./'
export ckpt_path='./ckpt'
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-online}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

# ========== Data ==========
export STEPCOUNT_TRAIN_DATA=${STEPCOUNT_TRAIN_DATA:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/StepCountQA-RL-Traj_0_10}
export BOK_TOTAL_STEPS=${BOK_TOTAL_STEPS:-49}
export TRAIN_OVERSAMPLE_NO_MASK_FACTOR=${TRAIN_OVERSAMPLE_NO_MASK_FACTOR:-1}
export TRAIN_HARD_OVERSAMPLE_FACTOR=${TRAIN_HARD_OVERSAMPLE_FACTOR:-1}
export TRAIN_SAVE_FREQ=${TRAIN_SAVE_FREQ:-20}
export TRAIN_SAVE_LIMIT=${TRAIN_SAVE_LIMIT:-6}
export TRAIN_VAL_FREQ=${TRAIN_VAL_FREQ:-20}

# ========== Reward & Trajectory ==========
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

# ========== Interleaved Rollout ==========
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

# ========== BoK-GRPO / DR-GRPO ==========
export STEPCOUNT_RL_MODE=${STEPCOUNT_RL_MODE:-bok_grpo}
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-bok_grpo}
export PROCESS_REWARD_ENABLE=${PROCESS_REWARD_ENABLE:-0}
# V31: LR uplift 1.25e-6 → 1.5e-6 for 32-GPU larger effective batch
export ACTOR_LR=${ACTOR_LR:-1.5e-6}
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
export BOK_SMART_FILTER_THRESHOLD=${BOK_SMART_FILTER_THRESHOLD:-0.965}
export BOK_ALLWRONG_CAP=${BOK_ALLWRONG_CAP:-1.0}
export BOK_ALLWRONG_ANSWER_THRESHOLD=${BOK_ALLWRONG_ANSWER_THRESHOLD:-0.5}
export BOK_ALLWRONG_NEG_ONLY=${BOK_ALLWRONG_NEG_ONLY:-0}
export BOK_WINNER_BOOST=${BOK_WINNER_BOOST:-0}
export BOK_EASY_SCALE=${BOK_EASY_SCALE:-1.0}
export BOK_QUALITY_BONUS=${BOK_QUALITY_BONUS:-0}

# ========== Gradient Safety ==========
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

# ========== Fail-fast ==========
export V30_FAILFAST_ENABLE=${V30_FAILFAST_ENABLE:-1}
export V30_FAILFAST_NAN_LIMIT=${V30_FAILFAST_NAN_LIMIT:-6}
export V30_FAILFAST_ENTROPY_CRITICAL=${V30_FAILFAST_ENTROPY_CRITICAL:-0.85}
export V30_FAILFAST_ENTROPY_CONSEC=${V30_FAILFAST_ENTROPY_CONSEC:-3}
export V30_FAILFAST_FORMAT_CRITICAL=${V30_FAILFAST_FORMAT_CRITICAL:-0.30}
export V30_FAILFAST_FORMAT_CONSEC=${V30_FAILFAST_FORMAT_CONSEC:-3}
export V30_FAILFAST_LOWVAR_CRITICAL=${V30_FAILFAST_LOWVAR_CRITICAL:-45}
export V30_FAILFAST_ZEROREWARD_CRITICAL=${V30_FAILFAST_ZEROREWARD_CRITICAL:-30}
export V30_FAILFAST_BOK_CONSEC=${V30_FAILFAST_BOK_CONSEC:-2}

# ========== 32-GPU Topology ==========
export V31_NNODES=${V31_NNODES:-${HOST_NUM:-4}}
export V31_N_GPUS_PER_NODE=${V31_N_GPUS_PER_NODE:-${HOST_GPU_NUM:-8}}
# Batch sizes scaled for 32 GPUs (4x of 8-GPU defaults)
export V31_ROLLOUT_BATCH_SIZE=${V31_ROLLOUT_BATCH_SIZE:-256}
export V31_VAL_BATCH_SIZE=${V31_VAL_BATCH_SIZE:-256}
export V31_GLOBAL_BATCH_SIZE=${V31_GLOBAL_BATCH_SIZE:-256}
export V31_MICRO_BATCH_UPDATE=${V31_MICRO_BATCH_UPDATE:-8}
export V31_MICRO_BATCH_EXP=${V31_MICRO_BATCH_EXP:-8}
export V31_GPU_MEM_UTIL=${V31_GPU_MEM_UTIL:-0.65}
export V31_MAX_MODEL_LEN=${V31_MAX_MODEL_LEN:-16384}
export V31_MAX_NUM_BATCHED_TOKENS=${V31_MAX_NUM_BATCHED_TOKENS:-32768}
export V31_ENFORCE_EAGER=${V31_ENFORCE_EAGER:-true}

# ========== Paths ==========
MODEL_PATH=${MODEL_PATH:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/models/StepCount-7b-SFT-30k-high-without-reasoning}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/examples/config.yaml}
REWARD_FN_PATH=${REWARD_FN_PATH:-${PROJECT_ROOT}/examples/reward_function/StepCount_mask_reward.py:compute_score}

# Experiment naming
export V31_EXPERIMENT_NAME=${V31_EXPERIMENT_NAME:-StepCount-7B_v31_dr_grpo_lr_uplift_32gpu_$(date +%Y%m%d_%H%M)}
export V31_SAVE_CHECKPOINT_PATH=${V31_SAVE_CHECKPOINT_PATH:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/checkpoints/${V31_EXPERIMENT_NAME}}

# ========== Validation ==========
for required_file in "${CONFIG_PATH}" "${PROJECT_ROOT}/examples/reward_function/StepCount_mask_reward.py"; do
    [[ -f "${required_file}" ]] || { echo "[FATAL] required file not found: ${required_file}" >&2; exit 1; }
done
for required_path in "${MODEL_PATH}" "${STEPCOUNT_TRAIN_DATA}" "/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/datasets/pixmo-test" "${STEPCOUNT_MASKS_METADATA}" "${STEPCOUNT_MASKS_DIR}"; do
    [[ -e "${required_path}" ]] || { echo "[FATAL] required path not found: ${required_path}" >&2; exit 1; }
done
[[ -z "${INTERLEAVED_PROCESS_PROMPT_FILE}" || -f "${INTERLEAVED_PROCESS_PROMPT_FILE}" ]] || { echo "[FATAL] process prompt not found: ${INTERLEAVED_PROCESS_PROMPT_FILE}" >&2; exit 1; }
[[ -z "${SYSTEM_PROMPT_FILE}" || -f "${SYSTEM_PROMPT_FILE}" ]] || { echo "[FATAL] system prompt not found: ${SYSTEM_PROMPT_FILE}" >&2; exit 1; }

RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE="${LOG_DIR}/v31_dr_grpo_lr_uplift_32gpu_${RUN_TS}.log"
MONITOR_SCRIPT="${PROJECT_ROOT}/tools/monitor_training_v2.py"
FAILFAST_SCRIPT="${PROJECT_ROOT}/tools/monitor_v28_failfast.py"
MONITOR_DIR="${LOG_DIR}/../monitor"
mkdir -p "${MONITOR_DIR}"
MONITOR_LOG="${MONITOR_DIR}/monitor_v31_${RUN_TS}.log"
MONITOR_JSON="${MONITOR_DIR}/metrics_v31_${RUN_TS}.jsonl"
FAILFAST_LOG="${MONITOR_DIR}/failfast_v31_${RUN_TS}.log"
export STOP_FILE="/tmp/v31_stop_${RUN_TS}"
rm -f "${STOP_FILE}"

echo "================================================================"
echo "[V31] DR-GRPO LR Uplift — 32 GPU (${V31_NNODES} nodes × ${V31_N_GPUS_PER_NODE} GPUs)"
echo "[V31] model=${MODEL_PATH}"
echo "[V31] data=${STEPCOUNT_TRAIN_DATA}"
echo "[V31] lr=${ACTOR_LR} kl=${KL_COEF} total_steps=${BOK_TOTAL_STEPS}"
echo "[V31] batch: rollout=${V31_ROLLOUT_BATCH_SIZE} global=${V31_GLOBAL_BATCH_SIZE} val=${V31_VAL_BATCH_SIZE}"
echo "[V31] BOK: smart_filter=${BOK_SMART_FILTER_THRESHOLD} tau=${BOK_TAU_INIT}->${BOK_TAU_FINAL} fallback=${BOK_FALLBACK_MODE}"
echo "[V31] grad: spike_thresh=${GRAD_SPIKE_THRESHOLD} spike_brake=${GRAD_SPIKE_BRAKE_MAX} nonfinite_brake=${GRAD_NONFINITE_BRAKE_MAX}"
echo "[V31] save: freq=${TRAIN_SAVE_FREQ} limit=${TRAIN_SAVE_LIMIT} path=${V31_SAVE_CHECKPOINT_PATH}"
echo "[V31] log=${LOG_FILE}"
echo "================================================================"

if [[ "${V30_DRY_RUN:-0}" == "1" ]]; then
    echo "[DryRun] All paths validated. Config:"
    echo "  model=${MODEL_PATH}"
    echo "  train_data=${STEPCOUNT_TRAIN_DATA}"
    echo "  masks_meta=${STEPCOUNT_MASKS_METADATA}"
    echo "  masks_dir=${STEPCOUNT_MASKS_DIR}"
    echo "  config=${CONFIG_PATH}"
    echo "  reward_fn=${REWARD_FN_PATH}"
    echo "  nnodes=${V31_NNODES} gpus_per_node=${V31_N_GPUS_PER_NODE}"
    echo "  lr=${ACTOR_LR} total_steps=${BOK_TOTAL_STEPS}"
    echo "  checkpoint_path=${V31_SAVE_CHECKPOINT_PATH}"
    exit 0
fi

# ========== Monitor & Failfast ==========
if [[ -f "${MONITOR_SCRIPT}" ]]; then
    nohup python3 "${MONITOR_SCRIPT}" \
        --log "${LOG_FILE}" \
        --out "${MONITOR_LOG}" \
        --stop-file "${STOP_FILE}" \
        --interval 60 --every 5 --summary-every 20 \
        --json-metrics "${MONITOR_JSON}" \
        > /dev/null 2>"${MONITOR_DIR}/monitor_v31_stderr_${RUN_TS}.log" &
    MONITOR_PID=$!
else
    MONITOR_PID=""
fi

if [[ "${V30_FAILFAST_ENABLE}" == "1" && -f "${FAILFAST_SCRIPT}" ]]; then
    nohup python3 "${FAILFAST_SCRIPT}" \
        --run-name V31 \
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
        > /dev/null 2>"${MONITOR_DIR}/failfast_v31_stderr_${RUN_TS}.log" &
    FAILFAST_PID=$!
else
    FAILFAST_PID=""
fi

cleanup_monitor() {
    if [[ -n "${MONITOR_PID:-}" ]]; then kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; fi
    if [[ -n "${FAILFAST_PID:-}" ]]; then kill "${FAILFAST_PID}" 2>/dev/null || true; wait "${FAILFAST_PID}" 2>/dev/null || true; fi
}
trap cleanup_monitor EXIT

# ========== Launch Training ==========
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
    worker.actor.micro_batch_size_per_device_for_update=${V31_MICRO_BATCH_UPDATE} \
    worker.actor.micro_batch_size_per_device_for_experience=${V31_MICRO_BATCH_EXP} \
    worker.rollout.gpu_memory_utilization=${V31_GPU_MEM_UTIL} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_model_len=${V31_MAX_MODEL_LEN} \
    worker.rollout.max_num_batched_tokens=${V31_MAX_NUM_BATCHED_TOKENS} \
    worker.rollout.enforce_eager=${V31_ENFORCE_EAGER} \
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
    trainer.experiment_name=${V31_EXPERIMENT_NAME} \
    trainer.logger=['console','wandb'] \
    trainer.save_checkpoint_path=${V31_SAVE_CHECKPOINT_PATH} \
    trainer.total_epochs=${STEPCOUNT_TOTAL_EPOCHS:-1} \
    trainer.save_freq=${TRAIN_SAVE_FREQ} \
    trainer.save_limit=${TRAIN_SAVE_LIMIT} \
    trainer.val_freq=${TRAIN_VAL_FREQ} \
    data.rollout_batch_size=${V31_ROLLOUT_BATCH_SIZE} \
    data.val_batch_size=${V31_VAL_BATCH_SIZE} \
    worker.actor.global_batch_size=${V31_GLOBAL_BATCH_SIZE} \
    trainer.nnodes=${V31_NNODES} \
    trainer.n_gpus_per_node=${V31_N_GPUS_PER_NODE} 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
echo "[V31] Training finished with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
