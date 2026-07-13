#!/bin/bash
# =============================================================================
# V36 8xH200 fast probe / high-efficiency launch.
#
# Defaults:
#   - 8 GPUs, batch=128, rollout_n=16.
#   - micro_update=4 after the Step-2 actor backward OOM observed with 8.
#   - vLLM blocks=20480 and gpu_mem=0.50 to preserve rollout OOM margin.
#   - Save every 5 steps, keep only the newest 3 checkpoints.
#   - Run 200 steps by default. Use V36_MAX_STEPS=35 for a short OOM probe.
#
# Usage:
#   bash examples/rl_launch/run_v36_8gpu_fast_probe.sh
#
# Optional overrides:
#   V36_MAX_STEPS=60 V31_MICRO_BATCH_UPDATE=8 bash examples/rl_launch/run_v36_8gpu_fast_probe.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_DIR}/examples/local_path_env.sh"

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

export TRAIN_SAVE_FREQ=${TRAIN_SAVE_FREQ:-5}
export TRAIN_SAVE_LIMIT=${TRAIN_SAVE_LIMIT:-3}
export V36_MAX_STEPS=${V36_MAX_STEPS:-200}

export V31_MICRO_BATCH_UPDATE=${V31_MICRO_BATCH_UPDATE:-4}
export V31_MICRO_BATCH_EXP=${V31_MICRO_BATCH_EXP:-8}
export V36_ENABLE_ULYSSES_SP=${V36_ENABLE_ULYSSES_SP:-0}
case "${V36_ENABLE_ULYSSES_SP}" in
  1|true|TRUE|yes|YES|on|ON)
    export V31_ULYSSES_SEQUENCE_PARALLEL_SIZE=${V31_ULYSSES_SEQUENCE_PARALLEL_SIZE:-${V36_ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}}
    ;;
  0|false|FALSE|no|NO|off|OFF)
    export V31_ULYSSES_SEQUENCE_PARALLEL_SIZE=1
    ;;
  *)
    echo "[run-v36-8gpu-fast][ERROR] V36_ENABLE_ULYSSES_SP must be 0/1/true/false, got: ${V36_ENABLE_ULYSSES_SP}" >&2
    exit 1
    ;;
esac
export EASYR1_VLLM_NUM_GPU_BLOCKS=${EASYR1_VLLM_NUM_GPU_BLOCKS:-20480}
export V31_GPU_MEM_UTIL=${V31_GPU_MEM_UTIL:-0.50}
export V31_MAX_NUM_BATCHED_TOKENS=${V31_MAX_NUM_BATCHED_TOKENS:-49152}
export V36_SAVE_ROOT=${V36_SAVE_ROOT:-${EASYR1_CHECKPOINT_ROOT}}

MODEL_PATH_ARG=${MODEL_PATH:-${STEPCOUNT_1M_RESUME_CKPT476_MODEL_PATH}}
DATA_PATH_ARG=${STEPCOUNT_TRAIN_DATA:-${STEPCOUNT_DENSE_11_50_MASKCOMPLETE_DATA}}
BASETAG=${BASETAG:-ckpt476_11_50_maskcomplete38332_bokgrpo_h200_8gpu_fastprobe}
LAUNCH_LABEL=${V36_LAUNCH_LABEL:-run-v36-8gpu-fast}

cd "${REPO_DIR}"

echo "[${LAUNCH_LABEL}] max_steps=${V36_MAX_STEPS} save_freq=${TRAIN_SAVE_FREQ} save_limit=${TRAIN_SAVE_LIMIT}"
echo "[${LAUNCH_LABEL}] micro=${V31_MICRO_BATCH_UPDATE}/${V31_MICRO_BATCH_EXP} ulysses_enable=${V36_ENABLE_ULYSSES_SP} ulysses_sp=${V31_ULYSSES_SEQUENCE_PARALLEL_SIZE} blocks=${EASYR1_VLLM_NUM_GPU_BLOCKS} gpu_mem=${V31_GPU_MEM_UTIL}"
echo "[${LAUNCH_LABEL}] save_root=${V36_SAVE_ROOT}"

exec bash "${SCRIPT_DIR}/mn_trainer_dense_v36.sh" 1 128 "${MODEL_PATH_ARG}" "${DATA_PATH_ARG}" "${BASETAG}"
