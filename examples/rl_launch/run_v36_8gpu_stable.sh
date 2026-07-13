#!/bin/bash
# =============================================================================
# V36 8xH200 clean-start launch with pluggable Ulysses sequence parallelism.
#
# Defaults:
#   - V36_ENABLE_ULYSSES_SP=0 for the Qwen2.5-VL stable path.
#   - With SP disabled, actor update micro defaults to 4.
#   - If experimental SP is explicitly enabled, actor update micro still
#     defaults to 4; raise it only after numerical equivalence is verified.
#
# Usage:
#   bash examples/rl_launch/run_v36_8gpu_stable.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export V36_ENABLE_ULYSSES_SP=${V36_ENABLE_ULYSSES_SP:-0}
export V36_ULYSSES_SEQUENCE_PARALLEL_SIZE=${V36_ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}
case "${V36_ENABLE_ULYSSES_SP}" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ "${V36_ACK_EXPERIMENTAL_VLM_SP:-0}" != "1" || "${V36_ALLOW_UNVALIDATED_QWEN25_VL_SP:-0}" != "1" ]]; then
      echo "[run-v36-8gpu-stable][ERROR] Qwen2.5-VL Ulysses is unvalidated in this local stack." >&2
      echo "Set both V36_ACK_EXPERIMENTAL_VLM_SP=1 and V36_ALLOW_UNVALIDATED_QWEN25_VL_SP=1 only for a short development probe; production must keep SP disabled." >&2
      exit 1
    fi
    _V36_DEFAULT_STABLE_MICRO_UPDATE=${V36_SP_MICRO_BATCH_UPDATE:-4}
    ;;
  0|false|FALSE|no|NO|off|OFF)
    _V36_DEFAULT_STABLE_MICRO_UPDATE=4
    ;;
  *)
    echo "[run-v36-8gpu-stable][ERROR] V36_ENABLE_ULYSSES_SP must be 0/1/true/false, got: ${V36_ENABLE_ULYSSES_SP}" >&2
    exit 1
    ;;
esac

export V31_MICRO_BATCH_UPDATE=${V31_MICRO_BATCH_UPDATE:-${_V36_DEFAULT_STABLE_MICRO_UPDATE}}
export V31_MICRO_BATCH_EXP=${V31_MICRO_BATCH_EXP:-8}
export EASYR1_ALLOW_ZERO_MM_LOGPROB=${EASYR1_ALLOW_ZERO_MM_LOGPROB:-0}
export EASYR1_VLLM_NUM_GPU_BLOCKS=${EASYR1_VLLM_NUM_GPU_BLOCKS:-20480}
export V31_GPU_MEM_UTIL=${V31_GPU_MEM_UTIL:-0.50}
export V31_MAX_NUM_BATCHED_TOKENS=${V31_MAX_NUM_BATCHED_TOKENS:-49152}
export TRAIN_SAVE_FREQ=${TRAIN_SAVE_FREQ:-20}
export TRAIN_SAVE_LIMIT=${TRAIN_SAVE_LIMIT:-3}
export V36_MAX_STEPS=${V36_MAX_STEPS:-200}
export TRAINER_VAL_BEFORE_TRAIN=${TRAINER_VAL_BEFORE_TRAIN:-false}
export BASETAG=${BASETAG:-ckpt476_11_50_maskcomplete38332_bokgrpo_h200_8gpu_stable}
export V36_LAUNCH_LABEL=${V36_LAUNCH_LABEL:-run-v36-8gpu-stable}

exec bash "${SCRIPT_DIR}/run_v36_8gpu_fast_probe.sh"
