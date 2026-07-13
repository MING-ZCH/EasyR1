#!/bin/bash
# =============================================================================
# V36 focused10k Ulysses-SP2 probe for 8xH200.
#
# This is an explicit development probe, not a supported training launch.
# Qwen2.5-VL mRoPE/image-feature alignment is not validated under Ulysses and
# may fail before completing a step. The default probe keeps micro_update=4;
# never use its output as a training checkpoint without SP1/SP2 equivalence.
#
# Usage:
#   V36_ACK_EXPERIMENTAL_VLM_SP=1 \
#   V36_ALLOW_UNVALIDATED_QWEN25_VL_SP=1 \
#     bash examples/rl_launch/run_v36_8gpu_focused10k_ulysses_sp2_probe.sh
#
# A full run is not supported until SP1/SP2 image-token and logprob equivalence
# has been demonstrated in the target environment.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${V36_ACK_EXPERIMENTAL_VLM_SP:-0}" != "1" || "${V36_ALLOW_UNVALIDATED_QWEN25_VL_SP:-0}" != "1" ]]; then
  echo "[run-v36-8gpu-focused10k-sp2][ERROR] This is an unvalidated VLM Ulysses probe." >&2
  echo "Set both V36_ACK_EXPERIMENTAL_VLM_SP=1 and V36_ALLOW_UNVALIDATED_QWEN25_VL_SP=1 after reading the migration guide." >&2
  exit 1
fi

export V36_ENABLE_ULYSSES_SP=${V36_ENABLE_ULYSSES_SP:-1}
export V36_ULYSSES_SEQUENCE_PARALLEL_SIZE=${V36_ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}
export V31_MICRO_BATCH_UPDATE=${V31_MICRO_BATCH_UPDATE:-4}
export V31_MICRO_BATCH_EXP=${V31_MICRO_BATCH_EXP:-8}
export V36_MAX_STEPS=${V36_MAX_STEPS:-5}
export TRAIN_SAVE_FREQ=${TRAIN_SAVE_FREQ:-5}
export TRAIN_VAL_FREQ=${TRAIN_VAL_FREQ:-5}
export BASETAG=${BASETAG:-ckpt476_11_30_quality_focused10k_bokgrpo_h200_8gpu_ulysses_sp2_probe}
export V36_LAUNCH_LABEL=${V36_LAUNCH_LABEL:-run-v36-8gpu-focused10k-sp2-probe}
export EASYR1_ALLOW_ZERO_MM_LOGPROB=0

exec bash "${SCRIPT_DIR}/run_v36_8gpu_stable_focused10k.sh"
