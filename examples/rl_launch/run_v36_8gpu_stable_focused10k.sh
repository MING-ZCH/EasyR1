#!/bin/bash
# =============================================================================
# V36 8xH200 stable launch on the quality-prioritized focused 10k subset.
#
# Dataset mix:
#   - 8,500 rows from answer 11-30
#   - 1,500 rows from answer 31-50 for long-count retention
#
# Usage:
#   bash examples/rl_launch/run_v36_8gpu_stable_focused10k.sh
#
# Ulysses SP plugin controls. Default stable path keeps SP disabled because the
# local Qwen2.5-VL mRoPE/image-feature path is not production-validated.
#   V36_ENABLE_ULYSSES_SP=1                 # explicitly enable actor/ref sequence parallel
#   V36_ULYSSES_SEQUENCE_PARALLEL_SIZE=2    # first candidate for Qwen2.5-VL-7B 8xH200
#   V36_ACK_EXPERIMENTAL_VLM_SP=1           # required safety acknowledgement
#   V36_ALLOW_UNVALIDATED_QWEN25_VL_SP=1     # second development-only acknowledgement
#   V31_MICRO_BATCH_UPDATE=4                # first test correctness before increasing micro
#
# Explicit SP2 probe:
#   V36_ACK_EXPERIMENTAL_VLM_SP=1 V36_ALLOW_UNVALIDATED_QWEN25_VL_SP=1 \
#     V36_ENABLE_ULYSSES_SP=1 \
#     V36_ULYSSES_SEQUENCE_PARALLEL_SIZE=2 bash examples/rl_launch/run_v36_8gpu_stable_focused10k.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_DIR}/examples/local_path_env.sh"

export STEPCOUNT_TRAIN_DATA=${STEPCOUNT_TRAIN_DATA:-${STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA}}
export BASETAG=${BASETAG:-ckpt476_11_30_quality_focused10k_bokgrpo_h200_8gpu_stable}
export V31_EXPERIMENT_NAME=${V31_EXPERIMENT_NAME:-StepCount-7B_v36_dense_11_30_quality_focused10k_ckpt476_8gpu_$(date +%Y%m%d_%H%M)}
export V36_MAX_STEPS=${V36_MAX_STEPS:-80}
export TRAIN_VAL_FREQ=${TRAIN_VAL_FREQ:-20}
export V36_LAUNCH_LABEL=${V36_LAUNCH_LABEL:-run-v36-8gpu-focused10k}

exec bash "${SCRIPT_DIR}/run_v36_8gpu_stable.sh"
