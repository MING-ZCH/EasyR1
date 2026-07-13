#!/bin/bash
# =============================================================================
# Resume V36 8xH200 training from an EasyR1/FSDP global_step checkpoint.
#
# Usage:
#   bash examples/rl_launch/resume_v36_8gpu_from_ckpt.sh /path/to/run/global_step_35 [stable|fast]
#
# Profiles:
#   stable: micro_update=4, blocks=20480, gpu_mem=0.50
#   fast:   micro_update=4, blocks=20480, gpu_mem=0.50
#
# By default, resumed checkpoints are saved back into the original run directory
# so TRAIN_SAVE_LIMIT keeps rolling within the same run.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_DIR}/examples/local_path_env.sh"

CKPT_PATH="${1:?usage: resume_v36_8gpu_from_ckpt.sh /path/to/global_step_N [stable|fast]}"
PROFILE="${2:-stable}"

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "[resume-v36-8gpu][ERROR] checkpoint dir does not exist: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ "$(basename "${CKPT_PATH}")" != global_step_* ]]; then
  echo "[resume-v36-8gpu][ERROR] checkpoint path must end with global_step_N: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -d "${CKPT_PATH}/actor" || ! -f "${CKPT_PATH}/dataloader.pt" ]]; then
  echo "[resume-v36-8gpu][ERROR] expected EasyR1 checkpoint with actor/ and dataloader.pt: ${CKPT_PATH}" >&2
  exit 1
fi

shopt -s nullglob
ACTOR_MODEL_SHARDS=("${CKPT_PATH}"/actor/model_world_size_*_rank_*.pt)
if [[ ${#ACTOR_MODEL_SHARDS[@]} -eq 0 ]]; then
  if [[ "${V32_DRY_RUN:-0}" == "1" && "${V36_ALLOW_INCOMPLETE_DRY_RUN_CHECKPOINT:-0}" == "1" ]]; then
    echo "[resume-v36-8gpu][WARN] dry-run is using an intentionally incomplete fake checkpoint." >&2
  else
    echo "[resume-v36-8gpu][ERROR] no actor FSDP model shards found in ${CKPT_PATH}/actor." >&2
    exit 1
  fi
else
  declare -A ACTOR_SHARD_RANKS=()
  for shard in "${ACTOR_MODEL_SHARDS[@]}"; do
    shard_name="$(basename "${shard}")"
    if [[ ! "${shard_name}" =~ ^model_world_size_([0-9]+)_rank_([0-9]+)\.pt$ ]]; then
      echo "[resume-v36-8gpu][ERROR] unrecognized actor shard name: ${shard_name}" >&2
      exit 1
    fi
    shard_world_size="${BASH_REMATCH[1]}"
    shard_rank="${BASH_REMATCH[2]}"
    if [[ "${shard_world_size}" != "8" ]]; then
      echo "[resume-v36-8gpu][ERROR] checkpoint shard world size is ${shard_world_size}; this launcher requires 8." >&2
      exit 1
    fi
    ACTOR_SHARD_RANKS["${shard_rank}"]=1
  done
  if [[ ${#ACTOR_MODEL_SHARDS[@]} -ne 8 || ${#ACTOR_SHARD_RANKS[@]} -ne 8 ]]; then
    echo "[resume-v36-8gpu][ERROR] expected 8 unique actor model shards, found ${#ACTOR_MODEL_SHARDS[@]}." >&2
    exit 1
  fi
  for shard_rank in {0..7}; do
    if [[ -z "${ACTOR_SHARD_RANKS[${shard_rank}]:-}" \
          || ! -f "${CKPT_PATH}/actor/optim_world_size_8_rank_${shard_rank}.pt" \
          || ! -f "${CKPT_PATH}/actor/extra_state_world_size_8_rank_${shard_rank}.pt" ]]; then
      echo "[resume-v36-8gpu][ERROR] incomplete actor model/optimizer/extra shard set for rank ${shard_rank}." >&2
      exit 1
    fi
  done
fi

RUN_DIR="$(dirname "${CKPT_PATH}")"
RUN_NAME="$(basename "${RUN_DIR}")"

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

export V36_LOAD_CHECKPOINT_PATH="${CKPT_PATH}"
export V31_SAVE_CHECKPOINT_PATH=${V31_SAVE_CHECKPOINT_PATH:-${RUN_DIR}}
export V31_EXPERIMENT_NAME=${V31_EXPERIMENT_NAME:-${RUN_NAME}}
export TRAIN_SAVE_FREQ=${TRAIN_SAVE_FREQ:-10}
export TRAIN_SAVE_LIMIT=${TRAIN_SAVE_LIMIT:-3}
export V36_ENABLE_ULYSSES_SP=${V36_ENABLE_ULYSSES_SP:-0}
export V36_ULYSSES_SEQUENCE_PARALLEL_SIZE=${V36_ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}

case "${PROFILE}" in
  stable)
    export V31_MICRO_BATCH_UPDATE=${V31_MICRO_BATCH_UPDATE:-4}
    export V31_MICRO_BATCH_EXP=${V31_MICRO_BATCH_EXP:-8}
    export EASYR1_VLLM_NUM_GPU_BLOCKS=${EASYR1_VLLM_NUM_GPU_BLOCKS:-20480}
    export V31_GPU_MEM_UTIL=${V31_GPU_MEM_UTIL:-0.50}
    export V31_MAX_NUM_BATCHED_TOKENS=${V31_MAX_NUM_BATCHED_TOKENS:-49152}
    ;;
  fast)
    case "${V36_ENABLE_ULYSSES_SP}" in
      1|true|TRUE|yes|YES|on|ON)
        _V36_RESUME_FAST_MICRO_UPDATE=${V36_SP_MICRO_BATCH_UPDATE:-4}
        ;;
      0|false|FALSE|no|NO|off|OFF)
        _V36_RESUME_FAST_MICRO_UPDATE=4
        ;;
      *)
        echo "[resume-v36-8gpu][ERROR] V36_ENABLE_ULYSSES_SP must be 0/1/true/false, got: ${V36_ENABLE_ULYSSES_SP}" >&2
        exit 1
        ;;
    esac
    export V31_MICRO_BATCH_UPDATE=${V31_MICRO_BATCH_UPDATE:-${_V36_RESUME_FAST_MICRO_UPDATE}}
    export V31_MICRO_BATCH_EXP=${V31_MICRO_BATCH_EXP:-8}
    export EASYR1_VLLM_NUM_GPU_BLOCKS=${EASYR1_VLLM_NUM_GPU_BLOCKS:-20480}
    export V31_GPU_MEM_UTIL=${V31_GPU_MEM_UTIL:-0.50}
    export V31_MAX_NUM_BATCHED_TOKENS=${V31_MAX_NUM_BATCHED_TOKENS:-49152}
    ;;
  *)
    echo "[resume-v36-8gpu][ERROR] unknown profile '${PROFILE}', use stable or fast." >&2
    exit 1
    ;;
esac

case "${V36_ENABLE_ULYSSES_SP}" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ "${V36_ACK_EXPERIMENTAL_VLM_SP:-0}" != "1" || "${V36_ALLOW_UNVALIDATED_QWEN25_VL_SP:-0}" != "1" ]]; then
      echo "[resume-v36-8gpu][ERROR] VLM Ulysses resume is experimental and disabled by default." >&2
      echo "Set both V36_ACK_EXPERIMENTAL_VLM_SP=1 and V36_ALLOW_UNVALIDATED_QWEN25_VL_SP=1 only for a short development probe." >&2
      exit 1
    fi
    export V31_ULYSSES_SEQUENCE_PARALLEL_SIZE=${V31_ULYSSES_SEQUENCE_PARALLEL_SIZE:-${V36_ULYSSES_SEQUENCE_PARALLEL_SIZE}}
    ;;
  0|false|FALSE|no|NO|off|OFF)
    export V31_ULYSSES_SEQUENCE_PARALLEL_SIZE=1
    ;;
  *)
    echo "[resume-v36-8gpu][ERROR] V36_ENABLE_ULYSSES_SP must be 0/1/true/false, got: ${V36_ENABLE_ULYSSES_SP}" >&2
    exit 1
    ;;
esac

MODEL_PATH_ARG=${MODEL_PATH:-${STEPCOUNT_1M_RESUME_CKPT476_MODEL_PATH}}
RUN_MANIFEST="${RUN_DIR}/v36_run_manifest.tsv"
manifest_value() {
  local key="$1"
  awk -F '\t' -v key="${key}" '$1 == key { print $2; exit }' "${RUN_MANIFEST}"
}

if [[ -f "${RUN_MANIFEST}" ]]; then
  MANIFEST_GPUS="$(manifest_value TOTAL_GPUS)"
  if [[ -z "${MANIFEST_GPUS}" ]]; then
    echo "[resume-v36-8gpu][ERROR] manifest has no TOTAL_GPUS: ${RUN_MANIFEST}" >&2
    exit 1
  fi
  if [[ "${MANIFEST_GPUS}" != "8" ]]; then
    echo "[resume-v36-8gpu][ERROR] checkpoint manifest world size is ${MANIFEST_GPUS} GPUs; this launcher requires an 8-GPU checkpoint." >&2
    exit 1
  fi
fi

if [[ -n "${STEPCOUNT_TRAIN_DATA:-}" ]]; then
  DATA_PATH_ARG="${STEPCOUNT_TRAIN_DATA}"
  DATA_SOURCE="explicit STEPCOUNT_TRAIN_DATA"
elif [[ -f "${RUN_MANIFEST}" ]]; then
  DATA_PATH_ARG="$(manifest_value STEPCOUNT_TRAIN_DATA)"
  DATA_SOURCE="${RUN_MANIFEST}"
  if [[ -z "${DATA_PATH_ARG}" ]]; then
    echo "[resume-v36-8gpu][ERROR] manifest has no STEPCOUNT_TRAIN_DATA: ${RUN_MANIFEST}" >&2
    exit 1
  fi
elif [[ "${RUN_NAME}" == *"11_30_quality_focused10k"* ]]; then
  DATA_PATH_ARG="${STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA}"
  DATA_SOURCE="legacy focused10k run-name fallback"
elif [[ "${RUN_NAME}" == *"11_50_maskcomplete"* ]]; then
  DATA_PATH_ARG="${STEPCOUNT_DENSE_11_50_MASKCOMPLETE_DATA}"
  DATA_SOURCE="legacy full-data run-name fallback"
else
  echo "[resume-v36-8gpu][ERROR] cannot determine the original training dataset from ${RUN_DIR}." >&2
  echo "Set STEPCOUNT_TRAIN_DATA explicitly, or resume a run containing v36_run_manifest.tsv." >&2
  exit 1
fi
if [[ ! -e "${DATA_PATH_ARG}" ]]; then
  echo "[resume-v36-8gpu][ERROR] selected training dataset does not exist: ${DATA_PATH_ARG}" >&2
  exit 1
fi
if [[ -n "${BASETAG:-}" ]]; then
  BASETAG="${BASETAG}"
elif [[ "${DATA_PATH_ARG}" == "${STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA}" ]]; then
  BASETAG="ckpt476_11_30_quality_focused10k_bokgrpo_h200_8gpu_resume_${PROFILE}"
else
  BASETAG="ckpt476_11_50_maskcomplete38332_bokgrpo_h200_8gpu_resume_${PROFILE}"
fi

cd "${REPO_DIR}"

echo "[resume-v36-8gpu] load=${V36_LOAD_CHECKPOINT_PATH}"
echo "[resume-v36-8gpu] save=${V31_SAVE_CHECKPOINT_PATH}"
echo "[resume-v36-8gpu] data=${DATA_PATH_ARG} source=${DATA_SOURCE}"
echo "[resume-v36-8gpu] profile=${PROFILE} micro=${V31_MICRO_BATCH_UPDATE}/${V31_MICRO_BATCH_EXP} ulysses_enable=${V36_ENABLE_ULYSSES_SP} ulysses_sp=${V31_ULYSSES_SEQUENCE_PARALLEL_SIZE} blocks=${EASYR1_VLLM_NUM_GPU_BLOCKS} gpu_mem=${V31_GPU_MEM_UTIL}"
echo "[resume-v36-8gpu] save_freq=${TRAIN_SAVE_FREQ} save_limit=${TRAIN_SAVE_LIMIT} max_steps=${V36_MAX_STEPS:-<full-epoch>}"

exec bash "${SCRIPT_DIR}/mn_trainer_dense_v36.sh" 1 128 "${MODEL_PATH_ARG}" "${DATA_PATH_ARG}" "${BASETAG}"
