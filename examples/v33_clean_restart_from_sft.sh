#!/bin/bash
# ================================================================
# V33 Clean Restart from SFT Base — V31 Training Params + V32 Infra Fixes
# ================================================================
# This is a CLEAN RESTART from StepCount-7b-SFT-30k-high-without-reasoning.
# No previous checkpoint is loaded.
#
# Merge Decision (V31 training params + V32 infrastructure):
#
# TRAINING PARAMS: Use V31 values
#   - LR: 1.5e-6 (V31's 32-GPU uplift; V32 reverted to 1e-6 conservatively)
#   - micro_batch_update: 8 (V31 original; V32 reduced to 4 as "OOM fix")
#
#   Why restoring V31 training params is correct:
#     V32 OOM at step 36 was caused by gpu_keepalive running in background
#     (~10GB persistent GPU occupancy). V31 ran to completion with micro_batch=8
#     without OOM — its poor results came from mask incompleteness + keepalive
#     interference, NOT from training hyperparameters.
#     Now gpu_keepalive is stopped during training, so GPU memory has full
#     headroom and micro_batch=8 is safe. micro_batch=8 also means better
#     gradient diversity and faster throughput per step.
#
# INFRA: Use V32 values (all improvements over V31)
#   - cuBLAS 12.6.4.1 explicit path (vs V31's pip 12.2/12.5)         [V32]
#   - Ray GCS: /dev/shm + 300s timeouts (vs V31 minimal)             [V32]
#   - Mask LRU cache (STEPCOUNT_MASK_CACHE=1, 300K cap)               [V32]
#   - REWARD_NUM_WORKERS=16 (vs V31 default 4)                        [V32]
#   - Correct masks: Full metadata (282,454) + Sharded files (282,455) [V32]
#   - Debug frequency 200 (vs V31's noisy 10)                         [V32]
#   - BOK_SMART_FILTER=0.955 (Phase0 diagnosis)                       [V32]
#
# Usage (on Pod launcher node, INDEX=0):
#   cd /apdcephfs/private_hyleochang/EasyR1-latest
#   bash examples/v33_clean_restart_from_sft.sh
#
# Dry run:
#   V32_DRY_RUN=1 bash examples/v33_clean_restart_from_sft.sh
# ================================================================

# ==================== V31 Training Params (restored) ====================
# Restore V31's LR uplift for 32-GPU: larger effective batch needs higher LR.
export ACTOR_LR=${ACTOR_LR:-1.5e-6}

# Restore V31's micro_batch=8. Safe now because gpu_keepalive is stopped
# (it was the real OOM cause in V32, not the batch size itself).
# micro_batch=8 gives better gradient diversity and faster throughput.
export V31_MICRO_BATCH_UPDATE=${V31_MICRO_BATCH_UPDATE:-8}

# ==================== New Experiment Name ====================
export V31_EXPERIMENT_NAME=${V31_EXPERIMENT_NAME:-StepCount-7B_v33_clean_full_masks_32gpu_$(date +%Y%m%d_%H%M)}
export V31_SAVE_CHECKPOINT_PATH=${V31_SAVE_CHECKPOINT_PATH:-/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/checkpoints/${V31_EXPERIMENT_NAME}}

# Explicitly unset checkpoint resume: start from SFT base (MODEL_PATH).
unset V32_LOAD_CHECKPOINT_PATH

echo "================================================================"
echo "[V33] Clean Restart — 32 GPU (V31 train-params + V32 infra, full-mask)"
echo "[V33] experiment_name=${V31_EXPERIMENT_NAME}"
echo "[V33] save_path=${V31_SAVE_CHECKPOINT_PATH}"
echo "[V33] No checkpoint loaded — starting from SFT base"
echo "[V33] Training params (V31): LR=${ACTOR_LR}  micro_batch_update=${V31_MICRO_BATCH_UPDATE}"
echo "[V33] Infra (V32):           cuBLAS=12.6.4.1  mask_cache=1  reward_workers=16"
echo "[V33]                        masks: Full-metadata(282,454) + Sharded-files(282,455)"
echo "================================================================"

exec bash "$(dirname "$0")/v32_sparse_0_10_stable_drfix.sh"
