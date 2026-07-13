#!/bin/bash
# ================================================================
# V32 Resume from Step 20 — OOM Fix Applied
# ================================================================
# Resumes V32 training from global_step_20 checkpoint.
# Fix: micro_batch_update 8→4 to prevent backward OOM on long sequences.
#
# Usage (on pod launcher node):
#   cd /apdcephfs/private_hyleochang/EasyR1-latest
#   bash examples/v32_resume_from_step20.sh
# ================================================================

# Resume from existing step_20 ckpt
export V32_LOAD_CHECKPOINT_PATH=/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/checkpoints/StepCount-7B_v32_sparse_0_10_stable_drfix_32gpu_20260531_0545/global_step_20

# Keep same experiment name & save path (continues into same dir)
export V31_EXPERIMENT_NAME=StepCount-7B_v32_sparse_0_10_stable_drfix_32gpu_20260531_0545
export V31_SAVE_CHECKPOINT_PATH=/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz/checkpoints/${V31_EXPERIMENT_NAME}

# OOM fix: micro_batch_update 8→4 (halves backward activation peak)
export V31_MICRO_BATCH_UPDATE=4

# Keep total steps = 49 (will continue from step 20 to 49)
export BOK_TOTAL_STEPS=49

echo "[V32-Resume] Loading from: ${V32_LOAD_CHECKPOINT_PATH}"
echo "[V32-Resume] micro_batch_update: 4 (OOM fix, was 8)"
echo "[V32-Resume] Will train steps 21-49"

exec bash "$(dirname "$0")/v32_sparse_0_10_stable_drfix.sh"
