# V17 Retrain Changelog

## Date: 2025-03-28

## Problem
V17 Exp-C (Mixed + Cap=1.0 + LR=1.5e-6) achieved **val_best = 0.8594** (historical best) around step 35,
but this checkpoint was NOT saved (save_freq=30 → only step_30/60/90; step_30/60 were lost).
Only the degraded step_90 (val=0.7647) remains available.

Additionally, the watchdog triggered EARLY_STOP at step=5 but the training process did NOT receive the signal
because `ray_trainer.py` had no mechanism to check the stop file.

## Changes Made

### 1. V17 Training Script: Save Strategy Improvement
**File:** `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v17.sh`

- `trainer.save_freq`: 30 → **10** (save every 10 steps → 9 checkpoints total for 90 steps)
- Added `trainer.save_limit=-1` (keep ALL checkpoints, override config.yaml default of 5)
- `export STOP_FILE` for training process to read watchdog signal

**Expected checkpoints:** step 10, 20, 30, 40, 50, 60, 70, 80, 90
**Disk impact:** ~15GB per checkpoint × 9 = ~135GB total (vs ~45GB previously)

### 2. Training Loop: Watchdog Early Stop Integration
**File:** `verl/trainer/ray_trainer.py`

Added stop file detection in the training loop (3 locations):
- **L1008-1021:** After each step's metrics logging, check `STOP_FILE` env var.
  If file exists → log reason → save checkpoint → set `_early_stopped` flag → break
- **L786:** Inner batch loop: skip remaining batches when early_stopped
- **L775:** Outer epoch loop: skip remaining epochs when early_stopped
- **L1040:** End-of-training: distinguishes "early-stopped" vs "completed" log message

**Mechanism:** `break`-based (no exception throwing), clean exit with checkpoint saved.

### 3. No Algorithm Changes
All BoK-GRPO parameters remain identical to the V17 initial run:
- LR=1.5e-6, Cap=1.0, bok_clip=4.0, tau_init=0.5, easy_threshold=0.50
- Mixed data (5792 samples, ~91 steps/epoch)
- Epochs=1, ppo_epochs=2, max_grad_norm=1.0

## Retrain Command
```bash
cd /mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest
bash examples/launch_v17_experiments.sh C
```

## Post-Training: Merge Checkpoints to HuggingFace Format
After training completes, merge FSDP sharded checkpoints to HuggingFace format:
```bash
# Find the new V17 save directory (with updated timestamp)
V17_SAVE=$(ls -td /mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest/save/StepCount-7B-SFT-30k_v17_* | head -1)

# Merge each checkpoint
for step_dir in ${V17_SAVE}/global_step_*/actor; do
    echo "Merging: ${step_dir}"
    python3 /mnt/shared-storage-user/zhangchenhao/work/EasyR1/scripts/model_merger.py \
        --local_dir "${step_dir}" 2>&1 | tail -3
done
```

## Evaluation Plan
1. Find best val checkpoint from training log
2. Evaluate on StepCount-Bench-500 using `eval_fix_pixels_one_count_per_time.py`
3. Compare with SFT baseline (checkpoint-3537) and V17 step_90

## Key Metrics to Monitor
| Metric | V17 Original | Target |
|--------|-------------|--------|
| Val best | 0.8594 (step ~35) | ≥ 0.8594 |
| Val final | 0.7647 (step 90) | ≥ 0.80 |
| NaN count | 0 | 0 |
| Watchdog response | NOT working | Working ✓ |
