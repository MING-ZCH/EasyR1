# V24 Smart Filter — Implementation Record

## Overview
V24 adds a **Smart Filter** to the All-Correct group routing that preserves point
training signal from groups where answer is correct but point quality is still low.

## Problem (V23)
- All-Correct filter: if all 16 trajectories have answer=1 → zero gradient
- ~23% of training data goes to zero gradient
- These groups pass low_var check (std > 1e-5) → point quality **varies**
- Zero gradient **wastes** the point direction signal in these groups

## Solution (V24)
Add a `group_mean` threshold check to All-Correct filter:

```
if all(answer=1):
    if group_mean >= smart_filter_threshold:  → zero gradient (truly mastered)
    else:                                     → release to Easy path (optimize point)
```

### Why group_mean works as proxy for mean(point)
When all answer=1, format≈1:
- `overall = 0.6×1 + 0.3×point + 0.1×1 = 0.7 + 0.3×point`
- `group_mean = 0.7 + 0.3 × mean(point)`
- `group_mean ≥ 0.955  ⟺  mean(point) ≥ 0.85`

## Changes Made

### File: `verl/trainer/core_algos.py`

1. **New env var** (L510):
   ```python
   _smart_filter_threshold = float(os.environ.get("BOK_SMART_FILTER_THRESHOLD", "0"))
   ```
   - Default 0 = disabled (V23 behavior)
   - Recommended V24 value: 0.955 (≈ mean(point) ≥ 0.85)

2. **New counter** (L544):
   ```python
   n_all_correct_released = 0  # V24: groups released to Easy by smart filter
   ```

3. **Modified All-Correct filter** (L593-601):
   ```python
   if pass_rate_full >= 1.0 - 1e-6:  # all K correct
       if _smart_filter_threshold > 0 and group_mean.item() < _smart_filter_threshold:
           n_all_correct_released += K   # release to Easy path
       else:
           n_all_correct_filtered += K
           continue                      # zero gradient
   ```

4. **Updated logging** (2 lines):
   - Main log: added `ac_released={n}/{bsz}`
   - Detail log: added `ac_released={n}/{bsz}` + `smart_filter_th={val:.3f}`

### Files NOT changed
- `ray_trainer.py` — no change needed (group_mean already computed in core_algos)
- Training scripts — add `BOK_SMART_FILTER_THRESHOLD=0.955` to V24 script

## Backward Compatibility
- Default `BOK_SMART_FILTER_THRESHOLD=0` → all all-correct groups filtered → V23 behavior
- No changes to any other routing path
- Counter `ac_released` shows 0 when disabled

## Expected Impact
- ~12% of data previously zero-gradient now gets Easy z-score training (point optimization)
- Easy path increases from ~43% to ~55%
- All-Correct filter decreases from ~23% to ~11%
- Effective gradient utilization: ~77% → ~89%

## V24 Training Script Config
```bash
export BOK_SMART_FILTER_THRESHOLD=0.955  # V24 smart filter (mean_point >= 0.85)
# All other env vars same as V23
```

## V24 Final Config (Updated: Soft Gate Removed)

### Why gate_mode=off (not soft)
Analysis showed soft gate (base=0.8) is harmful:
- **Easy path**: z-score rank unchanged (rank correlation = 1.0) → soft gate has NO effect
- **AC released**: z-score coefficient cancels out → soft gate has NO effect  
- **BoK path**: correct-wrong gap shrinks 40% (min gap: 0.30 → 0.18) → **HARMFUL**
  - Soft gate makes "answer_correct + point_bad" ≈ "answer_wrong + point_good"
  - BoK softmax gives less weight to correct trajectories → weaker learning signal
  - Directly contradicts pass@1 maximization objective

### V24 = V23 + Smart Filter ONLY (single-variable experiment)

| Parameter | V23 | V24 |
|-----------|-----|-----|
| `BOK_FILTER_ALL_CORRECT` | 1 | 1 |
| `BOK_SMART_FILTER_THRESHOLD` | 0 (disabled) | **0.955** |
| `BOK_EASY_SCALE` | 1.0 | 1.0 |
| `TRAJ_ANSWER_GATE_MODE` | off | **off** (unchanged) |

Threshold 0.955 maps to mean(point) ≥ 0.85 when gate=off:
- `group_mean = 0.7 + 0.3*mean(point)` → `0.955 = 0.7 + 0.3*0.85`

## V24 Final Parameter Update (LR + KL Restoration)

### Root Cause Analysis: V23 pixmo-test 79.77% < V12 81.47%

**Effective learning rate comparison:**
- V12: LR=1.5e-6 × ppo_epochs=2 = "3.0e-6 equivalent"
- V23: LR=1.0e-6 × ppo_epochs=1 = "1.0e-6 equivalent" (33% of V12!)
- V24: LR=1.5e-6 × ppo_epochs=1 = "1.5e-6 equivalent" (50% of V12, 150% of V23)

**NaN forensics (critical finding):**
- V12 had NaN from step 85 onwards (91/178 steps NaN!) — ppo_epochs=2 is the ROOT cause
- V22 had NaN from step 118 — also ppo_epochs=2
- All ppo_epochs=1 runs had ZERO NaN (V21r2, V23)
- LR and kl_coef are NOT NaN factors

**Safe to raise LR because:**
- NaN is caused exclusively by ppo_epochs=2 (stale 2nd epoch update)
- ppo_epochs=1 has been NaN-free across all runs regardless of LR
- V23 grad_norm peaked at only 2.29 (far below NaN territory)

### V24 Complete Diff from V23

| Parameter | V23 | V24 | Rationale |
|-----------|-----|-----|-----------|
| ACTOR_LR | 1e-6 | **1.5e-6** | Restore V12-level effective LR |
| kl_coef | 0.03 | **0.02** | Restore V12-level exploration |
| SMART_FILTER_THRESHOLD | 0 | **0.955** | Release low-point AC groups |
| ANSWER_GATE_MODE | off | off | Soft gate harms BoK contrast |
| ppo_epochs | 1 | 1 | Unchanged (NaN prevention) |
| FORMAT_REJECTION | 1 | 1 | Unchanged (format safety) |
