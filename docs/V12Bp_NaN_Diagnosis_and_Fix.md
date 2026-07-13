# V12-B' NaN Diagnosis & Fix (Step 104)

> Date: 2026-04-21
> Version: V12-B' (Config B')
> Failure: Step 104, training stopped at NaN

---

## 1. Failure Summary

| Metric | Value |
|--------|-------|
| Training Completion | 104/213 steps (48.9%) |
| First NaN | Step 104 |
| Total NaN Events | 22 (all from same root cause) |
| Final State | grad_norm = nan, grad_norm_ema = 1.881 |

---

## 2. Root Cause: BoK All-Correct Filtering Collapse

### The Problem Chain

```
Mixed data (28% hard) + BoK all_correct filtering
    ↓
Step 104: 272/1024 samples (26.6%) tagged as "all_correct" and filtered
    ↓
Only 752 samples get gradients (advantage heavily biased to negative cases)
    ↓
Advantage variance explodes: adv_std → entropy_loss = 1.121
    ↓
entropy_loss × coef + kl_loss in backward pass → NaN
```

### Step 104 Failure Metrics

| Metric | Value | Status |
|--------|:-----:|--------|
| `all_correct_filtered` | 272/1024 (26.6%) | **CRITICAL** |
| `ac_released` | 16/1024 (1.6%) | Only 1.6% recovery |
| `entropy_loss` | 1.121 | **EXTREME** (target <0.6) |
| `kl_loss` | 0.157 | High |
| `grad_norm` | **nan** | Failure |
| `grad_norm_ema` | 1.881 | EMA stable (spike detected) |
| `easy_drgrpo` | 416/1024 (40.6%) | Normal |
| `adv_mean` | -0.0118 | Balanced |
| `adv_std` | 0.8104 | Normal |

### Why This Happened

1. **BoK All-Correct Filter Design**
   - Purpose: Skip training on samples model already gets right
   - Logic: Filter if score ≥ BOK_EASY_SCORE_THRESHOLD (default 0.5)
   - Works well on easy-only data (few correct samples)

2. **Mixed Data Changed the Distribution**
   - mixed data: 28% hard + 72% samples from easy_plus_hard
   - Many samples have score ≥ 0.5 by chance
   - Step 104: 26.6% hit this threshold → massive filtering

3. **Filtering Breaks Advantage Normalization**
   - PPO relies on balanced advantage distribution
   - Filtering creates extreme imbalance: mostly negative advantages
   - Entropy loss explodes trying to regularize the distribution
   - kl_loss amplifies the instability

4. **Why ABSOLUTE_CAP=4.0 Didn't Help**
   - Cap applies **after** clipping to the gradient
   - NaN happens **during backward pass** when computing loss
   - Loss (entropy + kl) becomes NaN → backward propagates NaN

---

## 3. Entropy Degradation Timeline

| Step Range | Entropy | Status |
|:----------:|:-------:|--------|
| 0-79 | ~0.50 | Normal |
| 80-90 | 0.80-0.90 | Creeping up |
| 91-100 | 0.90-1.00 | Rising consistently |
| 101-103 | ~1.10 | High but stable |
| 104 | 1.121 | **NaN threshold** |

Entropy rises steadily from step 85 onward, indicating filtering effect accumulates.

---

## 4. Why V12-A (Easy-Only) Won't Have This Issue

- Easy-only: only 11,455 samples, 14% naturally hard
- BoK filtering affects fewer samples (lower baseline score)
- Advantage distribution stays balanced
- Entropy stays <0.6

---

## 5. The Fix: BOK_FILTER_ALL_CORRECT=0

### What Changed

| Parameter | Original | Fix |
|-----------|:--------:|:----:|
| `BOK_FILTER_ALL_CORRECT` | 1 (enabled) | **0 (disabled)** |
| All other params | Same | Same |

### Why This Works

1. **Removes aggressive filtering**
   - No more 26% of samples being filtered
   - Advantage distribution stays balanced
   - Entropy stays in control

2. **Trades redundant training for stability**
   - Some "already good" samples train again (small cost)
   - But since 28% are hard, plenty of hard data still trains
   - Net: more stable, acceptable redundancy

3. **Preserves all other BoK benefits**
   - DRGRPO fallback still works
   - AllWrong cap still works
   - Easy scaling still works

### Alternative Fixes (Ranked by confidence)

| Option | Parameter | Change | Risk |
|--------|-----------|:------:|:----:|
| **1. Disable Filter** | BOK_FILTER_ALL_CORRECT | 1→0 | LOW ✓ |
| 2. Raise threshold | BOK_EASY_THRESHOLD | 0.5→0.7 | MED |
| 3. Double confirmation | BOK_EASY_SCORE_THRESHOLD | add requirement | MED |
| 4. Lower kl | kl_coef | 0.02→0.03 | HIGH |
| 5. Less aggressive clip | max_grad_norm | 0.5→1.0 | HIGH |

---

## 6. Script Changes

**New script**: `qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v12_Bp_fix.sh`

Key changes:
```bash
# Disable all_correct filtering
export BOK_FILTER_ALL_CORRECT=${BOK_FILTER_ALL_CORRECT:-0}

# All other params preserved from V12-B'
LR=2e-6, clip=0.5, ppo=1, kl=0.02, BOK_CLIP=4.0, mixed data
```

---

## 7. Expected Outcome

**V12-B*-Fix** (with BOK_FILTER_ALL_CORRECT=0):
- Target: 82-84% (match V12-B' projection)
- NaN risk: <5% (if entropy still spikes, we have evidence)
- Format fail: <5%
- Total steps: 213 (no early stopping expected)

---

## 8. Post-Training Analysis

If **V12-B*-Fix** succeeds:
- Compare performance vs V12-A (easy-only baseline)
- Validates that mixed data + proper BoK tuning > easy-only
- Proves sign-GD (clip=0.5) advantage

If **V12-B*-Fix** still has issues:
- Fall back to Option 2: BOK_EASY_THRESHOLD=0.7 (less aggressive filter)
- Or Option 5: reduce clip to 1.0 (less extreme sign-GD)
