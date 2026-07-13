# V17/V18 First-Principles Analysis Report

## 1. Full Evaluation Comparison Table

### StepCount-Bench-500 (count range 11-50, dense objects)

| Version | Algorithm | Data | Acc (no history) | Acc (with_history_1) | Notes |
|---------|----------|------|:-:|:-:|-------|
| Qwen2.5-VL-7B base | — | — | 5.60% | — | Zero-shot |
| SFT-3537 (our SFT) | — | — | 9.60% | — | Best SFT baseline |
| V7 (standard GRPO) | GRPO | easy | 10.60% | 12.00% | |
| V11 (BoK-GRPO) | BoK-GRPO | easy | 12.00% | 13.00% | 1ep |
| **V12 (BoK-GRPO)** | BoK-GRPO | easy | **13.00%** | **12.00%** | **2ep, NaN@85** |
| V12-step132 | BoK-GRPO | easy | **15.45%** | 10.60% | **Best checkpoint** |
| V14-mixed | BoK-GRPO | mixed | 11.40% | 13.20% | 1ep, mixed data |
| V15 | BoK-GRPO | easy | 12.20% | — | 2ep |
| V16 | BoK-GRPO | easy | 13.20% | — | 2ep |
| **V17 (Cap=1.0)** | BoK-GRPO+Cap | mixed | **12.20%** | **11.40%** | **1ep, NaN@77** |

**Key Finding: V17 regressed from V12-step132's 15.45% to 12.20% on StepCount-Bench.**

### CountBench-491 (count range 0-10, sparse objects)

| Version | Acc | Notes |
|---------|:-:|-------|
| SFT-10k-high-1296 | 78.82% | SFT baseline |
| V12-step132 | **79.43%** | Best V12 checkpoint |
| V14-hard | 79.02% | |
| V14-mixed | 78.21% | |
| V15 | 79.02% | |
| V16 | 77.19% | Lower |
| **V17 (Cap=1.0)** | **79.43%** | **Ties V12-step132** |

**Key Finding: V17 CountBench = V12-step132 = 79.43%. No improvement, but no regression either.**

### PixMo-529 (count range 0-10, sparse objects)

| Version | Acc (no history) | Acc (with_history_1) | Notes |
|---------|:-:|:-:|-------|
| V8 | 80.15% | 79.21% | |
| V11 | 80.15% | 82.04% | |
| V12-step132 | **81.47%** | **82.80%** | Best V12 |
| V12-step178 | 82.04% | **83.36%** | **Overall best** |
| V14-mixed | 80.34% | 81.85% | |
| V15 | 80.15% | 81.66% | |
| V16 | 79.96% | 81.29% | |
| **V17** | — | **81.29%** | **Same as V16** |

**Key Finding: V17 PixMo 81.29% < V12-step178 83.36% (Δ = -2.1pp)**

---

## 2. V17 Training Dynamics Deep Analysis

### 2.1 Training answer_mean Trajectory
```
Step  1: answer=0.7310, fmt_fail=3.4%
Step  5: answer=0.6633, fmt_fail=2.6%   ← Early dip
Step 10: answer=0.6344, fmt_fail=2.6%   ← Historic low
Step 15: answer=0.6797, fmt_fail=1.0%
Step 20: answer=0.7125, fmt_fail=0.8%
Step 30: answer=0.6502, fmt_fail=5.7%   ← Another dip
Step 40: answer=0.6965, fmt_fail=4.0%
Step 50: answer=0.7264, fmt_fail=5.4%
Step 60: answer=0.7436, fmt_fail=1.9%   ← Peak
Step 70: answer=0.7069, fmt_fail=4.0%
Step 80: answer=0.7400, fmt_fail=6.3%
Step 90: answer=0.6433, fmt_fail=12.2%  ← COLLAPSE
```

### 2.2 Validation answer_mean Trajectory
```
Val call  1: answer=0.7812 (SFT starting point)
Val call  3: answer=0.7031
Val call 10: answer=0.7812
Val call 20: answer=0.7656
Val call 30: answer=0.6719  ← Dip
Val call 40: answer=0.6719  ← Stagnant
Val call 50: answer=0.8594  ← PEAK (historic best!)
Val call 60: answer=0.8125
Val call 63: answer=0.7647  ← FINAL (degraded)
```

### 2.3 Grad Norm Trajectory — The Smoking Gun
```
Step  1: grad_norm=0.977
Step  5: grad_norm=2.147
Step 10: grad_norm=3.674
Step 20: grad_norm=0.860
Step 30: grad_norm=6.675    ← SPIKE (but recovered)
Step 40: grad_norm=4.040
Step 45: grad_norm=5.390
Step 50: grad_norm=4.588
Step 55: grad_norm=3.282
Step 60: grad_norm=3.167
Step 65: grad_norm=2.703
Step 70: grad_norm=2.695
Step 75: grad_norm=1.824
Step 76: grad_norm=4.932    ← CRITICAL SPIKE
Step 77: NaN                ← IRREVERSIBLE FROM HERE
Step 78-80: NaN
Step 81: 2.009 (brief recovery)
Step 82-90: NaN             ← PERMANENT NaN
```

**Root Cause: Step 76 grad_norm spiked to 4.932 → Step 77 NaN → Unrecoverable.**
The earlier spike at step 30 (6.675) didn't cause NaN because the model was still near SFT init.
By step 76, cumulative parameter drift made the model fragile — a moderate 4.9x spike was enough to push into NaN territory.

### 2.4 BoK-GRPO Routing Statistics
```
Typical step: easy_drgrpo ≈ 400-550/1024 (43%)
              allwrong_capped ≈ 16-80/1024 (3.5%)
              all_correct_filtered ≈ 150-300/1024 (21%)
              → Only ~30% goes through actual BoK softmax ranking
              adv_mean stays near 0 (good normalization)
              adv_std ≈ 0.85-1.0 (healthy)
```

---

## 3. Why V17 NaN Occurred Despite Cap=1.0

### 3.1 Cap=1.0 Only Limits Advantage, Not Grad Norm

Cap=1.0 limits the AllWrong group advantage to max 1.0 (formerly unbounded).
But **NaN does NOT come from large advantages**. It comes from:
1. **Loss landscape instability** — the policy ratio `π/π_old` diverges
2. **bf16 overflow** — accumulated parameter updates create values exceeding bf16 range
3. **Cascading effect** — once grad_norm is inf/NaN on one GPU, FSDP aggregation propagates it

### 3.2 The Real NaN Trigger: Stale Rollouts + PPO Epoch 2

V17 uses `ppo_epochs=2`, meaning each batch is trained twice:
- **PPO Epoch 1**: policy ratios are close to 1.0 (fresh rollouts) → stable
- **PPO Epoch 2**: policy was already updated in epoch 1. Now `π_new/π_old` can be large → ratio clipping may not be sufficient → loss explodes

The grad norm at step 76 (4.932) suggest the PPO Epoch 2 of that batch encountered an extreme ratio.

### 3.3 Why V12 Also Had NaN at Step 85

V12 config: LR=2e-6, ppo_epochs=2, easy data, NO Cap
- V12 NaN triggered at step 85 with larger LR (2e-6 vs 1.5e-6)
- V17 NaN triggered at step 77 with smaller LR (1.5e-6) but mixed data

**Mixed data is harder** → more AllWrong groups → despite Cap, the correct-trajectory advantages can still be large (up to 4.0 cap in adv_range) → combined with ppo_epoch=2, creates instability.

---

## 4. V17 vs V12: Why V17 is Worse

| Factor | V12 (best=step132) | V17 | Impact |
|--------|:---:|:---:|--------|
| LR | 2e-6 | 1.5e-6 | V17 learns slower |
| Data | easy only (11455) | mixed (5792) | V17 sees harder data it can't learn from |
| NaN onset | Step 85 | Step 77 | V17 has less clean training |
| Total clean steps | 84 (2ep × 179steps, NaN at step 85) | 76 out of 90 | V17 gets 76 clean steps only |
| Val peak | ≈ step 132 | Val call 50 (≈step 50) | Both peak mid-training |
| Checkpoint saved at | step 132 (lucky!) | step 90 (NaN-corrupted!) | V12 used a better checkpoint |

**Critical insight**: V12's evaluation used the step-132 checkpoint which was pre-NaN.
V17's evaluation used step-90, which was DEEP IN NaN territory (NaN from step 77).
If V17 had been saved at step 50 (val=0.8594), it would likely outperform V12.

---

## 5. V18 Assessment (GradSpikeProtect)

### 5.1 V18 Config Changes from V17
- `GRAD_SPIKE_PROTECT: 0 → 1` (enabled, threshold=5x, cooldown=3, lr_factor=0.1)
- `BOK_TOTAL_STEPS: 178 → 90`
- `save_freq: 30 → 90` ← **PROBLEM**: only saves final checkpoint!
- Everything else identical (LR=1.5e-6, mixed data, Cap=1.0, ppo_epochs=2)

### 5.2 GradSpikeProtect Assessment

**Mechanism**: When grad_norm > 5× running_mean, reduces LR by 0.1x for 3 steps.

**Will it prevent NaN?**
- V17's fatal spike: step 76 at 4.932, but the running mean by that point was ~2.5x, so 5×2.5 = 12.5 threshold. The 4.932 would NOT trigger protection!
- Step 30 had grad_norm=6.675, running mean ~2x → threshold=10 → 6.675 < 10 → NOT triggered
- **The threshold=5x is too high for V17's spike pattern.**

**Prediction**: V18 will likely still experience NaN around step 70-85. The GradSpikeProtect may catch EXTREME spikes but won't prevent the gradual instability that leads to NaN.

### 5.3 V18 Current Status (Step 3)
```
Step 3: answer_mean=0.7538, format_fail=2.15%, spike_count=0
Easy: 432/1024, AllCorrect: 256/1024, AllWrong: 16/1024
No spikes detected yet (too early)
```

---

## 6. Root Cause Summary

The fundamental issue is NOT a single catastrophic event. It's **cumulative parameter drift + ppo_epochs=2**:

1. **Mixed data introduces unsolvable problems** → AllWrong groups generate capped but still nonzero gradients
2. **ppo_epochs=2 amplifies staleness** → second epoch has stale reference policies
3. **bf16 precision** → accumulated small errors compound over 70+ steps
4. **No early stopping at val peak** → the best model was never saved (step 90 checkpoint was NaN-corrupted)
5. **max_grad_norm=1.0 clips magnitude but not direction** → even clipped NaN gradients propagate

---

## 7. Recommended Optimization Strategy

### Tier 1: Must-Do (Immediate Impact)

1. **Reduce ppo_epochs from 2 → 1**
   - This is the single most impactful change. PPO epoch 2 is the primary NaN amplifier.
   - V9 (1ep, easy) and V11 (1ep, easy) both had 0 NaN.
   - Trade-off: slightly less sample efficiency, but stable training.

2. **Save checkpoints every 10 steps** (save_freq=10, save_limit=-1)
   - V17's val peak was at step 50 but only step 90 was saved.
   - This alone could have saved a 0.8594 checkpoint.
   - **V18 has save_freq=90 — MUST FIX immediately!**

3. **Use easy data only, not mixed**
   - V11 (easy, 1ep): Val=0.828, 0 NaN, StepCount-Bench=12.00%
   - V12 (easy, 2ep): Val=0.776 at NaN, StepCount-Bench=15.45% (at best ckpt)
   - V14-mixed (mixed, 1ep): Val=0.762, StepCount-Bench=11.40%
   - Mixed data hurts both stability and performance.

### Tier 2: High Impact

4. **Reduce max_grad_norm from 1.0 → 0.5**
   - V17's NaN started after grad_norm 4.932 (clipping at 1.0 was clearly insufficient)
   - Wait — max_grad_norm=1.0 should clip to at most 1.0. The logged values >1.0 suggest the grad_norm is the PRE-CLIP value.
   - Still, tighter clipping = smaller parameter updates = more stable.

5. **Lower LR to 1e-6** for longer training
   - V17's LR=1.5e-6 with mixed data was still too aggressive.
   - Try LR=1e-6 with 2 epochs of easy data (358 steps) for slower but stabler learning.

6. **Fix GradSpikeProtect threshold**: 5x → 3x
   - V17's fatal spike (4.932 vs running mean ~2.5) was only 2x.
   - Threshold=3x would have caught most dangerous spikes.

### Tier 3: Experimental (Research Ideas)

7. **Best-of-K without ppo_epochs=2**: The core BoK-GRPO idea (softmax ranking → advantage) is sound. The instability comes from PPO epoch 2, not from BoK itself.

8. **Moving average checkpoint (EMA)**: Keep an exponential moving average of weights. Even if training goes NaN, the EMA checkpoint would be near-peak performance.

9. **NaN recovery**: When grad_norm is NaN, rollback the parameter update and skip that batch entirely (currently the model uses "Skip update" but the damage to optimizer state may persist).

10. **Curriculum: Easy → Hard**: Train on easy data first (100 steps), then gradually introduce hard samples. This gives the model stable grounding before facing difficult cases.

---

## 8. Recommended Next V19 Configuration

```bash
# V19: Stability-first approach
export LR=1.5e-6           # Keep V17's LR
export PPO_EPOCHS=1         # ← KEY CHANGE: 1 instead of 2
export TOTAL_EPOCHS=2        # 2 epochs over easy data = 358 steps
export DATA=easy             # ← KEY CHANGE: easy only
export MAX_GRAD_NORM=0.5     # ← Tighter clipping
export GRAD_SPIKE_PROTECT=1  # Keep enabled with threshold=3x
export SAVE_FREQ=10          # Save every 10 steps
export SAVE_LIMIT=-1         # Keep all checkpoints
export ALLWRONG_CAP=1.0      # Keep Cap
export BOK_TAU=0.5           # Keep tau
export EASY_THRESHOLD=0.50   # Keep routing threshold
```

Expected outcome:
- 0 NaN (V9/V11 with 1ep easy had 0 NaN)
- Val peak achievable and saveable with save_freq=10
- StepCount-Bench: 12-15% (matching V12-step132 range)
- CountBench: 78-80%
- PixMo: 80-83%

---

## 9. For V18 Currently Running

**Urgent**: V18 has save_freq=90 (only saves at the very end). 
If NaN occurs at step 70+, the only saved checkpoint will be NaN-corrupted.

**Recommendation**: 
- If possible, modify V18 to save more frequently (but this requires restart).
- Otherwise, let V18 run as a "GradSpikeProtect validation experiment" — if it survives past step 77 without NaN, GradSpikeProtect is valuable.
- If V18 gets NaN anyway, that confirms GradSpikeProtect threshold=5x is insufficient.

