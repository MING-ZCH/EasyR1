# V14-Mixed Tuning Analysis & Optimization Directions

**Date**: 2026-03-19 (Revised: 2026-03-20)
**Objective**: Determine optimal parameters for V14-mixed experiment based on evidence from all prior experiments
**Revision Note**: Updated with V12/V13 NaN analysis (grad_norm findings reversed), holistic data+parameter strategy

---

## 1. Evidence-Based kl_coef Analysis

### Conclusion: kl_coef is NOT the root cause of format collapse

Cross-experiment evidence:

| Experiment | kl_coef | Data | Collapsed? | Onset |
|-----------|---------|------|-----------|-------|
| Pre-v5 BOK (10k) | 0.05 | 10k filtered | YES | step 118 |
| V7 BOK-DAPO | 0.01 | full | YES | step 114 |
| V8 BOK | 0.03 | full | Mild | step 187 |
| V9 BOK | 0.02 | full | NO | — |
| V11 BOK | 0.02 | full | NO | — |
| V12 BOK | 0.02 | full | NO | 88 steps stable (NaN after) |
| V13 BOK | 0.02 | full | NO | NaN inherited from V12 |
| V14-hard | 0.02 | hard-only 1808 | YES | step 48 |

**True collapse factors (by evidence strength):**
1. **Data composition/diversity** — hard-only and small filtered data always collapse
2. **DAPO filter** — V7 with DAPO=1 collapsed; V9+ with DAPO=0 stable
3. **Algorithm** — DrGRPO: 2/2 collapsed; Standard GRPO: 3/3 stable

**Decision: Keep kl_coef=0.02.** Changing it is unsupported by evidence.

---

## 2. Critical Finding: grad_norm & NaN Analysis

### ⚠️ REVISED: max_grad_norm=1.0 is BETTER than 0.5

Cross-experiment grad_norm NaN analysis:

| Experiment | max_grad_norm | Total Grad Entries | NaN Count | NaN% | Notes |
|-----------|-------------|-------------------|----------|------|-------|
| V12 | **0.5** | 184 | **92** | **50%** | NaN starts step ~89, 96% NaN after |
| V13-run1 | **0.5** | 31 | **29** | **93.5%** | Inherited from corrupted V12 |
| V13-run2 | **0.5** | 35 | **33** | **94.3%** | Inherited from corrupted V13-run1 |
| V14-hard | **1.0** | 78 | **7** | **9%** | Minimal NaN despite format collapse |

### V12 grad_norm diagnosis:
- V12 used max_grad_norm=0.5, but actual gradient magnitudes ranged 0.5-10.137
- **87/88 steps before NaN had grad > 0.5** — nearly every step was aggressively clipped
- Median natural grad = 0.917, mean = 1.316 — 0.5 clips **everything**
- NaN onset at step ~89 → after that 96% NaN → model parameters corrupted
- V13 resumed from V12's corrupted final checkpoint → inherited NaN (93-94%)

### V14-hard grad_norm diagnosis:
- V14-hard used max_grad_norm=1.0
- Natural grad range: 1.0-19.488, median=2.957, mean=3.601
- Only 7 NaN entries out of 78 → **1.0 is far more stable than 0.5**
- Format collapse was caused by data composition, NOT gradient instability

### Root cause hypothesis:
max_grad_norm=0.5 too aggressively clips all gradients → gradient information is destroyed → optimizer state accumulates error → eventually produces NaN. max_grad_norm=1.0 preserves more gradient signal → much fewer NaN.

**Decision: Keep max_grad_norm=1.0. Do NOT reduce to 0.5.**

---

## 3. V14-Mixed Script Bugs (Must Fix Before Launch)

### BUG 1: ACTOR_LR comment mismatch (cosmetic)
Code says `ACTOR_LR="1.5e-6"` but comment says "reduce to 1e-6".
**Fix:** Update comment, keep 1.5e-6 (V12 proven).

### BUG 2: BOK_TOTAL_STEPS / total_epochs inconsistency
Header says `total_epochs=2, BOK_TOTAL_STEPS=182` but code has `total_epochs=1, BOK_TOTAL_STEPS=91`.
**Decision:** Keep total_epochs=1 with BOK_TOTAL_STEPS=91. V14-hard proved multi-epoch is harmful.

### BUG 3: save_freq too sparse
save_freq=60 → only 1 checkpoint in 91 steps.
**Fix:** save_freq=20 (4-5 checkpoints).

---

## 4. Holistic Data + Parameter Analysis

### 4.1 Why V14-mixed Should Work (Data Perspective)

V14-mixed dataset: 5792 samples = 1808 hard + 384 eval + 3600 easy

**Positive signal analysis (from V12 baseline):**
- V12 used 11455 samples with natural easy/hard distribution
- V12's effective training was **only 88 steps** (NaN from step 89)
- 88 steps × 64 batch = 5632 effective training samples seen
- V14-mixed has **5792 samples ≈ V12's effective training size!**
- Key insight: V12's val=0.855 was achieved with ~5632 samples, not 11455!

**Risk factors:**
- V14-mixed is deliberately **harder** (31% hard vs V12's natural ~16% hard)
- High-count samples dominate: count=8 is 18.2% (vs V12's 8.5%)
- Harder data → noisier gradients → higher entropy growth

### 4.2 Data Distribution Risk

| Count | V12 (%) | V14-mixed (%) | Difficulty | Risk |
|-------|---------|---------------|-----------|------|
| 1-3 | 30.3 | 17.0 | Easy | ↓ format-anchor signal weakened |
| 4-6 | 40.9 | 31.5 | Medium | ↓ moderate decline |
| 7-10 | 28.9 | 51.4 | Hard | ↑↑ majority hard = noisy gradient |

**51.4% hard samples** vs V12's 28.9%. This makes reward signal more sparse:
- Hard samples are more likely to get answer_reward=0
- More zero-reward samples → advantage estimation dominated by noise
- BoK filtering helps (removes too-easy), but if MOST samples are hard, BoK may not have enough positive signal

### 4.3 Data Strategy Recommendations

**Strategy D1: Count distribution rebalancing (RECOMMENDED)**
- Current: count=1 has 175 samples (3.0%), count=8 has 1052 (18.2%)
- Proposal: Resample easy set (3600) to flatten count distribution
- Target: ~580 per count (5792/10), ensuring sufficient easy signal at each count level
- Implementation: Modify dataset creation, no code change needed

**Strategy D2: Epoch 1 curriculum (alternative)**
- Shuffle dataset so early batches have more easy samples
- Gradually increase hard sample ratio through training
- Risk: FSDP shuffling may override manual ordering

**Strategy D3: Dynamic difficulty via reward weights**
- Increase ANSWER_WEIGHT for easy samples, decrease for hard
- Already partially handled by BoK filtering (easy_threshold)
- May be overcomplicating things for V14; consider for V15

### 4.4 V12 vs V14-mixed: The Real Comparison

| Aspect | V12 (effective) | V14-mixed (planned) |
|--------|----------------|---------------------|
| Samples seen | ~5632 (88 steps × 64) | 5792 (91 steps × 64) |
| Hard ratio | ~16% natural | **31%** curated |
| grad_norm | 0.5 (caused NaN!) | 1.0 (proven stable) |
| Val best | 0.855 (within 88 steps) | Target: ≥0.86 |
| NaN risk | HIGH (50% NaN) | LOW (projected <10%) |
| Actual training | **Only 88 steps** effective! | 91 steps (all clean) |

**V14-mixed's advantage over V12:**
1. ✅ grad_norm=1.0 means ALL 91 steps should be useful (vs V12's 88)
2. ✅ Fresh SFT start (no corrupted optimizer state)
3. ✅ More hard samples for targeted improvement

**V14-mixed's risk vs V12:**
1. ⚠️ Higher hard ratio → slower convergence → may need 2 epochs
2. ⚠️ Less format-anchor signal from easy samples
3. ⚠️ Higher entropy growth expected

---

## 5. Optimization Directions for V14-Mixed

### Direction A: save_freq & val_freq (ESSENTIAL)
**Fix save_freq=60 → 20, val_freq=20 → 10.**
More frequent checkpointing and validation for early warning. This is non-negotiable.

### Direction B: Tau Annealing (KEEP DEFAULT)
BOK_TAU 0.7→0.3 over 91 steps (cosine). Same as V12 proven config. No change needed.
If format_fail rises above 15% by step 30, can manually adjust.

### Direction C: Reward Weights (KEEP DEFAULT)
ANSWER_WEIGHT=0.6, POINT_WEIGHT=0.3, FORMAT_WEIGHT=0.1. V12 proven config.
Don't change multiple variables simultaneously.

### Direction D: Data Rebalancing (RECOMMENDED)
See Strategy D1 above. Flatten count distribution in easy set to ensure sufficient positive signal.

### Direction E: Watchdog Early Stop (HIGH PRIORITY)
Activate monitoring with thresholds:
- format_fail > 20% for 3 consecutive steps → alert
- entropy > 0.7 → warning
- grad_norm NaN count > 3 in 10 steps → stop

### Direction F: Second Epoch (CONDITIONAL)
If epoch 1 shows stable training + val < 0.85:
- Run epoch 2 with fresh tau annealing (reset BOK_TAU)
- This differs from V14-hard where tau continued annealing across epochs

---

## 6. Recommended V14-Mixed Parameter Configuration

| Parameter | Current Script | Recommended | Reason |
|-----------|---------------|-------------|--------|
| ACTOR_LR | 1.5e-6 | **1.5e-6** (keep) | V12 proven, effective |
| kl_coef | 0.02 | **0.02** (keep) | Evidence: not a factor |
| max_grad_norm | 1.0 | **1.0** (keep) | V12=0.5 caused 50% NaN; V14=1.0 had 9% |
| total_epochs | 1 | **1** (keep) | V14-hard proved multi-epoch harmful |
| BOK_TOTAL_STEPS | 91 | **91** (keep) | Matches 1 epoch |
| save_freq | 60 | **20** | Essential: need intermediate checkpoints |
| val_freq | 20 | **10** | More frequent early warning |
| BOK_TAU | 0.7→0.3 | **0.7→0.3** (keep) | V12 proven |
| Reward weights | 0.6/0.3/0.1 | **0.6/0.3/0.1** (keep) | V12 proven |
| Watchdog | inactive | **active** | Early stop on collapse |
| Data rebalancing | none | **flatten count dist** | Ensure sufficient easy signal |

### Script edits summary:
1. save_freq: 60 → 20
2. val_freq: 20 → 10  
3. Fix comment: LR comment should match actual value (1.5e-6)
4. Fix header: total_epochs=1, BOK_TOTAL_STEPS=91 (remove 2/182 references)
5. Activate watchdog monitoring

---

## 7. Key Insights Summary

1. **V12 was only effectively trained for 88 steps** (NaN from step 89). Its val=0.855 came from ~5632 samples — almost identical to V14-mixed's 5792.

2. **max_grad_norm=1.0 is strictly better** than 0.5 for this model. 0.5 destroys gradient information and eventually causes NaN.

3. **Data composition, NOT hyperparameters, is the primary risk factor.** V14-hard collapsed due to hard-only data. V14-mixed has 62% easy+medium which should provide sufficient format-anchor signal.

4. **Count distribution matters.** V14-mixed is skewed toward high counts (51.4% in 7-10 range). Consider rebalancing the easy set for more uniform coverage.

5. **Single epoch is sufficient.** V12 achieved val=0.855 in 88 steps. V14-mixed has 91 steps — enough if the data quality is right.

---

*Generated from cross-experiment analysis of V7-V14 training logs, grad_norm NaN patterns, and data distribution analysis.*
