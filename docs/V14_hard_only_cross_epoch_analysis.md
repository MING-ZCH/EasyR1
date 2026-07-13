# V14-Hard-Only Cross-Epoch Analysis Report (FINAL)

**Date**: 2026-03-19 (Updated)  
**Experiment**: V14-hard-only (training_interleaved_traj_v14_hard_only)  
**Status**: CATASTROPHIC FAILURE — Training should be stopped immediately  
**Steps Analyzed**: 0-77 / 112 total (Epochs 1-3 of 4)

## Executive Summary

**V14-hard-only has experienced an irrecoverable format collapse starting at Step 48-50 (Epoch 2, steps 20-22).** The model mutated the JSON key `"point_2d"` to `"point_22"`, `"point_20"`, `"point_27"`, etc., causing format validation failure rate to jump from 9% to 98%+ within 3 steps. This is a self-reinforcing death spiral: format errors → zero format reward → less positive signal → more entropy → more format drift → more errors.

**Key numbers:**
- Val answer accuracy: NEVER exceeded SFT baseline (0.752)
- Format fail rate: 6.8% (E1) → 32.5% (E2) → 97.5% (E3)
- Entropy: 0.567 → 2.297 (+305%)
- KL: 0.003 → peak 0.824

## Configuration Summary

| Parameter | Value |
|-----------|-------|
| Dataset | 1808 hard-only samples |
| batch_size | 64, rollout_n=16 (1024 rollouts/step) |
| total_epochs | 4 (28 steps/epoch = 112 total) |
| ppo_epochs | 2, shuffle=True |
| LR | 1e-6 |
| kl_coef | 0.02 |
| BOK_TAU | 0.7→0.3 (cosine over 113 steps) |
| FORMAT_REJECTION | 0 (disabled) |
| DAPO_FILTER | 0 (disabled) |

---

## 1. CATASTROPHIC FORMAT COLLAPSE

### Timeline

| Step | Epoch | format_fail | Status |
|------|-------|-------------|--------|
| 1-28 | E1 | 6.8% avg | Normal |
| 29-44 | E2 early | ~8% avg | Normal |
| 45 | E2 | 8.5% | Normal |
| 46 | E2 | 5.1% | Normal |
| 47 | E2 | **9.3%** | Last normal step |
| **48** | **E2** | **26.6%** | **First spike** |
| **49** | **E2** | **53.8%** | **Halfway broken** |
| **50** | **E2** | **94.1%** | **Near-total collapse** |
| 51+ | E2-E3 | **98%+** | Fully broken, irreversible |

### Root Cause: Token-Level BPE Confusion

The model mutated `"point_2d"` → `"point_22"`, `"point_20"`, `"point_27"`, etc.

**All 12 corrupted variants found:**
| Variant | Occurrences | 
|---------|-------------|
| "point_22" | 495 |
| "point_20" | 311 |
| "point_27" | 165 |
| "point_24" | 123 |
| "point_26" | 105 |
| "point_25" | 99 |
| "point_21" | 96 |
| "point_28" | 73 |
| "point_23" | 54 |
| "point_29" | 39 |
| "point_2 " | 4 |
| "point_2." | 3 |

**Mechanism**: The BPE tokenizer encodes `"point_2d"` with `"point_2"` as a shared prefix token. As entropy rises (model becomes less confident), the next-token probability for `d` vs digits (`0-9`) becomes uncertain. Once a few rollouts output `"point_22"` instead of `"point_2d"`, the format validation fails → overall reward drops → positive signal decreases → entropy rises further → more corruption. Self-reinforcing loop.

### Turn-Level Corruption Propagation

The corruption started in **later turns** (turn 7+) and progressively spread to earlier turns:

| Step | Turn 1 | Turn 2 | Turn 3 | Turn 7 | Turn 10 |
|------|--------|--------|--------|--------|---------|
| 47 | point_2d | point_2d | point_2d | point_2d | — |
| 48 | point_2d | point_2d | point_2d | point_2d | point_2d |
| 49 | point_2d | point_2d | — | **point_22** | point_2d |
| 50 | point_2d | **point_21** | **point_21** | **point_28** | **point_27** |
| 55 | — | **point_28** | **point_28** | **point_27** | **point_27** |
| 60 | point_2d | **point_26** | **point_20** | **point_28** | **point_20** |
| 70 | **point_22** | **point_22** | **point_22** | **point_20** | **point_21** |

By Step 70, even Turn 1 is corrupted — the model has completely lost the `"point_2d"` key.

---

## 2. Validation Score Timeline

| Step | Epoch | Val_Ans | FmtFail | vs SFT |
|------|-------|---------|---------|--------|
| 0 | — | 0.7520 | 4.0% | baseline |
| 10 | E1 | 0.7364 | 4.0% | -2.1% |
| 20 | E1 | 0.7199 | 3.3% | -4.3% |
| 30 | E2 | 0.7403 | 3.8% | -1.6% |
| 40 | E2 | 0.7468 | 4.0% | -0.7% |
| 50 | E2 | 0.7333 | **99.7%** | -2.5% |
| 60 | E3 | 0.7538 | **99.7%** | +0.2% |
| 70 | E3 | 0.7438 | **99.7%** | -1.1% |

**Paradox**: Val answer_mean stays ~0.74 despite 99.7% format failure. This is because:
1. answer_mean is computed independently from format — it only checks `<answer>N</answer>` tags
2. The model still outputs reasonable count numbers even with broken JSON keys
3. Format reward is a separate component that goes to zero

**But val never improved.** Best val = 0.7538 at Step 60, essentially equal to SFT baseline (0.752). No checkpoint from this experiment is worth saving.

---

## 3. Entropy & KL Divergence

### Entropy Explosion

| Epoch | Start | End | Mean | Max | Change |
|-------|-------|-----|------|-----|--------|
| E1 (steps 1-28) | 0.567 | 0.705 | 0.607 | 0.705 | +24% |
| E2 (steps 29-56) | 0.663 | 1.198 | 0.908 | 1.240 | +81% |
| E3 (steps 57-76) | 1.247 | 2.297 | 1.784 | 2.297 | +84% |
| **Total** | **0.567** | **2.297** | — | — | **+305%** |

Normal RL training entropy: 0.5-0.8. This model has reached 2.297 — it's essentially random.

### KL Divergence

| Epoch | Start | End | Mean | Max |
|-------|-------|-----|------|-----|
| E1 | 0.003 | 0.269 | 0.084 | 0.291 |
| E2 | 0.149 | 0.591 | 0.306 | 0.591 |
| E3 | 0.695 | 0.495 | 0.560 | 0.824 |

KL with kl_coef=0.02 is far too weak to prevent divergence. Even at KL=0.824, the KL penalty term is only 0.02 × 0.824 = 0.016, negligible compared to policy gradient.

---

## 4. Train Reward Metrics

| Metric | Epoch 1 | Epoch 2 | Epoch 3 (partial) |
|--------|---------|---------|-------------------|
| answer_mean | 0.382 | 0.387 | 0.402 |
| format_fail | 6.8% | 32.5% | 97.5% |

**answer_mean is flat across all 3 epochs** (~0.39). The model never learned to answer hard questions better. The small apparent "improvement" in E3 (0.402) is noise — with 97.5% format failure, the computation is dominated by the few remaining correctly-formatted outputs.

---

## 5. BoK-GRPO Diagnostics

| Metric | E1 | E2 | E3 | Trend |
|--------|----|----|----|----|
| tau | 0.700→0.641 | 0.637→0.530 | 0.525→0.389 | Monotonic decay ✓ |
| low_var/1024 | ~80-112 | ~64-128 | ~80-144 | Stable |
| collapsed/1024 | ~0-6 | ~0-14 | ~6-16 | Rising |
| easy_drgrpo/1024 | ~80-128 | ~64-192 | ~64-192 | Volatile |
| all_correct/1024 | ~16-32 | ~16-32 | ~16-32 | Stable-low |

**Collapsed groups** (all responses identical = zero gradient) are rising from ~0 to ~16, consistent with entropy explosion causing the model to converge on a single (wrong) pattern.

---

## 6. Cross-Epoch Comparison: Same Data, Different Epochs

Since total_epochs=4 with shuffle=True and 1808 samples, each epoch sees the same 1808 samples in different order. The key question was: does re-exposure help?

### Answer: **NO — Re-exposure actively hurts.**

| Metric | E1 → E2 same data | E2 → E3 same data |
|--------|-------------------|-------------------|
| answer_mean | 0.382 → 0.387 (+1.3%) | 0.387 → 0.402 (+3.9%*) |
| format_fail | 6.8% → 32.5% (+376%) | 32.5% → 97.5% (+200%) |
| entropy | 0.607 → 0.908 (+50%) | 0.908 → 1.784 (+96%) |
| KL | 0.084 → 0.306 (+264%) | 0.306 → 0.560 (+83%) |

*E3 answer_mean increase is meaningless — only 2.5% of outputs have valid format.

**Each epoch accelerates the degradation.** The model doesn't "learn from mistakes" on hard data — it unlearns correct formatting and gains entropy without improving accuracy.

---

## 7. Comparison with V7 Failure Mode

This is the SECOND time we've seen `point_2d` key corruption:

| Aspect | V7 (BOK+DAPO+FmtRej) | V14-hard-only (BOK) |
|--------|----------------------|---------------------|
| Onset step | ~51 | ~48 |
| Corruption | point_2d → point_274.2 | point_2d → point_22/20/27... |
| Root cause | DAPO filter zeroed gradients | Entropy explosion on hard data |
| kl_coef | 0.01 | 0.02 |
| FORMAT_REJECTION | ON (made it worse) | OFF |
| Recovery possible? | No | No |

**Pattern**: Low kl_coef + insufficient positive reward signal → entropy grows → BPE token confusion → format collapse. The specific corruption differs (V7 had decimal coordinates in key, V14 has digit replacement) but mechanism is identical.

---

## 8. Diagnosis & Root Causes

### Primary Causes (ordered by importance)

1. **Insufficient reward signal on hard data**: Only ~38% answer accuracy means 62% of rollouts provide no useful positive gradient. RL needs sufficient "success rate" to learn.

2. **kl_coef=0.02 is too weak**: KL penalty of 0.02 × KL cannot prevent divergence when KL reaches 0.5-0.8. Need kl_coef ≥ 0.05.

3. **Multi-epoch amplifies errors**: Each epoch re-exposes the diverged model to the same data where it mostly fails, generating negative advantages that further push the model away from its (already suboptimal) policy.

4. **BPE vulnerability of "point_2d"**: The token `"point_2"` is a shared prefix with common number patterns. When entropy rises, the model confuses `d` (dimension) with digits. This is a structural weakness of the JSON format choice.

5. **No format safeguard**: FORMAT_REJECTION=0 means format failures still contribute to the training loss. With 98% failure rate, the model trains mostly on broken-format data.

### Self-Reinforcing Death Spiral

```
                     ┌──────────────────────────────────────┐
                     │  Hard data: 62% rollouts fail       │
                     │  → Gradient is mostly noise          │
                     └──────────┬───────────────────────────┘
                                │
                     ┌──────────▼───────────────────────────┐
                     │  Entropy rises (0.567 → 2.297)       │
                     │  Policy becomes uncertain             │
                     └──────────┬───────────────────────────┘
                                │
                     ┌──────────▼───────────────────────────┐
                     │  BPE confusion: point_2d → point_22  │
                     │  Later turns corrupted first          │
                     └──────────┬───────────────────────────┘
                                │
                     ┌──────────▼───────────────────────────┐
                     │  Format validation fails: 98%+       │
                     │  format reward = 0 for most samples  │
                     └──────────┬───────────────────────────┘
                                │
                     ┌──────────▼───────────────────────────┐
                     │  Overall reward drops                 │
                     │  → Less positive signal               │
                     │  → More entropy growth                │
                     │  → Loop back to top ↑                 │
                     └──────────────────────────────────────┘
```

---

## 9. Recommendations

### Immediate
1. **STOP V14-hard-only training** — Epochs 3-4 will continue degrading
2. **No checkpoint is worth saving** — Val never exceeded SFT baseline

### For Next Experiment (V14-mixed)
1. **Use mixed data (5792 samples)** — Easy data provides the positive reward signal that hard-only lacks
2. **total_epochs=2** (not 4) — reduce re-exposure risk
3. **kl_coef=0.05** (not 0.02) — stronger KL regularization to prevent divergence
4. **Consider format protection**:
   - Option A: Early stopping on format_fail > 20%
   - Option B: TRAJ_FORMAT_REJECTION=1 with DrGRPO fallback (not DAPO)
   - Option C: Replace `"point_2d"` with a BPE-stable key like `"xy"` or `"pos"`
5. **Monitor entropy** — if entropy exceeds 0.8, stop and diagnose

### Structural Improvements
1. **Rename JSON key**: Change `"point_2d"` to something BPE-safe (e.g., `"pos"`, `"xy"`, `"coord"`). This prevents the specific token confusion that caused collapse in both V7 and V14.
2. **Entropy penalty**: Add explicit entropy regularization term to prevent unbounded growth.
3. **Adaptive KL**: Implement target KL (e.g., target=0.05) with automatic kl_coef adjustment.
4. **Format monitoring early stop**: Automatically stop training if format_fail_rate exceeds threshold for 5 consecutive steps.

---

*Report generated from full log analysis of 77/112 training steps.*  
*Experiment outcome: CATASTROPHIC FAILURE — format collapse at Step 48-50.*
