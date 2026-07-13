# V24 Smart Filter Analysis: DAPO/Dr.GRPO Literature Review + Implementation Design

**Date**: 2026-04-12  
**Context**: V23 complete, all-correct filter wastes 23→27% gradient budget  
**Question**: Can we reference DAPO/Dr.GRPO's proven filter mechanisms to optimize BOK_FILTER_ALL_CORRECT?

---

## 1. Literature Review Summary

### 1.1 DAPO (arXiv:2503.14476) — Dynamic Sampling

**Mechanism**: Data-level filter. Over-sample prompts, discard groups where accuracy = 0% or 100%.

**Filter condition**: `0 < |{o_i | R_i=1}| < G` (at least one correct AND one wrong in group)

**Key point**: DAPO uses **binary rewards {−1, +1}**. When all correct or all wrong, reward std = 0, advantage = 0, gradient is TRULY zero. The filter removes these to REPLACE them via over-sampling → every batch has effective gradients.

**Evidence**: +8 points on AIME24 (42→50), the **single largest ablation improvement** in the paper.

### 1.2 Dr.GRPO (arXiv:2503.20783) — Difficulty Debiasing

**Mechanism**: Remove std normalization from GRPO, NOT filter.

**Two fixes**:
1. Remove response-length normalization (use `masked_sum / MAX_TOKENS` instead of `masked_mean`)
2. Remove difficulty bias: `advantage = R_i - mean(R)` instead of `(R_i - mean(R)) / std(R)`

**Key insight**: Dividing by std **amplifies** easy/hard groups (low std → large advantage). Removing std division naturally gives low-variance groups small advantages and high-variance groups large advantages — true difficulty-awareness WITHOUT filtering.

**Evidence**: Correct reasoning accuracy maintained while eliminating length inflation. 3-run statistical significance verified.

### 1.3 GRPO-LEAD — Difficulty-Aware Reweighting

**Mechanism**: Logistic function reweighting based on group pass rate.

`w(ρ) = A + (B-A) / (1 + exp[k(ρ-ρ₀)])`, with A=0.4, B=1.5, ρ₀=0.75, k=10.

**Evidence**: +2 pts on AIME24, +1.7 pts on AIME25.

---

## 2. Critical Gap: Our Case vs Literature

| Dimension | DAPO / Dr.GRPO | Our StepCount |
|-----------|---------------|---------------|
| **Reward type** | Binary {-1,1} or {0,1} | Continuous (0.6×answer + 0.3×point + 0.1×format) |
| **All-correct = zero gradient?** | YES (std=0) | **NO** (point variance exists) |
| **Filter replaces wasted samples?** | YES (over-sampling) | NO (trajectory generation too expensive) |
| **Task type** | Single-turn math reasoning | Multi-step trajectory counting |
| **Reward dimensions** | 1D | 3D (answer + point + format) |

**Core difference**: DAPO's filter removes truly zero-gradient groups. Our "all-correct" filter removes groups that **DO have valid gradient** (from point/format variance). This is fundamentally wasteful.

---

## 3. K=16 Exponential Strictness Problem

The user proposed: filter if `all(answer=1) AND all(point >= T)`.

With K=16, the "all K must satisfy" condition is exponentially strict:

| Condition | T | P(filter per group) | % of total filtered | Gradient recovered |
|-----------|---|--------------------|--------------------|-------------------|
| all(point≥T) | 0.80 | 0.0021 | 0.05% | 22.95% |
| all(point≥T) | 0.70 | 0.111 | 2.55% | 20.45% |
| all(point≥T) | 0.60 | 0.557 | 12.81% | 10.19% |
| **mean(point)≥T** | 0.85 | 0.703 | **16.17%** | **6.83%** |
| **mean(point)≥T** | 0.87 | 0.500 | **11.50%** | **11.50%** |

**Finding**: `all(point >= T)` with T≥0.75 barely filters anything (< 0.5%) — effectively becomes FILTER=0.

**Solution**: Use `mean(point) >= T` instead of `all(point >= T)` to avoid K-exponential strictness.

---

## 4. The Deeper Root Cause: Z-Score Amplification

Our `easy_drgrpo` path computes:
```python
adv = (score - group_mean) / (group_std + eps)  # z-score normalization
adv = clip(adv, -2.5, 2.5)
advantages[i] = adv * easy_scale
```

This is EXACTLY the "difficulty bias" Dr.GRPO identifies:
- Low-std all-correct groups (std=0.03): z-scores amplified to ±2.0 → adv = ±0.6 (with scale=0.3)
- High-std mixed groups (std=0.20): z-scores moderate ±1.0 → adv = ±0.3

**The low-var groups get LARGER advantages than high-var groups!** This is counterintuitive and wastes gradient on distinguishing nearly-identical trajectories.

---

## 5. Three Ranked Approaches

### Approach 1: Dr.GRPO-Inspired Easy Path (RECOMMENDED)

**Change**: Replace z-score normalization with raw centering in easy_drgrpo:

```python
# BEFORE (z-score):
adv = (score - group_mean) / (group_std + eps)
adv = clip(adv, -2.5, 2.5)
advantages[i] = adv * easy_scale  # scale=0.3

# AFTER (Dr.GRPO centering):
adv = score - group_mean  # NO std division!
advantages[i] = adv * easy_scale  # scale=2.0
```

**Advantage magnitudes with Dr.GRPO mode**:
| Group type | group_std | mean |raw_adv| | ×scale=2.0 | Signal quality |
|-----------|----------|----------------|----------|---------------|
| AC low-var | 0.03 | 0.024 | 0.048 | Noise (auto-filtered!) |
| AC med-var | 0.15 | 0.120 | 0.240 | Useful |
| Hard groups | 0.25 | 0.200 | 0.400 | Very useful |

Low-var all-correct groups get **automatically tiny** (0.048) advantages — no explicit filter needed!

**EASY_SCALE derivation**:
```
BoK aggregate: 22.1% × 1.07 (tau≈0.5) = 23.65
Easy aggregate: 65.6% × (0.175 × sqrt(2/π)) × SCALE = 65.6 × 0.14 × SCALE
Balance: SCALE = 23.65 / 9.18 = 2.58
Conservative (BoK slightly dominant): SCALE ≈ 2.0
```

**Pros**: 
- Addresses ROOT CAUSE (std amplification = Dr.GRPO's "difficulty bias")
- Proven effective in Dr.GRPO paper
- No new hyperparameters, no data pipeline changes
- Auto-filters low-var groups by math, not heuristic

**Cons**:
- Changes fundamental advantage formula
- SCALE needs careful tuning (2.0 is derived but not tested)

### Approach 2: Smart Filter with Group Mean (ALTERNATIVE)

**Change**: Modify all-correct filter condition:

```python
# BEFORE:
if pass_rate_full >= 1.0 - 1e-6:
    continue  # filter all answer-correct groups

# AFTER:
if pass_rate_full >= 1.0 - 1e-6:
    if point_scores is not None:
        group_points = torch.stack([point_scores[j] for j in indices])
        if group_points.mean() >= _point_threshold:  # 0.85
            continue  # filter only truly mastered groups
    else:
        continue  # fallback: original behavior
```

**Requires**: Passing `point_scores` from ray_trainer.py to core_algos.py (~15 lines of code)

**Effect**: Recovers ~7% gradient (from 23% → 16% filtered, with T=0.85)

**Pros**: DAPO validates that filtering "all-correct" groups helps; we just make the condition smarter
**Cons**: New hyperparameter (T), needs data pipeline change, doesn't address z-score amplification

### Approach 3: Simple FILTER=0 + SCALE=0.3 (SAFEST)

Same as previous analysis. Simplest but doesn't fix z-score amplification.

---

## 6. Recommended V24 Configuration

### Option A: Dr.GRPO-Mode (If comfortable changing advantage formula)
```bash
BOK_FILTER_ALL_CORRECT=0        # Remove binary filter
BOK_EASY_ADV_MODE=drgrpo        # NEW: use (score - mean) instead of z-score
BOK_EASY_SCALE=2.0              # Re-derived for raw centering
```
Expected: Low-var groups auto-damped to ~0.05, BoK:Easy ratio ~1.3:1

### Option B: Z-Score + Smart Filter (If prefer minimal formula change)
```bash
BOK_FILTER_ALL_CORRECT=2        # NEW: smart filter mode (0=off, 1=old, 2=smart)
BOK_AC_POINT_THRESHOLD=0.85     # NEW: group mean point threshold
BOK_EASY_SCALE=0.3              # Same z-score mode derivation
```
Expected: 16% filtered (vs 23%), 7% gradient recovered, BoK:Easy ratio ~1:1

### Option C: Simple (Maximum safety)
```bash
BOK_FILTER_ALL_CORRECT=0
BOK_EASY_SCALE=0.3
```

---

## 7. Implementation Cost

| Component | Option A (Dr.GRPO) | Option B (Smart Filter) | Option C (Simple) |
|-----------|-------------------|------------------------|------------------|
| core_algos.py | ~8 lines (add mode) | ~15 lines (add condition + point_scores param) | 0 lines |
| ray_trainer.py | 0 lines | ~5 lines (extract point_scores) | 0 lines |
| New env vars | 1 (EASY_ADV_MODE) | 2 (FILTER mode + threshold) | 0 |
| New hyperparams | 0 | 1 (point threshold) | 0 |
| Risk level | Medium | Medium | Low |

---

## 8. Final Recommendation

**Option A (Dr.GRPO-mode)** is the most principled choice because:
1. It addresses the ROOT CAUSE (z-score amplification) rather than the SYMPTOM (all-correct groups getting too much gradient)
2. Dr.GRPO paper proves removing std normalization is effective for GRPO-family algorithms
3. Low-var all-correct groups are automatically soft-filtered (adv ≈ 0.05 vs normal 0.24-0.40)
4. No new hyperparameters to tune
5. Clean theoretical justification from first principles

Options B and C are valid fallbacks if the Dr.GRPO-mode shows unexpected behavior (e.g., advantage scale mismatch).

---

## APPENDIX A: Deep Debate Round 2 (User Follow-Up)

### Q1 Clarification: Easy Path is Z-Score, NOT Dr.GRPO

**Code verification** (core_algos.py L599-604):
```python
# Easy DrGRPO path (CURRENT): Z-SCORE with std division!
adv = (scores[global_i] - group_mean) / (group_std + eps)
adv = max(-2.5, min(2.5, adv))
advantages_1d[global_i] = adv * _easy_scale
```

```python
# Low-var fallback (SEPARATE): TRUE Dr.GRPO without std!
advantages_1d[global_i] = scores[global_i] - batch_mean
```

The "Easy DrGRPO" is named misleadingly — it uses **z-score normalization** (÷group_std), NOT Dr.GRPO raw centering. True Dr.GRPO (score - mean, no std division) is only used in the low_var fallback path.

**Why this matters**: Z-score amplifies low-var groups (group_std=0.03 → z-scores hit clip boundaries at ±2.5). Dr.GRPO paper explicitly identifies this as "difficulty bias."

### All-Wrong Filter: Current is CAP, not FILTER

- **V23 behavior**: all-wrong groups get advantages **capped to ±1.0** (not zeroed)
- **Condition**: `all(answer_scores < 0.5)` → since answer∈{0,1}, equivalent to all(answer=0)
- **Verdict**: Keep CAP (don't change to filter). In multi-dimensional reward (answer+point+format), even all-wrong groups contain useful point quality signal. The 3% allwrong share is too small to matter either way.

### Mathematical Equivalence Discovery

Current filter: `all(overall_score > 0.5)` where `overall = 0.6×answer + 0.3×point + 0.1×format`

When answer=1: `overall ≥ 0.6 > 0.5` → always passes  
When answer=0: `overall ≤ 0.3+0.1 = 0.4 < 0.5` → never passes

**Therefore**: `all(overall > 0.5)` ≡ `all(answer = 1)` ≡ `all(answer=1) AND all(overall>0)`

The **only real change** in the smart filter is adding `mean(point) ≥ 0.85`.

### V24 Plan: Single-Variable Experiment

| What Changes | V24 |
|---|---|
| All-correct condition | `all(answer=1) AND mean(point)≥0.85` → filter |
| All-wrong | Keep cap ±1.0 (unchanged) |
| Easy DrGRPO formula | Keep z-score (unchanged) |
| EASY_SCALE | Keep 1.0 (unchanged) |
| Everything else | Identical to V23 |

**Rationale**: Only one variable changed → clean ablation.

### True Dr.GRPO for Easy Path: Deferred to V25

Theoretical advantage: eliminates difficulty bias, proven in paper.  
But: affects 44%+ of samples (vs filter affecting only 7%). Too large a change to combine with filter modification. If V24 breaks ceiling → filter was the issue, no need to change Easy path. If not → V25 adds Dr.GRPO mode.
