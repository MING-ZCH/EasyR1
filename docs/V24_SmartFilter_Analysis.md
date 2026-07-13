# V24 Smart Filter Analysis: All-Correct Filtering Deep Dive

**Date**: 2026-04-12  
**Context**: V23 complete (213/213 steps, 0 NaN, peak 0.7769, final 0.7637)  
**Question**: Should we modify `FILTER_ALL_CORRECT` to check `answer=1 AND point>=threshold`, or simply set FILTER=0?

---

## 1. Current Behavior (V23)

### Code Path (core_algos.py L583-591)
```python
if _filter_all_correct:
    pass_rate_full = (group_scores > _easy_score_threshold).float().mean().item()
    if pass_rate_full >= 1.0 - 1e-6:  # all K correct
        n_all_correct_filtered += K
        continue  # advantages stay 0 → zero gradient
```

**What "all correct" actually means**: `score > 0.5` where `score = 0.6×answer + 0.3×point + 0.1×format`.  
Since `answer_weight = 0.6 > 0.5`, any trajectory with `answer=1` automatically has `score ≥ 0.6 > 0.5`.  
→ The filter effectively checks: **all 16 trajectories answered correctly**.

### V23 Routing Evolution (Empirical Data)
| Phase | AC Filtered% | Useful (Easy+BoK)% | Trend |
|-------|-------------|---------------------|-------|
| Steps 1-50 | 21.9% | 64.8% | — |
| Steps 51-100 | 22.5% | 61.4% | ↓ |
| Steps 101-150 | 24.3% | 59.6% | ↓ |
| Steps 151-153 | 27.1% | 54.2% | ↓↓ |

**Critical**: AC filter rate is GROWING (21.9% → 27.1%), useful gradient budget SHRINKING.  
As the model improves, more prompts become "all correct" → more gradient wasted → self-limiting ceiling.

---

## 2. Three Options Analyzed

### Option A: Simple FILTER=0 + SCALE=0.3
- All previously-filtered groups fall through to `easy_drgrpo` path
- Advantage: `(score - group_mean) / (group_std + eps) × 0.3`, clipped [-2.5, 2.5]
- **Pro**: Simple, recovers 23% lost gradient, no new hyperparameters
- **Con**: Z-score normalization amplifies low-variance groups (see Section 3)

### Option B: Smart Threshold (answer=1 AND point>=T)
- Keep filter but refine: only filter groups where ALL trajectories have answer=1.0 AND point ≥ 0.85
- Groups with answer=1 but some point < 0.85 → not filtered → get easy_drgrpo gradient
- **Pro**: Theoretically cleaner gradient (removes only truly mastered prompts)
- **Con**: Requires passing `point_scores` to core_algos (new data pipeline), new hyperparam `T`

### Option C: Continuous Scaling (no binary filter)
- Replace binary filter with continuous dampening: `scale = max(0.1, 1.0 - pass_rate)`
- **Pro**: Smoothest gradient allocation
- **Con**: Most complex, interacts with existing `_easy_scale`

---

## 3. Deep Dive: Z-Score Amplification Problem

When FILTER=0, the all-correct groups enter `easy_drgrpo`:

```python
adv = (scores[global_i] - group_mean) / (group_std + eps)
adv = max(-2.5, min(2.5, adv))
advantages_1d[global_i] = adv * _easy_scale
```

**The z-score normalization divides by group_std**. For low-std all-correct groups:

| Group Type | group_std | Score Range | Z-scores | Adv (×0.3) |
|-----------|----------|-------------|----------|------------|
| Good-point AC (std=0.03) | 0.03 | [0.87, 0.93] | [-2.0, +2.0] | [-0.6, +0.6] |
| Mixed-point AC (std=0.15) | 0.15 | [0.70, 0.95] | [-1.0, +1.0] | [-0.3, +0.3] |
| Typical easy (std=0.20) | 0.20 | [0.40, 0.95] | [-1.5, +1.5] | [-0.45, +0.45] |

**Key insight**: Low-std groups get **LARGER** advantages than high-std groups after z-normalization!  
The good-point AC groups (std≈0.03) get amplified to ±0.6, while typical easy groups only get ±0.45.

**Is this noise or signal?**
- Good-point AC groups: the 0.87→0.93 difference is small and possibly generation noise
- Mixed-point AC groups: the 0.70→0.95 difference is clearly a meaningful strategy difference
- Smart threshold would filter the former but keep the latter

### Noise Impact Assessment

Estimated breakdown of V23's 23% all-correct groups:
- ~9% of total: very low variance (std < 0.05), good pointing → **noisy gradient**
- ~7% of total: moderate variance (0.05–0.15), mixed pointing → **useful gradient**  
- ~7% of total: high variance (> 0.15), some poor pointing → **useful gradient**

With FILTER=0 + SCALE=0.3:
- Noise contribution: 9% × 0.6 avg|adv| = 0.054 effective gradient
- Signal contribution: 14% × 0.45 avg|adv| = 0.063 effective gradient
- Noise:Signal ≈ 0.86:1 (nearly equal in magnitude!)

BUT: Over N training steps, SGD noise averages out as O(1/√N) while signal accumulates as O(N).  
After just ~10 steps: signal/noise = 10/√10 = 3.16:1. After 50 steps: 50/√50 = 7.07:1.

---

## 4. Why Smart Threshold is Theoretically Sound but Practically Unnecessary

### Argument FOR Smart Threshold:
- Removes ~9% noisy gradient from low-var, all-correct, good-pointing groups
- Keeps ~14% useful gradient from mixed-pointing groups
- Gradient quality improved by ~5-9%

### Arguments AGAINST Smart Threshold:
1. **SGD averaging**: Random noise cancels over steps; signal accumulates → noise becomes negligible
2. **Entropy benefit**: V23 entropy is stuck at 0.520. The "noisy" gradient from low-var groups acts as exploration noise, potentially beneficial for breaking the entropy stasis
3. **Implementation complexity**: Requires passing `point_scores` through the data pipeline (ray_trainer.py → core_algos.py), adds new env var `BOK_AC_POINT_THRESHOLD`
4. **Hyperparameter sensitivity**: T=0.85 is not principled; different datasets would need different thresholds
5. **Diminishing returns**: With EASY_SCALE=0.3, the maximum noise magnitude is ±0.75 (clipped ±2.5 × 0.3). This is already small relative to BoK path advantages (up to ±4.0).

### The Mathematical Auto-Filter

The easy_drgrpo z-normalization ALREADY provides a soft version of what the smart filter does:
- Low-var groups: z-scores hit clip boundaries → all get similar ±2.5 → gradient direction is RANDOM → averages out
- High-var groups: z-scores spread naturally → gradient direction is MEANINGFUL → accumulates

This is "automatic soft filtering" — the math already handles the quality distinction, just not perfectly but well enough.

---

## 5. EASY_SCALE Derivation

### Goal: Easy and BoK paths contribute equally to expected gradient magnitude

**Formula**: `EASY_SCALE = (BoK_share × BoK_mean_|adv|) / (Easy_share × Easy_mean_|zscore|)`

With FILTER=0, the routing changes:
- Previously AC (23%) flows into Easy → New Easy = 42.6% + 23.0% = **65.6%**
- BoK stays at ~22.1% (unchanged)
- LowVar stays at ~12.3% (unchanged)

**Parameters** (from V23 logs):
- BoK_share = 22.1%, BoK_mean_|adv| ≈ 0.83 (from softmax weighting)
- Easy_share = 65.6%, Easy_mean_|zscore| ≈ 0.895 (from z-score distribution)

**Calculation**:
```
EASY_SCALE = (22.1 × 0.83) / (65.6 × 0.895)
           = 18.34 / 58.71
           = 0.3125
```

**Sensitivity analysis** (varying assumptions ±20%):
| BoK_|adv| | Easy_|zscore| | EASY_SCALE |
|-----------|---------------|------------|
| 0.66 (low) | 1.07 (high) | 0.208 |
| 0.83 (mid) | 0.895 (mid) | **0.312** |
| 1.00 (high) | 0.72 (low) | 0.467 |

**Robust range**: 0.25 – 0.40. **Recommended: 0.3** (round number, conservative).

---

## 6. Definitive Recommendation

### V24 Configuration
```bash
BOK_FILTER_ALL_CORRECT=0    # Remove binary filter completely
BOK_EASY_SCALE=0.3           # Dampen inflated Easy path to match BoK gradient magnitude
```

### Why NOT Smart Threshold
1. Adds complexity for <5% theoretical improvement
2. SGD averaging eliminates the noise concern within ~10 steps
3. The exploration noise may actually HELP break the 0.78 ceiling
4. The z-score math already provides "automatic soft filtering"
5. Clean experiment design: change ONE thing (filter→off), compensate math (scale→0.3)

### Plan B (only if V24 shows instability)
If V24's validation curve shows more oscillation than V23 (suggesting gradient noise IS harmful), then for V25:
- Smart filter: `filter if all(answer ≥ 1.0) AND all(point ≥ 0.85)`
- Requires: ~15 lines code change in core_algos.py + ray_trainer.py
- Add `point_scores` extraction in ray_trainer.py L912-914
- Add `point_scores` parameter to `compute_bok_grpo_advantage()`

---

## 7. Expected Impact

| Metric | V23 (current) | V24 (predicted) | Reasoning |
|--------|--------------|-----------------|-----------|
| AC filtered | 23% → 27% growing | 0% | Filter removed |
| Useful gradient | 65% → 54% shrinking | 87%+ stable | All groups contribute |
| Easy path magnitude | ×1.0 (overweight) | ×0.3 (balanced) | Scale dampening |
| Gradient budget balance | Easy:BoK = 3.5:1 | Easy:BoK ≈ 1:1 | Correct allocation |
| Val ceiling | 0.78 (stuck) | TBD | More gradient + balance → ceiling break? |

The core hypothesis: the 0.78 ceiling is caused by (a) wasting 23% gradient on all-correct groups and (b) Easy path overweighting at 3.5:1. Fixing both simultaneously in V24 should either break the ceiling or prove the bottleneck is elsewhere (e.g., HM=0 visual attention issue).
