# BoK-GRPO v3: Algorithm Improvements for pass@32 → pass@1 Conversion

## Problem Analysis

Current BoK-GRPO routing distribution (V19, 56 steps average):

| Route | % of Batch | Gradient Contribution | pass@1 Impact |
|-------|-----------|----------------------|---------------|
| all_correct_filtered | 18.9% | Zero (wasted) | None |
| **easy_drgrpo** (>8/16) | **46.8%** | **Dominant** | Low: maintaining |
| **BoK_normal** (1-8/16) | **21.3%** | Medium | **Highest: critical zone** |
| allwrong_capped (0/16) | 2.9% | Weak (cap=1.0) | Low |
| low_var fallback | 10.1% | Dr.GRPO weak | Very low |

**Key insight**: Only 21.3% of the batch is in the "critical learning zone" where pass@32→pass@1 improvement happens. The gradient is dominated by easy_drgrpo (46.8%) which maintains performance but doesn't improve it.

**Bottleneck**: In BoK hard groups with few correct trajectories (e.g., 2/16), the positive gradient signal from rare correct trajectories is diluted by the dominant easy group gradient.

## Three V3 Improvements

### 1. Winner Amplification (`BOK_WINNER_BOOST`)

**Env var**: `BOK_WINNER_BOOST` (default: 0 = disabled)

**V20 setting**: `BOK_WINNER_BOOST=3.0`

Boosts correct-trajectory advantages in BoK hard groups proportional to their rarity:

```
boost = min(boost_max, sqrt(K / n_correct))
advantage_correct *= boost
```

| n_correct/16 | pass_rate | boost (max=3.0) |
|-------------|-----------|-----------------|
| 1/16 | 6.3% | 3.0 (capped from 4.0) |
| 2/16 | 12.5% | 2.83 |
| 4/16 | 25.0% | 2.0 |
| 8/16 | 50.0% | 1.41 |

**Expected effect**: 2-3x stronger gradient from rare correct trajectories, directly targeting the pass@32→pass@1 gap.

### 2. Easy Gradient Dampening (`BOK_EASY_SCALE`)

**Env var**: `BOK_EASY_SCALE` (default: 1.0 = no change)

**V20 setting**: `BOK_EASY_SCALE=0.5`

Scales easy_drgrpo advantages by a dampening factor:
```
advantage_easy *= easy_scale  # e.g., 0.5 = halve easy gradient
```

**Expected effect**: Shifts ~23% of gradient budget from easy → BoK critical zone. The BoK signal becomes relatively 2x stronger without increasing its absolute magnitude.

### 3. Quality-Ranked BoK (`BOK_QUALITY_BONUS`)

**Env var**: `BOK_QUALITY_BONUS` (default: 0 = disabled)

**V20 setting**: `BOK_QUALITY_BONUS=0.5`

Among correct trajectories in BoK groups, adds bonus proportional to relative point quality:
```
quality = overall_score - 0.6  # ≈ 0.3*point + 0.1*format component
relative_quality = (quality - min_quality) / (max_quality - min_quality)
advantage += quality_bonus * relative_quality
```

**Expected effect**: Model learns to follow the highest-quality correct path (best point predictions → less error accumulation), not just any correct path.

## Implementation Details

### Files Modified

1. **`verl/trainer/core_algos.py`**:
   - L515-521: New env var declarations
   - L598-602: Easy Gradient Dampening in easy_drgrpo path
   - L648-671: Winner Amplification + Quality Bonus after BoK softmax
   - L540: New counter `n_winner_boosted`
   - L751, L774: Updated diagnostic log output

2. **`examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v20.sh`**:
   - New env vars: `BOK_WINNER_BOOST=3.0`, `BOK_EASY_SCALE=0.5`, `BOK_QUALITY_BONUS=0.5`
   - RunConfig echo line for V3 params

3. **`tools/analyze_training.py`**:
   - Fixed pre-existing BoK-GRPO regex bug (adv_mean/adv_std used wrong group numbers)
   - Changed to named regex groups for robustness
   - Added `winner_boosted` tracking (backward compatible: V19=0)

### Backward Compatibility

- All three features default to OFF (0 or 1.0) and are controlled by environment variables
- No impact on non-BoK-GRPO training modes
- Analysis tool handles both old (no winner_boosted) and new log formats

### Bug Fix

Fixed pre-existing bug in `tools/analyze_training.py`: the BoK-GRPO regex parsed `allwrong_capped_bsz` as `adv_mean` due to incorrect group numbering. Now uses named regex groups (`(?P<name>...)`) which is immune to group ordering changes.

## Monitoring

The `[BoK-GRPO]` diagnostic line now includes `winner_boosted=N/1024`:
```
[BoK-GRPO] batch=1024 tau=0.450 step=15/213  low_var=96/1024  collapsed=2  ...  allwrong_capped=32/1024  winner_boosted=128/1024  adv_mean=0.0150 adv_std=1.0234
```

The `[BoK-GRPO-Detail]` line now includes V3 params:
```
... winner_boost=3.0 easy_scale=0.50 quality_bonus=0.5
```

## Recommended V20 Settings

```bash
export BOK_WINNER_BOOST=3.0          # Strong winner amplification
export BOK_EASY_SCALE=0.5            # Halve easy gradient
export BOK_QUALITY_BONUS=0.5         # Moderate quality differentiation
```

## What to Watch During V20 Training

1. **winner_boosted count**: Should be 15-25% of batch (BoK groups with some correct)
2. **adv_std**: Should increase slightly (from ~0.88 to ~1.0-1.2 due to winner boost)
3. **val/answer_reward**: Primary metric — should trend upward from 0.762 baseline
4. **grad_norm**: Should stay stable with bf16 (no FP16 subnormal issues)
5. **Emergency brake**: Should NOT activate if bf16+EMA fix works

## V3 Critical Bug Fix: Asymmetric Clip

### Problem Discovered
The global `advantages_1d.clamp(-bok_clip, bok_clip)` at line ~730 executed AFTER all V3 modifications,
completely neutralizing Winner Boost for the most important rare-correct groups (n=1-3/16):

| n_correct | Pre-clip adv | After V3 boost | After global clip (BUG) |
|-----------|-------------|----------------|------------------------|
| 1/16      | 4.0 (hit ceiling) | 12.0          | **4.0 (boost erased!)** |
| 2/16      | 4.0 (hit ceiling) | 11.3          | **4.0 (boost erased!)** |
| 5/16      | 1.9         | 3.4            | 3.4 (boost preserved) |

### Fix Applied
Asymmetric clip: `pos_clip = bok_clip * sqrt(winner_boost)` = 4.0 × √3.0 = **6.93**

```python
if bok_clip > 0:
    if _winner_boost > 1.0:
        pos_clip = bok_clip * math.sqrt(_winner_boost)
    else:
        pos_clip = bok_clip
    advantages_1d = advantages_1d.clamp(-bok_clip, pos_clip)
```

### Simulation Results (V2 → V3_bug → V3_fix)

| Metric | V2 (current) | V3_bug (sym clip) | V3_fix (asym clip) |
|--------|-------------|-------------------|-------------------|
| BoK |adv| mean | 1.170 | 1.421 | **1.614** |
| Easy |adv| mean | 0.782 | 0.391 | **0.391** |
| BoK/Easy ratio | 1.50x | 3.64x | **4.13x** |
| adv_max | 4.00 | 4.00 | **6.93** |
| Gradient budget: BoK | 44.4% | 64.1% | **67.0%** |
| Gradient budget: easy | 50.7% | 30.1% | **27.7%** |

### Winner Boost Effective Amplification by n_correct

| n/16 | Theoretical boost | V3_fix/V2 effective | 
|------|------------------|-------------------|
| 1 | 3.00x | **1.73x** |
| 2 | 2.83x | **1.73x** |
| 3 | 2.31x | **1.87x** |
| 4 | 2.00x | **2.09x** |
| 5 | 1.79x | **1.92x** |

### Safety Analysis
- Positive clip raised from 4.0 → 6.93 (+73%), negative clip unchanged at -4.0
- Only BoK groups benefit (easy caps at ±1.25, low_var ≈ ±0.5)
- PPO ratio clipping (1+clip_high=1.28) bounds effective gradient per token
- ~5% of batch tokens affected by higher ceiling → ~15% max total gradient increase
- GradSpikeProtect v2 emergency brake provides safety net
