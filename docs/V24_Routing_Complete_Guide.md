# BoK-GRPO Complete Routing Guide — All Thresholds & Sample Journey

## 1. All Thresholds Overview

### Env Var → Internal Variable → Purpose

| Env Var | Default | Internal Variable | What it compares | Level | Purpose |
|---------|---------|-------------------|-----------------|-------|---------|
| `BOK_LOW_VAR_THRESHOLD` | 1e-5 | `low_var_threshold` | group_std (of overall scores) | Group | Detects groups where all 16 trajectories have identical scores |
| `BOK_EASY_SCORE_THRESHOLD` | 0.5 | `_easy_score_threshold` | individual overall score | Trajectory | Binary: is this trajectory "correct"? (overall > 0.5 ⟺ answer=1) |
| `BOK_FILTER_ALL_CORRECT` | 1 | `_filter_all_correct` | boolean on/off | Switch | Enable/disable all-correct zero-gradient filter |
| `BOK_SMART_FILTER_THRESHOLD` | 0 (off) | `_smart_filter_threshold` | group_mean (of overall scores) | Group | V24: min quality for truly-mastered (0.955 ≈ mean_point ≥ 0.85) |
| `BOK_EASY_THRESHOLD` | 0.75 | `_easy_threshold` | pass_rate (fraction of correct trajectories) | Group | Easy vs BoK routing split (V23: 0.50) |
| `BOK_TAU` / `BOK_TAU_INIT/FINAL` | 0.5 | `bok_tau` | softmax temperature | Algorithm | BoK sharpness (lower = more selective) |
| `BOK_ALLWRONG_ANSWER_THRESHOLD` | 0.5 | `_allwrong_answer_threshold` | individual answer score | Trajectory | Detects: is this trajectory's answer correct? (answer ∈ {0,1}) |
| `BOK_ALLWRONG_CAP` | 0 (off) | `_allwrong_cap` | advantage magnitude | Algorithm | V23: 1.0. Caps advantages for all-wrong groups |
| `BOK_EASY_SCALE` | 1.0 | `_easy_scale` | advantage magnitude | Algorithm | Dampens Easy path gradient (0.4 = 40% strength) |

### Threshold Relationships Diagram

```
Individual Trajectory Level                        Group Level (K=16 trajectories)
─────────────────────────                          ──────────────────────────────

 overall_score                                     group_std (std of 16 overall_scores)
 ┌──────────────────────┐                          ┌──────────────────────────────┐
 │ = 0.6*answer         │                          │ ≤ 1e-5  → LOW_VAR path      │
 │ + 0.3*point          │                          │ > 1e-5  → normal routing     │
 │ + 0.1*format         │                          └──────────────────────────────┘
 │                      │
 │ answer=0 → [0, 0.4]  │                         pass_rate = count(overall > 0.5) / 16
 │ answer=1 → [0.6, 1.0]│                          ┌──────────────────────────────┐
 │        ↑ 0.5 gap ↑   │                          │ = 1.0    → ALL_CORRECT check │
 └──────────────────────┘                          │ > 0.5    → EASY path         │
                                                   │ ≤ 0.5    → BOK path          │
 answer_score (0 or 1)                             └──────────────────────────────┘
 ┌──────────────────────┐
 │ < 0.5 → "wrong"      │                         group_mean (mean of 16 overall_scores)
 │ ≥ 0.5 → "correct"    │                          ┌──────────────────────────────┐
 └──────────────────────┘                          │ ≥ 0.955 → truly mastered     │
     ↑ used by:                                    │ < 0.955 → point still weak   │
     ALLWRONG_ANSWER_THRESHOLD                     └──────────────────────────────┘
     WINNER_BOOST detection                             ↑ used by: SMART_FILTER
```

## 2. Complete Routing Flowchart

```
For each prompt → K=16 trajectories → compute group_scores, group_mean, group_std
│
├─ [GATE 1] group_std ≤ BOK_LOW_VAR_THRESHOLD (1e-5)?
│  │  Q: "Are all 16 trajectories nearly identical?"
│  │
│  ├─ YES → LOW_VAR path
│  │  │  All 16 scores differ by < 0.00001
│  │  │  Could be: all-correct-same-quality OR all-wrong-same-quality OR mixed-same-score
│  │  │
│  │  ├─ DAPO_FILTER=1? → zero gradient (skip group)
│  │  └─ DAPO_FILTER=0? → Dr.GRPO fallback: advantage = score - batch_mean
│  │
│  └─ NO → Continue to GATE 2
│
├─ [GATE 2] FILTER_ALL_CORRECT=1 AND all(overall > 0.5)?
│  │  Q: "Are all 16 trajectories' answers correct?"
│  │  (overall > 0.5 ⟺ answer = 1, proven mathematically)
│  │
│  ├─ YES (all answer=1) →
│  │  │
│  │  ├─ [GATE 2a] SMART_FILTER: group_mean < threshold (0.955)?
│  │  │  Q: "Are these correct answers backed by good point quality?"
│  │  │  │
│  │  │  ├─ YES (point weak, mean<0.955) → RELEASED to Easy path
│  │  │  │  (n_all_correct_released += K, falls through to GATE 3)
│  │  │  │
│  │  │  └─ NO (point good, mean≥0.955) → FILTERED: zero gradient
│  │  │     (n_all_correct_filtered += K, continue)
│  │  │
│  │  └─ [SMART_FILTER disabled (threshold=0)] → FILTERED: zero gradient
│  │
│  └─ NO (not all correct) → Continue to GATE 3
│
├─ [GATE 3] pass_rate > BOK_EASY_THRESHOLD (0.5)?
│  │  Q: "Do more than 50% of trajectories have correct answers?"
│  │  pass_rate = count(overall > 0.5) / K
│  │
│  ├─ YES (>8/16 correct) → EASY Z-SCORE path
│  │  advantage = (score_i - group_mean) / (group_std + eps)
│  │  clip to [-2.5, 2.5], then × EASY_SCALE
│  │
│  └─ NO (≤8/16 correct) → Continue to GATE 4 (BoK)
│
├─ [GATE 4] BoK Softmax (default path for hard groups)
│  │  z-normalize → divide by tau → softmax → (w - 1/K) × K
│  │
│  ├─ [POST-PROCESSING 4a] WINNER_BOOST > 0?
│  │  Q: "Boost rare correct trajectories?"
│  │  If 0 < n_correct < K: boost = min(boost_max, sqrt(K/n_correct))
│  │
│  ├─ [POST-PROCESSING 4b] QUALITY_BONUS > 0?
│  │  Q: "Rank correct trajectories by point quality?"
│  │
│  └─ [POST-PROCESSING 4c] ALLWRONG_CAP > 0 AND all(answer < 0.5)?
│     Q: "Are ALL 16 trajectories wrong?"
│     │
│     ├─ YES → Cap advantages to ±ALLWRONG_CAP (±1.0)
│     │  Preserves direction signal but limits magnitude
│     │
│     └─ NO → Normal BoK advantages (uncapped)
│
└─ Advantage computed for this group → next group
```

## 3. A Sample's Complete Journey (Concrete Example)

### Setup
- **Prompt**: "How many red circles are in this image?"
- **Ground truth**: 7 red circles
- **K=16 trajectories** generated by the model via rollout

### Example Trajectory Outcomes (16 paths)

| Trajectory | Answer | Point Score | Format | Overall Score |
|-----------|--------|-------------|--------|--------------|
| T1  | 1 (correct: 7) | 0.92 | 1 | 0.6+0.276+0.1 = 0.976 |
| T2  | 1 (correct: 7) | 0.85 | 1 | 0.6+0.255+0.1 = 0.955 |
| T3  | 1 (correct: 7) | 0.78 | 1 | 0.6+0.234+0.1 = 0.934 |
| T4  | 1 (correct: 7) | 0.70 | 1 | 0.6+0.210+0.1 = 0.910 |
| T5  | 1 (correct: 7) | 0.65 | 1 | 0.6+0.195+0.1 = 0.895 |
| T6  | 1 (correct: 7) | 0.88 | 1 | 0.6+0.264+0.1 = 0.964 |
| T7  | 1 (correct: 7) | 0.72 | 1 | 0.6+0.216+0.1 = 0.916 |
| T8  | 1 (correct: 7) | 0.80 | 1 | 0.6+0.240+0.1 = 0.940 |
| T9  | 1 (correct: 7) | 0.55 | 1 | 0.6+0.165+0.1 = 0.865 |
| T10 | 1 (correct: 7) | 0.60 | 1 | 0.6+0.180+0.1 = 0.880 |
| T11 | 1 (correct: 7) | 0.90 | 1 | 0.6+0.270+0.1 = 0.970 |
| T12 | 1 (correct: 7) | 0.75 | 1 | 0.6+0.225+0.1 = 0.925 |
| T13 | 1 (correct: 7) | 0.82 | 1 | 0.6+0.246+0.1 = 0.946 |
| T14 | 1 (correct: 7) | 0.68 | 1 | 0.6+0.204+0.1 = 0.904 |
| T15 | 1 (correct: 7) | 0.73 | 1 | 0.6+0.219+0.1 = 0.919 |
| T16 | 1 (correct: 7) | 0.50 | 1 | 0.6+0.150+0.1 = 0.850 |

### Step-by-Step Routing

**Group statistics:**
- group_scores: [0.976, 0.955, ..., 0.850]
- group_mean = 0.922
- group_std = 0.037
- batch_mean = 0.800 (across all 64 groups in batch)

**GATE 1: Low-Var?**
- group_std = 0.037 > 1e-5 → **NO** → continue

**GATE 2: All-Correct?**
- All 16 overall > 0.5 → pass_rate_full = 1.0 → **YES**
- FILTER_ALL_CORRECT = 1 → checking...

**GATE 2a: Smart Filter (V24)?**
- group_mean = 0.922
- SMART_FILTER_THRESHOLD = 0.955
- 0.922 < 0.955 → **YES, point quality below threshold!**
- **RELEASED** to Easy path (n_all_correct_released += 16)

**GATE 3: Easy?**
- pass_rate = 1.0 > 0.5 → **YES** → Easy z-score path

**Easy Z-Score Advantage:**
| Trajectory | score | adv = (score - 0.922) / 0.037 | clipped | final (×EASY_SCALE) |
|-----------|-------|-------------------------------|---------|---------------------|
| T1 (pt=0.92) | 0.976 | +1.46 | +1.46 | +1.46 |
| T16 (pt=0.50) | 0.850 | -1.95 | -1.95 | -1.95 |
| T9 (pt=0.55) | 0.865 | -1.54 | -1.54 | -1.54 |

**Result:** T1 (best point) gets positive gradient → model learns to point more accurately.
T16 (worst point) gets negative gradient → model learns to avoid sloppy pointing.

### V23 Comparison
In V23 (no smart filter), this same group would get **zero gradient** for all 16 trajectories.
The model would learn nothing about point quality improvement from this sample.

### What if this group had mean(point) = 0.90?
- group_mean = 0.7 + 0.3×0.90 = 0.970 > 0.955
- → **FILTERED** (zero gradient) — truly mastered, no need to train

## 4. Different Scenario Examples

### Scenario A: Mixed Group (BoK Path)
- 4/16 correct, 12/16 wrong → pass_rate = 0.25
- GATE 1: std >> 0.1 → NO
- GATE 2: pass_rate_full = 0.25 < 1.0 → NO
- GATE 3: 0.25 < 0.5 → NO → **BoK Softmax**
- BoK concentrates probability on the 4 correct trajectories

### Scenario B: Easy Group
- 12/16 correct, 4/16 wrong → pass_rate = 0.75
- GATE 1: std >> 0.1 → NO
- GATE 2: pass_rate_full = 0.75 < 1.0 → NO (not all correct)
- GATE 3: 0.75 > 0.5 → **Easy z-score**

### Scenario C: All Wrong (BoK + Cap)
- 0/16 correct → pass_rate = 0
- GATE 1: std ≈ 0.05 (point variance only) → NO
- GATE 2: pass_rate_full = 0 < 1.0 → NO
- GATE 3: 0 < 0.5 → NO → **BoK Softmax**
- POST-PROCESSING 4c: all(answer < 0.5) → YES → **Cap to ±1.0**
- Best wrong trajectory (highest point) gets capped positive advantage

### Scenario D: Low-Var (Identical Scores)
- All 16 trajectories: answer=1, point=0.85, format=1 → overall=0.955 for all
- GATE 1: std < 1e-5 → **YES** → Low-Var path
- Dr.GRPO fallback: advantage = 0.955 - 0.800 = +0.155 (gentle positive)

## 5. V23 → V24 Script Migration

```bash
# V23 config (for reference):
export BOK_FILTER_ALL_CORRECT=1           # filter all-correct groups
# BOK_SMART_FILTER_THRESHOLD not set → defaults to 0 (disabled)

# V24 Smart Filter config (recommended):
export BOK_FILTER_ALL_CORRECT=1           # keep filter ON
export BOK_SMART_FILTER_THRESHOLD=0.955   # NEW: release low-quality AC groups
# This means: only filter AC groups with mean(point) >= 0.85
```

### Alternative V24 config (aggressive, as in existing V24 script):
```bash
export BOK_FILTER_ALL_CORRECT=0           # disable filter entirely
export BOK_EASY_SCALE=0.4                 # dampen all Easy gradients instead
# All AC groups go to Easy path with dampened gradient
```
