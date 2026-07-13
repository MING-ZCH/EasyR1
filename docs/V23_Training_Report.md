# V23 Training Report (In-Progress)

**Status**: Step 153/213 (72%), actively running  
**Config**: ppo_epochs=1, FORMAT_REJECTION=1, resume from V21-R2 step-60  
**Started**: 2026-04-10 00:37

## Key Results

### NaN: 0 — ppo_epochs=1 Fix CONFIRMED
- 92 optimizer steps completed, 0 NaN
- V22 with ppo_epochs=2 had 91 NaN in 213 steps
- Grad norm stable: mean=1.03, max=2.29, only 4 spikes total

### Val/Answer Trajectory

| Step | V23 | V22 | Delta | Note |
|------|------|------|-------|------|
| 60 | 0.7580 | 0.7505 | **+0.0075** | V23 leads |
| 75 | 0.7316 | N/A | — | dip |
| 90 | 0.7543 | 0.7429 | **+0.0114** | V23 leads |
| 105 | 0.7618 | 0.7713 | -0.0095 | V22 leads |
| 120 | 0.7524 | 0.7675 | -0.0151 | V22 leads |
| **135** | **0.7769** | 0.7675 | **+0.0094** | **V23 PEAK = V22 PEAK** |
| 150 | 0.7580 | 0.7675 | -0.0095 | V22 leads |

**Peak: 0.7769 at step 135** (V22 peak: 0.7769 at step 165 — V23 matched peak 30 steps earlier)

### Entropy: Extremely Stable
- Range: [0.492, 0.540], latest: 0.535
- V22 entropy climbed to 0.942 (pathological)
- V21-R1 entropy exploded to 1.364 (format collapse)
- V23 entropy ~10x more stable than V22

### Training Reward Trends

| Phase | Overall | Answer | Point | Format Fail | Turns Exceeded |
|-------|---------|--------|-------|-------------|----------------|
| 1-30 | 0.7673 | 0.7467 | 0.8677 | 2.63% | 2.58% |
| 31-60 | 0.7676 | 0.7412 | 0.8500 | 2.89% | 2.84% |
| 61-90 | 0.7794 | 0.7506 | 0.8435 | 2.55% | 2.46% |
| 91+ | 0.7795 | 0.7537 | 0.8644 | 2.27% | 2.20% |

Trend: Answer reward slowly improving (0.747 → 0.754), format fail decreasing (2.6% → 2.3%)

### Point Accuracy (Val step 150)
- Step 1: 90.9% hit (no duplicates)
- Step 4: 63.6% hit (weakest — late-turn accuracy drop)
- Step 9-10: 100% hit
- Mask stats: 81.4% hit, 17.1% miss, 1.5% duplicate

### BoK-GRPO
- Tau annealing: 0.624 → 0.373 (correct)
- Easy(DrGRPO) routing: ~42% (moderate)
- All-correct: ~24% (increasing — model improving)
- All-wrong: ~3% (very low — model rarely fully fails)

## Observations

1. **ppo_epochs=1 is the correct setting** — 0 NaN, stable entropy, healthy training dynamics
2. **V23 matched V22's peak val (0.7769)** at step 135, 30 steps earlier
3. **Val oscillates** with ~0.04 amplitude — characteristic of RL training
4. **Step 135 checkpoint** is the current best candidate for evaluation
5. **Training still has 60 steps** — may produce another peak
6. **Point step 4 is the weakest** (63.6% hit) — potential focus for improvement

## Status: ONGOING
Awaiting training completion (est. step 213)
