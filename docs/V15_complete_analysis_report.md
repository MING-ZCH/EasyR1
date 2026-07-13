# V15 BoK-GRPO Complete Training Analysis Report

## 1. Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 5e-7 |
| PPO Epochs | 2 |
| Clip Ratio High | 0.28 |
| **eff_intensity** | **2.8e-7** |
| Max Grad Norm | 1.0 |
| Total Steps | 90 |
| NaN Count | **0** |
| Grad Spikes (>5.0) | 1 (max=8.20) |
| tau range | 0.500 → 0.300 |
| tau_bumped | First at S76 |

## 2. Validation Trajectory (Complete)

| Step | answer_reward | point_reward | format_reward | Δ vs SFT(0.756) |
|------|--------------|-------------|--------------|-----------------|
| S0   | **0.756**    | 0.919       | 0.962        | +0.000 (baseline) |
| S15  | 0.739        | 0.917       | 0.960        | -0.017          |
| S30  | 0.756        | 0.931       | 0.972        | +0.000          |
| S45  | **0.762** ←PEAK | 0.927    | 0.968        | **+0.006**      |
| S60  | 0.741 ←DIP   | 0.927       | 0.970        | -0.015          |
| S75  | 0.743        | 0.923       | 0.966        | -0.013          |
| S90  | **0.762** ←RECOVERY | 0.927 | 0.968        | **+0.006**      |

**Key observations:**
- Val oscillates in a **±0.02 band** around 0.750
- S60 dip is **within noise**, NOT a systematic degradation
- S90 = S45 = 0.762 (identical peak, perfect recovery)
- **Net improvement over SFT: only +0.006** (from 0.756 to 0.762)
- Point reward stable: 0.917-0.931 (marginal +1.2% improvement)
- Format reward stable: 0.960-0.972

## 3. Routing Analysis by Phase

### 3.1 Phase Summary Table

| Phase | tau | BoK% | DrGRPO% | AllCorr% | LowVar% | Score_mean | collapsed |
|-------|-----|------|---------|----------|---------|-----------|-----------|
| S1-10 (baseline) | 0.497 | 28.7 | 47.2 | 16.6 | 7.7 | 0.743 | 3.7 |
| S15-30 (early) | 0.472 | 31.2 | 46.4 | 14.6 | 5.7 | 0.723 | 3.5 |
| S31-45 (→peak) | 0.433 | 30.0 | 45.2 | 17.7 | 5.8 | 0.725 | 3.5 |
| S46-60 (→dip) | 0.373 | 28.2 | 45.0 | 19.2 | 7.3 | 0.737 | 2.9 |
| S55-62 (post-peak) | 0.356 | 26.0 | 49.5 | 18.2 | 6.6 | 0.751 | 2.8 |
| S63-75 (mid-late) | 0.327 | 24.3 | 47.8 | 18.9 | 7.9 | 0.754 | 2.5 |
| S76-90 (final) | 0.306 | 27.5 | 43.0 | 21.6 | 8.0 | 0.744 | 4.5 |

### 3.2 Key Routing Trends

**DrGRPO% (easy groups → standard GRPO path):**
- Spikes to ~50% in S55-62 window (normally ~44-47%)
- Coincides exactly with the val dip window
- Returns to ~43-47% afterwards → val recovers

**AllCorr% (all-correct groups → filtered out, no learning signal):**
- Gradually increasing: 16.6% → 17.7% → 19.2% → 21.6%
- By S76-90: **21.6%** of batch is all-correct → wasted capacity
- Extreme spikes: S76=34.4%, S88=31.2%

**BoK% (BoK softmax weighted path → strongest learning signal):**
- Declining trend: 28.7% → 26.0% → 24.3%
- Partial recovery in S76-90 to 27.5% (due to tau_bumped preventing further flattening)

**tau_bumped:**
- First appears at S76 (16 samples bumped)
- Becomes regular at S79+: indicates tau hit floor (0.30)
- S84-85: 32 samples bumped per step

## 4. S60 Val Dip Root Cause Diagnosis

### 4.1 Primary Cause: DrGRPO Routing Spike (NOT a real regression)

The S60 val dip from 0.762→0.741 is a **transient fluctuation**, not systematic degradation:

1. **DrGRPO% spike at S55-62 (~50%)**: More groups classified as "easy", reducing BoK's sophisticated weighting. Standard GRPO provides weaker learning signal for hard cases.
2. **Recovery mechanism**: DrGRPO% naturally fluctuates. By S63+, it returns to normal (~47%), and val recovers.
3. **Proof**: S90 val = 0.762 = exactly S45 peak. The model fully recovered.

### 4.2 Why the oscillation exists

The ±0.02 val band (0.739-0.762) exists because:
- **Small effective learning rate (eff_intensity=2.8e-7)** means each step makes tiny parameter changes
- **Training data batch composition varies** between steps, causing natural metric fluctuation
- **Val set is fixed (529 samples)** — small shifts in model behavior create visible but meaningless score changes
- With 529 val samples, ±1% accuracy = ±5 samples → Binomial standard error ≈ 0.019

### 4.3 Conclusion: S60 dip is statistical noise

The binomial standard error for 529 binary trials at p=0.75 is:
$$\sigma = \sqrt{\frac{p(1-p)}{n}} = \sqrt{\frac{0.75 \times 0.25}{529}} \approx 0.019$$

The S60 dip of -0.021 from peak is within 1.1σ — **entirely within expected variance**.

## 5. The Real Problem: Insufficient Learning

### 5.1 V15 vs Previous Experiments

| Version | eff_intensity | Best Val | Δ vs SFT | NaN Rate | Status |
|---------|--------------|----------|----------|----------|--------|
| V7 | 3.0e-7 | 0.773 | +0.017 | 0% | ✅ Best stable |
| V12 | 8.4e-7 | ~0.775 | +0.019 | 51.7% | ❌ NaN collapse |
| V14-hard | 5.6e-7 | — | — | 6.3% | ⚠️ Marginal |
| V14-mixed | 8.4e-7 | — | — | 30.0% | ❌ NaN |
| **V15** | **2.8e-7** | **0.762** | **+0.006** | **0%** | ⚠️ Too conservative |

### 5.2 Why V15 underperforms V7

Despite more training steps (90 vs ~178), V15's best val (0.762) is significantly below V7's best (0.773):

1. **LR too low**: V15 LR=5e-7 is half of V7's LR=1e-6
2. **PPO epochs don't compensate**: V15 has ppo_epochs=2 (vs V7's 1), but each epoch sees the same data → diminishing returns on policy improvement per step
3. **eff_intensity 2.8e-7 is the safe boundary but below the optimal learning zone**:
   - V7 at 3.0e-7 achieved 0.773 — right at the sweet spot
   - V15 at 2.8e-7 achieves only 0.762 — below the learning threshold

### 5.3 Overfitting Signal

- Train reward: ~0.97 (step 60) → near ceiling
- Val answer: 0.741 (step 60)
- **Gap: 0.23** → the model memorizes training patterns but fails on novel val patterns
- AllCorr% increasing monotonically (16.6% → 21.6%) confirms training data saturation

## 6. Recommendations for V16

### 6.1 Configuration Proposal

| Parameter | V15 (current) | V16 (proposed) | Rationale |
|-----------|---------------|----------------|-----------|
| LR | 5e-7 | **1e-6** | Match V7's effective learning |
| ppo_epochs | 2 | **1** | Reduce overfitting per step |
| clip_ratio_high | 0.28 | **0.28** | Keep conservative |
| **eff_intensity** | 2.8e-7 | **2.8e-7** | Same safe boundary as V7 |
| max_grad_norm | 1.0 | 1.0 | Keep |
| Total steps | 90 | **180+** | More steps at lower per-step intensity |

**Key insight**: V7's success was LR=1e-6 × ppo_epochs=1 = eff_intensity=2.8e-7. V15 tried to match this with LR=5e-7 × ppo_epochs=2 = 2.8e-7, but **ppo_epochs=2 is NOT equivalent to 2× LR**:
- ppo_epochs=2: same data twice → gradient direction doesn't change much, risk of overfitting
- LR=2×: every gradient update has bigger step → more responsive to new data each step

### 6.2 Alternative: Higher eff_intensity with NaN Guards

If we want to push beyond V7's 0.773 ceiling:

| Parameter | V16-aggressive | Rationale |
|-----------|---------------|-----------|
| LR | 1e-6 | V7-level |
| ppo_epochs | 2 | Doubled learning |
| clip_ratio_high | 0.25 | More conservative clipping |
| eff_intensity | 5.0e-7 | Between V7 and V14-hard |
| **NaN adaptive LR** | **Enabled** | From NaN report recommendations |
| **tau floor** | **0.35** | Prevent extreme softmax concentration |

### 6.3 Training Data Improvements

- **AllCorr% at 21.6% is wasteful**: Over 1/5 of compute produces zero learning signal
- Consider: filtering out "too easy" samples from training set, or increasing rollout diversity
- Consider: curriculum learning — phase out easy samples as training progresses

## 7. Summary

| Aspect | Finding |
|--------|---------|
| S60 Val Dip | **Statistical noise** (1.1σ, within expected variance) |
| NaN Safety | **Perfect** — zero NaN across 90 steps |
| Learning Effectiveness | **Insufficient** — only +0.006 over SFT baseline |
| Root Cause | eff_intensity=2.8e-7 with ppo_epochs=2 is less effective than ppo_epochs=1 at same eff |
| V15 Best Val | 0.762 (S45 and S90) — below V7's 0.773 |
| Recommendation | Return to V7-style config (LR=1e-6, ppo=1) or try moderate aggression with NaN guards |
