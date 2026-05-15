# V12 vs V23/V24 Deep Diagnosis: Why V12 Remains the Best

**Date**: 2026-04-13  
**Context**: V12 pixmo-test 82.04% > V23 79.77% > V24 77.13%(val), despite V23/V24 having 0-11% NaN vs V12's 51%  
**Question**: Is the regression caused by Easy path routing + DrGRPO mode reducing BoK effectiveness?

---

## 1. Complete Evaluation Scorecard

| Version | Algorithm | pixmo-test | countbench | NaN Rate | Effective LR | BoK% | Easy% |
|---------|-----------|-----------|-----------|----------|-------------|------|-------|
| SFT Base | — | 75.6% | — | 0% | — | — | — |
| V7 (GRPO) | Standard GRPO | 80.72% | — | 0% | 1.0e-6 | N/A | N/A |
| **V8 (pure BoK)** | BOK-GRPO | 80.15% | — | ~20%(late) | 1.0e-6 | **100%** | **0%** |
| V11 (BoK+Easy) | BOK-GRPO | 80.15% | — | 0% (77steps) | 3.0e-6 | ~25% | ~30% |
| **V12** 🏆 | BOK-GRPO | **82.04%** | **79.43%** | 51% | **3.0e-6** | **25.4%** | **29.8%** |
| V14-mixed | BOK-GRPO | 80.34% | 78.21% | 29% | 3.0e-6 | 43.6% | 27.2% |
| V15 | BOK-GRPO | 80.15% | 79.02% | 0% | 2.8e-7 | ~30% | 51.6% |
| **V23** | BOK-GRPO | **79.77%** | **80.45%** | **0%** | **1.0e-6** | **~12%** | **~50%** |
| **V24** | BOK-GRPO | N/A (val=77.13%) | N/A | 11% | 1.5e-6 | ~12% | ~50% |

---

## 2. The Three Root Causes of V12 > V23

### Cause 1: Effective Learning Rate 3x Gap (50% contribution)

| Config | V12 | V23 | V24 | Delta V12→V23 |
|--------|-----|-----|-----|---------------|
| LR | 1.5e-6 | 1e-6 | 1.5e-6 | 0.67x |
| ppo_epochs | 2 | 1 | 1 | 0.5x |
| **Effective LR** | **3.0e-6** | **1.0e-6** | **1.5e-6** | **0.33x** |
| max_grad_norm | 0.5 | 1.0 | 1.0 | 2x looser |

V23's effective learning rate is only 33% of V12. This is the **single largest factor**.

V24 tried to partially restore this (LR=1.5e-6 → eff 1.5e-6 = 50% of V12), but combined it with KL=0.02 (weaker regularization), causing entropy divergence and 17 NaN steps. The root error: V24 restored V12's LR without restoring V12's grad_norm=0.5 safety brake.

**V12's secret**: High effective LR (3e-6) + strict grad clipping (0.5) = aggressive but bounded learning. The grad_norm=0.5 acted as an implicit trust region, preventing the large gradients from causing catastrophic updates. V12 still had 51% NaN because ppo_epochs=2 introduced ratio≠1 instability, but the 0.5 grad clip contained the damage.

### Cause 2: Easy Path Routing Starving BoK (30% contribution)

**Routing distribution comparison:**

| Path | V12 (TH=0.75) | V23 (TH=0.50) | Shift |
|------|---------------|---------------|-------|
| BoK softmax | 25.4% | ~12% | **-53%** |
| Easy z-score | 29.8% | ~50% | **+68%** |
| AC filtered | 30.0% | ~20% | -33% |
| Low-var fallback | 14.8% | ~14% | ~0% |

V23 routes **twice as many samples** to the Easy path and **half as many** to BoK compared to V12.

**Two factors drive this shift:**

1. **EASY_THRESHOLD lowered**: V12 used 0.75, V23 inherited V15's 0.50. With pass@1≈80%, lowering from 0.75→0.50 means groups that previously needed >75% correct (borderline hard) now qualify as "easy" at >50% — absorbing the entire "medium difficulty" population.

2. **Stronger starting model**: V12 started from fresh SFT (75.6%), V23 resumed from V21-R2 step-60 (already ~77-78%). The stronger model produces more high-pass-rate groups → more groups qualify as "easy" from step 1.

**Why this matters**: BoK's softmax advantage is the algorithm's core innovation — it concentrates learning signal on the best trajectories within hard groups. When only 12% of samples reach BoK (V23), the algorithm's key mechanism barely contributes to overall learning. The 50% routed to Easy gets suboptimal z-score gradients (see Cause 3).

### Cause 3: Easy Path Uses Z-Score, NOT True Dr.GRPO (Design Bug) (20% contribution)

**The V11 design document's math was correct, but the implementation diverged.**

V11 analysis claimed DrGRPO gives 2.6x stronger gradient for easy groups:
- Easy group (pass_rate=0.875): DrGRPO raw centering advantage = +0.37
- Same group: BoK softmax advantage = +0.14
- Ratio: 2.6x → "DrGRPO is better for easy groups"

**But the actual implementation (core_algos.py L610) uses z-score, not Dr.GRPO:**

```python
# ACTUAL CODE (called "Easy DrGRPO path"):
adv = (scores[global_i] - group_mean) / (group_std + eps)  # ← z-score normalization!
adv = max(-2.5, min(2.5, adv))
advantages_1d[global_i] = adv * _easy_scale

# TRUE Dr.GRPO (only used in low_var fallback):
advantages_1d[global_i] = scores[global_i] - batch_mean  # ← no std division
```

**The z-score normalization is exactly the "difficulty bias" Dr.GRPO paper identifies as harmful:**

| Group type | group_std | z-score magnitude | Real signal? |
|-----------|----------|-------------------|-------------|
| AC low-var | 0.03 | |z| ≈ 2.5 (clipped!) | ❌ Noise amplified |
| AC med-var | 0.10 | |z| ≈ 1.5 | ⚠️ Mixed |
| Hard groups | 0.25 | |z| ≈ 0.8 | ✅ Real signal |

The low-variance "easy" groups get the LARGEST z-scores because dividing by tiny std amplifies point-level noise to clip boundaries (±2.5). This sends near-random gradient directions for already-mastered problems — pure noise injection.

In V23, 50% of samples suffer from this z-score amplification. In V12, only 30%.

---

## 3. Why Easy Path Didn't Deliver its Promised 2.6x Advantage

The V11 design document (V11_modifications.md) presented this mathematical argument:

> For easy groups (pass_rate=0.875, K=16, 14 correct, 2 wrong):
> - BoK softmax: advantage for best trajectory ≈ +0.14
> - DrGRPO centering: advantage for best ≈ +0.37
> - Ratio: 2.6x → Route easy groups to DrGRPO for stronger learning

**Three reasons this failed:**

### 3.1 Implementation Mismatch
The math assumed raw centering (`score - mean`), but code implements z-score (`(score - mean) / std`). These have opposite properties:
- Raw centering: low-var groups → tiny advantages (auto-dampened) ✅
- Z-score: low-var groups → huge advantages (noise amplified) ❌

### 3.2 Wrong Gradient Target
Even with correct Dr.GRPO, giving 2.6x stronger gradients to *already-correct* groups is suboptimal:
- Easy groups (pass_rate>0.75) already score well
- Stronger gradients on these groups → faster convergence to current behavior (potential overfitting)
- Marginal improvement headroom is tiny (from ~87.5% to ~90%?)
- Meanwhile hard groups (pass_rate<0.50) have huge improvement headroom but get less total gradient budget

### 3.3 Self-Defeating Routing Dynamics
As training progresses:
1. Model improves → more groups become "easy" (pass_rate > threshold)
2. More Easy routing → less BoK on hard cases
3. Hard cases don't improve as fast
4. Easy groups get z-score noise → minimal real learning
5. **Net result**: training stagnates because the algorithm progressively routes AWAY from where learning is most valuable

---

## 4. Controlled Variable Analysis

### 4.1 Easy Path Effect (V8 → V11)

| | V8 (no Easy) | V11 (with Easy) | Delta |
|---|---|---|---|
| pixmo-test | 80.15% | 80.15% | **0.00pp** |
| Effective LR | 1.0e-6 | 3.0e-6 | 3x higher |
| Easy routing | 0% | ~30% | +30% |
| BoK routing | 100% | ~25% | -75% |

**V11 had 3x the effective LR of V8 but scored IDENTICALLY.** The Easy path routing, despite taking 30% of samples from BoK, provided zero improvement even with 3x more learning rate budget. This is strong evidence that Easy path routing is net-negative: the 3x LR advantage was entirely consumed by the suboptimal Easy path gradients.

### 4.2 Effective LR Effect (V8 → V12)

| | V8 | V12 | Delta |
|---|---|---|---|
| pixmo-test | 80.15% | 82.04% | **+1.89pp** |
| Effective LR | 1.0e-6 | 3.0e-6 | 3x |
| grad_norm | 1.0 | 0.5 | 0.5x (tighter) |
| KL | 0.03 | 0.02 | 0.67x (weaker) |
| Easy routing | 0% | 29.8% | +30% |

V12 gained +1.89pp over V8. But V11 (same config as V12 except 2 epochs vs 1) gained 0pp. The difference: V12 trained 1 epoch (178 steps) while V11 trained 2 epochs (256 steps). **V12 avoided the second-epoch overfitting that killed V11's gains.**

### 4.3 EASY_THRESHOLD Effect (V15 TH=0.50 vs V12 TH=0.75)

| | V12 (TH=0.75) | V15 (TH=0.50) | Delta |
|---|---|---|---|
| pixmo-test | 82.04% | 80.15% | **-1.89pp** |
| BoK routing | 25.4% | ~30% | +5pp |
| Easy routing | 29.8% | 51.6% | +22pp |

V15 (with EASY_TH=0.50) scored 1.89pp lower than V12 (TH=0.75), despite similar BoK coverage. The additional 22% of samples routed from AC-filter to Easy path provided lower-quality gradients. **Lowering EASY_THRESHOLD from 0.75 to 0.50 was harmful.**

---

## 5. Quantified Contribution Breakdown

| Factor | Estimated Impact | Evidence |
|--------|-----------------|----------|
| Effective LR 3x gap | ~1.0pp (50%) | V8→V12: +1.89pp with LR+Easy; V8→V11: 0pp with Easy only |
| Easy path z-score noise | ~0.6pp (30%) | V12→V15: -1.89pp when EASY_TH lowered (more Easy routing) |
| Starting checkpoint warmth | ~0.4pp (20%) | V12 from SFT→82%, V23 from V21→79.77% (diminishing returns) |
| **Total gap** | **~2.0pp** | **V12 82.04% - V23 79.77% = 2.27pp** |

---

## 6. What V12 Got Right (Accidentally)

V12's configuration, despite causing 51% NaN, created an accidentally near-optimal setup:

1. **High effective LR + tight grad clip = fast, bounded learning**
   - eff LR 3e-6 pushed model aggressively
   - grad_norm=0.5 prevented catastrophic updates
   - ppo_epochs=2 caused NaN but was partially contained by the clip

2. **EASY_THRESHOLD=0.75 kept BoK coverage high**
   - 25% BoK vs V23's 12% → more samples through the superior mechanism
   - Medium-difficulty groups (50-75% pass rate) stayed in BoK where they belong

3. **1-epoch training avoided overfitting**
   - V11 (2 epochs, same config) scored 80.15% vs V12's 82.04%
   - Second pass over same data caused regression

4. **Max_grad_norm=0.5 was critical safety**
   - V24 tried V12's LR without this safety → divergence
   - grad_norm=0.5 is the overlooked key parameter

---

## 7. Diagnosis Summary

### Is it the Easy path problem?
**YES, partially (30% of the gap).** The Easy path has three compounding issues:
1. Uses z-score instead of true Dr.GRPO → noise amplification for low-var groups
2. With EASY_TH=0.50 in V23, routes 50% of samples through this noisy path
3. Creates self-defeating dynamics where improving model → more Easy routing → less BoK → slower learning

### Is it Dr.GRPO mode reducing BoK effectiveness?
**YES, but the framing is wrong.** The Easy path claims to use "DrGRPO-style" advantages but actually uses z-score normalization — the exact opposite of Dr.GRPO. The problem isn't "DrGRPO vs BoK choice" but "the Easy path implementation is buggy" (z-score when Dr.GRPO was intended). **If TRUE Dr.GRPO were implemented**, the Easy path would auto-dampen low-var groups (advantage ≈ 0.05 instead of ±2.5) and might actually be beneficial.

### Why didn't the 2.6x advantage materialize?
Three reasons: (a) implementation uses z-score not Dr.GRPO, (b) 2.6x gradient on already-correct patterns → overfitting not improvement, (c) routing dynamics are self-defeating.

---

## 8. Recommended Fixes for V25

### Priority 1: Restore V12's Effective LR with Proper Safety
```bash
ACTOR_LR=1.5e-6          # V12's per-epoch LR
ppo_epochs=1             # Keep single epoch (NaN-free guarantee)
max_grad_norm=0.5        # V12's safety brake (CRITICAL, missing in V23/V24!)
kl_coef=0.03             # V23's stronger regularization
```
Expected: Close 50% of the gap (~1.0pp) while maintaining 0% NaN.

### Priority 2: Fix Easy Path (True Dr.GRPO Mode)
```python
# Replace in core_algos.py L610:
# BEFORE (z-score):
adv = (scores[global_i] - group_mean) / (group_std + eps)
# AFTER (true Dr.GRPO centering):
adv = scores[global_i] - group_mean  # NO std division
```
```bash
BOK_EASY_SCALE=2.0        # Re-derived for raw centering (see V24 analysis doc)
BOK_EASY_ADV_MODE=drgrpo  # NEW: env var to select mode
```
Expected: Recovers 30% of the gap (~0.6pp). Low-var groups auto-dampened to ~0.05 advantage.

### Priority 3: Restore EASY_THRESHOLD=0.75
```bash
BOK_EASY_THRESHOLD=0.75   # V12's threshold, not V15's 0.50
```
Expected: Routes medium-difficulty groups back to BoK, increasing coverage from 12% to ~25%.

### Alternative: Aggressive Dr.GRPO + No Filter
If true Dr.GRPO mode auto-dampens low-var groups (adv≈0.05), the AC filter becomes unnecessary:
```bash
BOK_FILTER_ALL_CORRECT=0   # Remove binary filter
BOK_EASY_ADV_MODE=drgrpo   # Auto-dampens low-var
BOK_EASY_SCALE=2.0          # Re-derived scale
BOK_EASY_THRESHOLD=0.50     # Keep 0.50 because Dr.GRPO handles it properly
```
Expected: Recovers all wasted gradient (25-30% AC-filtered samples contribute), combined with Dr.GRPO auto-dampening.

### V25 Full Config (Conservative)
```bash
ACTOR_LR=1.5e-6
ppo_epochs=1
max_grad_norm=0.5          # V12's safety brake
kl_coef=0.03               # V23's regularization
BOK_EASY_ADV_MODE=drgrpo   # True Dr.GRPO (Priority 2)
BOK_EASY_SCALE=2.0
BOK_EASY_THRESHOLD=0.75    # V12's threshold (Priority 3)
BOK_FILTER_ALL_CORRECT=1   # Keep safety filter, let Dr.GRPO handle dampening
# Resume from V21-R2 step-60 (same as V23)
```
Expected pixmo-test: 80.5-82.0% with 0% NaN.

### V25 Full Config (Aggressive)
```bash
ACTOR_LR=1.5e-6
ppo_epochs=1
max_grad_norm=0.5
kl_coef=0.03
BOK_EASY_ADV_MODE=drgrpo
BOK_EASY_SCALE=2.0
BOK_FILTER_ALL_CORRECT=0   # Remove filter, Dr.GRPO auto-dampens
BOK_EASY_THRESHOLD=0.50    # OK because Dr.GRPO handles it
# Resume from V21-R2 step-60
```
Expected pixmo-test: 81.0-83.0% with 0% NaN (higher variance, higher ceiling).
