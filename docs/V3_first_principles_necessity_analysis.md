# V3 BoK-GRPO Improvements: First-Principles Necessity Analysis

> Date: 2026-03-31
> Question: Is V3 (Easy Dampening + Winner Boost + Quality Bonus) truly necessary,
> or is it treating symptoms of a simpler problem?

---

## 1. The User's Challenge (Exact Question)

> "如果仅仅因为BoK/Easy梯度比问题，那我原版bok-grpo不采用dr.grpo走easy更新
> 不就可以实现不抢占easy题目对难题的更新梯度吗？"

Translation: "If the problem is just gradient budget imbalance, why not just remove
easy_drgrpo routing entirely? Pure BoK already concentrates gradient on hard groups."

---

## 2. History: Why Was Easy_DrGRPO Added?

### V11 Motivation (docs/V11_modifications.md)

| Observation | Detail |
|-------------|--------|
| V10 diagnosis | "BOK softmax 在 easy groups 上给出微弱梯度（+0.14 vs DrGRPO +0.37）" |
| Root cause | PPO dead code (ppo_epochs=1) + BoK weak easy signal |
| V11 solution | Route easy groups (pass_rate > 0.75) to DrGRPO for 2.6x stronger gradient |
| Stated goal | "加速正确路径巩固" (accelerate correct path consolidation) |

### Threshold Evolution

| Version | EASY_TH | Easy% of batch | BoK% | Performance | Status |
|---------|---------|---------------|------|-------------|--------|
| V7-V10 | N/A | 0% | ~80% | Weak BoK | Crash/NaN |
| **V12** | **0.75** | **~41%** | **~28%** | **Best (15.45%)** | ✅ |
| V15/V16 | 0.50 | ~87% | ~6% | 12-13% | Declining |
| V17 | 0.50 | ~87% | ~6% | 12.20% (regressed) | NaN@77 |
| V19 | 0.50 | ~47% | ~21% | In progress | FP16 crash |

**Key finding**: V12 (TH=0.75) achieved the best StepCount-500 score. Lowering to
TH=0.50 correlated with performance REGRESSION, not improvement.

---

## 3. Quantitative Analysis: BoK vs DrGRPO Per-Group Gradient

Monte Carlo simulation (500 groups per n_correct, τ=0.48, CLIP=4.0):

| n_correct/16 | BoK |adv|_total | DrGRPO |adv|_total | BoK_winner | DrGRPO_winner | Dr/BoK |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **17.45** | 6.53 | **+4.000** | +2.500 | 0.37x |
| 2 | **20.40** | 10.25 | **+4.000** | +2.500 | 0.50x |
| 3 | **22.47** | 12.42 | **+3.695** | +2.070 | 0.55x |
| 4 | **20.89** | 13.79 | **+2.611** | +1.724 | 0.66x |
| 5 | **19.09** | 14.78 | +1.909 | +1.478 | 0.77x |
| 6 | **17.35** | 15.44 | +1.446 | +1.287 | 0.89x |
| **7** | 15.64 | **15.83** | +1.117 | **+1.131** | **1.01x** ← crossover |
| 8 | 13.95 | **15.96** | +0.872 | **+0.998** | 1.14x |
| 10 | 10.56 | **15.46** | +0.528 | **+0.773** | 1.46x |
| 12 | 7.12 | **13.83** | +0.297 | **+0.576** | 1.94x |
| 14 | 3.97 | **10.27** | +0.128 | **+0.377** | 2.59x |

### Core Insight

**Crossover at n=7/16 (pass_rate=43.75%)**:
- n ≤ 6: BoK gives STRONGER total gradient than DrGRPO (BoK is better for hard groups)
- n ≥ 7: DrGRPO gives stronger gradient (DrGRPO is better for easy groups)
- n = 14: DrGRPO gives 2.59x more gradient than BoK

**This means BoK ALREADY naturally concentrates gradient on hard groups.** The question
is whether easy groups need the extra DrGRPO gradient at all.

---

## 4. Quality Differentiation: BoK vs DrGRPO

Surprising finding from simulation (14/16 correct group):

```
BoK advantages:    range=[-0.192, +0.480], spread=0.671
DrGRPO advantages: range=[+0.205, +0.525], spread=0.320
DrGRPO/BoK spread ratio: 0.5x (DrGRPO has LESS spread)
```

**BoK gives MORE quality differentiation among correct trajectories (wider spread)**,
not less. BoK softmax even assigns NEGATIVE advantage to the worst correct trajectories,
creating stronger quality pressure than DrGRPO.

However, BoK gives much smaller MAGNITUDE (+0.14 mean vs +0.37 DrGRPO). So the
quality signal is more discriminating but weaker overall.

---

## 5. Evaluation Evidence

| Benchmark | Type | V12 (TH=0.75) | V17 (TH=0.50) | SFT |
|-----------|------|:-:|:-:|:-:|
| CountBench-491 | easy (sparse) | 79.43% | 79.43% | 78.82% |
| PixMo-529 | easy (sparse) | 82.80% | 81.29% | — |
| **StepCount-500** | **hard (dense)** | **15.45%** | **12.20%** | **9.60%** |

**Easy performance (CountBench/PixMo) is naturally stable regardless of routing config.**
**Hard performance (StepCount) declined when more groups were routed to DrGRPO.**

This directly supports the user's hypothesis: easy gradient is not needed for
maintenance, and routing too much to DrGRPO hurts hard performance.

---

## 6. Answer: Is V3 Necessary?

### What V3 was trying to fix

| V3 Component | Fixes | Via |
|--------------|-------|-----|
| Easy Dampening (0.5x) | Easy groups dominate gradient (47% budget) | Halve easy signal |
| Winner Boost (3.0x) | Rare correct trajectories not amplified enough | Multiply winner advantages |
| Quality Bonus (+0.5) | No differentiation among correct paths | Add quality-proportional bonus |

### Are these fixes addressing the ROOT CAUSE?

**NO. V3 is treating a SYMPTOM.**

The root cause is: **EASY_THRESHOLD=0.50 is too aggressive** — it routes too many
groups through DrGRPO, stealing gradient budget from BoK hard groups.

| Root cause | Symptom | V3's treatment |
|-----------|---------|---------------|
| TH=0.50 routes 47% to easy | Easy dominates gradient | Halve easy (0.5x) — partial fix |
| BoK gets only 21% of budget | Hard signal diluted | Winner boost — fails due to clip bug |
| N/A | Quality differentiation | Bonus — BoK already differentiates more |

### The simpler fix: Raise EASY_THRESHOLD back to 0.75

This DIRECTLY addresses the root cause:
- Fewer groups routed to DrGRPO → more BoK groups
- V12 (TH=0.75) was the best-performing config historically
- No new hyperparameters needed

---

## 7. Recommended V20 Configuration

### Option 1 (Conservative — Recommended): V12 recipe + Winner Boost

```
EASY_THRESHOLD = 0.75          # Restore V12's proven balance
BOK_WINNER_BOOST = 3.0         # Keep: amplifies rare-correct in hard groups
BOK_EASY_SCALE = 1.0           # REMOVE: unnecessary when TH=0.75
BOK_QUALITY_BONUS = 0.0        # REMOVE: BoK already differentiates
```

Rationale:
- V12 threshold was historically the best-performing config
- Winner Boost addresses a genuinely SEPARATE issue (rare-correct signal in BoK)
- Asymmetric clip fix ensures the boost actually takes effect
- Simplest config with strongest evidence

### Option 2 (Aggressive): Pure BoK

```
EASY_THRESHOLD = 1.01          # Disable easy_drgrpo entirely 
BOK_WINNER_BOOST = 3.0         # Keep
BOK_EASY_SCALE = 1.0           # N/A
BOK_QUALITY_BONUS = 0.0        # N/A
```

Rationale:
- Maximum hard focus (100% BoK). Evidence shows easy performance is self-sustained.
- Risk: total gradient magnitude is ~1.8x lower than V12 config (402 vs 553). May
  slow convergence rate. No historical precedent with clean training.

### Option 3 (Current V3): TH=0.50 + Dampening + Boost

```
EASY_THRESHOLD = 0.50
BOK_WINNER_BOOST = 3.0
BOK_EASY_SCALE = 0.5
BOK_QUALITY_BONUS = 0.5
```

Assessment: Adds 3 new hyperparameters to compensate for a threshold choice. Over-
engineered. Gradient budget shift (67% BoK) is achieved more cleanly by Option 1.

---

## 8. What Winner Boost ACTUALLY Fixes (Separate from Easy Routing)

Winner Boost addresses a real issue INDEPENDENT of easy routing:

In hard BoK groups (2/16 correct), the correct trajectory already gets adv=4.0 (clip).
But in the batch-level loss averaging over 1024 samples, this one trajectory's signal
is diluted to 4.0/1024 = 0.004 per token.

**Without Winner Boost**: 2 correct × 4.0 = 8.0 total positive signal per hard group
**With Winner Boost 3x + asymmetric clip**: 2 × 6.93 = 13.86 total (+73%)

This helps regardless of whether easy routing is on or off. It's a genuine improvement
for the pass@32→pass@1 gap bridging.

### With asymmetric clip fix:
| n_correct | V2 advantage | V3_fix advantage | Effective boost |
|-----------|-------------|-----------------|----------------|
| 1/16 | 4.00 | 6.93 | 1.73x |
| 2/16 | 4.00 | 6.93 | 1.73x |
| 4/16 | 2.61 | 5.46 | 2.09x |
| 5/16 | 1.91 | 3.67 | 1.92x |

---

## 9. Final Recommendation

**V20 should use: Option 1 (V12 recipe + Winner Boost + Asymmetric Clip)**

Changes from current V19 script:
1. `BOK_EASY_THRESHOLD=0.75` (was 0.50) — restore V12 balance
2. `BOK_WINNER_BOOST=3.0` (keep) — amplify rare-correct
3. `BOK_EASY_SCALE=1.0` or remove (was 0.5) — unnecessary dampening
4. `BOK_QUALITY_BONUS=0.0` or remove (was 0.5) — marginal, adds complexity
5. Asymmetric clip in core_algos.py (already applied) — makes boost effective

This is the **simplest config that addresses the core issue with historical evidence support.**
