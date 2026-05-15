# V12-Update Deep Analysis & V12-B'/V12-A Config Design

> Date: 2025-07-25
> Scope: V12-Update diagnosis, cross-version comparison, Config B'/A derivation
> Status: V12-B' and V12-A scripts ready for launch

---

## 1. V12-Update Training Summary (Apr 17-19)

| Metric | Value |
|--------|-------|
| Best Val Answer | 76.75% (= SFT baseline) |
| NaN% | 57% |
| First NaN | Step 74 |
| Effective Steps | ~43 |
| Grad Norm (pre-NaN) | avg=2.432, max=3.966, spike@s67=25.87 |

**Result: Training failed to improve over SFT baseline.**

---

## 2. Root Cause: Multiplicative Oversample Amplification

### The Hidden Hard Ratio Problem

V12-Update script had:
- `TRAIN_HARD_OVERSAMPLE_FACTOR=2`
- `TRAIN_OVERSAMPLE_NO_MASK_FACTOR=3`

`_build_sample_weights()` (data_loader.py L90-200) applies **multiplicative** factors:
- 561 samples were both no-mask AND hard → weight = 3 × 2 = **6.0**

Actual effective weights:
```
easy (with mask):     9,838 × 1 = 9,838
hard (with mask):     3,242 × 2 = 6,484  
hard (no mask):         561 × 6 = 3,366
─────────────────────────────────────────
Total hard effective:            = 9,850 / 19,688 = 50.03%
```

**50% hard** (vs intended ~28%) + kl=0.02 + ppo=2 → catastrophic gradient spike.

### Cascade Timeline
```
s55: grad_norm=5.58 (rising)
s60: grad_norm=8.883
s67: grad_norm=25.87 (spike)
s73: grad_norm=4.038 (damaged model)
s74: NaN (permanent)
s87: format collapse (JSON key "point_2d"→"point_22", 5%→40% fail)
```

---

## 3. Cross-Version Comparison (Key Versions)

| Version | Data | hard% | LR | clip | ppo | kl | BOK_CLIP | Best Eval% | NaN% |
|---------|------|:-----:|:---:|:----:|:---:|:---:|:--------:|:---------:|:----:|
| V7-std | easy | 14%* | 1e-6 | 1.0 | 1 | 0.05 | N/A | 78.64% | 0% |
| V8 | easy | 14%* | 1e-6 | 1.0 | 1 | 0.03 | 4.0 | 80.15% | 0% |
| **V12** | easy | 14%* | 1.5e-6 | **0.5** | **2** | 0.02 | 3.0 | **82.04%** | 51% |
| V21-R1/R2 | mixed | ~13% | 1e-6 | 1.0 | 1 | 0.03 | 3.0 | 76.75% | 0% |
| **V23** | mixed | ~13% | 1e-6 | 1.0 | 1 | 0.03 | 4.0 | **79.77%** | 0% |
| V24 | mixed | ~13% | 1.5e-6 | 1.0 | 1 | 0.02 | 3.0 | 77.13% | 0% |
| V25 | mixed | ~13% | 1.5e-6 | 1.0(BUG) | 1 | 0.03 | 3.0 | 77.69% | ~60% |
| V12-Update | mixed | **50%** | 1.5e-6 | 0.5 | 2 | 0.02 | 3.0 | 76.75% | 57% |

*Note: "easy-only" dataset (11,455) naturally contains 1,616 hard samples (14.1%)

---

## 4. Why V12 (130.5e-6 total update) > V23 (216e-6 total update)

### Key Finding: Update QUALITY > Update QUANTITY

**Factor 1: Sign-GD Direction Efficiency (clip=0.5)**
- V23 (clip=1.0): gradient = real_direction × real_magnitude → noisy samples dominate
- V12 (clip=0.5): gradient = real_direction × 0.5 (constant) → equal-weight per-param updates
- Equivalent to extreme adaptive LR (like Adam but more aggressive)
- In RL (high gradient variance), direction matters more than magnitude

**Factor 2: ppo=2 Double-Pass Efficiency**
- First pass: learn broad direction
- Second pass: refine on same batch with updated policy (off-policy self-correction)
- More sample-efficient than seeing new batches (no re-rollout needed)
- Tradeoff: instability (51% NaN)

**Factor 3: Data Composition is Nearly Identical**
- V12 easy-only: 11,455 samples including 1,616 hard (14%)
- V23 easy+hard: 13,647 samples including ~3,803 hard (28%)
- V12 already trains on hard samples! The difference is only extra duplication.

**Factor 4: V24 proves kl=0.02 + clip=1.0 is unstable**
- V24 (kl=0.02, clip=1.0): entropy→0.918, result=77.13%
- Sign-GD (clip=0.5) provides implicit regularization that makes kl=0.02 safe

---

## 5. Config B' Design (V12-B')

### Goal: Match V23 total update quantity with V12's update quality

```
V23 total update: 213 steps × 1.016e-6/step = 216e-6
Config B' target: 213 steps × (0.5 × 2e-6) = 213e-6 ✓
```

### Parameters

| Parameter | Value | Rationale |
|-----------|:-----:|-----------|
| LR | **2e-6** | Match V23 total update via 213 × 1.0e-6 = 213e-6 |
| clip | **0.5** | Sign-GD (V12 proven) |
| ppo | **1** | Eliminate NaN risk (V12's 51% NaN from ppo=2) |
| kl | **0.02** | Safe with sign-GD (V24 proves kl=0.02+clip=1.0 fails) |
| BOK_CLIP | **4.0** | More tolerant for mixed hard samples |
| ABSOLUTE_CAP | **4.0** | Safety net |
| Data | mixed (28% hard) | More hard training than V12's 14% |

### Risk Assessment
- NaN risk: LOW (10%) — V11/V12 at clip=0.5+kl=0.02 had 0%/51% NaN, 51% was ppo=2
- Entropy risk: LOW (15%) — sign-GD prevents V24-style entropy collapse
- Performance: Expected 82-84%

### Novel Aspects (Never Tested Before)
1. LR=2e-6 (historical max was 1.5e-6)
2. BOK_CLIP=4.0 + clip=0.5 (historically BOK_CLIP=4.0 only with clip=1.0)

---

## 6. Config A Design (V12-A)

### Goal: Clean V23-style baseline with cold start

| Parameter | Value | Rationale |
|-----------|:-----:|-----------|
| LR | 1e-6 | V23 proven |
| clip | 1.0 | Standard (V23) |
| ppo | 1 | Standard |
| kl | 0.03 | V23 proven |
| BOK_CLIP | 3.0 | V12 value (V12>V8 by 1.89pp with 3.0 vs 4.0) |
| ABSOLUTE_CAP | 5.0 | Conservative |
| Data | easy-only | Clean baseline |

Expected: ~80% (similar to V23, but cold start vs warm may differ)

---

## 7. Script Files

- **V12-B'**: `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v12_Bp.sh`
- **V12-A**: `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v12_A.sh`

---

## 8. Hard Data Definition (Corrected)

**CRITICAL**: "Hard" = SFT base model pass@1 **wrong** samples, ALL within count 0-10 range.
- NOT "high count 11-50+" as previously assumed
- SFT eval: 11,455 total → 9,839 correct (85.89%), 1,616 wrong
- hard_only: 1,616 train-hard + 192 external (V12 wrong on pixmo-test+countbench) = 1,808
- easy_plus_hard: full easy (11,455, includes 1,616 hard) + hard_only (1,808) + 192×2 extra = 13,647
- 1,616 hard samples appear TWICE in easy_plus_hard dataset
