# Cross-Experiment Analysis: V7 → V15 BoK-GRPO Optimization Direction

## 1. Verified Configuration Table (from log headers)

| Exp | Algorithm | LR | ppo | clip_high | grad_norm | eff_intensity | τ_init→final | gate | fmt_rej | Data | Total Steps |
|-----|-----------|------|-----|-----------|-----------|--------------|-------------|------|---------|------|------------|
| V7-BoK | bok_grpo | 1e-6 | 1 | 0.28 | 1.0 | 2.8e-7 | 0.8→0.3 | soft | 1 | 0_10 | 178(crash@92) |
| V7-GRPO | grpo | 1e-6 | 1 | 0.30 | 1.0 | 3.0e-7 | — | soft | 1 | 0_10 | 178 |
| V8 | bok_grpo | 1e-6 | 1 | 0.28 | 1.0 | 2.8e-7 | 0.8→0.5 | soft | 1 | 0_10 | 178 |
| V11 | bok_grpo | 1.5e-6 | 2 | 0.28 | **0.5** | 8.4e-7 | 0.7→0.3 | off | 0 | 0_10 | 256(crash@77) |
| V12 | bok_grpo | 1.5e-6 | 2 | 0.28 | **0.5** | 8.4e-7 | 0.7→0.3 | off | 0 | 0_10 | 178 |
| V13 | bok_grpo | 1.5e-6 | 2 | 0.28 | **0.5** | 8.4e-7 | 0.7→0.3 | off | 0 | 0_10 | 356(stop@211) |
| V14-hard | bok_grpo | 1e-6 | 2 | 0.28 | 1.0 | 5.6e-7 | 0.7→0.3 | off | 0 | **hard_only** | 113(crash@76) |
| V14-mixed | bok_grpo | 1.5e-6 | 2 | 0.28 | 1.0 | 8.4e-7 | 0.7→0.3 | off | 0 | **mixed_h+e** | 91 |
| V15 | bok_grpo | **5e-7** | 2 | 0.28 | 1.0 | 2.8e-7 | **0.5→0.3** | off | 0 | **mixed_h+e** | 90 |

### Configuration Evolution Timeline
```
Phase 1 (V7/V8):  gate=soft, fmt_rej=1, answer_weight=0.7, easy_threshold=N/A
Phase 2 (V11-V14): gate=off, fmt_rej=0, answer_weight=0.6, easy_threshold=0.75
Phase 3 (V15):     gate=off, fmt_rej=0, answer_weight=0.6, easy_threshold=0.50
```

## 2. Validation Performance Comparison

### 2.1 Performance Summary

| Exp | Baseline | Peak Val | Peak Step | Final Val | NaN Skips | Status |
|-----|---------|---------|-----------|-----------|-----------|--------|
| V7-BoK | 0.756 | 0.758 | S40 | — | 0 | ❌ Format collapse @S92 |
| **V7-GRPO** | 0.756 | **0.773** | S140 | 0.760 | 0 | ✅ Completed |
| V8 | 0.756 | 0.760 | S80 | 0.737 | **72** | ⚠️ Completed w/ NaN |
| V11 | 0.756 | 0.764 | S66 | 0.764 | 0 | ❌ Crash @S77 |
| **V12** | 0.756 | **0.775** | S165 | 0.775 | **273** | ⚠️ Completed w/ heavy NaN |
| V13* | (0.779) | **0.788** | S180 | 0.764 | **107** | ❌ Stop @S211 |
| V14-hard | 0.756 | 0.758 | S60 | — | 21 | ❌ Format collapse |
| V14-mixed | 0.756 | 0.762 | S45/S90 | 0.762 | **75** | ⚠️ Completed w/ NaN |
| **V15** | 0.756 | **0.762** | S45/S90 | 0.762 | **0** | ✅ Clean completion |

*V13 resumed from V12 checkpoint at S178

### 2.2 Val Trajectory Visualization

```
Val answer_reward vs Training Steps

0.79 |                                            *V13(S180)
0.78 |                                            
0.77 |                          *V7-GRPO(S140)  *V12(S165)
0.76 |--*---*------*------------*------*--------*---*---*--- V14m/V15(S45,S90)
     |  | V11(S66) |            |      |                  
0.75 |  |          *V8(S80)     |      |
0.74 |  |          |            |      *V15(S60 dip)
0.73 |  *V15(S15)  |            |      
     |             |            |
0.70 |  *V7BoK(S60 collapse)    |
     |             |            |
0.53 |        *V7BoK(S70 crash) |
     +---+----+----+----+----+----+----+----+----+
         20   40   60   80  100  120  140  160  180  Steps
```

## 3. Critical Findings

### Finding 1: DATA is the ceiling — not LR

| Data Type | Experiments | Best Val | Ceiling |
|-----------|-----------|---------|---------|
| **0_10 (standard)** | V7-GRPO, V8, V12, V13 | 0.788 (V13) | **~0.775** |
| **mixed_hard+easy** | V14-mixed, V15 | 0.762 | **~0.762** |
| **hard_only** | V14-hard | 0.758 | **~0.758** |

**Evidence:**
- V14-mixed (eff=8.4e-7, NaN) peaks at 0.762
- V15 (eff=2.8e-7, clean) also peaks at 0.762
- Despite 3× difference in learning intensity → same ceiling

**Implication:** The mixed_hard_easy dataset has a **lower generalization ceiling (~0.762)** than the standard 0_10 data (~0.775). The curriculum change hurts.

### Finding 2: Standard GRPO outperformed early BoK-GRPO

| Comparison | Peak | Stability |
|-----------|------|-----------|
| V7-GRPO (standard GRPO) | **0.773** | ✅ Stable |
| V7-BoK (bok_grpo, gate=soft) | 0.758 | ❌ Format collapse |
| V8-BoK (bok_grpo, gate=soft) | 0.760 | ⚠️ 72 NaN skips |

**Root cause:** Phase 1 BoK (gate=soft, format_rejection=1) had implementation issues:
- `gate=soft` created gradient bottleneck at point-gating layer
- `format_rejection=1` zeroed reward for format errors → training signal loss
- tau 0.8→0.3 was too aggressive, causing format collapse

### Finding 3: eff_intensity 8.4e-7 learns fast but unstable

V12 achieved the best from-scratch peak (0.775) at eff=8.4e-7, but with 273 NaN skip updates:
- 273/178 steps had NaN gradients → **1.5 NaN per step** on average
- Despite this, the model still climbed — NaN skip guard worked as a safety net
- **Key insight:** Skip-update mechanism effectively acts as implicit gradient clipping

### Finding 4: ppo_epochs=2 is NOT equivalent to 2× LR

| Config A | Config B | Same eff? | Result |
|----------|----------|-----------|--------|
| V7-BoK: LR=1e-6, ppo=1 | — | 2.8e-7 | Peak 0.758 |
| V15: LR=5e-7, ppo=2 | — | 2.8e-7 | Peak 0.762 |
| V12: LR=1.5e-6, ppo=2 | — | 8.4e-7 | Peak 0.775 |
| V14-hard: LR=1e-6, ppo=2 | — | 5.6e-7 | Peak 0.758 |

V7-BoK and V15 have the same eff_intensity but different results:
- V7-BoK crashed (format collapse, gate=soft issue)
- V15 was stable but learned slowly
- The difference is NOT from eff_intensity but from other params (gate, fmt_rej, data, tau schedule)

### Finding 5: tau_init matters for BoK exploitation quality

| tau_init | Exps | BoK% @ start | Behavior |
|----------|------|--------------|----------|
| 0.8 | V7, V8 | ~15-20% | Very flat softmax → BoK degenerates to uniform |
| 0.7 | V11-V14 | ~20-25% | Moderate selectivity |
| 0.5 | V15 | ~25-30% | Good selectivity from start |

Higher tau_init → flatter softmax → BoK provides weaker signal. V15's tau_init=0.5 was a correct improvement.

## 4. Optimization Direction Analysis

### 4.1 What worked across experiments

1. **gate=off, format_rejection=0** (Phase 2+): Eliminated format collapse from V7-BoK
2. **Standard 0_10 data**: Higher ceiling than mixed/hard data
3. **NaN skip guard in dp_actor.py**: Enabled V12 to survive 273 NaN events and still learn
4. **tau_init=0.5** (V15): Better BoK selectivity from start
5. **easy_threshold=0.50** (V15): Catches more "easy" groups, reduces BoK computation waste

### 4.2 What didn't work

1. **mixed_hard_easy data**: Lower ceiling (0.762 vs 0.775)
2. **hard_only data**: Format collapse, too narrow distribution
3. **gate=soft**: Gradient bottleneck caused format collapse
4. **grad_norm=0.5** (V11-V13): Too aggressive gradient clipping (V11 crash, V12/V13 NaN storms)
5. **ppo_epochs=2 at low LR**: Returns diminish faster than expected

### 4.3 Optimal direction for V16

**Goal:** Exceed V12's 0.775 and approach V13's 0.788. Stable (zero NaN).

**Recommended configs (Priority order):**

#### Option A: "V7-GRPO Improved" (Conservative, proven baseline)
```
Algorithm: grpo (standard GRPO, not BoK!)
LR: 1e-6, ppo_epochs: 1, clip_high: 0.30
Data: standard 0_10
Steps: 200+
eff_intensity: 3.0e-7
Expected peak: ~0.773 (matches V7-GRPO)
Risk: Low
```
**Rationale:** V7-GRPO already reached 0.773 with zero NaN. Reproduce this as a ceiling benchmark before adding BoK complexity.

#### Option B: "V12 with NaN Guards" (Aggressive, highest potential)
```
Algorithm: bok_grpo
LR: 1e-6, ppo_epochs: 2, clip_high: 0.28
Data: standard 0_10
grad_norm: 1.0 (NOT 0.5)
tau: 0.5 → 0.35 (V15-style start, higher floor)
Steps: 178
NaN adaptive LR: enabled
eff_intensity: 5.6e-7
Expected peak: 0.775+
Risk: Medium (NaN possible but guarded)
```
**Key changes vs V12:**
- grad_norm=1.0 (V12 used 0.5 which may have caused more NaN)
- tau_init=0.5 instead of 0.7 (better BoK selectivity)
- tau_min=0.35 (prevent extreme softmax concentration)
- NaN adaptive LR from NaN report (halve LR on NaN, recover after 5 clean steps)

#### Option C: "Between V12 and V15" (Moderate risk)
```
Algorithm: bok_grpo
LR: 1e-6, ppo_epochs: 1, clip_high: 0.28
Data: standard 0_10
tau: 0.5 → 0.30
Steps: 178
eff_intensity: 2.8e-7
Expected peak: 0.770-0.775
Risk: Low
```
**Rationale:** V7-BoK at eff=2.8e-7 crashed due to gate=soft, NOT due to LR. With Phase 2 fixes (gate=off, fmt_rej=0), this should work cleanly and match V7-GRPO's 0.773.

### 4.4 Key decision: Standard 0_10 data vs mixed data

**Strongly recommend reverting to standard 0_10 data for V16:**
- Proven ceiling: 0.775-0.788
- Mixed data ceiling: limited at 0.762
- Hard-only data: too narrow, format collapse risk
- After establishing a strong V16 baseline, consider curriculum learning in V17

## 5. Experiment Genealogy

```
SFT-3537 (base, val=0.756)
├── V7-BoK (gate=soft) → crash@S92 ❌
├── V7-GRPO (standard) → 0.773@S140 ✅ [BEST STABLE]
├── V8-BoK (τ→0.5) → 0.760@S80 ⚠️
├── V11 (gate=off,ppo=2) → 0.764@S66, crash@S77 ❌
├── V12 (gate=off,ppo=2) → 0.775@S165 ⚠️ [BEST BoK FROM-SCRATCH]
│   └── V13 (resume V12) → 0.788@S180 ⚠️ [ABSOLUTE BEST, degraded later]
├── V14-hard (hard-only) → crash format ❌
├── V14-mixed (mixed data) → 0.762@S90 ⚠️
└── V15 (low LR, mixed) → 0.762@S90 ✅ [MOST STABLE, but ceiling limited]
```

## 6. Conclusions

1. **The primary bottleneck is data, not algorithm**: Standard 0_10 data ceiling (~0.775) > mixed data ceiling (~0.762)
2. **Standard GRPO is a strong baseline**: V7-GRPO (0.773) outperforms most BoK variants
3. **BoK-GRPO with correct settings can match/exceed GRPO**: V12 reached 0.775, but needs NaN management
4. **V16 should revert to standard 0_10 data** and either:
   a. Run Option C (bok_grpo, LR=1e-6, ppo=1) for safe 0.773+ target
   b. Run Option B (bok_grpo, LR=1e-6, ppo=2, NaN guards) for 0.775+ target
5. **The path to 0.79+ val** likely requires curriculum learning (V13 showed 0.788 through resume→continue) rather than single-run optimization

## 7. Deep Dive: Why Standard 0_10 Data Has Higher Ceiling

### 7.1 Dataset Distribution Comparison

| Dataset | Samples | Easy(1-3) | Medium(4-6) | Hard(7-10) |
|---------|---------|-----------|-------------|------------|
| **0_10 (standard)** | **11,455** | 30.2% | 40.8% | 28.9% |
| mixed_hard_easy | 5,792 | 17.0% | 31.6% | **51.5%** |
| hard_only | 1,808 | 17.9% | 40.2% | **41.9%** |

### 7.2 Why Easy Samples Help Despite AllCorr Filtering

**Misconception:** "All-correct samples waste compute, so removing them should help."

**Reality:** Easy samples serve 3 critical functions even when filtered:

1. **Not all easy samples are all-correct.** At SFT baseline pass@1=75.6%, even count=1 samples might get 12/16 or 14/16 correct. These "almost-all-correct" easy samples still provide gradient signal that reinforces basic counting patterns — a form of baseline maintenance.

2. **KL regularization anchor.** The AllCorr-filtered samples (advantage=0) still contribute to the KL divergence term in PPO loss. The model computes log_prob on these tokens → KL penalty constrains the policy from drifting too far from reference on easy patterns → prevents catastrophic forgetting.

3. **Batch diversity for BoK routing.** With 1024 samples per batch:
   - Standard 0_10: ~300 easy, ~400 medium, ~300 hard → diverse routing
   - Mixed: ~170 easy, ~320 medium, ~530 hard → hard-dominated gradients
   
   The mixed data's hard bias causes gradient concentration on hard patterns. Since hard counting (7-10 objects) requires different visual strategies than easy counting (1-3), gradient bias toward hard patterns can HURT generalization on the medium range (4-6) which is 40% of validation.

4. **Training data diversity = exploration diversity.** With 11,455 samples vs 5,792, the standard dataset exposes the model to 2× more unique images per epoch. Each unique image provides different visual features → richer representation learning.

### 7.3 Evidence: Medium-range counting is the ceiling

The val answer_reward measures accuracy across ALL count ranges. The model likely:
- Gets 1-3 correct ~90%+ (easy)
- Gets 7-10 correct ~50-60% (hard, limited ceiling due to visual complexity)
- Gets 4-6 correct ~70-80% (medium, most improvable range)

The mixed data's 31.6% medium coverage vs standard's 40.8% means **less training on the most improvable range**. This directly limits the val ceiling.

## 8. V12 NaN Phase Analysis: The "NaN Learning" Paradox

### 8.1 V12 Val vs NaN Timeline

```
V12 Val:                    NaN Status:
S0:   0.756                 Clean
S33:  0.752                 Clean (slow decline)
S66:  0.754                 Clean (flat)
          ← NaN onset at ~Step 83 →
S99:  0.752                 NaN (every step 83-178)
S132: 0.771 (+0.019!)       NaN (every step still skipping)
S165: 0.775 (+0.004!)       NaN (continuing)
S178: 0.775                 NaN (final)
```

### 8.2 The Paradox

V12's biggest improvement (+0.019, from 0.752→0.771) occurred **during the NaN phase** when gradients were supposedly being skipped.

**Possible explanations:**

1. **Partial FSDP shard updates:** With FULL_SHARD across 4 GPUs, each rank computes grad_norm independently. 3 skips per step (out of 4 ranks) means **1 rank may still update**. This creates an asymmetric model state where some parameter shards evolve while others freeze — a crude form of parameter noise injection.

2. **ppo_epoch selective survival:** With ppo_epochs=2, one epoch may produce NaN (skipped) while the other produces finite gradients (applied). The surviving epoch provides filtered, high-quality gradients.

3. **NaN skip as extreme gradient selection:** Only the "safest" gradient directions (those that don't overflow bf16) get applied. This naturally filters out the most extreme parameter updates, keeping only conservative improvements.

### 8.3 Key Implication for V16

V12's accidental discovery suggests that **aggressive learning + NaN-skip safety net** can outperform **conservative learning + zero NaN**:
- V12: eff=8.4e-7 + 273 NaN skips → val=0.775
- V15: eff=2.8e-7 + 0 NaN skips → val=0.762

The NaN skip mechanism effectively provides adaptive gradient clipping: when gradients are too large (NaN), skip entirely; when they're normal, apply fully. This is more aggressive than standard gradient clipping (which scales down but still applies).

## 9. NaN Risk Assessment: V15 Config + Standard 0_10 Data

### 9.1 Verdict: VERY LOW RISK (No NaN Expected)

| Factor | Assessment | Reasoning |
|--------|-----------|-----------|
| eff_intensity | 2.8e-7 | Safe boundary, proven by V7/V15 |
| gate mode | off | Eliminates V8's NaN source |
| Data variance | Lower than mixed | More easy samples = lower gradient variance |
| grad_norm | 1.0 | Generous ceiling, won't clip prematurely |
| tau range | 0.5→0.3 | V15 proven safe with same range |

### 9.2 Evidence

- V15 (eff=2.8e-7, mixed data, gate=off): **0 NaN** in 90 steps
- V7-BoK (eff=2.8e-7, 0_10 data, gate=soft): **0 NaN** in 92 steps (crashed for other reasons)
- V8 (eff=2.8e-7, 0_10 data, gate=soft): 72 NaN skips ← gate=soft caused this, NOT the data
- Standard 0_10 has MORE easy samples (30.2% vs 17%) → lower gradient variance → SAFER

### 9.3 Conclusion

V15 config (LR=5e-7, ppo=2, clip=0.28, gate=off) + standard 0_10 data → NaN probability < 1%.

However, the **expected ceiling is 0.762-0.770**, not 0.775:
- Same eff_intensity (2.8e-7) was too conservative to reach V12's 0.775
- Recommend: increase LR to 1e-6 with ppo=1 (→ same eff but more responsive per step) or ppo=2 (→ eff=5.6e-7, moderate risk)

## 10. Updated V16 Recommendations

### Priority 1: Revert to Standard 0_10 Data
This alone could raise ceiling from 0.762 to ~0.775.

### Priority 2: Optimal Learning Configuration

| Config | LR | ppo | eff | Expected Peak | NaN Risk | Recommendation |
|--------|-----|-----|------|-------------|----------|---------------|
| **V16-A** | 1e-6 | 1 | 2.8e-7 | 0.770-0.775 | ~0% | ✅ Safe bet |
| **V16-B** | 1e-6 | 2 | 5.6e-7 | 0.775-0.780 | ~5% | ⭐ Best risk/reward |
| **V16-C** | 1.5e-6 | 2 | 8.4e-7 | 0.775+ | ~30% | ⚠️ Only with NaN guards |

**Rationale for V16-B as top pick:**
- V14-hard at eff=5.6e-7 had 21 NaN skips — but used hard-only data (much higher variance)
- With balanced 0_10 data, eff=5.6e-7 should be safe
- ppo=2 provides second gradient pass for deeper exploitation
- Combines V15's improvements (gate=off, tau=0.5→0.3, easy_threshold=0.50) with V12's data

### Priority 3: Algorithm Code Optimizations (for future work)
1. **AllCorr recycling:** Instead of zeroing advantages, use AllCorr groups for KL-only regularization with explicit weight
2. **Adaptive easy_threshold:** Start at 0.75, decay to 0.50 based on training progress
3. **Curriculum scheduling:** After reaching 0.775, switch to hard-biased data for fine-grained improvement

## 11. Hard Data Evaluation: V12 vs V14-mixed (NEW)

Direct evaluation of RL models on the 1,808 SFT-incorrect hard samples reveals:

| Metric | V12 | V14-mixed |
|--------|-----|-----------|
| Accuracy | **31.3%** | 27.2% |
| Mean |error| | **1.36** | 1.98 |
| Round=2 pathology | 3.1% | **16.7%** |
| Large under-count (Δ≤-4) | 9 | **206** |

**Key Finding: V14's failure is caused by "lazy counting" reward hacking** — the model terminates 302 trajectories in just 2 rounds (1 point + answer "2"), achieving 0% accuracy on GT≥3 samples. This accounts for 92.7% of V14's large under-count errors.

When excluding the round=2 pathology, V14's error patterns become nearly identical to V12's (±1 errors: 79.8% vs 80.7%, Mean|error|: 1.38 vs 1.36), confirming the early termination is the primary failure mode, not inferior counting ability.

**Hard-data paradox confirmed: training on hard-biased data teaches the model to give up, not to try harder.**

Full analysis: `docs/hard_data_eval_analysis_report.md`
