# V16 Complete Training Analysis Report
> BOK-GRPO with KL Penalty | 90/90 Steps | Mixed Data  
> Generated: 2025-03-25

## 1. Experiment Configuration

| Parameter | Value |
|-----------|-------|
| LR | 1e-6 |
| ppo_epochs | 2 |
| clip_ratio_high | 0.28 |
| grad_clip_norm | 1.0 |
| KL_coef | **0.03** |
| τ schedule | 0.5→0.3 cosine |
| BOK_CLIP | 4.0 |
| BOK_EASY_THRESHOLD | 0.50 |
| Data | Mixed (5792 samples) |
| Total steps | 90 |
| val_freq | 15 |
| save_freq | 30 |
| eff_intensity | 5.6e-7 |

## 2. Training Score Trajectory (batch_mean)

### Phase Summary
| Phase | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| Phase 1 (S1-30) | 0.7309 | 0.0308 | 0.6646 | 0.7829 |
| Phase 2 (S31-60) | 0.7420 | 0.0275 | 0.6882 | 0.7931 |
| Phase 3 (S61-90) | **0.7576** | 0.0298 | 0.6957 | **0.8143** |

### 10-Step Rolling Average
| Window | Avg Score | Trend |
|--------|-----------|-------|
| S1-S10 | 0.7461 | baseline |
| S11-S20 | 0.7320 | ↓ dip |
| S21-S30 | 0.7145 | ↓ trough |
| S31-S40 | 0.7322 | ↑ recovery |
| S41-S50 | 0.7454 | ↑ near baseline |
| S51-S60 | 0.7484 | ↑ slight gain |
| S61-S70 | **0.7621** | ↑ **peak window** |
| S71-S80 | **0.7649** | ↑ **highest** |
| S81-S90 | 0.7458 | ↓ decline |

**Observation**: Training score follows U-shape → peak → decline. Phase 3 is overall strongest but collapses in S81-90.

## 3. Validation Trajectory (Critical)

| Step | answer_reward | Δ baseline | format_reward | point_reward |
|------|--------------|------------|---------------|-------------|
| S0 (baseline) | 0.756 | — | 0.962 | 0.919 |
| S15 | 0.754 | -0.002 | 0.966 | 0.923 |
| S30 | 0.750 | -0.006 | 0.966 | 0.924 |
| S45 | 0.752 | -0.004 | 0.970 | 0.927 |
| **S60** | **0.762** | **+0.006** | 0.972 | 0.929 |
| S75 | 0.758 | +0.002 | **0.979** | **0.935** |
| S90 | 0.754 | -0.002 | 0.974 | 0.932 |

### Key Findings:
1. **Best checkpoint: S60** (answer=0.762, +0.6% over baseline)
2. **S90 is BELOW baseline** (0.754 vs 0.756)
3. Format/Point rewards **monotonically improve** throughout training
4. Answer reward follows inverted-U: rise S30→S60, then decline S60→S90
5. **Only S90 checkpoint saved** — optimal S60 checkpoint was NOT saved (save_freq=30 → saved S30, S60 overlap with val, but actual save was S90 only!)

### Saved Checkpoint Status:
```
save/StepCount-7B-SFT-30k_v16_.../global_step_90/  ← ONLY this checkpoint
```
⚠️ **CRITICAL**: save_freq=30 should have saved S30, S60, S90, but only S90 remains. If S30 and S60 were overwritten/cleaned, the optimal S60 checkpoint is LOST.

## 4. KL Divergence Analysis

### Phase-wise KL Growth
| Phase | Avg KL | Max KL | Spikes >0.1 | Growth vs P1 |
|-------|--------|--------|-------------|-------------|
| Phase 1 (S1-30) | 0.0155 | 0.052 | 0 | 1.0x |
| Phase 2 (S31-60) | 0.0423 | 0.162 | 1 | **2.7x** |
| Phase 3 (S61-90) | 0.0828 | **0.205** | **10** | **5.3x** |

### All KL Spikes >0.1
| Step | KL Value | Phase |
|------|----------|-------|
| S53 | 0.121 | P2 |
| S56 | **0.162** | P2 |
| S62 | 0.111 | P3 |
| S65 | 0.133 | P3 |
| S72 | 0.102 | P3 |
| S74 | 0.109 | P3 |
| S79 | **0.205** ← max | P3 |
| S81 | 0.111 | P3 |
| S82 | **0.183** | P3 |
| S86 | **0.164** | P3 |
| S89 | 0.122 | P3 |
| S90 | 0.132 | P3 |

**Diagnosis**: KL divergence grows **exponentially** throughout training. Phase 3 has **10 spikes >0.1**, with maximum 0.205 at S79. The KL_coef=0.03 penalty is clearly **insufficient** to contain policy drift at this learning rate and training length.

## 5. Routing Distribution Evolution

| Phase | EasyDrGRPO | AllCorrect | Interpretation |
|-------|-----------|------------|---------------|
| Phase 1 | 483/1024 (47.1%) | 161/1024 (15.7%) | Normal distribution |
| Phase 2 | 451/1024 (44.1%) | 209/1024 (20.4%) | AllCorrect ↑ (+4.7%) |
| Phase 3 | 449/1024 (43.8%) | 222/1024 (21.7%) | AllCorrect ↑↑ (+6.0%) |

**Observation**: AllCorrect fraction grows steadily (15.7% → 21.7%), indicating the model learns to solve more problems fully correctly. EasyDrGRPO slightly decreases, suggesting fewer "easy-medium" solutions and more polarization (either fully correct or hard-fail).

## 6. Comparison with Historical Experiments

| Experiment | Internal Val | Benchmark (pixmo-test) | KL Stability |
|-----------|-------------|----------------------|-------------|
| SFT Baseline | 0.756 | 75.6% | N/A |
| V12 S178 | — | **82.04%** | Moderate |
| V14-mixed S90 | — | 80.34% | Unstable (no KL penalty) |
| V15 S90 | — | 80.15% | Stable |
| **V16 S60** | **0.762** | **TBD** | Moderate |
| **V16 S90** | 0.754 | **TBD** | **Unstable** |

## 7. Root Cause Analysis

### Why V16 Peaks at S60 Then Declines:

1. **KL penalty creates a "sweet spot" window**: Around S50-65, the model has learned enough from rewards while policy hasn't drifted too far. KL penalty successfully regularizes.

2. **After S65, KL penalty loses effectiveness**: Policy drift accelerates (5.3x Phase 1), creating a positive feedback loop — larger KL → larger policy updates → even larger KL.

3. **adv_mean consistently negative**: KL penalty biases advantages negative, meaning the model is penalized even for good trajectories. This erodes good behavior learned earlier.

4. **tau_bumped at S76+**: τ hitting adaptive floor means the temperature schedule has exhausted its range, removing one regularization mechanism.

### Fundamental Issue:
KL_coef=0.03 is **too weak for 90 steps at LR=1e-6**. It works for ~60 steps, then the cumulative policy drift overwhelms the penalty.

## 8. Verdict & Recommendations

### Rating: 🟡 MARGINAL

**Positive signals:**
- Format/point rewards improve monotonically → structural learning works
- AllCorrect fraction grows → model learns to solve more problems fully
- S60 val=0.762 > baseline 0.756 → KL penalty CAN help when properly timed
- Training score (batch_mean) highest in Phase 3 → model capacity not saturated

**Negative signals:**
- S90 val=0.754 < baseline 0.756 → final model slightly WORSE
- KL instability in Phase 3 → 10 spikes >0.1, max 0.205
- Only S90 checkpoint saved → optimal S60 lost
- adv_mean consistently negative → KL penalty over-regularizes

### Actionable Recommendations:

1. **Run S90 checkpoint through benchmark eval** to get actual pixmo-test score. Internal val is not directly comparable with V12's benchmark score.

2. **For V17, try stronger KL control:**
   - Option A: Early stop at 60 steps with KL_coef=0.03
   - Option B: Increase KL_coef to 0.05-0.08 for full 90 steps
   - Option C: Adaptive KL coefficient — increase KL_coef when KL > threshold (e.g., double coef when KL > 0.08)

3. **Save more checkpoints**: Change save_freq=15 to match val_freq, ensuring the best validation checkpoint is preserved.

4. **Consider KL target approach**: Instead of fixed KL penalty, use KL target (e.g., target_kl=0.05) with adaptive coefficient.

5. **V12 remains BEST**: V12 S178 at 82.04% benchmark is still unbeaten. V16's contribution is showing that KL penalty CAN help in the 30-60 step window.

---
*Report generated from full V16 training log (132,195 lines, 90/90 steps)*
