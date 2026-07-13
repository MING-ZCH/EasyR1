# V14-Hard-Only Experiment — Deep Analysis Report

> Experiment: `training_interleaved_traj_v14_hard_only_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej0_bok_grpo_20260318_162557`
> Duration: 2026-03-18 16:26 → 2026-03-19 02:54 (~10.5 hours)
> Model: StepCount-7B-SFT-30k-high/checkpoint-3537 (SFT base)
> Algorithm: BoK-GRPO, LR=1e-6, τ annealing 0.7→0.3 (cosine), 4 GPUs
> Dataset: 1808 hard-only samples, ROLLOUT_N=16, global_batch=64
> Total Steps: 32 completed (1808/64 ≈ 28 steps/epoch, but log shows 32 steps = ~1.14 epochs)

---

## 1. Val Score Timeline (pixmo-test, 529 samples)

| Step | Val reward_score | Δ from SFT | Trend |
|------|-----------------|------------|-------|
| **0** (pre-train) | **0.756** | baseline | — |
| **10** | **0.739** | -1.7pp | ↓ declining |
| **20** | **0.732** | -2.4pp | ↓ continued decline |
| **30** | **0.749** | -0.7pp | ↑ partial recovery |

### Key Observation
- **Val NEVER exceeded the SFT baseline (0.756)**
- Pattern: initial decline → partial recovery (common in RL; model explores before exploiting)
- Best during training: Step 30 (0.749), still below SFT
- The checkpoint saved at global_step_28 is the only saved checkpoint

### Val Detailed Metrics (Step 0 → Step 30)
| Metric | Step 0 | Step 10 | Step 20 | Step 30 |
|--------|--------|---------|---------|---------|
| answer_reward | 0.756 | 0.739 | 0.732 | 0.749 |
| point_reward | 0.919 | — | — | — |
| format_reward | 0.962 | — | — | — |
| consistency_violation | 0.072 | — | — | — |
| format_fail_rate | 0.038 | — | — | — |

---

## 2. Training Reward Statistics (Train Worker pid=2285018)

### Answer Mean Evolution
| Phase | Calls | answer_mean range | overall_mean range | Trend |
|-------|-------|-------------------|-------------------|-------|
| Early (1-10) | 1-10 | 0.347-0.428 | 0.488-0.535 | Flat oscillation |
| Mid (11-20) | 11-20 | 0.318-0.451 | 0.467-0.563 | Slight uptick |
| Late (21-32) | 21-32 | 0.349-0.434 | 0.486-0.548 | Flat / slight decline |

### Key Training Reward Numbers
- **answer_mean**: oscillates 0.32-0.45 (never sustained improvement)
- **point_mean**: oscillates 0.57-0.67 (relatively stable)
- **format_fail_rate**: 3-11% (non-trivial; these trajectories get 0 reward)
- **turns_exceeded_rate**: 3-10% (same as format_fail — the main format failure IS turns_exceeded)

### Training vs Val Gap
- **Train answer_mean** ≈ 0.37 (hard data, GT=5-10 dominant)
- **Val answer_mean** ≈ 0.74 (pixmo-test, includes easy GT=1-4)
- Gap = 0.37 → hard data is genuinely difficult; model struggles to improve on it

---

## 3. BoK-GRPO Algorithm Diagnostics

### τ Annealing Progress
| Step | τ | Progress | Phase |
|------|---|----------|-------|
| 1 | 0.700 | 0.9% | Max exploration |
| 10 | 0.692 | 8.9% | Early phase |
| 20 | 0.669 | 17.9% | Still high exploration |
| 30 | 0.633 | 26.8% | Moderate |
| 32 | 0.625 | 28.6% | Training ended here |

**Note**: τ only annealed from 0.700 → 0.625 (10.7% of planned 0.7→0.3 range). Training ended at step 32/112, meaning only 28.6% of the planned annealing schedule was completed. This means the BoK selectivity never fully kicked in.

### Group Routing Statistics
| Metric | Min | Max | Typical | Interpretation |
|--------|-----|-----|---------|---------------|
| easy_drgrpo | 32/1024 | 176/1024 | ~96-128 | 3-17% groups classified easy → DrGRPO |
| all_correct_filtered | 0/1024 | 64/1024 | ~16-32 | 0-6% trivially solved (zeroed out) |
| low_var | 0/1024 | 64/1024 | ~16-32 | 0-6% low-variance groups → fallback |
| collapsed | 0 | 10 | ~5-7 | <1% degenerate weights |
| tau_bumped | 0/1024 | 0/1024 | 0 | Never triggered (τ=0.625 >> τ_min=0.08) |
| n_zero_std | 0-4/64 | — | ~0-1 | Rare all-same-reward groups |

### Advantage Statistics
| Metric | Typical Range | Interpretation |
|--------|--------------|---------------|
| adv_mean | -0.17 to -0.06 | Consistently negative (more wrong than right) |
| adv_std | 0.93-1.11 | Well-calibrated around 1.0 |
| adv_range | [-2.500, 3.000] | Hitting both clip boundaries |
| batch_mean (reward) | 0.47-0.56 | Oscillating, no trend |
| batch_std (reward) | 0.29-0.34 | Moderate variance |

### Diagnosis
- BoK-GRPO algorithm is **mechanically healthy**: no collapse, no degeneration, well-calibrated advantages
- **Problem is upstream**: the model can't improve answer accuracy on hard data → reward signal stays flat → no learning signal for BoK to exploit
- τ annealing barely progressed (28.6%) — if training continued, selectivity would increase, but the underlying answer accuracy needs to improve first

---

## 4. Training Loss & PPO Metrics

### Loss Timeline (31 steps)
| Phase | pg_loss | kl_loss | entropy_loss | grad_norm |
|-------|---------|---------|-------------|-----------|
| Step 0 | 0.083 | 0.003 | 0.567 | 2.957 |
| Step 5 | 0.097 | 0.014 | 0.578 | 1.149 |
| Step 10 | 0.105 | 0.051 | 0.603 | 1.259 |
| Step 15 | 0.106 | 0.103 | 0.597 | 1.579 |
| Step 20 | 0.062 | 0.141 | 0.622 | 3.304 |
| Step 25 | 0.051 | 0.138 | 0.663 | 2.324 |
| Step 30 | 0.098 | 0.224 | 0.708 | 4.102 |

### Key Trends
1. **KL divergence is exploding**: 0.003 → 0.224 (**75× increase** in 31 steps)
   - This means the model is rapidly drifting from the SFT reference
   - KL_coef = 2e-2, so KL penalty in loss = 2e-2 × 0.224 = 0.00448 (still small vs pg_loss ~0.1)
   - **Problem**: KL penalty is too weak to prevent drift

2. **Entropy is rising**: 0.567 → 0.708 (**+25%**)
   - Model is becoming MORE uncertain, not more confident
   - This is the opposite of what successful RL training looks like
   - Indicates the model is "exploring" without finding useful patterns

3. **pg_loss is oscillating**: 0.05-0.16, no clear trend
   - This is expected for RL with flat reward signal
   - The model is being updated but not consistently improving

4. **grad_norm is volatile**: 1.1-6.3
   - Some spikes (Step 16: 6.323, Step 28: 4.641) but max_grad_norm=1.0 should clip these
   - No NaN events — LR=1e-6 is stable

---

## 5. NaN / Error Analysis

**Zero NaN/Inf events** throughout the entire 32-step training run.
- LR=1e-6 + BOK_CLIP=3.0 + max_grad_norm=1.0 combination is confirmed stable
- V14's NaN fix (reducing LR from 1.5e-6) is validated

---

## 6. Sample Output Analysis (Val Samples at Step 30)

### Failed Sample: GT=6 chairs, Predicted=4
**Error Pattern**: The model only pointed 4 chairs in the first turn, then immediately went to `<think>...<answer>4</answer>`. It failed to detect 2 occluded/partially visible chairs.

**Root Cause**: **Visual perception error** — not an RL training issue. The model genuinely can't see the remaining chairs. This aligns with SFT baseline already being 0.756 (not perfect).

### Successful Sample: GT=10 people (Mao portraits), Predicted=10
The model correctly pointed all 10 portraits in a single turn, then verified in reasoning. Grid-like arrangement → easy for the model.

### Failure Mode Taxonomy (from format_fail analysis)
- **Primary failure**: `turns_exceeded` (model keeps pointing without stopping → exceeds max 11 turns)
- **Rate**: 3-10% of training samples, consistent throughout
- This is the "over-pointing" problem: model doesn't learn when to stop pointing

---

## 7. Comprehensive Diagnosis

### Why V14-Hard-Only Failed

**Root Cause: Insufficient Learning Signal on Hard Data**

1. **Hard data pass rate is too low**: Train answer_mean ≈ 0.37 means only ~37% of trajectories get correct answers. With 16 rollouts per prompt, that's ~6 correct out of 16 on average.

2. **High variance in hard data rewards**: batch_std=0.29-0.34 seems reasonable, but the low pass rate means most "learning" comes from comparing wrong answers of different severity, not from contrasting correct vs wrong.

3. **KL divergence explosion** (0.003 → 0.224): The BoK-GRPO advantage signal pushes the model away from SFT, but the reward signal isn't good enough to guide it to a better place. Result: the model drifts into a worse region of policy space.

4. **Entropy increase** (0.567 → 0.708): Confirms the model is becoming less certain. RL exploration is not finding reward-improving behaviors.

5. **τ annealing incomplete** (28.6% progress): Training ended before BoK selectivity could fully kick in. Even if it had, the underlying problem (low pass rate on hard data) would persist.

### What Worked
- ✅ NaN stability: LR=1e-6 completely eliminates NaN
- ✅ BoK-GRPO mechanics: algorithm runs correctly, no degeneration
- ✅ Format compliance: ~95% of trajectories complete properly
- ✅ Point accuracy: point_mean stays ~0.6 on hard data (decent for hard GT=5-10)

### What Didn't Work
- ❌ Val score declined from SFT baseline (0.756 → 0.732 at worst)
- ❌ Training reward shows no sustained improvement
- ❌ KL divergence explodes without proportional reward improvement
- ❌ Model becomes less certain (entropy rises) instead of more focused

### Comparison to V12 (Best Experiment, val=0.771)
| Metric | V12 | V14-hard-only | Interpretation |
|--------|-----|---------------|---------------|
| Dataset | 3607 full (easy+hard) | 1808 hard-only | Half the data, all hard |
| Steps completed | ~132 (2+ epochs) | 32 (~1.1 epoch) | Much less training |
| Val best | 0.771 | 0.749 | V12 much better |
| easy_drgrpo rate | 41% | 3-17% | V12 has more "easy wins" |
| all_correct_filtered | 28% | 0-6% | V12 filters more trivial groups |
| Train answer_mean | ~0.55 | ~0.37 | V12 gets more correct trajectories |

**Key Insight**: V12 succeeded because easy data provided strong positive gradient signal (high pass rate → clear correct/wrong contrast). Hard-only data doesn't have this — pass rate is too low for effective BoK-GRPO learning.

---

## 8. Recommendations for Next Steps

### Immediate: Start V14-mixed Training
The mixed dataset (5792 samples = 1808 hard + 384 eval_extra + 3600 easy) directly addresses the core problem:
- Easy data provides reliable positive learning signal (pass rate ~80-95% for GT=1-4)
- Hard data oversampling ensures the model also sees difficult cases
- Expected train answer_mean: ~0.55-0.60 (similar to V12)

### Key Parameters to Adjust from V14-hard-only Evidence
1. **KL management**: Consider lowering kl_coef from 0.02 → 0.01, or implement adaptive KL. KL=0.224 at step 30 is too high — model drifts without benefit.
2. **BOK_TOTAL_STEPS=182**: Correct for 2 epochs of mixed data. τ will reach ~0.3 by end.
3. **Save more frequently**: save_freq=30 gives ~6 checkpoints. Consider save_freq=20 for finer granularity since val is non-monotonic.
4. **Monitor format_fail_rate**: If >5% sustained, the "over-pointing" problem needs addressing (could add explicit stopping reward or reduce max_turns penalty).

### Medium-term
- **P1 (Remove *K scaling)**: The adv_range always hits [-2.500, 3.000] clip bounds. Removing *K scaling would produce more reasonable advantage magnitudes, potentially reducing KL drift.
- Consider **early stopping** with patience=3 validation checks if val keeps declining.
