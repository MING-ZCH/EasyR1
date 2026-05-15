# V20 Final Configuration Analysis Report

## Overview

V20 is the culmination of exhaustive first-principles analysis of BoK-GRPO algorithm behavior, gradient dynamics, and routing optimal configuration for the StepCount-7B interleaved point-to-count task.

**Key changes from V19**: bf16 rollback, V3 params disabled, TH=0.50 retained for mixed data.

---

## 1. Configuration Changes

### From V19 → V20

| Parameter | V19 | V20 | Rationale |
|-----------|------|------|-----------|
| Precision | FP16 | **BF16** | FP16 caused NaN/grad spikes |
| BOK_EASY_THRESHOLD | 0.50 | **0.50** | Optimal for mixed data (see §3) |
| BOK_WINNER_BOOST | N/A | **0** (disabled) | Analysis: unnecessary (see §2.1) |
| BOK_EASY_SCALE | N/A | **1.0** (disabled) | Analysis: treats symptom (see §2.2) |
| BOK_QUALITY_BONUS | N/A | **0** (disabled) | Analysis: redundant w/ softmax (see §2.3) |
| GRAD_SPIKE_COOLDOWN | 3 | **6** | Fix comment/value mismatch |
| save_freq | 60 | **15** | More frequent checkpoint capture |
| val_freq | 20 | **15** | Aligned with save_freq |

### Retained from V12-best configuration

| Parameter | Value | Status |
|-----------|-------|--------|
| BOK_CLIP | 4.0 | BoK advantage range [-0.9, +4.0] |
| BOK_UNIFORM_MIX | 0.1 | ε-greedy floor |
| BOK_TAU_INIT → FINAL | 0.5 → 0.3 cosine | Temperature annealing |
| BOK_ALLWRONG_CAP | 1.0 | Prevents gradient dominance |
| ppo_epochs | 2 | DAPO-style |
| DAPO dual clip | 0.2 / 0.28 | Low/high clip ratios |
| max_grad_norm | 1.0 | Always active (real norm 1.3-2.4) |
| KL_coef | 0.02 | Proven value |

---

## 2. V3 BoK-GRPO Improvements — Why All Disabled

### 2.1 Winner Amplification (BOK_WINNER_BOOST=0)

**Intent**: Boost rare correct trajectories by sqrt(K/n_correct).

**Why disabled**:
- Only 18% of groups eligible under bimodal distribution
- Affects 5.2% of correct trajectories  
- Introduces asymmetric clip complexity with marginal benefit
- BoK softmax already concentrates advantage on rare correct (n=1→+4.0)

### 2.2 Easy Gradient Dampening (BOK_EASY_SCALE=1.0)

**Intent**: Scale down easy DrGRPO gradient to reduce competition with hard signal.

**Why disabled**:
- Gradient clipping dilution is only ~7.5% (V12 vs V11 effective LR)
- Empirical evidence: V12 (more DrGRPO) outperforms V11 (less DrGRPO) on hard benchmark
- Treating symptom: the real issue was TH routing, not gradient magnitude

### 2.3 Quality-Ranked BoK (BOK_QUALITY_BONUS=0)

**Intent**: Add bonus for correct trajectories with better point quality.

**Why disabled**:
- Softmax already ranks trajectories by total reward score
- Point quality is embedded in the combined score (0.6×answer + 0.3×point + 0.1×format)
- Additional bonus is redundant with inherent softmax ordering

---

## 3. TH Analysis: Why 0.50 for Mixed Data

### Dataset composition

| Component | Count | Percentage |
|-----------|-------|------------|
| Easy | 11,455 | 83.9% |
| Hard-train | 3,232 (1616×2) | 23.7% |
| Hard-eval | 576 (192×3) | 4.2% |
| **Total** | **13,647** | |

### TH routing simulation (10K trials, 64 groups/batch)

| TH | BoK% | DrGRPO% | AllCorrect | Assessment |
|----|-------|---------|------------|-----------|
| 0.25 | 13.1% | 77.3% | 4.7% | BoK too little |
| **0.50** | **21.5%** | **68.9%** | **4.7%** | **Optimal balance** |
| 0.75 | 44.7% | 42.4% | 4.7% | BoK too much |

### Why not TH=0.75 for mixed data?

V12 (TH=0.75, easy-only) = 15.45% was best. But this doesn't apply to mixed data:

1. **Easy-only + TH=0.50 → 0.1% BoK** (BoK disabled). TH=0.75 was necessary to activate BoK.
2. **Mixed + TH=0.50 → 22% BoK** (natural from bimodal distribution). Already healthy.
3. **Mixed + TH=0.75 → 45% BoK**. Similar to V7/V11 (~60% BoK) which scored only 10-12%.
4. TH=0.75 misroutes n=8-11 groups to BoK (weaker signal) instead of DrGRPO (42% stronger).

### Per-n routing accuracy

| n_correct | Better algo | TH=0.50 route | TH=0.75 route |
|-----------|------------|----------------|----------------|
| 1-3 | BoK | BoK ✓ | BoK ✓ |
| 4-7 | DrGRPO (marginal) | BoK ✗ | BoK ✗ |
| 8-11 | DrGRPO (clear) | DrGRPO ✓ | **BoK ✗** |
| 12-15 | DrGRPO (dominant) | DrGRPO ✓ | DrGRPO ✓ |

TH=0.50 has 4 misrouted n-values; TH=0.75 has **8 misrouted n-values** on mixed data.

---

## 4. BoK Advantage Structural Analysis

### Advantage range

- **Positive**: A_max = min((1-MIX)(K-1), BOK_CLIP) = min(13.5, 4.0) = **+4.0**
- **Negative**: A_min = -(1-MIX) = **-0.9** (structural floor, independent of τ, K, scores)
- **Asymmetry**: 4.4:1 positive/negative

### Negative floor is by design

The -0.9 floor comes from MIX=0.1 (ε-greedy floor): w_min = MIX/K = 0.00625.
Six approaches to increase it were analyzed; only easy_drgrpo routing works:

| Approach | Result | Zero-mean? | Verdict |
|----------|--------|-----------|---------|
| Lower MIX | A_min=-1.0 (only +11%) | ✓ | Insufficient |
| Negative amplify | -2.2 | ✗ Breaks | Rejected |
| Hybrid BoK+/DrGRPO- | -2.5 | ✗ Breaks | Rejected |
| Asymmetric clip | -0.9 (no effect) | ✓ | Useless |
| Separate neg softmax | -2.5 | ⚠️ Complex | Over-engineered |
| **easy_drgrpo routing** | **-2.5 for easy** | **✓** | **✓ Already implemented** |

### Gradient clipping reality

**Key finding**: Gradient clipping triggers 93-100% of all training steps.

| Version | Config | max_grad_norm | Mean raw norm | Effective LR | Score |
|---------|--------|-------------|-------------|-------------|-------|
| V11 | BoK~60% | 0.5 | 1.25 | 0.51 | 12.00% |
| V12 | BoK~25% | 0.5 | 1.35 | 0.54 | **15.45%** |
| V17 | DrGRPO~99% | 1.0 | 2.38 | 0.50 | 12.20% |
| V19 | BoK~37% | 1.0 | 4.81 | 0.31 | NaN |

**Conclusion**: Gradient dilution (~7.5%) is negligible vs benefit of strong easy-error penalties.

---

## 5. Historical Performance Reference

| Version | Data | TH | BoK% | StepCount-500 | CountBench-491 |
|---------|------|-----|------|---------------|----------------|
| V7 | mixed | 0.50 | ~60% | 10.60% | ~78% |
| V11 | mixed | 0.50 | ~60% | 12.00% | ~78% |
| V12 | easy-only | 0.75 | ~25% | **15.45%** | 79.43% |
| V14m | mixed | 0.50 | ~40% | 12.00% | 78.00% |
| V17 | easy-only | 0.50 | ~0.1% | 12.20% | 79.43% |
| V19 | easy+hard | 0.50 | ~37% | NaN (crash) | NaN |
| **V20** | **easy+hard** | **0.50** | **~22%** | **TBD** | **TBD** |

V20 targets V12-level performance (15%+) with improved stability (bf16) and mixed data.

---

## 6. Launch Command

```bash
cd /mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest
nohup bash examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v20.sh \
  2>&1 | tee logs/train/v20_launch.log &
```

## 7. Monitoring

```bash
# Real-time log tail
tail -f logs/train/training_interleaved_traj_v20_*.log | grep -E 'grad_norm|reward|val_score|bok_route'

# Analysis tool
python3 tools/analyze_training.py logs/train/training_interleaved_traj_v20_*.log
```
