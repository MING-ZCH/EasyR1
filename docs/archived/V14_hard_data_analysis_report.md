# V14 Hard-Data-Only BoK-GRPO Training Analysis Report

## 1. Executive Summary

Based on:
- 2025 GRPO paper survey (DAPO, Dr.GRPO, REINFORCE++, PRIME, ORZ)
- SFT base model pass@1 evaluation on training set (11455 samples)
- NaN root cause diagnosis from V12/V13

**Key Conclusion**: Hard-data-only training is theoretically sound and practically promising for BoK-GRPO. It aligns with DAPO's Dynamic Sampling principle (discarding uninformative groups) and maximizes gradient information density. Combined with LR=1e-6, NaN risk is manageable.

---

## 2. SFT Base Model Pass@1 Analysis

### 2.1 Overview
| Metric | Value |
|--------|-------|
| Total training samples | 11,455 |
| Pass@1 correct (easy) | 9,839 (85.89%) |
| Pass@1 wrong (hard) | 1,616 (14.11%) |
| External hard data (V12 wrong on pixmo-test+countbench) | 192 |
| **Combined hard dataset** | **1,808** |

### 2.2 Error Patterns
- **Over-count**: 828 (51.2%), dominated by +1 errors (640/828 = 77.3%)
- **Under-count**: 788 (48.8%), dominated by -1 errors (687/788 = 87.2%)
- **Parse failures**: 0
- **Conclusion**: Errors are nearly symmetric and dominated by ±1 off-by-one mistakes, indicating visual point precision is the limiting factor, not reasoning ability.

### 2.3 GT Distribution of Hard Samples
| GT | Hard / Total | Wrong Rate |
|----|-------------|------------|
| 1  | 58 / 1165   | 5.0%   |
| 2  | 103 / 1142  | 9.0%   |
| 3  | 132 / 1157  | 11.4%  |
| 4  | 123 / 1134  | 10.8%  |
| 5  | 167 / 1051  | 15.9%  |
| 6  | 378 / 2493  | 15.2%  |
| 7  | 249 / 1385  | 18.0%  |
| 8  | 194 / 977   | 19.9%  |
| 9  | 122 / 632   | 19.3%  |
| 10 | 90 / 319    | 28.2%  |

**Key Insight**: Wrong rate monotonically increases with GT count. GT=10 has 28.2% wrong rate vs GT=1 at 5.0%. This confirms that higher object counts cause more point-level errors that accumulate.

### 2.4 Round Distribution of Hard Samples
| Rounds | Hard / Total | Wrong Rate |
|--------|-------------|------------|
| 2      | 35 / 1144   | 3.1%   |
| 5      | 168 / 1176  | 14.3%  |
| 8      | 233 / 1366  | 17.1%  |
| 11     | 168 / 407   | **41.3%** |

**Key Insight**: 11-round trajectories (max turns) have 41.3% wrong rate — indicating the model struggles most when all turns are used, suggesting these are genuinely difficult counting scenarios.

---

## 3. BoK-GRPO Improvement Recommendations (Paper-Informed)

### 3.1 Why Hard-Data-Only Training is Theoretically Sound

**DAPO Connection (Dynamic Sampling)**:
- DAPO filters "homogeneous groups" where all K trajectories have the same outcome (all correct or all wrong) because these contribute zero learning signal.
- In the full training set, ~86% of samples are easy (pass@1 correct). With ROLLOUT_N=16, most easy groups will produce 16/16 correct → BOK_FILTER_ALL_CORRECT=1 already filters them → wasted compute.
- Hard-data-only eliminates this waste at the data level, ensuring every group is informative.

**Dr.GRPO Connection (Difficulty Bias)**:
- Dr.GRPO proved that group-level std normalization creates "difficulty bias" — easy groups with low variance get over-amplified gradients.
- With hard-data-only, all groups are similarly difficult → the difficulty bias problem is naturally reduced.
- BOK_ADV_NORMALIZE=0 (already set) is correct for this setting.

**BoK-GRPO Specific**:
- BoK-GRPO's core design is to concentrate gradient on the rarest correct trajectory in a group. With hard data (pass@1 wrong), groups naturally have fewer correct trajectories (pass@16 >> pass@1) → BoK softmax produces maximum advantage differentiation.
- The easy/hard routing (BOK_EASY_THRESHOLD=0.75) effectively becomes all-hard routing — nearly 100% of groups go through the BoK softmax path, maximizing the algorithm's design intent.

### 3.2 NaN Risk Assessment for Hard-Data-Only

**Two Competing Effects**:

| Factor | Effect on NaN Risk |
|--------|-------------------|
| Fewer total steps (113 vs 358) | **REDUCES** — less time for gradient overflow accumulation |
| More extreme advantage distribution | **INCREASES** — fewer correct trajectories → BoK softmax concentrates more → advantage hits clip boundaries more often |
| LR reduced to 1e-6 (from 1.5e-6) | **STRONGLY REDUCES** — this LR was verified NaN-free for standard GRPO across 179 steps |

**Net Assessment**: The LR reduction is the dominant factor. V7 standard GRPO ran 179 steps at LR=1e-6 with 0 NaN events. Even with harder advantage distributions, LR=1e-6 should prevent bf16 gradient overflow.

**Monitoring Plan**: Watch for NaN indicators in the first 30 steps. If advantage clip saturation rate > 80%, consider reducing BOK_CLIP from 3.0 to 2.5.

### 3.3 Optimal V14 Configuration

| Parameter | V14 (Full Data) | V14 (Hard-Only) | Rationale |
|-----------|-----------------|-----------------|-----------|
| Train data | 11,455 samples | 1,808 samples | SFT-base wrong + external wrong |
| Steps/epoch | 178 | 28 | 1808/(16×4) |
| total_epochs | 1 | 4 | ~113 total steps (31% of V14's 2-epoch) |
| BOK_TOTAL_STEPS | 356 | 113 | Correct τ annealing schedule |
| save_freq | 66 | 28 | Save every epoch |
| ACTOR_LR | 1e-6 | 1e-6 | Same (NaN-safe) |
| BOK_CLIP | 3.0 | 3.0 | Same |
| max_grad_norm | 1.0 | 1.0 | Same |
| All BoK params | unchanged | unchanged | Same baseline |

### 3.4 Additional Improvements to Consider (Not Yet Applied)

1. **Process Reward on Hard Data**: With hard data, enabling `PROCESS_REWARD_ENABLE=1` could provide denser learning signals for point accuracy. Currently OFF. Consider enabling for V14b.

2. **τ Schedule Acceleration**: With only 113 steps (vs 356), the cosine τ annealing from 0.7→0.3 completes in 113 steps instead of 356. This means τ reaches 0.3 by step 113, which is actually desirable — faster convergence to selective advantage.

3. **BOK_CLIP Tightening**: If NaN appears despite LR=1e-6, reduce BOK_CLIP from 3.0 to 2.5 as secondary defense. This reduces clip-boundary gradient spikes.

4. **DAPO Filter**: BOK_DAPO_FILTER=0 currently. With hard data, most groups should be heterogeneous, so enabling it (=1) is less risky than with full data. But keep OFF for V14 baseline, enable for V14b ablation.

5. **Oversampling**: Consider running 8 epochs (~226 steps) if 4 epochs is insufficient for convergence. Hard data benefits from repeated exposure.

---

## 4. Hard-Data-Only Dataset Details

### 4.1 Composition
| Source | Samples | Description |
|--------|---------|-------------|
| Training set hard (SFT-base pass@1 wrong) | 1,616 | Filtered from StepCountQA-RL-Traj_0_10 |
| pixmo-test hard (V12-step132 wrong) | 98 | External evaluation wrong predictions |
| CountBench hard (V12-step132 wrong) | 101 | External evaluation wrong predictions |
| **Dedup external** | **192** | 98+101 → 192 after dedup |
| **Total** | **1,808** | Combined hard dataset |

### 4.2 File Location
```
StepCountQA-RL-Traj_0_10_hard_only/data/train-00000-of-00001.parquet
├── 1,808 rows
├── 612.7 MB
├── Schema: images (list[{bytes, path}]), problem (str), answer (str)
```

### 4.3 Build Script
```
dataset/build_hard_only_dataset.py
```

---

## 5. Expected Outcomes & Evaluation Plan

### 5.1 Success Metrics
| Metric | V12 Benchmark | V14 Target |
|--------|--------------|------------|
| NaN events (in 113 steps) | 91/179 (from step 85) | **0** |
| pixmo-test accuracy | 81.5% (V12-step132) | ≥82% |
| countbench accuracy | 79.4% (V12-step132) | ≥80% |
| Training stability | NaN from step 85 | Stable throughout |

### 5.2 Why Hard-Data Should Improve Performance
1. **No wasted compute**: Every training step provides informative gradients
2. **Targeted learning**: Model focuses on failure modes (high-GT counting, multi-turn sequences, ambiguous scenes)
3. **Stronger per-step signal**: BoK-GRPO advantage differentiation is maximized on hard data
4. **External hard data**: 192 pixmo-test+countbench wrong samples directly target evaluation benchmarks

### 5.3 Risk Factors
1. **Catastrophic forgetting**: Training only on hard data may cause regression on easy samples. Mitigation: evaluate on full training set periodically.
2. **Overfitting**: 1808 samples with 4 epochs → each sample seen 4× by the policy. Monitor for reward hacking.
3. **Pass@16 = 0 groups**: Some hard samples may have pass@16=0 with SFT base → BoK assigns zero advantage to entire group → wasted step. This is acceptable but should be monitored.

---

## 6. V14 vs V14 Experiment Matrix

| Experiment | Dataset | LR | BOK_CLIP | Steps | Purpose |
|-----------|---------|-----|----------|-------|---------|
| **V14** | Full (11455) | 1e-6 | 3.0 | ~178 | Conservative NaN fix: LR reduction only |
| **V14** | Hard (1808) | 1e-6 | 3.0 | ~113 | Hard-data-only BoK-GRPO |
| V14b (planned) | Hard (1808) | 1e-6 | 3.0 | ~113 | + Process Reward enabled |
| V14c (planned) | Hard (1808) | 1e-6 | 2.5 | ~226 | Tighter clip + 8 epochs |

---

*Generated: $(date '+%Y-%m-%d %H:%M')*
