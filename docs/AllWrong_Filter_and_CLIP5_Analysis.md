# AllWrong (0/16) Filter & BOK_CLIP=5 Deep Analysis — Revised

Date: 2025-07-25
Status: Analysis complete, critical finding: "easy-only" dataset contains hard questions

## Critical Correction

**Previous analysis was wrong.** "Easy-only" dataset (`StepCountQA-RL-Traj_0_10`) means
*count range 0-10*, NOT *easy questions*. It contains ALL 11,455 questions including:
- ~9,650 questions the model gets right (pass@1 correct)
- ~1,250 medium-hard questions (pass@1 wrong but pass@32 correct, p_q ≈ 0.10)
- ~550 very hard questions (pass@32 also wrong, p_q ≈ 0.03)

## 1. 0/16 All-Wrong Sample Analysis (Revised)

### 1.1 Frequency: Much Higher Than Initially Estimated

| Group | p_q | N questions | P(0/16) | 0/16 per epoch | Per step (178 steps) |
|-------|-----|------------|---------|----------------|---------------------|
| A (easy) | 0.925 | 9,650 | ~0% | 0 | 0 |
| B (med-hard) | 0.10 | 1,250 | 18.5% | 231 | 1.3 |
| C (very hard) | 0.03 | 550 | 61.4% | 337 | 1.9 |
| **Total** | | | | **568** | **3.2** |

**Every step has ~3.2 groups that are answer-all-wrong entering BoK softmax.**

### 1.2 Impact with TRAJ_SOFT_ANSWER_DECAY=1

V17 uses `TRAJ_SOFT_ANSWER_DECAY=1` (alpha=8.0, cap=0.4), which means:
- Wrong answers get non-zero score: error=-1 → 0.20, error=-2 → 0.04
- `overall = 0.6*answer_decay + 0.2*point + 0.2*format`
- All-wrong groups always have variance → always enter BoK softmax
- BoK softmax selects the "least wrong" trajectory → +4.0 advantage

**Is this harmful?** Nuanced answer:
- With soft_decay: BoK selects "closest to correct" → somewhat useful signal
- Without soft_decay: BoK selects "best point score" → harmful (unrelated to answer)
- But +4.0 is too strong for an all-wrong group — same magnitude as 1/16 correct

### 1.3 Recommended Fix

**Option A (Recommended): AllWrong advantage cap**
```python
# After BoK softmax computation, before final clipping:
if all answer_scores in group == 0:  # binary correctness, not soft_decay
    advantages = advantages.clamp(-1.0, 1.0)  # reduced cap for all-wrong
```
This preserves the relative ordering signal from soft_decay while preventing
the +4.0 magnitude that competes with truly correct trajectories.

**Option B: AllWrong DrGRPO fallback**
```python
if all answer_scores in group == 0:
    route to DrGRPO instead of BoK → all negative advantages
```
More aggressive — removes all positive signal from all-wrong groups.

**Priority: MEDIUM** — affects ~3.2 groups/step, but soft_decay mitigates worst case.

## 2. BOK_CLIP=5 Analysis (Revised)

### 2.1 Corrected Impact for V17

Previous: "CLIP=4→5 has zero effect on easy-only V17" — WRONG.

V17 dataset has ~1,800 hard questions that enter BoK path:
- Per step: ~10 hard questions × 82% non-0/16 = ~8 BoK groups from hard questions
- Groups affected by CLIP=4 (k≤3/16): ~4 per step
- Groups uniquely affected by CLIP=4→5 (k=3): ~1 per step

### 2.2 Quantitative Impact

| Metric | CLIP=3 | CLIP=4 | CLIP=5 |
|--------|--------|--------|--------|
| BoK adv_std | 1.446 | 1.535 | 1.590 |
| Total adv_std | 0.916 | 0.959 | 0.985 |
| Drift rate change | baseline | +4.7% | +7.5% |
| Groups getting more signal | - | k≤5 | k=3 (1/step) |

### 2.3 Decision: Maintain CLIP=4.0

- CLIP=4→5 benefits ~1 group/step with +0.33 more advantage (4.0→4.33 for k=3/16)
- Cost: +3% drift rate → ~3 fewer steps before NaN risk
- Net effect: marginal and uncertain
- **Recommendation: Keep CLIP=4.0. Future ablation with hard-data experiments.**

## 3. Summary

| Item | Finding | Recommendation | Priority |
|------|---------|---------------|----------|
| 0/16 frequency | ~3.2 groups/step | Implement AllWrong advantage cap | **Medium** |
| soft_decay impact | Makes 0/16 less harmful but still overpowered | Cap at ±1.0 for all-wrong | Medium |
| BOK_CLIP=5 | ~1 group/step affected | Keep CLIP=4.0 | Low |

## 4. Key Insight for Training

The "easy-only" dataset is a misnomer for BoK-GRPO purposes. ~16% of questions
are hard enough that the model rarely/never solves them in 16 rollouts. These
hard questions dominate the BoK path (since easy questions go through DrGRPO).
Understanding this data composition is critical for any BoK hyperparameter tuning.
