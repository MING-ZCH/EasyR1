# AllWrong Advantage Cap — Implementation Documentation

## Overview

When **all K trajectories** in a group have wrong answers (0/K correct), BoK-GRPO's softmax weighting assigns the "best wrong" trajectory a positive advantage up to `+bok_clip` (default: +4.0). This is overpowered — it pushes the model to strongly repeat a wrong answer pattern, even though the trajectory is fundamentally incorrect.

**AllWrong Advantage Cap** limits the maximum |advantage| for these groups to a configurable cap (default: ±1.5), preserving the directional signal ("closer to correct is better") while preventing overpowered gradient pushing.

## Impact Analysis

- **Frequency**: ~3.2 AllWrong groups per step (568/epoch with K=16, dataset=StepCountQA-RL-Traj_0_10)
- **Before cap**: best wrong trajectory gets +4.0 advantage
- **After cap (1.5)**: best wrong limited to +1.5 → **62.5% magnitude reduction**
- **Direction preserved**: relative ordering within AllWrong group unchanged
- **No impact on**: Easy groups, AllCorrect groups, or DrGRPO-routed groups

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `BOK_ALLWRONG_CAP` | `1.5` (v17.sh) | Max |advantage| for AllWrong groups. 0 = disabled |
| `BOK_ALLWRONG_ANSWER_THRESHOLD` | `0.5` | answer_score < this = wrong answer |

## Files Modified

### 1. `verl/trainer/core_algos.py`
- **Function signature**: Added `answer_scores: torch.Tensor = None` parameter
- **Env var reading**: `BOK_ALLWRONG_CAP`, `BOK_ALLWRONG_ANSWER_THRESHOLD`
- **Counter**: `n_allwrong_capped` tracking
- **AllWrong cap logic**: After BoK softmax advantage computation, detect AllWrong groups and clamp advantages to ±cap
- **Detection**: Uses `answer_scores` tensor if available (precise), falls back to `overall_scores` if not (backward-compatible)
- **Diagnostics**: Added to both summary and detailed logging

### 2. `verl/trainer/ray_trainer.py`
- **L898-902**: Extract per-sample `reward_metrics["answer"]` list before `reduce_metrics()` destroys it; store as `batch.batch["answer_scores"]` tensor
- **L180-185**: Pass `answer_scores` from `data.batch` to `compute_bok_grpo_advantage`

### 3. `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v17.sh`
- Added `BOK_ALLWRONG_CAP=1.5` and `BOK_ALLWRONG_ANSWER_THRESHOLD=0.5` env vars

## Detection Logic

```
For each BoK-path group:
  if BOK_ALLWRONG_CAP > 0:
    if answer_scores provided:
      is_allwrong = all(answer_scores[j] < threshold for j in group)
    else:
      is_allwrong = all(overall_scores[j] < threshold for j in group)
    if is_allwrong:
      advantages[j] = clamp(advantages[j], -cap, +cap) for all j in group
```

## Verified Behavior (Simulation)

### K=16, GT=5, All wrong (errors: -1,-2,-3,+1,+2)
```
Cap OFF: best_wrong_adv = +4.000, worst_wrong_adv = -0.900
Cap ON:  best_wrong_adv = +1.500, worst_wrong_adv = -0.900
→ 62.5% reduction on positive extreme, direction preserved
```

### K=4, Mixed batch (AllWrong + Easy + AllCorrect)
```
AllWrong:    cap_on=[+1.500, -0.895, -0.068, -0.895]  ← capped
Easy(3/4):   cap_on=[+0.684, +0.354, +0.448, -1.485]  ← unaffected
AllCorrect:  cap_on=[+0.000, +0.000, +0.000, +0.000]  ← filtered (existing)
```
