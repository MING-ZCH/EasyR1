# V24 Design Notes

## Summary

V24 keeps the V23 stability fix and only changes the reward-routing pieces that limited further answer gains after the V23 peak.

## What V23 Achieved

- Completed all 213 steps with **0 NaN**.
- Prevented format collapse: validation format-fail stayed around **2-3%**.
- Reached **best val/answer = 0.7769 at step 135**, matching the best V22 peak while staying numerically stable.

## Why V23 Still Plateaued

The completed V23 run showed two structural bottlenecks:

1. **All-correct groups were filtered to zero gradient**.
   - Around **23%** of BoK groups were `all_correct_filtered`.
   - Those groups still contain point-quality differences, but V23 discarded that signal.

2. **Correct answers were not coupled to point quality**.
   - With `TRAJ_ANSWER_GATE_MODE=off`, a correct answer received full answer reward even if the point trajectory was weaker.
   - This reduced pressure to convert strong point trajectories into stronger final counting behavior.

## V24 Changes

### 1. Enable soft answer gate

```bash
TRAJ_ANSWER_GATE_MODE=soft
TRAJ_SOFT_GATE_BASE=0.8
```

This keeps the reward dense while making correct-answer reward sensitive to point quality.

### 2. Recover gradients for easy / all-correct groups

```bash
BOK_FILTER_ALL_CORRECT=0
BOK_EASY_SCALE=0.4
```

This preserves some gradient on easy groups without letting them dominate hard groups.

### 3. Increase checkpoint density

```bash
trainer.save_freq=30
```

V23 peaked at step 135, but only steps 120 and 180 were saved. V24 saves more densely to reduce checkpoint miss risk.

## What Stays Unchanged

- `ppo_epochs=1`
- `FORMAT_REJECTION=1`
- `ANSWER_WEIGHT=0.6`
- `POINT_WEIGHT=0.3`
- `TRAJECTORY_FORMAT_WEIGHT=0.1`
- `BOK_TAU_INIT=0.7`, `BOK_TAU_FINAL=0.3`
- `BOK_WINNER_BOOST=0`
- `BOK_QUALITY_BONUS=0`
- `kl_coef=0.02` (restored from V12; changed from V23's 0.03)

## Resume Strategy

V24 resumes from V21-R2 step-60:

```bash
save/StepCount-7B-SFT-30k_v21_mask_reward_v4bok_grpo_hm0_gateoff_bok_grpo_20260406_0311/global_step_60
```

Reason:
- V21-R2 is a clean ppo_epochs=1 run with 0 NaN, val/answer=0.7561 at step 60;
- Saves ~2.4h training time vs starting from SFT;
- 60-step KL offset is tolerable: ref_policy is always SFT (checkpoint-3537);
- tau cosine resumes from progress=60/213≈0.28 (tau≈0.63), providing smooth continuation.

## Expected Outcome

If the hypothesis is correct, V24 should:

- keep **0 NaN** and low format-fail;
- improve point-to-answer transfer after step 120;
- exceed the V23 best checkpoint more sustainably than the short-lived step-135 spike.

## Launch Command

```bash
bash examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v24.sh
```
