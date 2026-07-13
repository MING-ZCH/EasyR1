# BOK-GRPO-Step Log Diagnosis (2026-04-27)

## Executive Conclusion

The existing EasyR1 StepCount logs support implementing `bok_grpo_step`, but only in a conservative form: final-answer trajectory outcome must remain the primary optimization signal, while point-step rewards should be used as gated local auxiliary credit. The logs do not support replacing outcome optimization with pure step-level GRPO.

The core evidence is:

1. Point quality and final answer correctness are correlated but clearly not equivalent. In v23/v24, validation `point_reward` reaches roughly `0.9230-0.9388`, while `answer_reward` remains around `0.7580-0.7637`.
2. v27 shows that point degradation can pull the whole trajectory down: final validation has `answer_reward=0.7089`, `point_reward=0.7197`, and `consistency_violation_reward=0.1191`.
3. Smart-filter releases are real and frequent. v24 step 213 has `ac_released=80/1024`, v27 step 213 has `ac_released=48/1024`, and v27b step 61 has `ac_released=96/1024`. These are all-correct groups whose point quality is still below the smart-filter threshold.
4. The current `bok_grpo` estimator sums token-level rewards into a trajectory score. Even when process reward is enabled, BoK still performs trajectory-level credit assignment. The existing `grpo_step` aligns by absolute reward-bearing token position, which is unsuitable for variable-length interleaved trajectories.

## Evidence by Run

### v12: Historical Strong Baseline

Log file: `logs/train/training_interleaved_traj_v12_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej0_bok_grpo_20260315_013620.log`

Final validation evidence:

- `val/answer_reward=0.775`
- `val/point_reward=0.936`
- `val/format_fail_reward=0.021`
- `val/consistency_violation_reward=0.059`

Interpretation: v12 proves that trajectory-level BoK can produce strong final-answer performance, but the gap between point and answer rewards shows that high point quality alone does not solve accumulated counting errors.

### v23: Strict Stable Anchor

Log file: `logs/train/training_interleaved_traj_v23_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej1_bok_grpo_20260410_003702.log`

Step 213 validation:

- `val/answer_reward=0.7637`
- `val/point_reward=0.9230`
- `val/format_fail_reward=0.0340`
- `val/consistency_violation_reward=0.0529`

Final BoK diagnostics:

- `easy_drgrpo=464/1024`
- `all_correct_filtered=192/1024`
- `allwrong_capped=48/1024`

Interpretation: v23 has strong point reward but only moderate answer reward. This argues against naive process-reward maximization, while supporting outcome-primary training with finer point credit.

### v24: Smart Filter Evidence

Log file: `logs/train/training_interleaved_traj_v24_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej1_bok_grpo_20260412_045606.log`

Step 213 validation:

- `val/answer_reward=0.7580`
- `val/point_reward=0.9388`
- `val/format_fail_reward=0.0208`

Final BoK diagnostics:

- `all_correct_filtered=304/1024`
- `ac_released=80/1024`
- `smart_filter_th=0.955`

Interpretation: `ac_released` proves that some all-answer-correct groups still contain low-quality point trajectories. The original BoK routing can decide whether such groups should keep training, but it cannot identify which semantic point step deserves credit.

### v27: High-LR / Point Degradation Evidence

Log file: `logs/train/training_interleaved_traj_v27_bok_grpo_bok_grpo_20260425_041701.log`

Validation trend:

- step 160: `answer_reward=0.7259`, `point_reward=0.7192`
- step 180: `answer_reward=0.7164`, `point_reward=0.7203`
- step 213: `answer_reward=0.7089`, `point_reward=0.7197`
- step 213: `consistency_violation_reward=0.1191`

Stability signals:

- terminal entropy is around `0.95-0.99`
- logs contain `grad_norm: !!float 'nan'`
- validation `format_fail_reward` remains around `0.0265`, so this is not a pure format-collapse failure

Interpretation: v27 failed through training dynamics and point/answer degradation, not because strict JSON was ineffective. Any step-level method must use conservative LR, stronger KL, smaller grad norm, and clipped auxiliary step advantages.

### v27b: Easy-Data Early Evidence

Log file: `logs/train/training_interleaved_traj_v27b_2gpu_h200_bok_grpo_bok_grpo_20260427_071318.log`

Early validation:

- step 0: `answer_reward=0.7429`, `point_reward=0.7187`
- step 20: `answer_reward=0.7713`, `point_reward=0.7207`
- step 60: `answer_reward=0.7656`, `point_reward=0.7227`

BoK early evidence:

- step 61: `ac_released=96/1024`
- entropy is around `0.46-0.50`, much lower than v27 terminal entropy

Interpretation: easy-data and strict JSON can recover final-answer behavior early, but point reward remains low. This is the ideal target for a small semantic point-step auxiliary rather than a large process-reward objective.

## Decision

Implement `bok_grpo_step` as an isolated estimator with these constraints:

- Preserve BoK-GRPO as the trajectory-level base advantage.
- Add semantic point-step auxiliary only on point-generation spans.
- Align by point order within the trajectory, not by absolute token position.
- Gate the step auxiliary by final answer correctness by default.
- Keep strict JSON and format rejection enabled.
- Use conservative v29 stability defaults.

## Success Criteria for v29

The v29 experiment should be considered promising only if:

1. `val/answer_reward` does not regress below the v27b early baseline.
2. `val/point_reward` improves beyond the v27b approximate `0.72` level.
3. `val/consistency_violation_reward` does not increase while point reward improves.
4. `val/format_fail_reward` remains low, ideally near `0.02-0.04`.
5. `[BoK-GRPO-Step] point_positions`, `step_values`, and `active_spans` are nonzero in training logs.
