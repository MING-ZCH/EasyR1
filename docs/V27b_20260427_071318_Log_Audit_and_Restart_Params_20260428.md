# V27b Log Audit and Restart Parameters

Date: 2026-04-28
Run log: `logs/train/training_interleaved_traj_v27b_easy_data_bok_grpo_bok_grpo_20260427_071318.log`

## Executive Judgment

The v27b run is an effective stability improvement over the previous unstable v27-style behavior. It reached step 106/178 before the external interruption, with no positive non-finite gradient count, no entropy explosion, and no format collapse. The best validation answer reward improved from 0.7429 at step 0 to 0.7750 at step 100, while validation format reward stayed in the 0.9716-0.9754 band.

The run did not yet prove a breakthrough over the historical best easy-data peaks. The validation point reward was nearly flat, moving from 0.7187 to 0.7219. This suggests the next bottleneck is not basic format survival, but unlocking more point-quality learning from answer-correct groups.

## Evidence

Validation summaries:

| Step | Answer | Point | Format | Format Fail | Overall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7429 | 0.7187 | 0.9716 | 0.0284 | 0.7429 |
| 20 | 0.7713 | 0.7207 | 0.9754 | 0.0246 | 0.7713 |
| 40 | 0.7599 | 0.7209 | 0.9716 | 0.0284 | 0.7599 |
| 60 | 0.7656 | 0.7227 | 0.9735 | 0.0265 | 0.7656 |
| 80 | 0.7618 | 0.7201 | 0.9716 | 0.0284 | 0.7618 |
| 100 | 0.7750 | 0.7219 | 0.9754 | 0.0246 | 0.7750 |

Training reward windows:

| Window | Overall | Answer | Point | Format Fail |
| --- | ---: | ---: | ---: | ---: |
| First 10 reward calls | 0.8408 | 0.8151 | 0.8434 | 0.0124 |
| Last 10 reward calls | 0.8576 | 0.8380 | 0.8515 | 0.0061 |

Stability signals:

| Signal | Result |
| --- | --- |
| Final observed training step | 106/178 |
| Non-finite gradient count | 0 positive events |
| Entropy loss | stable around 0.46-0.51, no upward explosion |
| Grad spikes | 2 detected and skipped by cooldown-only protection |
| Grad spike examples | step 78 ratio 4.28x, step 101 ratio 4.47x |
| Final LR after cooldown | restored to 1.5e-6 |
| Fail-fast trigger | none observed before interruption |

BoK-GRPO routing windows:

| Window | Low Var | Collapsed | Easy Routed | All-Correct Filtered | AC Released | All-Wrong Capped | Adv Std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First 10 BoK rows | 123.2 | 0.7 | 531.2 | 203.2 | 41.6 | 9.6 | 0.7975 |
| Last 10 BoK rows | 132.8 | 1.1 | 457.6 | 284.8 | 43.2 | 22.4 | 0.7407 |

The smart filter is working: all-correct groups are not blindly removed, and about 40-45 rollouts per recent batch remain released for point-quality learning. However, point validation stayed flat, so the restart raises the threshold modestly to release more answer-correct groups whose point quality is still below the desired level.

## Restart Parameter Changes

Applied to `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_easy_data.sh`:

| Parameter | Previous Effective Value | Restart Value | Reason |
| --- | ---: | ---: | --- |
| `ACTOR_LR` | 1.5e-6 | 1.5e-6 | Do not raise to 2e-6; two >4x grad spikes already occurred at 1.5e-6, but cooldown handled them. |
| `BOK_SMART_FILTER_THRESHOLD` | 0.955 | 0.965 | Preserve the smart filter but release more answer-correct groups with point_mean below about 0.883. |
| `V27B_FAILFAST_NAN_LIMIT` | 10 | 3 | Catch non-finite cascades earlier. |
| `V27B_FAILFAST_ENTROPY_CRITICAL` | 0.95 | 0.85 | Earlier stop for v27-like entropy drift; observed stable entropy was far below this. |
| `V27B_FAILFAST_FORMAT_CRITICAL` | 0.35 | 0.30 | Earlier stop for format collapse; observed format fail was about 0.02-0.03 on validation. |
| `V27B_FAILFAST_ZEROREWARD_CRITICAL` | 35 | 30 | Match the stricter failure guard used by the stable line. |
| `trainer.save_freq` | 120 in the interrupted run, 80 in current script | 40 | Avoid losing useful checkpoints when an external interruption happens before step 120. |
| `trainer.save_limit` | -1 | 5 | Keep restart storage bounded while preserving all expected checkpoints for a 178-step run. |

## Dry-Run Confirmation

The updated script dry-run reports:

```text
[RunConfig] mode=bok_grpo adv=bok_grpo lr=1.5e-6 max_grad_norm=1.0
[RunConfig] easy_threshold=0.50 smart_filter=0.965
[RunConfig] FailFast enable=1 entropy>0.85x3 format>0.30x3 nan_limit=3
[RunConfig] save_freq=40 save_limit=5
[DryRun] strict_json=1 format_rejection=1
[DryRun] lr=1.5e-6 ppo_epochs=1 kl=0.03 smart_filter=0.965
[DryRun] save_freq=40 save_limit=5
```

## Recommendation

Restart v27b from SFT cold start with the updated script. This is still an ablation/stress line, not the main v28 line. The expected outcome is stable full-epoch completion with checkpoints at steps 40, 80, 120, and 160, and a better chance of point-quality improvement without increasing LR risk.
