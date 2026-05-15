# V27b Training Analysis

Date: 2026-05-01
Run log: `logs/train/training_interleaved_traj_v27b_easy_data_bok_grpo_bok_grpo_20260429_221003.log`
Monitor log: `logs/monitor/monitor_v27b_20260429_221003.log`
Script: `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_easy_data.sh`

## Executive Summary

V27b is a successful easy-data BoK-GRPO run up to its validation peak. It reached `val/answer_reward=0.7958` at step 120, higher than the prior v12/v23 reference range reported in experiment notes, while keeping format healthy (`val/format_reward=0.9792`) and improving point reward to `0.7275`. The run then declined to `0.7845` at step 140 and `0.7694` at step 160.

The non-finite-gradient issue is real but late. The first unique non-finite optimizer step was 145, after the step-120 validation peak. Non-finite gradients were skipped by the gradient guard, and LR was reduced to `1.5e-7` during cooldown, so these events mostly prevented useful updates rather than directly corrupting the already-learned checkpoint. The practical failure was operational: `save_freq=90` saved only `global_step_90`, while the best observed validation point was step 120. The fail-fast monitor later stopped the job at monitor count 10 before a final checkpoint could be saved.

## Validation Curve

| Step | Answer | Point | Format |
| ---: | ---: | ---: | ---: |
| 0 | 0.7429 | 0.7187 | 0.9716 |
| 20 | 0.7353 | 0.7195 | 0.9735 |
| 40 | 0.7599 | 0.7182 | 0.9698 |
| 60 | 0.7599 | 0.7187 | 0.9698 |
| 80 | 0.7467 | 0.7185 | 0.9754 |
| 100 | 0.7788 | 0.7228 | 0.9716 |
| 120 | 0.7958 | 0.7275 | 0.9792 |
| 140 | 0.7845 | 0.7245 | 0.9622 |
| 160 | 0.7694 | 0.7225 | 0.9584 |

## Gradient Health

Unique non-finite optimizer steps observed in the train log:

| Opt Step | Nonfinite Count | LR |
| ---: | ---: | ---: |
| 145 | 1 | 1.50e-06 |
| 148 | 2 | 1.50e-07 |
| 155 | 3 | 1.50e-07 |
| 157 | 4 | 1.50e-07 |
| 162 | 5 | 1.50e-07 |

The monitor reported 10 non-finite events because the log contains both the rank-0 line and repeated multi-rank cluster lines. The optimizer-side counter reached 5 unique skipped updates by step 162.

Spike events were present before NaN onset: 7 spike detections by the late stage. The largest late spike cluster occurred around steps 124-128, including `grad_norm=34.6856` at spike #6. After the first non-finite gradient, cooldown held LR at `1.5e-7`; entropy did not show the v27-style extreme drift and remained roughly in the 0.64-0.78 band late in the run.

## BoK Routing Around Peak and Decline

| Step | Tau | Low Var | Easy DrGRPO | All-Correct Filtered | AC Released | Adv Std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 120 | 0.396 | 160/1024 | 432/1024 | 304/1024 | 48/1024 | 0.7059 |
| 140 | 0.343 | 192/1024 | 496/1024 | 288/1024 | 64/1024 | 0.6986 |
| 160 | 0.310 | 304/1024 | 448/1024 | 160/1024 | 80/1024 | 0.7469 |
| 162 | 0.308 | 160/1024 | 448/1024 | 224/1024 | 80/1024 | 0.8082 |

The smart filter threshold `0.965` worked as intended: it continued releasing all-correct but point-improvable groups to the Easy/DrGRPO path. At the step-120 peak, the run had a strong answer score and the best observed point score, with no non-finite gradients yet.

## Interpretation

The main gain came before NaN: steps 100-120 improved answer reward from 0.7788 to 0.7958 and point reward from 0.7228 to 0.7275 while format became cleaner. This supports the v27b recipe: easy-only data, strict JSON, format rejection, answer-primary BoK-GRPO, `BOK_SMART_FILTER_THRESHOLD=0.965`, and LR `1.5e-6` can produce a strong sparse-counting checkpoint.

The late decline is consistent with optimizer instability and over-training after the peak. Step 140 remained strong, but format and answer started to soften. Step 160 showed a clearer validation decline, while train reward remained high, suggesting that continuing updates after step 120 did not improve validation generalization.

## Script Changes Applied

The v27b launch script was updated for future reruns:

| Setting | Old Default | New Default | Reason |
| --- | ---: | ---: | --- |
| `TRAIN_SAVE_FREQ` | 90 | 20 | Save every validation point, including possible peaks such as step 120. |
| `TRAIN_SAVE_LIMIT` | 5 | 8 | Keep the recent validation-aligned checkpoint window without unbounded storage. |
| `V27B_FAILFAST_ENABLE` | 1 | 0 | Do not stop training when NaN count exceeds a threshold; use the regular monitor for logging. |

The regular monitor remains enabled and records NaN, spike, validation, and summary events. It does not create a stop file unless explicitly launched with `--auto-stop`, which the v27b script does not use.

## Recommended Follow-Up

For this run, evaluate the existing `global_step_90` checkpoint because it is the only saved checkpoint. For the next v27b rerun, use the patched script and inspect checkpoints at steps 100, 120, and 140 first. If external evaluation confirms step 120 is the best region, consider lowering LR slightly or stopping around step 120-140 for this easy-only recipe.
