# V30 V27-Lifted Stable Design

Date: 2026-04-29
Source run: `logs/train/training_interleaved_traj_v27_bok_grpo_bok_grpo_20260425_041701.log`
New script: `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v30_v27_lifted_stable.sh`

## Executive Judgment

The v27 sparse-guarded run should not be dismissed because of its late non-finite gradients. It completed the planned 213-step easy-plus-hard epoch and saved `global_step_213`; the final weights remained loadable because non-finite gradients were skipped rather than applied. The external evaluation reported by the user, PixMo-test 81.66% and CountBench 77.9%, suggests that v27 learned useful counting behavior despite unhealthy optimizer dynamics.

The useful part to transfer is not the NaN behavior. The transferable recipe is mixed easy-plus-hard data, strict JSON reward enforcement, BoK-GRPO with smart all-correct filtering, full-epoch exposure, and the v27/v23 clipping backbone. The part to change is optimizer safety: v27 started non-finite skips at optimizer step 92 and ended with 94 non-finite counts, so a new run needs lower LR, earlier checkpointing, conditional LR brakes, and fail-fast monitoring.

## V27 Evidence

Validation curve from the training log:

| Step | Answer | Point | Format | Format Fail |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7561 | 0.7092 | 0.9622 | 0.0378 |
| 20 | 0.7448 | 0.7130 | 0.9679 | 0.0321 |
| 40 | 0.7429 | 0.7161 | 0.9641 | 0.0359 |
| 60 | 0.7524 | 0.6523 | 0.6446 | 0.3554 |
| 80 | 0.7316 | 0.7179 | 0.9716 | 0.0284 |
| 100 | 0.7108 | 0.7238 | 0.9773 | 0.0227 |
| 120 | 0.7335 | 0.7207 | 0.9716 | 0.0284 |
| 140 | 0.7316 | 0.7239 | 0.9773 | 0.0227 |
| 160 | 0.7259 | 0.7192 | 0.9716 | 0.0284 |
| 180 | 0.7164 | 0.7203 | 0.9735 | 0.0265 |
| 200 | 0.7164 | 0.7213 | 0.9735 | 0.0265 |
| 213 | 0.7089 | 0.7197 | 0.9735 | 0.0265 |

Optimizer health:

| Signal | Observation |
| --- | --- |
| First non-finite gradient | optimizer step 92 |
| Final non-finite count | 94 |
| Explicit spike detections | 7 |
| Early spike cluster | 5 spikes before optimizer step 50 |
| Largest early spike | grad_norm 9.0235, ratio 7.55x |
| Final entropy summary | 0.96676 |
| Final checkpoint | `global_step_213` saved |

The training validation reward did not explain the high external score by itself. Internal PixMo validation answer reward declined after the early stage, while external evaluation was strong. The likely explanation is that easy-plus-hard full-epoch training improved out-of-distribution counting behavior, while the small internal validation slice was noisy and the optimizer became unhealthy late in the epoch.

## Transferred Factors

| Factor | V27 Behavior | V30 Decision |
| --- | --- | --- |
| Training data | `StepCountQA-RL-Traj_0_10_easy_plus_hard` | Keep as default |
| Epoch length | 213 optimizer steps | Keep `BOK_TOTAL_STEPS=213` |
| Estimator | `bok_grpo` | Keep for direct ablation comparability |
| Strict JSON | Enabled | Keep enabled |
| Format rejection | Enabled | Keep enabled |
| BoK clip | 4.0 | Keep 4.0 |
| Smart filter | 0.955 | Raise to 0.965 for more point-quality learning |
| Per-turn tokens | 1800 | Keep 1800 to avoid extra over-generation pressure |

## Stability Changes

| Parameter | V27 | V30 | Reason |
| --- | ---: | ---: | --- |
| `ACTOR_LR` | 1.5e-6 | 1.25e-6 | Preserve learning strength but reduce risk on mixed easy-plus-hard data. |
| `GRAD_SPIKE_BRAKE_MAX` | 6 | 4 | v27 had 5 spikes before step 50; brake earlier only when instability repeats. |
| `GRAD_SPIKE_BRAKE_WINDOW` | 40 | 50 | Capture the early v27 spike cluster. |
| `GRAD_NONFINITE_COOLDOWN` | implicit/default | 12 | Give more recovery time after non-finite gradients. |
| `GRAD_NONFINITE_LR_FACTOR` | implicit/default 0.1 | 0.05 | Use stronger temporary backoff after non-finite gradients. |
| `GRAD_NONFINITE_BRAKE_MAX` | default 3 | 2 | Halve the base LR earlier if non-finite gradients repeat. |
| `V30_FAILFAST_NAN_LIMIT` | none | 6 log events | Allows the non-finite brake to act, then stops if the cascade continues. |
| `V30_FAILFAST_ENTROPY_CRITICAL` | none | 0.85 x3 | Stops v27-like late entropy drift. |
| `V30_FAILFAST_FORMAT_CRITICAL` | none | 0.30 x3 | Stops sustained format collapse, but not a single noisy validation spike. |
| `TRAIN_SAVE_FREQ` | 60 | 40 | Saves checkpoints before and after the historical step-92 onset. |
| `TRAIN_SAVE_LIMIT` | -1 | 6 | Bounds storage while retaining useful checkpoints. |

## Expected Readout

The first decision point is step 80. A healthy V30 run should have no non-finite gradients before step 80, format reward near or above 0.97 outside isolated spikes, and point reward at least matching v27's 0.718-0.724 validation band. The second decision point is step 120: if V30 avoids the v27 step-92 non-finite cascade and keeps answer reward within the v27/v27b band, it should be a better candidate for external evaluation than the original v27 final checkpoint.

If V30 still triggers fail-fast near step 90-110, the next conservative adjustment should be `ACTOR_LR=1e-6` with the same easy-plus-hard data and smart filter, rather than raising LR.
