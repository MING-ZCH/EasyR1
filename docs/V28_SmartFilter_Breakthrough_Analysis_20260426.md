# V28 Smart Filter and Breakthrough Direction Analysis (2026-04-26)

## Executive conclusion

`BOK_SMART_FILTER_THRESHOLD=0.955` is valid as a point-quality gradient recovery mechanism. The previous decision to disable it in the V28 main run over-attributed V24/V27 entropy growth to smart filtering. The stronger causal explanation is the combination of higher learning rate, weaker or stressed KL dynamics, and the historical loose-format reward loophole. V28 should therefore use the V23-stable optimizer backbone while keeping smart filtering enabled:

- `ACTOR_LR=1e-6`
- `ppo_epochs=1`
- `kl_coef=0.03`
- `max_grad_norm=1.0`
- `BOK_SMART_FILTER_THRESHOLD=0.955`
- `TRAJ_POINT_STRICT_JSON=1`
- `GRAD_SPIKE_ABSOLUTE_CAP=0`

## First-principles interpretation

The counting task fails mainly through accumulated spatial errors, not through inability to output a final number. In all-correct answer groups, the final answer reward is saturated and cannot distinguish better from worse pointing. If these groups are fully filtered, the model loses gradient on exactly the cases where it already has the answer right but still needs better spatial grounding.

When all 16 trajectories answer correctly and format is valid:

`overall_mean = 0.6 * answer + 0.3 * point + 0.1 * format = 0.7 + 0.3 * mean(point)`

Therefore:

`BOK_SMART_FILTER_THRESHOLD=0.955` means `mean(point) >= 0.85`.

The filter behavior is:

- If all answers are correct and `mean(point) >= 0.85`, the group is truly mastered and can be filtered.
- If all answers are correct but `mean(point) < 0.85`, the group is released to Easy/Dr.GRPO so point and format differences continue to produce gradients.

This is exactly aligned with StepCount's objective: preserve final answer correctness while reducing miss/duplicate accumulation.

## Direct evidence from logs and docs

### 1. The implementation is semantically correct

`verl/trainer/core_algos.py` explicitly maps `0.955` to `mean(point) >= 0.85` under valid answer/format groups, and releases all-correct but point-low groups to Easy/Dr.GRPO.

### 2. V24 and V27 actually activated smart filter

Log evidence:

| Run | `smart_filter_th` | Avg all-correct filtered | Avg all-correct released | Interpretation |
|---|---:|---:|---:|---|
| V23 | old/disabled | 23.0% | 0.0% | All answer-correct groups are dropped. |
| V24 | 0.955 | 20.4% | 4.6% | Smart filter recovered point-gradient budget. |
| V27 latest | 0.955 | 15.4% | 3.4% | Smart filter is active under strict JSON. |

Latest monitor snapshot for V27 step 89 shows even stronger local activation:

- `allcorrect_filtered=112/1024`
- `ac_released=96/1024`
- `easy_drgrpo=576/1024`
- step 80-84 to 85-89 point trend: `0.8056 -> 0.8158` (`+0.0102`)

This supports the user's hypothesis that smart filter is useful for point learning.

### 3. Validation evidence is positive but not yet decisive

V23 vs V24 validation aggregation:

| Run | Best answer | Final answer | Mean answer | Mean point | Mean format fail |
|---|---:|---:|---:|---:|---:|
| V23 | 0.7769@135 | 0.7637@213 | 0.7574 | 0.9298 | 0.0277 |
| V24 | 0.7713@165 | 0.7580@213 | 0.7604 | 0.9319 | 0.0268 |

V24 has slightly higher mean answer and mean point, but lower peak answer. Because V24 changed more than smart filter (`lr=1.5e-6`, weaker `kl=0.02`, and warm start), it cannot be used as a clean proof that smart filter alone improves final validation accuracy. It does prove that smart filter is not equivalent to a destructive format-collapse mechanism.

### 4. Entropy growth is confounded, not proof against smart filter

V24 entropy growth happened with `lr=1.5e-6` and `kl=0.02`. V27 entropy growth happened with `lr=1.5e-6` under strict JSON and high KL pressure. V23 was stable with `lr=1e-6`, `kl=0.03`, and no strict JSON. Therefore the clean next test is not `smart_filter=0`; it is `smart_filter=0.955` under the V23-stable LR/KL backbone.

## Corrected V28 decision

The corrected main V28 run should be:

`V23 stable optimizer + strict JSON + smart_filter=0.955`

This isolates the useful point-gradient recovery mechanism from the high-LR/high-entropy confound.

## Breakthrough directions

### Direction 1: State-aware point reward and answer coupling

Problem: final answer can be correct even when point quality is mediocre; answer reward saturates too early.

Proposed direction:

- Use a soft answer gate: correct answer reward should depend on point coverage quality.
- Increase point influence only inside answer-correct groups, not globally.
- Add an explicit coverage/completeness term: reward unique mask coverage and penalize misses/duplicates at the trajectory level.

Expected effect: convert pass@K latent spatial capability into pass@1 by making the best trajectory's pointing path consistently higher reward even when all answers are correct.

### Direction 2: Dynamic smart filtering instead of a fixed 0.955 threshold

Problem: `0.955` is a fixed proxy for `mean(point)=0.85`; it may be too strict early and too loose late.

Proposed direction:

- Start around `0.93` (`mean(point)≈0.77`) to release more weak point groups early.
- Anneal to `0.955-0.970` as training progresses.
- Track `ac_released`, train point, entropy, and validation answer; keep released share in a target band such as 3-8%.

Expected effect: preserve the benefit of smart filtering while preventing excessive Easy/Dr.GRPO pressure.

### Direction 3: Hard-case curriculum from actual wrong predictions

Problem: historical wrong-prediction mining shows errors concentrate at higher counts, especially 7, 8, and 10. Random easy+hard mixing may not apply gradient where pass@1 fails.

Proposed direction:

- Build a small high-quality replay buffer from V12/V23/V27 wrong predictions.
- Oversample high-count and near-miss spatial cases but maintain enough easy cases to preserve format.
- Require mask-backed samples for RL point reward; for maskless hard examples, use `stat_mask_sim` instead of `count_iou` to avoid inflated point reward.

Expected effect: raise sparse-count accuracy above the 0.78 plateau by targeting the actual error distribution.

### Direction 4: Process prompt with explicit count state

Problem: the model currently infers count state mostly from red dots. As dots accumulate, visual state becomes noisy and misses compound.

Proposed direction:

- Inject a lightweight textual state into process prompts: counted count, remaining instruction, and last point coordinate.
- Keep the final answer format unchanged.
- Test on high-count easy samples first.

Expected effect: reduce mid-trajectory confusion and duplicate/miss accumulation without changing the model architecture.

### Direction 5: Turn-level credit assignment for the interleaved policy

Problem: a full trajectory receives one scalar advantage, but the error is often localized to one wrong point turn.

Proposed direction:

- Add per-turn point advantages for point tokens while keeping terminal answer reward for answer tokens.
- Use trajectory-level answer reward as a gate, but use mask point reward to assign local credit.
- Avoid full GSPO until scalar reward is stable; start with point-token auxiliary loss or weighted log-prob terms.

Expected effect: faster correction of visual offset errors, lower variance than relying only on terminal trajectory reward.

### Direction 6: Evaluation-aligned validation and checkpoint selection

Problem: historical peaks are narrow. V25 achieved a high answer peak but invalid format; V23 peaked at step 135 and then drifted.

Proposed direction:

- Save and validate every 10 steps during the high-risk window.
- Select checkpoints by a constrained metric: answer score first, then format fail below threshold, then entropy band.
- Add an automatic stop rule when entropy rises above a configured band without validation gain.

Expected effect: avoid losing the best checkpoint and prevent training from drifting after the useful learning window.

## Immediate experimental recommendation

Run V28 main as the corrected configuration:

- `ACTOR_LR=1e-6`
- `BOK_SMART_FILTER_THRESHOLD=0.955`
- `TRAJ_POINT_STRICT_JSON=1`
- `kl_coef=0.03`
- `ppo_epochs=1`
- `max_grad_norm=1.0`

Success criteria by step 40-80:

- train `format_fail_rate < 0.1`
- entropy remains below `0.65`
- `ac_released` stays non-zero and typically around 3-8%
- train point reward improves without answer collapse
- validation recovers above SFT baseline and trends toward V23 peak

If entropy still grows under `lr=1e-6`, then test dynamic smart threshold or `BOK_EASY_SCALE=0.5`, not immediate removal of smart filtering.
