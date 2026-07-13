# BOK-GRPO-Step Algorithm Design and Implementation (2026-04-27)

## Motivation

StepCount trains an interleaved visual point-to-count policy. A rollout is a full trajectory: the model points to one object, receives an updated marked image, points again, and finally emits an answer. Evaluation uses final answer pass@1, but training with only final answer reward is sparse. Point rewards provide dense visual grounding, but historical logs show that point reward cannot replace final answer reward.

`bok_grpo_step` is designed to combine both signals without letting the local signal dominate the global objective.

The method has three goals:

1. Keep trajectory-level BoK-GRPO as the primary signal for converting pass@K ability into pass@1 behavior.
2. Add semantic point-step credit assignment to point-generation spans.
3. Gate local point advantages by final answer correctness or outcome quality to reduce reward hacking.

## Related Work Positioning

The method is inspired by but distinct from general outcome-plus-process reward training. The literature and project scan suggests the following positioning:

- Tool-use RL and ToolRL-style results support the importance of reward granularity for multi-step action trajectories.
- PRIME-like methods support combining outcome and process rewards.
- PURE, ReTool, RAGEN, and RAGEN-2 warn that naive process reward summation can cause reward hacking, template collapse, or final-answer regression.
- Agent Lightning and Agentic-R emphasize that local action credit should be calibrated by global utility.
- DeepEyes and DeepEyesV2 show that VLM active perception can be improved with RL, but visual-action reward must be constrained by task outcome.

The correct novelty claim is not broad invention of process reward. The useful contribution is the StepCount-specific composition:

> outcome-primary Best-of-K trajectory optimization plus gated semantic point-step auxiliary credit for interleaved visual point-to-count trajectories.

## Algorithm

For one prompt group, let there be `K` sampled trajectories. For trajectory `i`, define:

- `R_i`: total trajectory reward
- `c_i`: final answer correctness or answer score
- `p_{i,t}`: process reward for the semantic point step `t`

### 1. Outcome Advantage

The base signal is the existing BoK-GRPO advantage:

```text
w_i = softmax(normalize(R_group) / tau)_i
A_out_i = K * (mix(w_i, uniform) - 1/K)
```

All existing BoK controls remain active: tau scheduling, uniform mixing, low-variance fallback, smart filter, all-correct filtering, all-wrong capping, and advantage clipping.

### 2. Semantic Step Alignment

The reward worker exposes point boundary token positions for each rollout:

```text
positions_i = sorted(token positions where point-step rewards are placed)
```

The `t`th point in each rollout is aligned as semantic step `t`, regardless of absolute token position. This is necessary because interleaved rollouts have different generation lengths.

### 3. Point-Step Advantage

For each prompt group and semantic step `t`, normalize the point rewards across rollouts that contain that step:

```text
A_step_{i,t} = normalize({p_{j,t}: j in same prompt group})
```

If group-level variance is too low, the implementation falls back to batch-level same-step normalization. If that is also low variance, the step auxiliary becomes zero.

### 4. Gated Composition

The final token-level advantage is:

```text
A(tokens in point step t) = A_out_i + lambda * g_i * clip(A_step_{i,t})
A(tokens outside point spans) = A_out_i
```

The default gate is final answer correctness:

```text
g_i = c_i in [0, 1]
```

This ensures the local point reward does not override the final-answer objective.

## Implementation Details

### Reward Manager

File: `verl/workers/reward/function.py`

When `PROCESS_REWARD_ENABLE=1`, the reward manager already places `_step_rewards` at `</point>` token positions. The implementation now additionally records internal point-step token positions:

```text
_point_step_token_positions: List[List[int]]
```

This internal field is removed before public metric reduction in training, and validation skips all internal metrics whose names start with `_`.

### Trainer

File: `verl/trainer/ray_trainer.py`

New estimator enum:

```text
BOK_GRPO_STEP = "bok_grpo_step"
```

Before reward metrics are reduced, the trainer converts `_point_step_token_positions` into:

```text
batch.batch["point_step_mask"]
```

The new estimator branch is only used when `algorithm.adv_estimator=bok_grpo_step`.

### Core Algorithm

File: `verl/trainer/core_algos.py`

New function:

```text
compute_bok_grpo_step_advantage(...)
```

It first calls the existing `compute_bok_grpo_advantage(...)` to preserve all outcome-level behavior, then adds the semantic point-step auxiliary on point spans. Existing `bok_grpo` and `grpo_step` behavior is unchanged.

## Default v29 Configuration

Script:

```text
examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v29_bok_grpo_step_easy_data.sh
```

Key defaults:

```text
ADV_ESTIMATOR=bok_grpo_step
PROCESS_REWARD_ENABLE=1
TRAJ_POINT_STRICT_JSON=1
TRAJ_FORMAT_REJECTION=1
ANSWER_WEIGHT=0.6
POINT_WEIGHT=0.3
TRAJECTORY_FORMAT_WEIGHT=0.1
ACTOR_LR=1e-6
algorithm.kl_coef=5e-2
worker.actor.max_grad_norm=0.5
BOK_SMART_FILTER_THRESHOLD=0.955
BOK_ALLWRONG_NEG_ONLY=1
BOK_STEP_LAMBDA=0.2
BOK_STEP_GATE_MODE=answer
BOK_STEP_CLIP=2.0
BOK_STEP_TOTAL_CLIP=4.0
BOK_STEP_MAX_SEMANTIC_STEPS=11
```

## Isolation Guarantees

- The original `compute_bok_grpo_advantage` is not modified.
- The original `compute_grpo_step_level_advantage` is not modified.
- Reward `overall` semantics are not modified.
- Validation still scores final answer through the existing eval path.
- The new behavior is active only with `ADV_ESTIMATOR=bok_grpo_step` and useful only when `PROCESS_REWARD_ENABLE=1`.

## Risk Controls

1. Outcome-primary objective: answer and non-point spans keep the BoK outcome signal.
2. Answer gate: point auxiliary is gated by final answer correctness by default.
3. Strict JSON: malformed point outputs cannot receive normal point credit.
4. Format rejection: malformed trajectories are zeroed in training.
5. Step clipping: `BOK_STEP_CLIP` limits local normalized advantages.
6. Total clipping: `BOK_STEP_TOTAL_CLIP` limits composed token advantages.
7. Conservative v29 optimization: lower LR, stronger KL, and smaller max grad norm.

## Ablation Plan

Recommended experiment order:

1. v27b baseline: `bok_grpo`, `PROCESS_REWARD_ENABLE=0`.
2. v29 default: `bok_grpo_step`, `BOK_STEP_LAMBDA=0.2`, `BOK_STEP_GATE_MODE=answer`.
3. Lambda sweep: `0.1`, `0.2`, `0.3`.
4. Gate sweep: `answer`, `positive_outcome`, `none`.
5. Negative control: process-only `grpo_step` or a large `BOK_STEP_LAMBDA`.
6. Smart-filter ablation: `BOK_SMART_FILTER_THRESHOLD=0` vs `0.955`.

Primary metrics:

- `val/answer_reward`: the decisive evaluation metric.
- `val/point_reward`: process quality.
- `val/consistency_violation_reward`: point-count and final-answer consistency.
- `val/format_fail_reward` and `val/turns_exceeded_reward`: format and stopping safety.
- `[BoK-GRPO-Step] point_positions`, `step_values`, `active_spans`: confirms the new credit assignment is active.

## Paper-Style Method Text

A concise method paragraph:

```text
We propose BOK-GRPO-Step, an outcome-primary reinforcement learning objective
for interleaved visual point-to-count trajectories. The method preserves the
Best-of-K trajectory-level advantage used to transfer pass@K capability into
pass@1 behavior, while adding a gated semantic point-step auxiliary advantage
on point-generation spans. Unlike token-position process reward methods, our
step auxiliary aligns the kth emitted point across rollouts of the same prompt,
making credit assignment invariant to variable generation length. The local
point advantage is gated by final answer correctness, preventing process reward
from overriding the evaluation objective.
```

Formula summary:

```text
A_i^out = BOK({R_j}_{j=1}^K)_i
A_{i,t}^{step} = Normalize({p_{j,t}: j in group})_i
A_{i,tokens in point t} = A_i^out + lambda * g_i * clip(A_{i,t}^{step})
A_{i,tokens outside points} = A_i^out
```

## Expected Outcome

If the hypothesis is correct, v29 should show:

- `val/answer_reward` at least matching v27b early behavior, then moving toward v12/v25-quality peaks.
- `val/point_reward` recovering beyond the v27b approximate `0.72` level.
- No increase in `consistency_violation_reward`.
- Low and stable `format_fail_reward`.
- Nonzero `[BoK-GRPO-Step]` diagnostics in training logs.
