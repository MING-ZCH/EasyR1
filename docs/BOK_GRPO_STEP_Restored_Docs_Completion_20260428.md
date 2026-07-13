# BOK-GRPO-Step Restored Docs Completion Note (2026-04-28)

## Scope

This note records the final reconciliation against the restored source documents:

- `docs/BOK_GRPO_Trajectory_Step_Hybrid_CN_20260427.md`
- `docs/GRPO_AgenticRL_Trajectory_Step_Reward_Literature_Debate_CN_20260427.md`

The implementation was checked specifically for the documented StepCount requirement: keep BoK trajectory outcome optimization as the primary objective while adding a bounded, soft-gated point-step auxiliary path that uses process reward positions and does not affect existing training modes.

## Completion Summary

The `bok_grpo_step` algorithm is implemented as an isolated estimator rather than a replacement for existing `bok_grpo` or `grpo_step`.

Implemented requirements:

- `AdvantageEstimator.BOK_GRPO_STEP = "bok_grpo_step"` is available in `verl/trainer/ray_trainer.py`.
- `verl/trainer/core_algos.py` contains `compute_bok_grpo_step_advantage(...)`.
- The estimator first computes the original BoK trajectory advantage, then adds a point-step auxiliary term only on semantic point spans.
- Reward worker process positions are passed through `_point_step_token_positions` and converted into `batch.batch["point_step_mask"]` before advantage computation.
- Point-step alignment uses semantic point order within each rollout, not absolute token index.
- The final answer tail remains controlled by the trajectory BoK advantage.
- The step auxiliary is clipped and the composed hybrid advantage is clipped.
- Existing `bok_grpo`, `grpo_step`, validation metrics, and baseline scripts remain isolated.

## Restored-Document Hyperparameter Alignment

The restored documents recommend canonical environment names. The estimator supports those names with backward-compatible v29 aliases:

| Restored-document name | Default | Backward-compatible alias |
| --- | ---: | --- |
| `BOK_STEP_WEIGHT` | `0.2` | `BOK_STEP_LAMBDA` |
| `BOK_STEP_GATE` | `answer_soft` | `BOK_STEP_GATE_MODE` |
| `BOK_STEP_MIN_GATE` | `0.2` | `BOK_STEP_GATE_FLOOR` |
| `BOK_STEP_CLIP` | `2.5` | same |
| `BOK_HYBRID_CLIP` | `4.0` | `BOK_STEP_TOTAL_CLIP` |
| `BOK_STEP_EXCLUDE_FINAL` | `1` | same |

Supported gate modes include `none`, `answer_soft`, `soft_count_closeness`, `answer_hard`, `traj_positive`, and the earlier aliases `positive_outcome`, `outcome_sigmoid`, and `traj_soft`.

## Launch Scripts

The documented algorithm-version entry point has been added:

```text
examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_bok_step_hybrid.sh
```

It delegates to the isolated v29 implementation script while preserving the v27b baseline:

```text
examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v29_bok_grpo_step_easy_data.sh
```

The documented entry sets the expected defaults:

```text
ADV_ESTIMATOR=bok_grpo_step
PROCESS_REWARD_ENABLE=1
TRAJ_RETURN_POINT_STEP_SCORES=1
TRAJ_POINT_STRICT_JSON=1
TRAJ_FORMAT_REJECTION=1
BOK_STEP_WEIGHT=0.2
BOK_STEP_GATE=answer_soft
BOK_STEP_MIN_GATE=0.2
BOK_STEP_CLIP=2.5
BOK_HYBRID_CLIP=4.0
BOK_STEP_EXCLUDE_FINAL=1
ANSWER_WEIGHT=0.6
POINT_WEIGHT=0.3
TRAJECTORY_FORMAT_WEIGHT=0.1
```

## Verification

The following lightweight checks were run after the restored-document alignment:

```text
python3 -m py_compile verl/trainer/core_algos.py verl/trainer/ray_trainer.py verl/workers/reward/function.py
```

Result:

```text
PY_COMPILE_OK
```

```text
bash -n examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v29_bok_grpo_step_easy_data.sh
bash -n examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_bok_step_hybrid.sh
```

Result:

```text
HYBRID_SCRIPTS_BASH_N_OK
```

```text
BOK_STEP_HYBRID_DRY_RUN=1 bash examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_bok_step_hybrid.sh
```

Observed key dry-run configuration:

```text
mode=bok_grpo_step adv=bok_grpo_step
bok_step_weight=0.2 gate=answer_soft min_gate=0.2 step_clip=2.5 hybrid_clip=4.0 exclude_final=1
strict_json=1 format_rejection=1
lr=1e-6 ppo_epochs=1 kl=0.05 smart_filter=0.955
```

## Conclusion

The restored-document `bok_grpo_step` algorithm development is complete at the code and launch-script level. The remaining question is empirical: whether the hybrid improves point grounding and final answer pass@1 over the v27b BoK baseline without increasing format failures, consistency violations, entropy spikes, or non-finite gradients.
