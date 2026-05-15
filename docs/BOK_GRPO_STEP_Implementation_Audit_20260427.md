# BOK-GRPO-Step Implementation Audit (2026-04-27)

## Scope

This audit checks whether the implemented `bok_grpo_step` matches the design requirements in the restored source documents:

- `docs/BOK_GRPO_Trajectory_Step_Hybrid_CN_20260427.md`
- `docs/GRPO_AgenticRL_Trajectory_Step_Reward_Literature_Debate_CN_20260427.md`

Both documents recommend an isolated outcome-primary hybrid estimator, process reward enabled, point-step auxiliary credit only on point spans, final-answer leakage excluded from step credit, soft answer/trajectory gating, clipped step and hybrid advantages, and an independent launch script for the algorithm version. The current implementation has been re-checked against those requirements after the documents were restored.

## Implementation Coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| Keep trajectory-level BoK outcome as the primary signal | Implemented | `compute_bok_grpo_step_advantage(...)` first calls `compute_bok_grpo_advantage(...)` and uses its result as the base advantage. |
| Keep the original `bok_grpo` behavior unchanged | Implemented | `compute_bok_grpo_advantage(...)` was not rewritten; `bok_grpo_step` is a new function and new enum branch. |
| Keep the original `grpo_step` behavior unchanged | Implemented | `compute_grpo_step_level_advantage(...)` was not modified. |
| Add an isolated estimator name | Implemented | `AdvantageEstimator.BOK_GRPO_STEP = "bok_grpo_step"` in `verl/trainer/ray_trainer.py`. |
| Activate only by explicit configuration | Implemented | New branch is used only when `algorithm.adv_estimator=bok_grpo_step`; v27b and existing scripts are unchanged. |
| Use process reward positions for point-step credit | Implemented | `verl/workers/reward/function.py` records `_point_step_token_positions` when `PROCESS_REWARD_ENABLE=1`; trainer converts it to `batch.batch["point_step_mask"]`. |
| Preserve zero-reward point steps in alignment | Implemented | The explicit `point_step_mask` is based on tag positions, not reward value, so zero-reward point steps still participate in semantic alignment. |
| Align by semantic point order, not absolute token index | Implemented | `sample_positions[row_idx]` is sorted per response; step `t` is compared across rollout rows by list index. |
| Apply step auxiliary only on point spans | Implemented | The function updates token ranges ending at each point position; answer/non-point tail remains outcome advantage. |
| Gate local step auxiliary by global correctness | Implemented | Default `BOK_STEP_GATE=answer_soft` with `BOK_STEP_MIN_GATE=0.2`; legacy `BOK_STEP_GATE_MODE` and `BOK_STEP_GATE_FLOOR` remain supported. Trainer passes `answer_scores` from reward metrics. |
| Keep answer span outcome-primary | Implemented | Tokens after the final point are not overwritten by step auxiliary and keep the BoK outcome advantage. |
| Exclude final answer reward from step auxiliary | Implemented | `BOK_STEP_EXCLUDE_FINAL=1` filters point-step positions before the final answer boundary; the answer tail remains controlled by trajectory BoK. |
| Clip local step advantage | Implemented | `BOK_STEP_CLIP=2.5` controls normalized step auxiliary clipping by default. |
| Clip composed token advantage | Implemented | `BOK_HYBRID_CLIP=4.0` controls final token-level advantage clipping by default; legacy `BOK_STEP_TOTAL_CLIP` remains supported. |
| Use restored-document environment names | Implemented | `BOK_STEP_WEIGHT`, `BOK_STEP_GATE`, `BOK_STEP_MIN_GATE`, `BOK_HYBRID_CLIP`, and `BOK_STEP_EXCLUDE_FINAL` are canonical, with earlier v29 aliases preserved. |
| Keep strict JSON and format rejection | Implemented in v29 script | `TRAJ_POINT_STRICT_JSON=1` and `TRAJ_FORMAT_REJECTION=1`. |
| Keep final answer reward dominant | Implemented in v29 script | `ANSWER_WEIGHT=0.6`, `POINT_WEIGHT=0.3`, `TRAJECTORY_FORMAT_WEIGHT=0.1`. |
| Keep smart filter for all-correct but point-low groups | Implemented in v29 script | `BOK_SMART_FILTER_THRESHOLD=0.955`. |
| Avoid rewarding all-wrong groups positively | Implemented in v29 script | `BOK_ALLWRONG_NEG_ONLY=1`; existing BoK all-wrong cap path supports this. |
| Use conservative stability defaults | Implemented in v29 script | `ACTOR_LR=1e-6`, `algorithm.kl_coef=5e-2`, `worker.actor.max_grad_norm=0.5`. |
| Exclude no-sequence/no-mask cases from step auxiliary | Mostly implemented by reward dataflow | `_step_rewards` is emitted only when `step_scores` is non-empty. If no sequence/mask can produce step scores, no point-step positions are exposed. Mask-derived no-GT cases can still participate when sequence masks exist, which is intentional for StepCount trajectory scoring. |
| Add diagnostics for whether the new estimator is active | Implemented | `[BoK-GRPO-Step]` logs `point_positions`, `step_values`, `active_spans`, `gate_mean`, `aux_mean`, and `aux_std`. |
| Keep validation metric reduction safe | Implemented | Validation skips reward metric keys starting with `_`; training pops `_point_step_token_positions` before `reduce_metrics`. |

## Files Changed or Added

- `verl/trainer/core_algos.py`: adds `compute_bok_grpo_step_advantage(...)`.
- `verl/trainer/ray_trainer.py`: adds `BOK_GRPO_STEP`, constructs `point_step_mask`, filters internal validation metrics, and routes to the new estimator.
- `verl/workers/reward/function.py`: records internal point-step token positions for process reward mode.
- `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v29_bok_grpo_step_easy_data.sh`: isolated v29 implementation launch script with restored-document default hyperparameters.
- `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_bok_step_hybrid.sh`: documented algorithm-version launch entry; delegates to the isolated v29 implementation while preserving the v27b baseline.
- `docs/BOK_GRPO_STEP_Log_Diagnosis_20260427.md`: log-based motivation and support evidence.
- `docs/BOK_GRPO_STEP_Algorithm_Design_20260427.md`: algorithm and paper-style method documentation.

## Verification Performed

- Python syntax check:

```text
python3 -m py_compile verl/trainer/ray_trainer.py verl/trainer/core_algos.py verl/workers/reward/function.py
```

Result: `PY_COMPILE_OK`.

- Estimator smoke test with explicit semantic point mask and grouped non-tensor-style UIDs.

Result: `BOK_GRPO_STEP_SMOKE_OK (4, 8)`.

- Shell syntax check:

```text
bash -n examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v29_bok_grpo_step_easy_data.sh
bash -n examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_bok_step_hybrid.sh
```

Result: `HYBRID_SCRIPTS_BASH_N_OK`.

- v29 dry run:

```text
V29_DRY_RUN=1 bash examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v29_bok_grpo_step_easy_data.sh
```

Result: paths and critical hyperparameters validated; no trainer was started.

- documented hybrid dry run:

```text
BOK_STEP_HYBRID_DRY_RUN=1 bash examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_bok_step_hybrid.sh
```

Result: the launch entry resolves to `mode=bok_grpo_step`, `adv=bok_grpo_step`, `PROCESS_REWARD_ENABLE=1`, strict JSON/format rejection enabled, and the restored-document defaults `BOK_STEP_WEIGHT=0.2`, `BOK_STEP_GATE=answer_soft`, `BOK_STEP_MIN_GATE=0.2`, `BOK_STEP_CLIP=2.5`, `BOK_HYBRID_CLIP=4.0`, `BOK_STEP_EXCLUDE_FINAL=1`.

## Known Limitations

1. The current trainer receives point-step positions through an internal reward metric field. This is intentionally minimal and isolated, but a future cleaner interface could store process-step metadata directly in the `DataProto`.
2. The point auxiliary is span-level over tokens from the previous point boundary to the current point boundary. This matches the interleaved generation surface, but it is still an approximation of the exact model decision that selected the coordinate.
3. The implementation does not claim general novelty over all outcome-plus-process reward methods. The contribution is the StepCount-specific semantic point-order alignment and gated composition with BoK outcome optimization.
4. If `PROCESS_REWARD_ENABLE=0`, the estimator falls back to detecting nonzero reward positions; the v29 script keeps process reward enabled, which is the intended path.

## Audit Conclusion

The implementation matches the restored-document `bok_grpo_step` design: trajectory outcome remains primary, semantic point-step credit is added only as a soft-gated local auxiliary, final-answer leakage is excluded from the step path, strict format controls remain enabled, and existing training modes are not changed. The documented algorithm-version launch script has been added. The remaining validation is empirical: the hybrid run must prove that point reward improves without reducing final answer pass@1 or increasing consistency/format failures.
