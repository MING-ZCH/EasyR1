# V28 Complete Algorithm and Training-Parameter Analysis (2026-04-26)

## 0. Scope

This document consolidates the current StepCount trajectory-RL evidence after reviewing historical logs, analysis docs, scripts, reward code, BoK-GRPO implementation, gradient protection, and prompt files.

Primary objective: improve StepCount interleaved point-to-count pass@1 while avoiding the historical failure modes:

1. format cliff,
2. NaN / non-finite optimizer state,
3. entropy explosion,
4. point miss/duplicate accumulation,
5. answer-correct but spatially-poor trajectories saturating reward.

## 1. First-principles diagnosis

Counting accuracy in this task is not only a final-number problem. The final answer is the terminal observable, but the causal path is spatial coverage:

`correct answer = unique target coverage - missed objects - duplicate counted objects - premature stop / over-stop`

The RL objective must therefore optimize two layers simultaneously:

- terminal correctness: answer count is correct;
- process correctness: each point shifts attention to a new target object.

The historical plateau around `0.775-0.783` is consistent with point-error accumulation: when the answer reward saturates, the model no longer receives enough pressure to improve point quality on already-answer-correct examples.

## 2. Corrected smart-filter conclusion

`BOK_SMART_FILTER_THRESHOLD=0.955` should be kept in the V28 main run.

### 2.1 Mechanism

For an all-answer-correct, valid-format group:

`overall_mean = 0.6 * answer + 0.3 * point + 0.1 * format = 0.7 + 0.3 * mean(point)`

Thus:

`0.955 = 0.7 + 0.3 * 0.85`

So `BOK_SMART_FILTER_THRESHOLD=0.955` means:

- filter all-correct groups only when `mean(point) >= 0.85`;
- release all-correct but point-low groups into Easy/Dr.GRPO when `mean(point) < 0.85`.

This is the right semantics for StepCount: do not waste gradients on truly mastered samples, but keep improving spatial grounding where answer is already correct and point quality is not.

### 2.2 Evidence

Historical routing aggregation:

| Run | `smart_filter_th` | Avg all-correct filtered | Avg all-correct released | Key interpretation |
|---|---:|---:|---:|---|
| V23 | disabled / old behavior | 23.0% | 0.0% | Answer-correct groups lose point gradients. |
| V24 | 0.955 | 20.4% | 4.6% | Smart filter recovers point-gradient budget. |
| V27 latest | 0.955 | 15.4% | 3.4% | Smart filter active under strict JSON. |

Latest V27 monitor at step 89:

- `allcorrect_filtered=112/1024`
- `ac_released=96/1024`
- `easy_drgrpo=576/1024`
- train point trend from step 80-84 to 85-89: `0.8056 -> 0.8158`

The current evidence supports the user's hypothesis: smart filter is useful as a point-learning mechanism. Entropy growth in V24/V27 is confounded by higher LR, weaker/stressed KL dynamics, and historical parser/format issues; it is not clean evidence against smart filtering.

## 3. Newly found high-impact issue: truncated process prompt

The active file `examples/format_prompt/StepCount_interleaved_process_prompt.txt` was incomplete and ended mid-sentence:

`If all target objects have been marked with red dots and counting i`

This is a direct prompt-level risk for stop-format failure and final-answer uncertainty in later turns. It can also make strict JSON more important than necessary because the model is under-instructed during process turns.

### 3.1 Fix applied

The process prompt was backed up as:

`examples/format_prompt/StepCount_interleaved_process_prompt.txt.bak_truncated_20260426`

The active prompt now explicitly states:

- do not count red-dot-marked objects;
- point exactly one next unmarked object when objects remain;
- when complete, output `<think>All target objects have been counted.</think><answer>n</answer>`;
- output no extra text after `</answer>`.

This fix is likely high-leverage because the task is multi-turn and the final-answer instruction must be present in the repeated process prompt, not only in the first-turn prompt.

## 4. Current V28 main configuration

V28 should be interpreted as:

`V23-stable optimizer + strict JSON reward + smart_filter=0.955 + complete process prompt`

| Parameter | V28 value | Reason |
|---|---:|---|
| `ACTOR_LR` | `1e-6` | V23-stable backbone; avoids v27 high-LR entropy drift. |
| `ppo_epochs` | `1` | Best historical NaN control; avoids stale rollout amplification. |
| `kl_coef` | `0.03` | More stable than `0.02`; V23-safe regularization. |
| `max_grad_norm` | `1.0` | V23-stable; avoids V12-style excessive sign-like clipping. |
| `BOK_SMART_FILTER_THRESHOLD` | `0.955` | Restores point gradients on answer-correct / point-low groups. |
| `BOK_CLIP` | `4.0` | Stable historical BoK ceiling; bounds advantage spikes. |
| `BOK_TAU_INIT -> FINAL` | `0.7 -> 0.3` | Safe exploration-to-selection schedule. |
| `BOK_EASY_THRESHOLD` | `0.50` | Proven routing balance in V23/V24 line. |
| `BOK_ALLWRONG_CAP` | `1.0` | Prevents all-wrong groups from dominating. |
| `BOK_ALLWRONG_NEG_ONLY` | `0` | Keep current V27-compatible behavior; test neg-only as later ablation. |
| `TRAJ_POINT_STRICT_JSON` | `1` | Core fix for parser reward hacking. |
| `TRAJ_FORMAT_REJECTION` | `1` | Invalid trajectories receive zero reward. |
| `TRAJ_ANSWER_GATE_MODE` | `off` | Keep final answer primary in V28 main; test soft gate later. |
| `TRAJ_SOFT_ANSWER_DECAY` | `1` | Gives ranked signal among wrong answers. |
| `ANSWER/POINT/FORMAT weights` | `0.6 / 0.3 / 0.1` | Keep stable baseline before reward-shape ablation. |
| `GRAD_SPIKE_THRESHOLD` | `3.0` | Conservative EMA spike guard. |
| `GRAD_SPIKE_ABSOLUTE_CAP` | `0` | Avoid skip-step bursts from post-hoc absolute caps. |
| `GRAD_NONFINITE_BRAKE_MAX` | `999` | Emergency LR-drop brake disabled by default; use skip + temporary cooldown, and fail fast on repeated non-finite events. |
| `gpu_memory_utilization` | `0.65` | H200-specific stable utilization. |
| `val_freq / save_freq` | `10 / 10` | Needed because historical best checkpoints are narrow. |

## 5. Algorithmic parameter interactions

### 5.1 LR, KL, and entropy

Entropy growth is not controlled by one knob. It is the product of:

- update size (`lr`, `max_grad_norm`, `ppo_epochs`),
- KL strength,
- advantage variance,
- format-invalid zeroing,
- reward parser loopholes,
- checkpoint starting point.

Historical interpretation:

- V23: `lr=1e-6`, `ppo=1`, `kl=0.03` was stable.
- V24: `lr=1.5e-6`, `kl=0.02`, smart filter active, entropy rose.
- V27: `lr=1.5e-6`, `kl=0.03`, strict JSON, smart filter active, entropy still rose by step 89.

Therefore the clean next run should lower LR back to V23 while keeping smart filter. If entropy still grows under `lr=1e-6`, then the next suspect becomes Easy/Dr.GRPO released-group pressure, not LR.

### 5.2 All-correct groups

All-correct filtering is not automatically good in a multi-dimensional reward. In binary reward tasks, all-correct means zero useful gradient. In StepCount, all-answer-correct can still contain point/format variance.

Recommended rule:

- Keep `BOK_FILTER_ALL_CORRECT=1` to remove truly mastered groups.
- Keep `BOK_SMART_FILTER_THRESHOLD=0.955` to recover point gradients for answer-correct / point-low groups.
- Monitor `ac_released`; target band: `3-8%` of rollouts.

### 5.3 Easy/Dr.GRPO path

Released all-correct groups go to Easy/Dr.GRPO, where point/format differences become gradient. This is useful, but too much Easy pressure can increase policy drift.

Control knobs:

- `BOK_EASY_SCALE`: currently `1.0`; future safe ablation `0.5-0.7` if entropy grows under stable LR.
- Dynamic smart threshold: maintain `ac_released` in a target band rather than fixed threshold.

### 5.4 BoK softmax path

BoK softmax is most valuable when a group contains few correct trajectories. It transfers pass@K ability into pass@1 by increasing probability of the best sampled path.

Current controls are reasonable:

- `BOK_TAU_INIT=0.7`, `BOK_TAU_FINAL=0.3`
- `BOK_UNIFORM_MIX=0.1`
- `BOK_CLIP=4.0`
- `BOK_LOGIT_CAP=12.0` in code

Future option:

- lower final tau only after strict-format stability is proven;
- test `BOK_WINNER_BOOST=1.3-1.5` only on stable runs.

### 5.5 All-wrong groups

All-wrong groups still contain soft-answer and point-quality ranking signal. Historical docs recommend capping rather than filtering. Current V28 keeps `BOK_ALLWRONG_CAP=1.0` and `BOK_ALLWRONG_NEG_ONLY=0` for compatibility.

Future ablation:

- `BOK_ALLWRONG_NEG_ONLY=1` may reduce reinforcement of wrong answers, but it could also remove useful closest-wrong curriculum signal. Test only after V28 main establishes a clean baseline.

### 5.6 Reward weights and answer gate

Current weights keep answer primary:

`overall = 0.6 * answer + 0.3 * point + 0.1 * format`

This is stable but creates saturation. Better future reward design should not globally reduce answer too early. Instead, make answer-correct trajectories separable by point quality.

Safe next ablation after V28 main:

- keep global weights unchanged;
- enable a mild correct-answer soft gate: `TRAJ_ANSWER_GATE_MODE=soft`, `TRAJ_SOFT_GATE_BASE=0.85-0.90`;
- or add a correct-answer-only point-quality bonus inside BoK ranking.

## 6. Data strategy

### 6.1 Sparse target benchmark objective

For `0-10` sparse objects, the current bottleneck is likely not the presence of hard samples alone. V23 easy+hard was stable and competitive. The issue is whether hard samples are mask-backed and whether their reward distribution matches the intended spatial objective.

### 6.2 Wrong-prediction mining

Historical wrong-prediction mining found errors concentrated at higher counts, especially `7`, `8`, and `10`. This is exactly the regime where sequential miss accumulation matters.

Recommended data direction:

1. Mine V23/V27 wrong predictions on pixmo-test/countbench-like held-out evals.
2. Add only mask-backed hard samples to RL if possible.
3. For maskless hard samples, avoid inflated `count_iou`; use `stat_mask_sim` or answer-only training mode.
4. Maintain enough easy samples to preserve format and final-answer behavior.

### 6.3 Curriculum schedule

Recommended curriculum after V28 main:

| Phase | Data | Goal |
|---|---|---|
| Phase A | easy+hard current dataset | Stable strict-format point learning. |
| Phase B | add mined wrong high-count samples at low ratio | Target actual pass@1 failures. |
| Phase C | hard-count oversampling with mask-backed samples | Improve dense-count generalization. |

## 7. Monitoring and stop rules

V28 should be judged by constrained metrics, not answer score alone.

### 7.1 Healthy training bands

| Metric | Healthy target |
|---|---:|
| train `format_fail_rate` | `< 0.10`, ideally `< 0.05` |
| validation `format_fail_reward` | `< 0.05`, no spike above `0.15` |
| entropy | stable near `0.50-0.65` |
| KL | no sustained runaway; interpret with entropy |
| `ac_released` | non-zero, target `3-8%` |
| train point reward | should improve without answer drop |
| non-finite events | zero; first event triggers brake |
| EMA spike count | low and non-accelerating |

### 7.2 Fail-fast rules

Stop or branch if any condition holds:

1. `format_fail_rate > 0.30` for 3 consecutive RewardHealth calls.
2. entropy `> 0.80` before validation improves.
3. validation answer below step-0 baseline for two consecutive validations after step 40.
4. any non-finite event after strict JSON and LR=1e-6.
5. `ac_released=0` for most steps, because smart filter is not contributing.
6. `ac_released>12%` with rising entropy, because Easy/Dr.GRPO pressure may be too high.

## 8. Recommended ablation order

### Main run: V28

Run this first:

- `lr=1e-6`
- `smart_filter=0.955`
- strict JSON
- fixed process prompt
- no absolute cap
- dense val/save

This is the cleanest synthesis of historical evidence.

### Ablation A: smart-filter control

Only if V28 is stable but inconclusive:

- same as V28, but `BOK_SMART_FILTER_THRESHOLD=0`

Purpose: isolate smart filter under the same stable LR/KL/prompt conditions.

### Ablation B: dynamic smart threshold

If V28 improves point but entropy still drifts:

- start `smart_threshold=0.93`
- anneal to `0.955` or `0.97`
- or adapt threshold to keep `ac_released` in `3-8%`

Purpose: retain point-gradient recovery while controlling Easy pressure.

### Ablation C: Easy scale

If `ac_released` is high and entropy grows:

- `BOK_EASY_SCALE=0.5` or `0.7`

Purpose: damp released/easy gradients without disabling smart filter.

### Ablation D: point-aware answer gate

If V28 is stable but answer accuracy remains capped:

- `TRAJ_ANSWER_GATE_MODE=soft`
- `TRAJ_SOFT_GATE_BASE=0.85-0.90`

Purpose: make correct-answer trajectories separable by point quality.

### Ablation E: hard-case data replay

If point and format are stable but sparse-val answer is still below target:

- add mined wrong samples with mask-backed reward;
- use low ratio first;
- avoid uncalibrated `count_iou` inflation for maskless examples.

## 9. Breakthrough directions

### 9.1 Dynamic attention-state prompting

The process prompt is now fixed, but the rollout still does not explicitly inject the current count state. A future high-impact change is to add dynamic text such as:

`You have already counted k objects. Continue with count_number k+1. Do not point to red dots.`

This reduces burden on visual-only red-dot memory.

### 9.2 Turn-level credit assignment

Current trajectory reward provides one scalar advantage for a whole trajectory. A point miss at turn 4 can be hidden by a later correct answer or by soft decay.

Future direction:

- use `TRAJ_RETURN_POINT_STEP_SCORES=1`;
- apply auxiliary token-level loss or per-turn advantage on point outputs;
- keep terminal answer reward for answer tokens.

This is likely a true algorithmic breakthrough because it matches the causal structure of interleaved counting.

### 9.3 Point-quality-aware BoK ranking

Current code already supports `BOK_QUALITY_BONUS`, disabled by default. A safer future route is:

- enable quality bonus only for answer-correct trajectories;
- keep global reward weights unchanged;
- use small bonus first, e.g. `0.1-0.2`.

This directly attacks the answer-correct / point-poor saturation problem.

### 9.4 Coverage reward

Add a trajectory-level coverage metric:

- unique mask hits / target count;
- duplicate penalty;
- over-point penalty;
- premature answer penalty if unhit masks remain.

Current point reward already approximates this, but exposing coverage explicitly in logs and ranking would improve diagnosability and learning pressure.

### 9.5 Selective high-count replay

Higher counts have more accumulated error. A targeted replay buffer for counts `7-10` can increase gradient density exactly where sparse benchmark failures occur.

Risk: overfitting and format drift. Mitigation: keep replay ratio small and require strict JSON/format constraints.

## 10. Final recommendation

Do not disable smart filter in V28. The corrected main path is:

`V28 = V23-stable LR/KL + strict JSON + smart_filter=0.955 + fixed process prompt + dense monitoring`

This is the most evidence-aligned next experiment. If it fails, the next branches should adjust dynamic smart threshold or Easy scale before removing smart filtering.
