# BoK-GRPO & StepCount Trajectory Reward — Full Parameter Reference (V14-mixed)

> Generated: 2025-06-16 | Script: `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v14_mixed.sh`
> Code: `verl/trainer/core_algos.py` (L424-735) | `examples/reward_function/StepCount_mask_reward.py`

---

## Table of Contents
1. [BoK-GRPO Algorithm Parameters — Active](#1-bok-grpo-algorithm-parameters--active)
2. [BoK-GRPO Algorithm Parameters — Disabled/Reserved](#2-bok-grpo-algorithm-parameters--disabledreserved)
3. [Reward / Trajectory Parameters — Active](#3-reward--trajectory-parameters--active)
4. [Reward / Trajectory Parameters — Disabled/Reserved](#4-reward--trajectory-parameters--disabledreserved)
5. [Training Infrastructure Parameters](#5-training-infrastructure-parameters)
6. [Interleaved Rollout Parameters](#6-interleaved-rollout-parameters)
7. [Debug / Logging Parameters](#7-debug--logging-parameters)

---

## 1. BoK-GRPO Algorithm Parameters — Active

These parameters are actively used in the current BoK-GRPO advantage computation and directly affect gradient signals.

### 1.1 `BOK_TAU` (Temperature)
| Field | Value |
|-------|-------|
| **Default** | `0.3` |
| **Script Line** | L179 |
| **Code Location** | `core_algos.py` L429, L592 |
| **Type** | `float` |
| **Status** | **ACTIVE** (overridden by τ-annealing when TAU_INIT/FINAL are set) |
| **Description** | Base softmax temperature for BoK advantage computation. Lower τ = more selective (concentrates weight on best trajectories). Higher τ = more uniform (approaches GRPO). Used as `effective_tau = max(bok_tau, eps)` in softmax: `w = softmax(z_norm / effective_tau)`. When τ-annealing is active, this value is dynamically overridden. |
| **Impact** | τ=0.3 gives ~3.3x selectivity over uniform; τ=0.1 gives ~10x. Controls how aggressively the best trajectory dominates gradient. |
| **When to Change** | Lower (0.1-0.2) for more aggressive BoK selection on hard problems. Higher (0.5-1.0) for more exploration. |

### 1.2 `BOK_CLIP` (Advantage Clipping)
| Field | Value |
|-------|-------|
| **Default** | `3.0` |
| **Script Line** | L180 |
| **Code Location** | `core_algos.py` L430, L664-665 |
| **Type** | `float` |
| **Status** | **ACTIVE** |
| **Description** | Clips final advantage values to `[-BOK_CLIP, BOK_CLIP]`. Prevents extreme gradients from blowing up training. Applied after all advantage computation (both BoK softmax and DrGRPO fallback). |
| **Impact** | V14 uses 3.0 (same as V12/V13). Historical: V7 used 2.5. With LR=1e-6, clip=3.0 is stable. |
| **When to Change** | Reduce to 2.0-2.5 if NaN reappears. Increase to 5.0 for stronger gradient (only with very low LR). |

### 1.3 `BOK_TAU_INIT` / `BOK_TAU_FINAL` (τ Annealing Range)
| Field | Value |
|-------|-------|
| **Default** | `TAU_INIT=0.7`, `TAU_FINAL=0.3` |
| **Script Lines** | L183-184 |
| **Code Location** | `core_algos.py` L434-435, L475-480 |
| **Type** | `float` / `float` |
| **Status** | **ACTIVE** (both must be > 0 to enable annealing) |
| **Description** | τ cosine annealing from `TAU_INIT` → `TAU_FINAL` over `BOK_TOTAL_STEPS` steps. At step 0: τ=0.7 (near-uniform, safe exploration). At final step: τ=0.3 (selective BoK). Formula: `τ = τ_final + (τ_init - τ_final) * 0.5 * (1 + cos(π * progress))`. |
| **Impact** | Enables curriculum-like behavior: early training is explorative (high τ), late training is selective (low τ). |
| **When to Change** | TAU_INIT: lower for faster selectivity ramp. TAU_FINAL: lower for more aggressive late-stage BoK (0.1-0.2). |

### 1.4 `BOK_TOTAL_STEPS` (Annealing Horizon)
| Field | Value |
|-------|-------|
| **Default** | `182` |
| **Script Line** | L187 |
| **Code Location** | `core_algos.py` L437, L475-476 |
| **Type** | `int` |
| **Status** | **ACTIVE** |
| **Description** | Total training steps for τ annealing schedule. `progress = min(1.0, global_step / total_steps)`. For V14-mixed: 5792 samples / (16 rollout × 4 GPU) ≈ 91 steps/epoch × 2 epochs = 182. |
| **Impact** | Must match actual training steps for correct annealing. If too low, τ reaches TAU_FINAL prematurely. |
| **When to Change** | Always recalculate: `total_steps = dataset_size / (rollout_n × n_gpus) × total_epochs`. |

### 1.5 `BOK_TAU_SCHEDULE` (Annealing Curve)
| Field | Value |
|-------|-------|
| **Default** | `cosine` |
| **Script Line** | L189 |
| **Code Location** | `core_algos.py` L474, L477-480 |
| **Type** | `string` ("cosine" or "linear") |
| **Status** | **ACTIVE** |
| **Description** | Annealing curve shape. `cosine`: slow start → fast middle → slow end (standard ML warmup style). `linear`: constant rate. |
| **Impact** | Cosine provides smoother transitions and is standard practice. Linear is simpler but less stable at boundaries. |
| **When to Change** | Switch to "linear" for debugging or when cosine schedule causes issues at start/end. |

### 1.6 `BOK_LOW_VAR_THRESHOLD` (Low-Variance Detection)
| Field | Value |
|-------|-------|
| **Default** | `3e-4` |
| **Script Line** | L191 |
| **Code Location** | `core_algos.py` L484-486 |
| **Type** | `float` |
| **Status** | **ACTIVE** |
| **Description** | If a group's reward standard deviation < this threshold, the group is classified as "low variance" and routed to fallback mode (see `BOK_FALLBACK_MODE`). Prevents division-by-zero and meaningless softmax on near-identical rewards. |
| **Evidence** | V14-hard-only: `low_var=0-64/1024` (0-6% of groups). Higher on hard-only data (more uniform wrong answers). |
| **When to Change** | Increase (1e-3) to route more groups to fallback. Decrease (1e-4) for stricter detection. |

### 1.7 `BOK_FALLBACK_MODE` (Low-Var Group Strategy)
| Field | Value |
|-------|-------|
| **Default** | `drgrpo` |
| **Script Line** | L194 |
| **Code Location** | `core_algos.py` L492, L563-590 |
| **Type** | `string` ("zscore" / "drgrpo" / "clip_std") |
| **Status** | **ACTIVE** |
| **Description** | Strategy for groups with variance < `BOK_LOW_VAR_THRESHOLD`: (1) `zscore`: standard z-normalization (unstable when std≈0); (2) `drgrpo`: Dr.GRPO mean-subtraction without std division (stable, default since V11); (3) `clip_std`: z-normalize with `std = max(std, BOK_MIN_BATCH_STD)` floor. |
| **Impact** | `drgrpo` is most stable; produces ~2.6x stronger gradients than z-score for low-var groups. |
| **When to Change** | Switch to `clip_std` if drgrpo produces too-large advantages for low-var groups. |

### 1.8 `BOK_EASY_THRESHOLD` (Easy/Hard Group Routing)
| Field | Value |
|-------|-------|
| **Default** | `0.75` |
| **Script Line** | L198 |
| **Code Location** | `core_algos.py` L501, L537-560 |
| **Type** | `float` (0.0-1.0) |
| **Status** | **ACTIVE** |
| **Description** | Groups with `pass_rate > BOK_EASY_THRESHOLD` are classified as "easy" and use DrGRPO advantage (z-norm, clip[-2.5, 2.5]) instead of BoK softmax. Rationale: softmax on easy groups wastes capacity; DrGRPO provides cleaner signal. |
| **Evidence** | V14-hard-only: `easy_drgrpo=32-176/1024` (3-17% of groups routed to easy). V12: `easy_drgrpo=416/1024` (41%). |
| **Impact** | Higher threshold = fewer groups classified as easy → more BoK softmax usage. 0.75 means >12/16 trajectories must succeed. |
| **When to Change** | Lower (0.5) to route more groups through DrGRPO. Higher (0.9) for near-pure BoK. |

### 1.9 `BOK_EASY_SCORE_THRESHOLD` (Easy Score Gate)
| Field | Value |
|-------|-------|
| **Default** | `0.5` |
| **Script Line** | L199 |
| **Code Location** | `core_algos.py` L502, L539 |
| **Type** | `float` |
| **Status** | **ACTIVE** |
| **Description** | Minimum mean reward score for a group to qualify as "easy". Works in conjunction with `BOK_EASY_THRESHOLD`. A group must satisfy BOTH `pass_rate > EASY_THRESHOLD` AND `mean_score > EASY_SCORE_THRESHOLD` to be classified as easy. |
| **Impact** | Prevents groups with high pass rate but low quality (e.g., all trajectories barely above 0) from getting DrGRPO treatment. |
| **When to Change** | Raise to 0.7 for stricter easy classification. Lower to 0.3 for more lenient. |

### 1.10 `BOK_FILTER_ALL_CORRECT` (Trivial Group Filtering)
| Field | Value |
|-------|-------|
| **Default** | `1` (ON) |
| **Script Line** | L201 |
| **Code Location** | `core_algos.py` L504, L509-523 |
| **Type** | `int` (0/1) |
| **Status** | **ACTIVE** |
| **Description** | When enabled, groups where ALL K trajectories are correct (reward > 0) receive zero advantage. Rationale: trivially-solved prompts provide no useful gradient signal (can't rank among all-correct). |
| **Evidence** | V14-hard-only: `all_correct_filtered=0-48/1024` (0-5%). V12 (full data): `all_correct_filtered=288/1024` (28%). |
| **Impact** | Reduces wasted gradient on easy problems. More impactful on mixed datasets (many easy samples). |
| **When to Change** | Set to 0 to include all-correct groups (standard GRPO behavior). |

---

## 2. BoK-GRPO Algorithm Parameters — Disabled/Reserved

These parameters exist in the script with `[DISABLED]` annotation. They are preserved for future experiments but currently have no effect on training.

### 2.1 `BOK_ADV_NORMALIZE` [DISABLED]
| Field | Value |
|-------|-------|
| **Default** | `0` (OFF) |
| **Script Line** | L204 |
| **Code Location** | `core_algos.py` L483, L626-632 |
| **Type** | `int` (0/1) |
| **Status** | **DISABLED** — Never activated in V12-V14 |
| **Description** | Per-group advantage post-normalization to zero-mean unit-variance. When ON: after computing softmax-based advantage, normalize each group's advantages to μ=0, σ=1. |
| **Why Disabled** | Dr.GRPO paper shows per-group normalization reduces gradient quality. Current BoK softmax already produces well-calibrated advantages. |
| **Future Use** | **P2 Improvement Plan**: May test `BOK_ADV_NORMALIZE=1` for asymmetric advantage (suppress wrong trajectories more aggressively). |
| **Evidence** | V14-hard-only: `adv_normalize=False` (always OFF). |

### 2.2 `BOK_UNIFORM_MIX` [DISABLED]
| Field | Value |
|-------|-------|
| **Default** | `0.1` |
| **Script Line** | L206 |
| **Code Location** | `core_algos.py` (used in anti-collapse mixing logic) |
| **Type** | `float` (0.0-1.0) |
| **Status** | **DISABLED** — Rarely triggers |
| **Description** | Anti-collapse mechanism: when softmax weights become degenerate (max_weight > 0.99 or entropy < 0.1), mix in uniform distribution: `w = (1-α)*w_softmax + α*w_uniform`. Prevents mode collapse. |
| **Why Disabled** | V14-hard-only: `collapsed=0-8/1024` (< 1% of groups). τ-annealing already prevents collapse by starting with high τ. |
| **Future Use** | May be needed if τ is set very low (< 0.1) or on extremely skewed reward distributions. |

### 2.3 `BOK_LOGIT_CAP` [DISABLED]
| Field | Value |
|-------|-------|
| **Default** | `12.0` |
| **Script Line** | L208 |
| **Code Location** | `core_algos.py` L487, (logit capping before softmax) |
| **Type** | `float` |
| **Status** | **DISABLED** — Redundant with τ-adaptive |
| **Description** | Caps raw logits (z_norm / τ) before softmax to prevent numerical overflow. `logits = clamp(logits, -LOGIT_CAP, LOGIT_CAP)`. |
| **Why Disabled** | With τ ≥ 0.3 and typical z-scores in [-3, 3], logits max out at ~10. LOGIT_CAP=12.0 is never hit. BOK_TAU_ADAPTIVE provides dynamic safety. |
| **Future Use** | Relevant if τ drops below 0.1 (logits could exceed 30). |

### 2.4 `BOK_TAU_ADAPTIVE` + `BOK_TAU_MIN` [DISABLED]
| Field | Value |
|-------|-------|
| **Default** | `TAU_ADAPTIVE=1` (ON in code), `TAU_MIN=0.08` |
| **Script Lines** | L210-211 |
| **Code Location** | `core_algos.py` L488-489 |
| **Type** | `int` (0/1) / `float` |
| **Status** | **DISABLED** — Never activates (τ > TAU_MIN always) |
| **Description** | Safety net: if effective τ drops below `TAU_MIN`, bump it back up. Prevents numerical instability from extremely low τ. Tracked as `tau_bumped` counter. |
| **Why Disabled** | τ-annealing floor is TAU_FINAL=0.3 >> TAU_MIN=0.08. The adaptive bump never triggers. |
| **Evidence** | V14-hard-only: `tau_bumped=0/1024` (never activated). |
| **Future Use** | Relevant if TAU_FINAL is set below 0.1 in aggressive experiments. |

### 2.5 `BOK_DAPO_FILTER` + `BOK_DAPO_AUTO_DISABLE_THRESHOLD` [DISABLED]
| Field | Value |
|-------|-------|
| **Default** | `DAPO_FILTER=0` (OFF), `AUTO_DISABLE=0.5` |
| **Script Lines** | L213-214 |
| **Code Location** | `core_algos.py` L493-497 |
| **Type** | `int` (0/1) / `float` |
| **Status** | **DISABLED** — OFF since V11 |
| **Description** | DAPO-style dynamic sampling: zero-out groups where all trajectories are homogeneous (all correct or all wrong). `AUTO_DISABLE_THRESHOLD`: if >50% of groups would be filtered, auto-disable DAPO to prevent gradient starvation. |
| **Why Disabled** | In V10-V11, DAPO + FORMAT_REJECTION caused "death spiral": filtered groups → fewer gradient batches → model diverges. Disabled for safety. |
| **Evidence** | V14-hard-only: `dapo_filtered=0/1024` (always zero, feature OFF). |
| **Future Use** | May re-enable for DAPO clipping experiments (with FORMAT_REJECTION=0 to avoid death spiral). |

### 2.6 `BOK_MIN_BATCH_STD` [DISABLED]
| Field | Value |
|-------|-------|
| **Default** | `0.1` |
| **Script Line** | L216 |
| **Code Location** | `core_algos.py` L494 |
| **Type** | `float` |
| **Status** | **DISABLED** — Only relevant for `clip_std` fallback |
| **Description** | Floor value for batch standard deviation in `clip_std` fallback mode. Used as `std = max(std, MIN_BATCH_STD)` to prevent division by near-zero std. |
| **Why Disabled** | Current fallback is `drgrpo` (no std division). Only activates when `BOK_FALLBACK_MODE=clip_std`. |
| **Future Use** | Needed if switching fallback to `clip_std`. |

---

## 3. Reward / Trajectory Parameters — Active

These parameters control the trajectory-level reward computation in `StepCount_mask_reward.py`.

### 3.1 `ANSWER_WEIGHT` / `POINT_WEIGHT` / `TRAJECTORY_FORMAT_WEIGHT`
| Field | Value |
|-------|-------|
| **Defaults** | `0.6` / `0.3` / `0.1` |
| **Script Lines** | L235-237 |
| **Code Location** | Passed as `reward_function_kwargs` to reward function |
| **Status** | **ACTIVE** |
| **Description** | Weights for the three reward components: `total_reward = ANSWER_WEIGHT × answer_score + POINT_WEIGHT × point_score + FORMAT_WEIGHT × format_score`. |
| **Impact** | Answer is primary (60%); point provides dense signal (30%); format ensures structural compliance (10%). V11 increased POINT_WEIGHT from 0.2→0.3 due to point degradation in V10. |
| **When to Change** | Increase POINT_WEIGHT to 0.4 if point accuracy drops significantly. Decrease ANSWER_WEIGHT if answer reward is too dominant. |

### 3.2 `TRAJ_RETURN_POINT_STEP_SCORES`
| Field | Value |
|-------|-------|
| **Default** | `1` (ON) |
| **Script Line** | L70 |
| **Code Location** | `StepCount_mask_reward.py` L1732 |
| **Status** | **ACTIVE** |
| **Description** | When ON, returns per-step point scores in the reward output for process reward / GSPO compatibility. Required for `PROCESS_REWARD_ENABLE=1`. |
| **When to Change** | Set to 0 only if PROCESS_REWARD is disabled AND no diagnostics need step-level scores. |

### 3.3 `TRAJ_MISS_DECAY_ENABLE`
| Field | Value |
|-------|-------|
| **Default** | `1` (ON) |
| **Script Line** | L71 |
| **Code Location** | `StepCount_mask_reward.py` L1151 |
| **Status** | **ACTIVE** |
| **Description** | Dense mask point reward: instead of binary 0/1 hit/miss, uses exponential decay based on distance to nearest GT mask. Closer misses get partial credit. When OFF (=0), reverts to original binary 0/1 point scoring. |
| **Sub-params** (code-only, not in script) | `TRAJ_MISS_DECAY_ALPHA=20.0`, `TRAJ_MISS_DECAY_ALPHA_PENALTY=50.0`, `TRAJ_MISS_DECAY_FAILURE_THRESHOLD=0.02` |
| **When to Change** | Set to 0 for ablation study comparing dense vs binary point reward. |

### 3.4 `TRAJ_CONSISTENCY_PENALTY`
| Field | Value |
|-------|-------|
| **Default** | `0.5` |
| **Script Line** | L72 |
| **Code Location** | `StepCount_mask_reward.py` L1902 |
| **Status** | **ACTIVE** |
| **Description** | Multiplier penalty when predicted count (answer) ≠ predicted point count. If `pred_answer != len(pred_points)`, the reward is scaled by `(1 - CONSISTENCY_PENALTY)` = 0.5. Encourages model to count points correctly. |
| **Impact** | With 0.5: 50% penalty for inconsistent answer vs point count. Higher = stricter. |
| **When to Change** | Increase to 0.7-0.8 for stricter consistency requirement. Set to 0 to disable. |

### 3.5 `TRAJ_EVAL_ANSWER_ONLY_ON_NO_MASK`
| Field | Value |
|-------|-------|
| **Default** | `1` (ON) |
| **Script Line** | L73 |
| **Code Location** | `StepCount_mask_reward.py` L1826 |
| **Status** | **ACTIVE** |
| **Description** | During evaluation, if no mask data is available for a sample (e.g., pixmo-test has no mask annotations), only evaluate answer correctness (skip point-gate, point reward, etc.). Prevents penalizing model for missing mask data in eval. |
| **When to Change** | Set to 0 if eval set has mask data AND you want point-level eval metrics. |

### 3.6 `TRAJ_ANSWER_GATE_MODE`
| Field | Value |
|-------|-------|
| **Default** | `off` |
| **Script Line** | L74 |
| **Code Location** | `StepCount_mask_reward.py` L1830 |
| **Status** | **ACTIVE** (currently set to `off`) |
| **Description** | Controls answer reward gating based on point accuracy: `off` = no gating (answer reward independent of point accuracy), `soft` = multiply answer reward by `soft_base + (1-soft_base) × point_score`, `hard` = zero answer reward if `point_score < GATE_THRESHOLD`. |
| **Impact** | `off` maximizes correct-wrong gap for BoK-GRPO (V9 Opt-C finding). `soft`/`hard` couple answer to point quality. |
| **When to Change** | Switch to `soft` to penalize "lucky guesses" (correct answer but wrong points). |

### 3.7 `TRAJ_SOFT_ANSWER_DECAY`
| Field | Value |
|-------|-------|
| **Default** | `1` (ON) |
| **Script Line** | L81 |
| **Code Location** | `StepCount_mask_reward.py` L1869 |
| **Status** | **ACTIVE** |
| **Description** | Wrong answers get distance-proportional partial credit instead of 0. Formula: `wrong_score = max(0, exp(-alpha × |pred - gt| / gt))`, capped at `DECAY_CAP`. Closer wrong answers get more credit, encouraging the model to be "close" even when wrong. |
| **When to Change** | Set to 0 for binary (correct=1, wrong=0) ablation. |

### 3.8 `TRAJ_ANSWER_DECAY_ALPHA`
| Field | Value |
|-------|-------|
| **Default** | `8.0` |
| **Script Line** | L82 |
| **Code Location** | `StepCount_mask_reward.py` L1878 |
| **Status** | **ACTIVE** |
| **Description** | Exponential decay steepness for wrong answer partial credit. Higher α = faster decay (less credit for far-off answers). V9 Opt-C: increased from 5.0→8.0 to reduce partial credit for high-GT wrong answers. |
| **Sub-behavior** | For undercounting with GT > `TRAJ_UNDER_ALPHA_GT_THRESHOLD`: α is scaled by `TRAJ_UNDER_ALPHA_GT_SCALE` (0.5), making undercounting less penalized at high GT. |
| **When to Change** | Lower (3-5) for more lenient partial credit; higher (10-15) for stricter. |

### 3.9 `TRAJ_ANSWER_DECAY_CAP`
| Field | Value |
|-------|-------|
| **Default** | `0.4` |
| **Script Line** | L83 |
| **Code Location** | `StepCount_mask_reward.py` L1891 |
| **Status** | **ACTIVE** |
| **Description** | Maximum partial credit for wrong answers. Ensures `wrong_score ≤ 0.4` (while correct = 1.0), maintaining a gap ≥ 0.48 between correct and best-wrong. Prevents wrong answers from getting too much reward. |
| **When to Change** | Lower to 0.2 for larger correct-wrong gap. Raise to 0.6 for more lenient (dangerous — reduces separation). |

### 3.10 `TRAJ_EXTRA_POINT_PENALTY_LAMBDA`
| Field | Value |
|-------|-------|
| **Default** | `1.0` |
| **Script Line** | L84 |
| **Code Location** | `StepCount_mask_reward.py` L1212 |
| **Status** | **ACTIVE** |
| **Description** | V11-fix: penalizes extra points beyond GT count. If model predicts more points than GT, penalty = `lambda × (extra_count / gt_count)`. Eliminates overcounting bias observed in V10. |
| **When to Change** | Reduce (0.5) for softer penalty. Set to 0 to disable (not recommended — overcounting will return). |

### 3.11 `TRAJ_UNDER_ALPHA_GT_SCALE` + `TRAJ_UNDER_ALPHA_GT_THRESHOLD`
| Field | Value |
|-------|-------|
| **Defaults** | `SCALE=0.5`, `THRESHOLD=5` |
| **Script Lines** | L85-86 |
| **Code Location** | `StepCount_mask_reward.py` L1883-1884 |
| **Status** | **ACTIVE** |
| **Description** | V11-fix: for undercounting (pred < gt) when GT > THRESHOLD (5), the decay alpha is scaled by SCALE (0.5), making undercounting less penalized. Rationale: at high GT (6-10), undercounting by 1-2 is less severe than at GT=2. |
| **When to Change** | Set SCALE=1.0 to disable leniency. Lower THRESHOLD to 3 to apply leniency earlier. |

---

## 4. Reward / Trajectory Parameters — Disabled/Reserved

### 4.1 `TRAJ_ANSWER_GATE_THRESHOLD` [DISABLED with gate_mode=off]
| Field | Value |
|-------|-------|
| **Default** | `0.4` |
| **Script Line** | L76 |
| **Code Location** | `StepCount_mask_reward.py` L1692, L1855-1863 |
| **Status** | **DISABLED** — Only active when `TRAJ_ANSWER_GATE_MODE=hard` |
| **Description** | Hard gate threshold: if point_score < this threshold, answer reward is zeroed. Requires `GATE_MODE=hard`. |
| **Future Use** | Re-enable if switching to hard gating mode for strict quality control. |

### 4.2 `TRAJ_SOFT_GATE_BASE` [DISABLED with gate_mode=off]
| Field | Value |
|-------|-------|
| **Default** | `0.8` |
| **Script Line** | L78 |
| **Code Location** | `StepCount_mask_reward.py` L1848 |
| **Status** | **DISABLED** — Only active when `TRAJ_ANSWER_GATE_MODE=soft` |
| **Description** | Soft gate base: answer_reward = `soft_base + (1 - soft_base) × point_score`. With base=0.8, even poor point accuracy still gets 80% of answer reward. |
| **Future Use** | Re-enable with `GATE_MODE=soft` for gentler point-answer coupling. |

### 4.3 `TRAJ_FORMAT_REJECTION` [DISABLED]
| Field | Value |
|-------|-------|
| **Default** | `0` (OFF) |
| **Script Line** | L80 |
| **Code Location** | `StepCount_mask_reward.py` L1916 |
| **Status** | **DISABLED** — OFF since V11 |
| **Description** | When ON (=1), trajectories with format_score ≤ 0.0 receive complete reward rejection (total_reward = 0). Designed to enforce strict format compliance. |
| **Why Disabled** | Caused "death spiral" with DAPO in V10-V11: rejected trajectories → fewer gradient samples → model quality drops → more format failures. |
| **Future Use** | May re-enable in late-stage fine-tuning (not early RL training) when format compliance is already high (>95%). |

---

## 5. Training Infrastructure Parameters

### 5.1 `STEPCOUNT_RL_MODE`
| Field | Value |
|-------|-------|
| **Default** | `bok_grpo` |
| **Options** | `grpo_standard`, `drgrpo`, `gspo`, `process_reward`, `dapo`, `bok_grpo` |
| **Script Line** | L102 |
| **Description** | Master switch controlling which advantage estimator and training algorithm is used. Each mode configures: ADV_ESTIMATOR, PROCESS_REWARD_ENABLE, ACTOR_LR, ROLLOUT_TEMPERATURE, CLIP_RATIO_*, DISABLE_KL. |

### 5.2 `ADV_ESTIMATOR`
| Field | Value |
|-------|-------|
| **Default** | `bok_grpo` (auto-set by mode) |
| **Description** | Advantage estimator identifier. Passed to `verl.trainer.main` as `algorithm.adv_estimator`. Dispatches to `compute_bok_grpo_advantage()` in core_algos.py. |

### 5.3 `ACTOR_LR`
| Field | Value |
|-------|-------|
| **Default** | `1e-6` |
| **Description** | Actor learning rate. V14: reduced from 1.5e-6 to 1e-6 (NaN-free). V7 GRPO verified stable at 1e-6. With lr_warmup_ratio=0.05. |

### 5.4 `PROCESS_REWARD_ENABLE`
| Field | Value |
|-------|-------|
| **Default** | `0` (OFF for bok_grpo) |
| **Description** | Whether to use per-turn process reward. OFF for BoK-GRPO because `token_level_rewards.sum(dim=-1)` cancels token distribution, making process reward equivalent to outcome reward. |

### 5.5 `CLIP_RATIO_LOW` / `CLIP_RATIO_HIGH` / `CLIP_RATIO_DUAL`
| Field | Value |
|-------|-------|
| **Defaults** | `0.2` / `0.28` / `3.0` |
| **Description** | PPO-style clipping ratios. `LOW`: lower bound for importance ratio clip. `HIGH`: upper bound. `DUAL`: dual clip threshold for extremely high ratios. Standard PPO uses 0.2/0.3 symmetric. |

### 5.6 `DISABLE_KL`
| Field | Value |
|-------|-------|
| **Default** | `false` |
| **Description** | Whether to disable KL divergence penalty. When false, KL coef = 2e-2. GSPO and DAPO disable KL for different reasons. |

### 5.7 `ROLLOUT_TEMPERATURE`
| Field | Value |
|-------|-------|
| **Default** | `1.0` |
| **Description** | Sampling temperature for rollout generation. Keep at 1.0 for controlled experiments (seed provides diversity). |

### 5.8 `ROLLOUT_N`
| Field | Value |
|-------|-------|
| **Default** | `16` |
| **Description** | Number of rollout trajectories per prompt. GRPO group size = 16. Must match advantage estimator's assumption. |

---

## 6. Interleaved Rollout Parameters

### 6.1 `INTERLEAVED_MAX_TURNS`
| **Default** | `11` |
| **Description** | Maximum trajectory turns (10 point turns + 1 answer turn). Exceeding → default wrong. |

### 6.2 `INTERLEAVED_PER_TURN_MAX_TOKENS` / `INTERLEAVED_ANSWER_TURN_MAX_TOKENS`
| **Defaults** | `1800` / `1800` |
| **Description** | Max token count per intermediate point turn / final answer turn. |

### 6.3 `INTERLEAVED_HISTORY_MODE`
| **Default** | `0` |
| **Options** | `0`=no text history, `1`=last 1 turn, `-1`=full history, `N`=last N turns |
| **Description** | How much previous turn text is included in the context for each new turn. `0` = only image feedback, no text history. |

### 6.4 `INTERLEAVED_FIRST_TURN_PROMPT_FILE` / `INTERLEAVED_PROCESS_PROMPT_FILE` / `SYSTEM_PROMPT_FILE`
| **Description** | Prompt template files. First turn prompt is empty (disabled v7+, system prompt used instead). Process prompt is injected for intermediate turns. System prompt is injected via `data.system_prompt_file`. |

---

## 7. Debug / Logging Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STEPCOUNT_TRAJ_REASON_DEBUG` | `1` | Enable trajectory reasoning debug output |
| `STEPCOUNT_TRAJ_REASON_DEBUG_EVERY` | `10` | Log every N steps |
| `STEPCOUNT_TRAJ_EVENT_LOG` | `1` | Enable trajectory event logging |
| `STEPCOUNT_TRAJ_EVENT_LOG_EVERY` | `10` | Log events every N steps |
| `STEPCOUNT_TRAJ_EVENT_LOG_MAX` | `50` | Max events per log batch |
| `EASYR1_REWARD_DEBUG_EVERY` | `10` | Reward debug output frequency |
| `EASYR1_REWARD_SAMPLE_DEBUG` | `1` | Enable per-sample reward debug |
| `EASYR1_REWARD_SAMPLE_DEBUG_MAX` | `1` | Max samples to debug per step |
| `EASYR1_REWARD_HEALTH_DEBUG` | `1` | Enable reward health monitoring |
| `EASYR1_REWARD_HEALTH_DEBUG_EVERY` | `1` | Health check frequency |
| `INTERLEAVED_DEBUG` | `1` | Enable interleaved rollout debug |
| `INTERLEAVED_DEBUG_PRINT_CHARS` | `50` | Max chars to print per debug msg |
| `STEPCOUNT_MASK_DEBUG` | `1` | Enable mask matching debug stats |
| `STEPCOUNT_MASK_DEBUG_EVERY` | `10` | Mask debug frequency |
| `STEPCOUNT_MASK_REQUIRE` | `1` | Require mask data (abort if unavailable) |
| `STEPCOUNT_MASK_PREFILL_BY_TURN` | `1` | Pre-fill used_sample_ids by turn |
| `STEPCOUNT_MASK_LOG_CONFIG` | `1` | Log mask config at startup |
| `STEPCOUNT_FORCE_TRAJECTORY_FOR_NUMERIC_GT` | `1` | Treat numeric GT as trajectory mode |

---

## Appendix: Parameter Quick Reference Card

### BoK-GRPO (Active)
```
BOK_TAU=0.3            BOK_CLIP=3.0           BOK_TAU_INIT=0.7
BOK_TAU_FINAL=0.3      BOK_TOTAL_STEPS=182    BOK_TAU_SCHEDULE=cosine
BOK_LOW_VAR_THRESHOLD=3e-4  BOK_FALLBACK_MODE=drgrpo
BOK_EASY_THRESHOLD=0.75    BOK_EASY_SCORE_THRESHOLD=0.5
BOK_FILTER_ALL_CORRECT=1
```

### BoK-GRPO (Disabled)
```
BOK_ADV_NORMALIZE=0         # [P2 future]
BOK_UNIFORM_MIX=0.1         # [anti-collapse safety]
BOK_LOGIT_CAP=12.0           # [overflow safety]
BOK_TAU_ADAPTIVE=1, BOK_TAU_MIN=0.08  # [τ floor safety]
BOK_DAPO_FILTER=0, BOK_DAPO_AUTO_DISABLE_THRESHOLD=0.5  # [DAPO sampling]
BOK_MIN_BATCH_STD=0.1       # [clip_std fallback]
```

### Reward (Active)
```
ANSWER_WEIGHT=0.6  POINT_WEIGHT=0.3  TRAJECTORY_FORMAT_WEIGHT=0.1
TRAJ_MISS_DECAY_ENABLE=1  TRAJ_CONSISTENCY_PENALTY=0.5
TRAJ_EVAL_ANSWER_ONLY_ON_NO_MASK=1  TRAJ_ANSWER_GATE_MODE=off
TRAJ_SOFT_ANSWER_DECAY=1  TRAJ_ANSWER_DECAY_ALPHA=8.0  TRAJ_ANSWER_DECAY_CAP=0.4
TRAJ_EXTRA_POINT_PENALTY_LAMBDA=1.0
TRAJ_UNDER_ALPHA_GT_SCALE=0.5  TRAJ_UNDER_ALPHA_GT_THRESHOLD=5
```

### Reward (Disabled)
```
TRAJ_ANSWER_GATE_THRESHOLD=0.4   # [needs gate_mode=hard]
TRAJ_SOFT_GATE_BASE=0.8          # [needs gate_mode=soft]
TRAJ_FORMAT_REJECTION=0          # [death spiral risk]
```
