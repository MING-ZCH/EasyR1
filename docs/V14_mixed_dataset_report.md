# V14-Mixed Dataset & Training Script Report (V2)

## Background

V14-hard-only experiment showed val answer_reward declining (0.756 → 0.732 over 20 steps), indicating catastrophic forgetting on pure hard data. Decision: switch to mixed hard+easy dataset starting from SFT base.

## Mixed Dataset Design

### Composition (Total: 5792 samples)

| Component | Count | Description |
|-----------|-------|-------------|
| All Hard | 1808 | 1616 training wrong + 192 external wrong |
| Eval Hard Extra | 384 | 192 external wrong × 2 extra copies (total 3x with hard) |
| Easy (weighted) | 3600 | Sampled from 9839 SFT-correct, weighted by error rate |

### Easy Sampling Weights (by GT error rate)

| GT | SFT Accuracy | Error Rate | Weight | Available | Sampled |
|----|-------------|------------|--------|-----------|---------|
| 1  | 95.0% | 5.0% | 0.33 | 1107 | 117 |
| 2  | 91.0% | 9.0% | 0.59 | 1039 | 212 |
| 3  | 88.6% | 11.4% | 0.75 | 1025 | 268 |
| 4  | 89.2% | 10.8% | 0.71 | 1011 | 254 |
| 5  | 84.1% | 15.9% | 1.04 | 884 | 374 |
| 6  | 84.8% | 15.2% | 1.00 | 2115 | 358 |
| 7  | 82.0% | 18.0% | 1.18 | 1136 | 495 |
| 8  | 80.1% | 19.9% | 1.30 | 783 | **783** (all) |
| 9  | 80.7% | 19.3% | 1.26 | 510 | **510** (all) |
| 10 | 71.8% | 28.2% | 1.85 | 229 | **229** (all) |

### Final GT Distribution

| GT | Count | Notes |
|----|-------|-------|
| 1  | 175   | 117 easy + 58 hard |
| 2  | 357   | 212 easy + 103 hard + 42 eval_extra |
| 3  | 451   | 268 easy + 132 hard + 51 eval_extra |
| 4  | 431   | 254 easy + 123 hard + 54 eval_extra |
| 5  | 595   | 374 easy + 167 hard + 54 eval_extra |
| 6  | 802   | 358 easy + 378 hard + 66 eval_extra |
| 7  | 819   | 495 easy + 249 hard + 75 eval_extra |
| 8  | 1052  | 783 easy + 194 hard + 75 eval_extra |
| 9  | 680   | 510 easy + 122 hard + 48 eval_extra |
| 10 | 430   | 229 easy + 90 hard + 111 eval_extra |

## Training Script: V14-mixed

**File**: `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v14_mixed.sh`

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dataset | StepCountQA-RL-Traj_0_10_mixed_hard_easy (5792) | 1808 hard + 384 eval_extra + 3600 easy |
| Model | SFT base (checkpoint-3537) | Fresh start |
| total_epochs | 2 | ~182 steps total |
| BOK_TOTAL_STEPS | 182 | For tau annealing schedule |
| save_freq | 30 | ~6 val checkpoints |
| LR | 1e-6 | NaN-safe |
| BOK_CLIP | 3.0 | Standard |
| RL Mode | bok_grpo | Best-of-K GRPO |

### Steps Calculation
- 5792 / (16 rollouts x 4 GPUs) = 90.5 -> 91 steps/epoch
- 2 epochs = 182 steps
- Val checkpoints at: step 30, 60, 90, 120, 150, 180

## Build Script

**File**: `dataset/build_mixed_dataset.py`
**Output**: `dataset/StepCountQA-RL-Traj_0_10_mixed_hard_easy/data/train-00000-of-00001.parquet`
**Size**: 2042.6 MB

## Expected Behavior

| Aspect | V14-hard-only | V14-mixed (expected) |
|--------|---------------|----------------------|
| Val trend | Declining (-2.4pp/20 steps) | Stable or improving |
| Catastrophic forgetting | Yes (pure hard) | Mitigated (easy maintains baseline) |
| BoK softmax path | ~86% | ~40-60% (balanced easy/hard routing) |
| Hard data exposure | 100% | 37.8% (1808+384 of 5792) |
| Training steps | 113 (4 epochs) | 182 (2 epochs) |

## Parameter Cleanup (V14-mixed)

### Removed Parameters (Dead/Redundant)

| Parameter | Default | Reason for Removal |
|-----------|---------|-------------------|
| `BOK_LOGIT_CAP=12.0` | 12.0 | Redundant with `BOK_TAU_ADAPTIVE`. V14-hard log: `tau_bumped=0/1024` — tau_adaptive always fires before logit_cap matters |
| `BOK_TAU_ADAPTIVE=1` | 1 in code | Safety net that never activates (`tau_bumped=0/1024`). Code default=1 is fine, no need to export |
| `BOK_TAU_MIN=0.08` | 0.08 in code | Only used by tau_adaptive. Code default sufficient |
| `BOK_ADV_NORMALIZE=0` | 0 in code | Permanently OFF (Dr.GRPO paper finding). Code default=0 |
| `BOK_DAPO_FILTER=0` | 0 in code | Permanently OFF since V11 (caused death spiral in V10). Code default=0 |
| `BOK_DAPO_AUTO_DISABLE_THRESHOLD=0.5` | 0.5 in code | Only relevant when DAPO_FILTER=1, which is OFF |
| `BOK_MIN_BATCH_STD=0.1` | 0.1 in code | Only used in `clip_std` fallback mode; current mode=`drgrpo` |
| `BOK_UNIFORM_MIX=0.1` | 0.1 in code | Rarely triggers (`collapsed=0-8/1024`). Code default sufficient |
| `TRAJ_ANSWER_GATE_THRESHOLD=0.4` | - | Dead with `TRAJ_ANSWER_GATE_MODE=off` |
| `TRAJ_SOFT_GATE_BASE=0.8` | - | Dead with `TRAJ_ANSWER_GATE_MODE=off` |
| `TRAJ_FORMAT_REJECTION=0` | - | Always 0, skipped in reward code |

### Retained Active Parameters

**BoK-GRPO Core (10 params):**
- `BOK_TAU=0.3` — Base tau (overridden by annealing)
- `BOK_CLIP=3.0` — Advantage clipping bound
- `BOK_TAU_INIT=0.7` — Annealing start temperature
- `BOK_TAU_FINAL=0.3` — Annealing end temperature
- `BOK_TOTAL_STEPS=182` — Total steps for annealing schedule
- `BOK_TAU_SCHEDULE=cosine` — Annealing schedule type
- `BOK_LOW_VAR_THRESHOLD=3e-4` — Low-variance group detection
- `BOK_FALLBACK_MODE=drgrpo` — Low-var fallback strategy
- `BOK_EASY_THRESHOLD=0.75` — Difficulty-aware routing threshold
- `BOK_EASY_SCORE_THRESHOLD=0.5` — Score threshold for pass/fail
- `BOK_FILTER_ALL_CORRECT=1` — Filter trivially easy groups

**Reward/Trajectory (11 params):**
- `TRAJ_ANSWER_GATE_MODE=off` — Gating strategy (off=no gating)
- `TRAJ_CONSISTENCY_PENALTY=0.5` — Answer/point count mismatch penalty
- `TRAJ_EVAL_ANSWER_ONLY_ON_NO_MASK=1` — Eval-only: skip gating without mask
- `TRAJ_SOFT_ANSWER_DECAY=1` — Wrong answer gets partial reward
- `TRAJ_ANSWER_DECAY_ALPHA=8.0` — Decay curve steepness
- `TRAJ_ANSWER_DECAY_CAP=0.4` — Wrong answer reward cap
- `TRAJ_MISS_DECAY_ENABLE=1` — Dense mask point reward
- `TRAJ_EXTRA_POINT_PENALTY_LAMBDA=1.0` — Overcounting penalty
- `TRAJ_UNDER_ALPHA_GT_SCALE=0.5` — Undercounting at high GT
- `TRAJ_UNDER_ALPHA_GT_THRESHOLD=5` — GT threshold for scaled undercounting
- `TRAJ_RETURN_POINT_STEP_SCORES=1` — Return step scores for diagnostics

### Evidence from V14-hard-only Logs (20 steps)

| Metric | Value | Implication |
|--------|-------|-------------|
| `tau_bumped` | 0/1024 (always) | BOK_TAU_ADAPTIVE never activates |
| `collapsed` | 0-8/1024 (rare) | BOK_UNIFORM_MIX rarely triggers |
| `low_var` | 0-64/1024 (0-6%) | Low-var is minor factor |
| `dapo_filtered` | 0/1024 (always) | DAPO filter correctly OFF |
| `easy_drgrpo` | 32-176/1024 (3-17%) | Difficulty routing active |
| `all_correct_filtered` | 0-48/1024 (0-5%) | All-correct filter active |

### Net Result
- Removed 11 dead parameters, retained 22 active ones
- Script reduced from 401 to 383 lines
- Syntax validated: OK
