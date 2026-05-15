#!/bin/bash
# ================================================================
# V17 Experiment Launch Scripts — Cap=1.0 Single-Variable Experiment
# ================================================================
# Strategy: Keep V16 parameters as baseline, ONLY add Cap=1.0.
#   Do NOT copy V12 parameters — that's untested combination.
#   Let Cap=1.0 prove its value on the existing V16 config.
#
# V17 base config (= V16 + Cap=1.0 + grad_clip=1.0):
#   LR=1e-6, ppo_epochs=2, grad_clip=1.0 (V14-V16 proven)
#   bok_clip=4.0, easy_thresh=0.50, tau_init=0.5 (V15/V16 values)
#   KL=0.02, AllWrong Cap=1.0 (NEW), GradSpikeProtect=OFF
#   ANSWER_WEIGHT=0.6, POINT_WEIGHT=0.3
#
# Historical context:
#   V12 (BEST train=0.914, val=0.776): LR=1.5e-6, clip=0.5, bok_clip=3.0
#     → NaN at step 85, train reward inflated vs val
#   V16 (train=0.828, val=0.828): LR=1e-6, clip=1.0, bok_clip=4.0
#     → 0 NaN, train≈val (no overfitting)
#   V11 (val=0.828): LR=1.5e-6, clip=0.5, 1 epoch, 0 NaN
#
# Dataset step counts (batch=64 prompts/step, 4 GPUs):
#   easy_0_10:   11455 samples => 179 steps/epoch
#   hard_only:    1808 samples =>  29 steps/epoch
#   mixed:        5792 samples =>  91 steps/epoch
#   dense_11_50:  9508 samples => 149 steps/epoch
#
# Usage:
#   cd /path/to/EasyR1-latest
#   bash examples/launch_v17_experiments.sh [A|B|C|D]
# ================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

V17_SCRIPT="examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v17.sh"

EXP="${1:-help}"

case "${EXP}" in
# ================================================================
# Exp-A: Mixed + V16 Config + Cap=1.0 — PRIMARY (RUN FIRST)
# ================================================================
# Single-variable experiment: V16 config + Cap=1.0 on mixed data.
# v17.sh defaults already have Cap=1.0 + V16 params.
# Mixed data gives balanced easy+hard curriculum.
# 2 epochs x 91 steps = 182 total training steps.
# Goal: Prove Cap=1.0 improves over V16's 0.828 val baseline.
A|a)
    echo "[Exp-A] PRIMARY: Mixed + Cap=1.0 (V16 config, single-variable test)"
    echo "  LR=1e-6 | bok_clip=4.0 | easy_thresh=0.50 | tau=0.5 | Cap=1.0"
    STEPCOUNT_TRAIN_DATA=/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_0_10_mixed_hard_easy \
    STEPCOUNT_TOTAL_EPOCHS=2 \
    BOK_TOTAL_STEPS=182 \
    bash "${V17_SCRIPT}"
    ;;

# ================================================================
# Exp-B: Easy + V16 Config + Cap=1.0 — Data Ablation
# ================================================================
# Same as A but easy-only data. Isolates data influence.
# Direct comparison with V16 (same data, same params, +Cap=1.0).
# 2 epochs to match V16's training budget.
B|b)
    echo "[Exp-B] ABLATION: Easy + Cap=1.0 (V16 exact config + Cap)"
    echo "  LR=1e-6 | bok_clip=4.0 | easy_thresh=0.50 | tau=0.5 | Cap=1.0"
    STEPCOUNT_TOTAL_EPOCHS=2 \
    BOK_TOTAL_STEPS=358 \
    bash "${V17_SCRIPT}"
    ;;

# ================================================================
# Exp-C: Mixed + Cap=1.0 + LR=1.5e-6 — High LR Test
# ================================================================
# Tests whether Cap=1.0 makes V12's LR=1.5e-6 safe.
# All other params stay at V16 values (bok_clip=4.0, etc).
# If NaN within 50 steps → Cap alone can't save high LR.
# If stable and better → LR was the main bottleneck.
C|c)
    echo "[Exp-C] HIGH-LR: Mixed + Cap=1.0 + LR=1.5e-6 (risky but informative)"
    echo "  LR=1.5e-6 | bok_clip=4.0 | easy_thresh=0.50 | tau=0.5 | Cap=1.0"
    ACTOR_LR=1.5e-6 \
    STEPCOUNT_TRAIN_DATA=/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_0_10_mixed_hard_easy \
    STEPCOUNT_TOTAL_EPOCHS=1 \
    BOK_TOTAL_STEPS=91 \
    bash "${V17_SCRIPT}"
    ;;

# ================================================================
# Exp-D: Hard-only + Cap=0.8 — Hard Specialization
# ================================================================
# 1808 hard samples, 4 epochs = 116 steps.
# Tighter Cap=0.8 for hard data (more AllWrong groups).
# Lower LR=1e-6 (default) for stability.
D|d)
    echo "[Exp-D] HARD-ONLY: Hard + 4 Epochs + Cap=0.8"
    echo "  LR=1e-6 | bok_clip=4.0 | easy_thresh=0.50 | tau=0.5 | Cap=0.8"
    BOK_ALLWRONG_CAP=0.8 \
    STEPCOUNT_TRAIN_DATA=/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_0_10_hard_only \
    STEPCOUNT_TOTAL_EPOCHS=4 \
    BOK_TOTAL_STEPS=116 \
    bash "${V17_SCRIPT}"
    ;;

# ================================================================
# Help
# ================================================================
*)
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  V17 Experiment Matrix — Cap=1.0 Single-Variable Design            ║"
    echo "║  Base: V16 params + AllWrong Cap=1.0 + grad_clip=1.0              ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Usage: bash examples/launch_v17_experiments.sh [A|B|C|D]"
    echo ""
    echo "  A  Mixed + Cap=1.0         Primary test (mixed data)  [RECOMMENDED]"
    echo "  B  Easy  + Cap=1.0         Data ablation (V16 repro)  [BASELINE]"
    echo "  C  Mixed + Cap=1.0 +LR1.5  High LR test              [RISKY]"
    echo "  D  Hard  + Cap=0.8         Hard specialization        [FOCUSED]"
    echo ""
    echo "Recommended: A first (mixed, primary), then B (baseline comparison)"
    echo "Run C only if A succeeds without NaN — tests LR upper bound"
    echo ""
    echo "V17 base config (all experiments share):"
    echo "  bok_clip=4.0 | easy_thresh=0.50 | tau=0.5 | KL=0.02 | ppo_ep=2"
    echo "  grad_clip=1.0 | Cap=1.0 | GradSpikeProtect=OFF"
    echo ""
    echo "Monitor: If NaN in first 50 steps of C → LR=1.5e-6 is too high"
    ;;
esac
