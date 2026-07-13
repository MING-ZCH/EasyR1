# V23 Optimization Analysis: How to Improve Point & Answer

## Current V23 Status
- **Peak val/answer**: 0.7769 (step 135) = V22 peak
- **Point accuracy**: 81.4% hit, 17.1% miss, 1.5% dup
- **Entropy**: 0.49-0.54 (extremely stable, possibly too stable)
- **NaN**: 0 (ppo_epochs=1 fix successful)

## Identified Bottlenecks (Priority Order)

### B1: ~30% of Training Batch Produces No/Weak Gradient for Point

**Evidence**: From BoK-GRPO diagnostics:
- all_correct_filtered: ~19% → zero gradient (16/16 trajectories correct)
- low_var fallback: ~11% → DrGRPO z-norm (very weak signal)
- Total: ~30% of batch contributes nothing to point improvement

**Why it matters**: These "easy" groups all get answer=1.0 but have varying point quality. With BOK_FILTER_ALL_CORRECT=1, the model never receives gradient to improve point accuracy on samples it already counts correctly. Since pass@16 ≈ 80%+, most groups are "easy" and their point signal is wasted.

**Current config**:
```
BOK_FILTER_ALL_CORRECT=1  # ← zero gradient for all-correct groups
BOK_EASY_SCALE=1.0        # ← no scaling applied
BOK_WINNER_BOOST=0        # ← no winner boost
```

### B2: Binary Answer (0.6) Dominates BoK Ranking, Overshadowing Point (0.3)

**Evidence**: 
- answer_weight=0.6, point_weight=0.3, format_weight=0.1
- answer_score is binary (1 or soft_decay ≤ 0.4)
- Two trajectories both answering correctly: overall difference = 0.3 × Δpoint ≤ 0.3
- Two trajectories with different answers: overall difference ≥ 0.6 × (1.0 - 0.4) = 0.36

**Why it matters**: BoK sorts by overall score. In groups where most/all trajectories answer correctly, the only differentiating signal is point quality at 0.3× weight. This is too weak to drive effective BoK selection.

**Current config**:
```
ANSWER_WEIGHT=0.6
POINT_WEIGHT=0.3
TRAJECTORY_FORMAT_WEIGHT=0.1
TRAJ_ANSWER_GATE_MODE=off  # answer not linked to point quality
TRAJ_SOFT_ANSWER_DECAY=1, alpha=8.0, cap=0.4  # wrong answer gets partial reward
```

### B3: HISTORY_MODE=0 — Model Has No Text Memory Across Turns

**Evidence**:
- Step 4 hit rate: 63.6% (worst) — mid-trajectory confusion
- Step 2 hit rate: 72.7% (second worst)
- Steps 9-10: 100% (small object count = easy to see what's left)

**Why it matters**: With HISTORY_MODE=0, each turn only gets:
`base_prompt(image+question) + process_prompt`
No text from previous turns. Model must infer counting state purely from red dots on image, which becomes increasingly difficult as more dots appear.

**Current config**:
```
INTERLEAVED_HISTORY_MODE=0  # no text history across turns
```

### B4: BoK Tau Too High in Early Training — Selection Too Flat

**Evidence**:
- Step 61 (training start): tau=0.624
- Step 130: tau=0.432
- For K=16, softmax(score/0.6) gives much flatter distribution than softmax(score/0.2)

**Current config**:
```
BOK_TAU_INIT=0.7  → BOK_TAU_FINAL=0.3 (cosine over 213 steps)
```

### B5: Entropy Stasis — Possible Under-Exploration

**Evidence**:
- Entropy: 0.49 → 0.54, net change +0.01 in 92 opt steps
- kl_coef=0.03 keeps policy close to reference
- No entropy bonus (default 0)
- Model may be stuck in local optimum for pointing strategy

---

## Optimization Recommendations

### Tier 1: High Impact, Low Risk (Recommended for V24)

#### O1: Remove All-Correct Filtering + Add Easy Scale
```bash
export BOK_FILTER_ALL_CORRECT=0  # Keep gradients for all-correct groups
export BOK_EASY_SCALE=0.3        # Reduce but don't eliminate easy gradient
```
**Effect**: ~19% of batch regains point-quality gradient. Model learns to point better even on "easy" counting tasks.
**Risk**: Low — just adds more gradient signal, doesn't destabilize.

#### O2: Rebalance Reward Weights
```bash
export ANSWER_WEIGHT=0.5
export POINT_WEIGHT=0.4
export TRAJECTORY_FORMAT_WEIGHT=0.1
```
**Effect**: Point quality has 33% more influence (0.4 vs 0.3) on BoK ranking. Two trajectories both correct now differ by up to 0.4 × Δpoint.
**Risk**: Low — answer still dominant at 0.5. V22 used 0.6:0.3:0.1 and had same comments in script noting GSPO mode used 0.5:0.4:0.1.

#### O3: Enable Soft Answer Gate  
```bash
export TRAJ_ANSWER_GATE_MODE=soft
export TRAJ_SOFT_GATE_BASE=0.7
```
**Effect**: answer_score = 0.7 + 0.3 × point_quality when correct. This links answer reward to point quality — perfect pointing gets full 1.0 answer, poor pointing gets 0.7.
**Risk**: Medium — changes reward landscape. Existing correct-but-poor-point trajectories penalized.

### Tier 2: Medium Impact, Medium Risk

#### O4: Lower BoK Tau Schedule
```bash
export BOK_TAU_INIT=0.5
export BOK_TAU_FINAL=0.15
```
**Effect**: Sharper BoK selection earlier. At step 61, tau drops from 0.62 to ~0.45. At step 135, from 0.40 to ~0.25.
**Risk**: Medium — too sharp can cause training instability. Monitor advantage variance.

#### O5: Increase History Mode
```bash
export INTERLEAVED_HISTORY_MODE=2  # Keep last 2 turns of text
```
**Effect**: Model can see "I counted object at (x,y) in step N-1 and N-2" → better spatial reasoning → fewer misses.
**Risk**: Medium — increases KV cache usage and compute time. Need to verify VLLM can handle it.

#### O6: Winner Boost for Hard Groups
```bash
export BOK_WINNER_BOOST=2.0  # Boost rare correct trajectories in hard groups
```
**Effect**: In groups where only 1-3/16 trajectories succeed, the correct ones get 2× advantage boost. Accelerates learning on hard samples.
**Risk**: Low-medium — only affects hard groups (minority of batch).

### Tier 3: Experimental

#### O7: Entropy Bonus
```bash
# Add to algorithm config (code change needed)
entropy_bonus_coeff=0.005
```
**Effect**: Explicit entropy preservation to encourage exploration.
**Risk**: Higher — needs code modification, may conflict with kl_coef.

#### O8: Process Prompt Enhancement (No Training Change)
Add counting state to process prompt:
```
You have already counted {n} objects so far. Continue...
```
**Risk**: Requires code change in trajectory rollout to inject dynamic count.

---

## Recommended V24 Configuration

### Variant A: Conservative (Tier 1 only)
```bash
# Changes from V23:
BOK_FILTER_ALL_CORRECT=0
BOK_EASY_SCALE=0.3
ANSWER_WEIGHT=0.5
POINT_WEIGHT=0.4
```
Expected: +1-3% val/answer improvement from better point gradient utilization.

### Variant B: Moderate (Tier 1 + select Tier 2)
```bash
# All of Variant A plus:
BOK_TAU_INIT=0.5
BOK_TAU_FINAL=0.15
BOK_WINNER_BOOST=2.0
```
Expected: +2-5% val/answer, sharper BoK selection on hard groups.

### Variant C: Aggressive (Tier 1+2, two changes)
```bash
# All of Variant B plus:
TRAJ_ANSWER_GATE_MODE=soft
TRAJ_SOFT_GATE_BASE=0.7
INTERLEAVED_HISTORY_MODE=2
```
Expected: Higher ceiling but more variables to debug.

---

## BoK-GRPO Routing Analysis Under Each Variant

| Component | V23 (current) | V24-A | V24-B |
|-----------|--------------|-------|-------|
| all_correct | 19% → 0 gradient | 19% → 0.3× gradient | Same |
| easy_drgrpo | 47% → 1.0× DrGRPO | 47% → 1.0× DrGRPO | 47% → 1.0× DrGRPO |
| bok_softmax | 23% → full BoK | 23% → full BoK | 23% → sharper BoK |
| low_var | 11% → DrGRPO | 11% → DrGRPO | 11% → DrGRPO |
| allwrong | 3% → capped | 3% → capped | 3% → capped |
| **Effective gradient batch** | **~81%** | **~100%** | **~100%** |

## Summary

The core insight is: **V23's answer accuracy ceiling (~0.78) is limited by point miss rate (17.1%), and point improvement is limited by insufficient gradient signal** — 30% zero-gradient, binary answer dominates ranking, and no text context for multi-turn reasoning.

Priority for next experiment:
1. **O1 (all_correct filter off)** + **O2 (rebalance weights)** = minimal risk, maximum point gradient boost
2. Monitor effect on val, then consider O4 (tau) and O5 (history mode) in subsequent iteration

