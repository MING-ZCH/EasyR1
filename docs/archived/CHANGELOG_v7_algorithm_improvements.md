# Changelog: v7 Algorithm Improvements

> **Date**: 2025-03-05 (during v6 training step ~29)
> **Applies to next training run**: v7
> **Files modified**: verl/trainer/core_algos.py, examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj.sh
> **Backups**: *.bak_v6 files created for both

---

## Summary

Addresses the **low_var=100% problem** in BoK-GRPO and temperature-induced format corruption.

### Root Cause Analysis

| Metric | v6 Training (29 steps) |
|--------|----------------------|
| low_var | **1024/1024 (100%)** at ALL 3 diagnostic logs (steps 1, 11, 21) |
| format_fail | 4.7% ~ 14.1% (avg 8.0%) - temp=1.2 causing corruption |
| pg_loss | [-0.052, 0.056] - non-zero (fallback IS working) |
| reward_mean | ~0.73 avg, positive trend |

**Why low_var=100%?** The counting task produces near-deterministic trajectories:
- For easy prompts: all 16 rollouts succeed with identical rewards (group_std = 0)
- For hard prompts: all 16 rollouts fail with identical rewards (group_std = 0)
- BoK-GRPO softmax path NEVER activates; entirely batch-level z-normalization
- temp=1.2 doesn't create point coordinate diversity (visual grounding dominates)

---

## Changes

### 1. Temperature Fix: 1.2 -> 1.0
- **File**: launch script line 149
- **Rationale**: temp=1.2 increases format_fail (8.0% avg) without improving diversity

### 2. Dr.GRPO-Enhanced Fallback (BOK_FALLBACK_MODE)
- **File**: core_algos.py - compute_bok_grpo_advantage()
- Modes: zscore (v6 default) | **drgrpo** (v7 default) | clip_std
- drgrpo: score - batch_mean (no std division), removes difficulty bias
- Note: gradient magnitude drops ~3-5x; may need LR increase to 5e-6

### 3. DAPO Dynamic Sampling Option (BOK_DAPO_FILTER)
- Default OFF (0) - our counting task has ~100% homogeneous groups
- When ON: zero-out advantages for homogeneous groups

### 4. Enhanced Per-Group Diagnostics
- New [BoK-GRPO-Detail] log every 10 calls
- Reports: group_std distribution (min/p25/median/p75/max), n_zero_std, score range

---

## New Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| BOK_FALLBACK_MODE | drgrpo | Low-var fallback: zscore / drgrpo / clip_std |
| BOK_DAPO_FILTER | 0 | Filter homogeneous groups: 0=off, 1=on |
| BOK_MIN_BATCH_STD | 0.1 | Min batch std for clip_std mode |

## Recommended v7 Configuration
- BOK_FALLBACK_MODE=drgrpo (default), temp=1.0 (default)
- If pg_loss too small: ACTOR_LR=5e-6
- Monitor [BoK-GRPO-Detail] for group_std patterns

## Risk Assessment
- drgrpo reduces gradient magnitude -> may need LR increase
- temp=1.0 is fine since 1.2 didnt help actual diversity
- Can A/B test: BOK_FALLBACK_MODE=zscore for v6 behavior
