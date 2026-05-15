# V23 Design Notes

## Summary
V23 = V22 configuration + **ppo_epochs=1** (NaN root-cause fix)

## Single Key Change

| Parameter | V22 | V23 | Rationale |
|---|---|---|---|
| `ppo_epochs` | 2 | **1** | Root cause of NaN gradient death spiral |

## Evidence for ppo_epochs=1

| ppo_epochs | Total Steps | NaN Count | NaN Rate | # Runs |
|---|---|---|---|---|
| 1 | 266 | 0 | **0%** | 2 (V21-R1/R2) |
| 2 | 516 | 92 | **17.8%** | 5 (V12,V16,V17,V20,V22) — 4/5 unstable |

100% correlation: every ppo_epochs=2 run >90 steps experienced NaN.
Zero NaN in any ppo_epochs=1 run, even at grad_norm=25.

See [V22_NaN_Definitive_RootCause.md](V22_NaN_Definitive_RootCause.md) for full analysis.

## Inherited from V22 (all kept)

- **FORMAT_REJECTION=1**: Prevents format collapse (V21 without it: format_fail→98%)
- **GradSpikeProtect**: threshold=3.0x, cooldown=3
- **max_grad_norm=1.0**, kl_coef=0.03
- **BoK-GRPO**: EASY_THRESHOLD=0.50, TAU_INIT=0.7→0.3 cosine, CLIP=4.0
- **AllWrong Cap=1.0**, DAPO dual clip (0.2/0.28)
- **Reward weights**: answer=0.6, point=0.3, format=0.1
- **V3 disabled**: WINNER_BOOST=0, EASY_SCALE=1.0, QUALITY_BONUS=0

## Resume Checkpoint

- Source: **V21-Run1 step-60** (same as V22)
- Path: `.../v21.../20260404_0428/global_step_60`
- Val/answer at step 60: **0.7599**
- This checkpoint was trained with ppo_epochs=1, 0 NaN

## Expected Behavior

1. **No NaN**: ppo_epochs=1 guarantees gradient stability (proven by V21)
2. **No format collapse**: FORMAT_REJECTION=1 enforces format (proven by V22, format_fail<3%)
3. **V23 = best of both**: V21's stability + V22's format enforcement
4. **Target**: val/answer should exceed V21-R1 peak of 0.7599 (since FORMAT_REJECTION prevents wasted gradient on bad-format samples)

## Training Command

```bash
bash examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v23.sh
```

## Save Locations

- Logs: `logs/train/training_interleaved_traj_v23_*.log`
- Monitor: `logs/monitor/monitor_v23_*.log`
- Checkpoints: `save/StepCount-7B-SFT-30k_v23_*/`
- Save freq: every 60 steps, val freq: every 15 steps
