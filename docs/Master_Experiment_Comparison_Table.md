# StepCount RL Master Experiment Comparison Table

Updated: 2026-04-27  
Scope: V12-V28 trajectory-mode StepCount RL experiments.  
Principle: direct training logs are authoritative for actual runtime behavior; scripts are secondary because several defaults were edited after some runs.

## 1. Direct log comparison table

| Version / log | Data | LR | PPO | KL | max_grad | Grad guard / BoK routing | Strict point JSON | Best parsed val | Last parsed val | Max train fmt_fail | Grad / NaN signal | Conclusion |
|---|---|---:|---:|---:|---:|---|---:|---|---|---:|---|---|
| V12 | easy-only | 1.5e-6 | 2 | 0.02 | 0.5 | none | 0 | direct `val/reward_score=0.7750` | direct `0.7750` | 0.078 | 273 non-finite skip lines | Strong sparse baseline, but optimizer is not numerically stable. |
| V12-update | easy+hard | 1.5e-6 | 1 | 0.03 | 0.5 | format rejection introduced | 0 | n/a | n/a | n/a | 0 non-finite, 0 spikes in parsed short run | Intermediate short-lived run; no reliable validation conclusion. |
| V12-fix | easy+hard | 1e-6 | 2 | 0.03 | 0.5 | thresh=3.5, abs_cap=4.0 | 0 / old loose parser | `0.7845@200`, fmt_fail=0.533 | `0.7750@213`, fmt_fail=1.000 | 1.000 | 241 non-finite lines, 1 EMA spike, 22 abs-cap skips, 1 non-finite brake | Failed: answer stayed high while trajectory format collapsed. |
| V13 | easy-only warm / variants | 1.5e-6 | 2 | 0.02 | 0.5-1.0 | none | 0 | not parseable in current summary | partial | 0.062-0.078 | 92-107 non-finite lines | Warm-start variants still inherited ppo=2 instability. |
| V14-hard | hard-only | 1e-6 | 2 | 0.02 | 1.0 | none | 0 | not parseable | partial | 1.000 | 21 non-finite lines | Hard-only destabilized format early. |
| V14-mixed | mixed subset | 1.5e-6 | 2 | 0.02 | 1.0 | none | 0 | not parseable | partial | 0.087-0.239 | 0-75 non-finite lines | Mixed data is not fatal alone, but ppo=2 + high LR is risky. |
| V15 | mixed subset | 5e-7 | 2 | 0.02 | 1.0 | none | 0 | not parseable | partial | 0.101 | 0 | Stable but too slow / low ceiling. |
| V16 | mixed subset | 1e-6 | 2 | 0.03 | 1.0 | none | 0 | not parseable | partial | 0.097 | 8 non-finite lines | Better KL, still ppo=2 risk. |
| V17 | easy/mixed line | 1.5e-6 | 2 | 0.02 | 1.0 | none | 0 | not parseable | partial | 0.174 | 38 non-finite lines | High LR + ppo=2 still risky. |
| V18 | easy+hard line | 1e-6 | 2 | 0.02 | 1.0 | early guard | 0 | not parseable | partial | 0.078 | 0 | Guard helps NaN, but ppo=2 is not ideal. |
| V19 | easy+hard line | 1e-6 | 2 | 0.02 | 1.0 | guard variant | 0 | not parseable | partial | 0.969 | 8 EMA spikes | Format collapse can happen without non-finite gradients. |
| V20 | easy+hard | 1e-6 | 2 | 0.02 | 1.0 | guard variant | 0 | `0.7694@60`, fmt_fail=0.025 | `0.7524@120`, fmt_fail=0.301 | 0.438 | 17 non-finite lines, 3 EMA spikes | Partial improvement, late format drift. |
| V21-run1 | easy+hard | 1e-6 | 2 | 0.03 | 1.0 | guard | 0 | `0.7599@60`, fmt_fail=0.026 | `0.7543@120`, fmt_fail=0.947 | 0.984 | 16 EMA spikes, 1 emergency brake | KL alone cannot fix the loose-format reward loophole; run was not gradient-clean. |
| V21-run2 | easy+hard | 1e-6 | 1 | 0.03 | 1.0 | guard | 0 | `0.7675@105`, fmt_fail=0.028 | `0.7561@120`, fmt_fail=1.000 | 1.000 | 0 non-finite, 5 EMA spikes | ppo=1 removes most NaN risk, but not format hacking. |
| V22 | easy+hard | 1e-6 | 2 | 0.03 | 1.0 | thresh=3.0 | 0 | `0.7769@165`, fmt_fail=0.025 | `0.7580@213`, fmt_fail=0.026 | 0.078 | 271 non-finite lines, 3 EMA spikes | Good val peak, but ppo=2 still not clean. |
| V23 | easy+hard | 1e-6 | 1 | 0.03 | 1.0 | thresh=3.0, no abs cap | 0 | `0.7769@135`, fmt_fail=0.028 | `0.7637@213`, fmt_fail=0.034 | 0.062 | 0 non-finite lines, 6 EMA spikes | Best verified optimizer backbone despite a few skipped spike updates. |
| V23+hard | hard-only warm | 1e-6 | 1 | 0.03 | 1.0 | thresh=3.0 | 0 | `0.7637@15`, fmt_fail=0.019 | `0.7637@45`, fmt_fail=0.074 | 0.189 | 4 non-finite lines | Hard-only does not improve the sparse benchmark. |
| V24 | easy+hard | 1.5e-6 | 1 | 0.02 | 1.0 | thresh=3.0, smart_filter=0.955 runtime-confirmed | 0 | `0.7713@165`, fmt_fail=0.026 | `0.7580@213`, fmt_fail=0.021 | 0.078 | 34 non-finite lines, 4 EMA spikes | Smart filter works as a point-gradient release mechanism, but LR=1.5e-6 + KL=0.02 has entropy risk. |
| V25 | easy+hard | 1.5e-6 | 1 | 0.03 | 1.0 | thresh=3.0, no abs cap; runtime smart_filter_th=0.000 | 0 | `0.7826@150`, fmt_fail=0.936 | `0.7769@213`, fmt_fail=0.938 | 1.000 | 138 non-finite lines, 5 EMA spikes | High ceiling, but invalid-format reward loophole dominates. |
| V26 | easy+hard | 1.5e-6 | 1 | 0.03 | 1.0 | thresh=3.5, abs_cap=2.5; runtime smart_filter_th=0.000 | 0 | `0.7713@100`, fmt_fail=0.998 | `0.7713@100`, fmt_fail=0.998 | 1.000 | 14 non-finite lines, 9 abs-cap skips, 1 spike-rate brake | Failed: abs cap did not solve the loose-format reward loophole. |
| V27 latest | easy+hard | 1.5e-6 | 1 | 0.03 | 1.0 | thresh=3.0, smart_filter=0.955, no abs cap | 1 | `0.7561@0`, fmt_fail=0.038 | `0.7164@180`, fmt_fail=0.027 | ~0.47 train / val spike 0.355 | 132 non-finite lines, 7 EMA spikes, 1 emergency brake, entropy~0.94-1.02 | Failed: strict JSON prevents the old format cliff, but high-LR entropy/non-finite instability degrades validation. |
| V27b latest | easy-only | 2e-6 | 1 | 0.03 | 1.0 | thresh=3.0, no abs cap | 1 | pending | pending | pending | pending | Useful ablation, not the main mixed-data run. |
| V28 final design | easy+hard | 1e-6 | 1 | 0.03 | 1.0 | thresh=3.0, smart=0.955, abs_cap=0, temporary cooldown only, fail-fast monitor | 1 | pending | pending | pending | pending | V23-stable optimizer backbone + strict JSON + original non-truncated process prompt + smart-filter point-gradient recovery. |

## 2. Corrections from the 2026-04-26 audit

1. V27 latest is worse than the previously recorded step-140 snapshot: the latest parsed validation is `0.7164@180`, with 132 non-finite lines and 7 EMA spikes.
2. V12-update was missing; it is now included as a short intermediate run without reliable validation.
3. V21-run1 was not gradient-clean: it had 16 EMA spike skips and one emergency brake.
4. V19 had 8 EMA spike skips, so the prior `0` stability signal was incomplete.
5. V23 remains the stability anchor, but the precise statement is `0` non-finite lines with 6 skipped EMA spikes, not completely spike-free.
6. V24 runtime logs confirm `smart_filter_th=0.955`; V25 and V26 runtime logs show `smart_filter_th=0.000` despite script comments/defaults, so their failures are not evidence against smart filtering.
7. V26's brake was a spike-rate emergency brake, not a non-finite brake.

## 2.5 Reward-function version and point-metric comparability

Historical point metrics are not fully comparable across all rows. V23-V25 logs use the `StepCount_mask_reward_v4` line, while V26/V27/V28 use the current `StepCount_mask_reward.py` line with stricter parsing and diagnostics. Therefore:

1. Do not interpret V25's higher early `point_reward` as pure model-quality improvement. Part of the gap is reward-function / parser-version artifact.
2. V25's best answer score (`0.7826@150`) occurred while validation format was already collapsed (`fmt_fail≈0.936`), so it is not a healthy target state.
3. V28 should be judged under the current strict reward function: final answer pass@1 first, then format health, then point reward as a diagnostic signal.
4. Warm-starting from a checkpoint trained under the old reward may improve early format familiarity, but it also introduces reward-version and KL/reference mismatch. The V28 main run therefore uses the SFT checkpoint with the current reward and stability controls.

## 3. First-principles causal conclusions

### A. Mixed data is not the primary root cause

Mixed/easy+hard data appears in both good and bad runs:

- Good/stable backbone: V23 (`0.7769@135`, max train `format_fail_rate=0.062`, `0` non-finite lines).
- Bad/collapsed runs: V12-fix, V25, and V26.

Therefore mixed data is a stressor and improves distribution coverage, but it is not sufficient to explain collapse.

### B. LR is not the only cause, but high LR is the main amplifier after strict JSON

- V25 reached the highest parsed answer score (`0.7826@150`) but did so through a near-complete format collapse under the loose parser.
- V27 removed the old loose-parser reward loophole with strict JSON, but `lr=1.5e-6` still amplified low-variance / zero-reward batches into entropy drift and non-finite skips.
- V23 shows that `lr=1e-6`, `ppo_epochs=1`, and `kl=0.03` are the strongest verified stability backbone.

### C. The old format failure mechanism was loose point parsing

The decisive old-pattern signature is: answer and point remain high while `format_fail_rate` becomes near 1.0.

- V26 final captured train health: `answer_mean=0.7485`, `point_mean=0.7831`, `format_fail_rate=0.9111`, `overall_mean=0.0711`.
- V12-fix final validation: `format_fail_reward=1.0` while `val/reward_score=0.7750`.

This means the model learned semantic count/point behavior while violating the required trajectory format. `TRAJ_POINT_STRICT_JSON=1` is therefore a necessary core fix.

### D. V27's NaN mechanism is different from V26's format cliff

V27's latest log separates the failure modes:

1. strict JSON prevents the V26-style hard format cliff;
2. the active process prompt had been vulnerable to truncation / format ambiguity and has now been restored to the original non-truncated process format;
3. `lr=1.5e-6` amplifies stochastic bad batches;
4. entropy climbs into the `0.94-1.02` band;
5. non-finite gradients start at `opt_step=92` and eventually reach 132 skip lines.

The correct V28 response is not to disable smart filtering; it is to keep smart filtering under the V23-stable optimizer backbone and use fail-fast monitoring for entropy / format / non-finite recurrence.

## 4. Proven effective methods

| Method | Status | Evidence |
|---|---|---|
| `ppo_epochs=1` | Proven | V23 and V21-run2 greatly reduce non-finite risk versus ppo=2 lines. |
| `actor_lr=1e-6` | Proven stability anchor | V23 is the best verified stable optimizer backbone. |
| `kl_coef=0.03` | Proven safer than 0.02 | V23 is stable; lower-KL variants show more entropy risk. |
| `GRAD_SPIKE_THRESHOLD=3.0` | Useful | Skips isolated bad updates; V23 succeeds with this guard. |
| `GRAD_SPIKE_ABSOLUTE_CAP=0` | Recommended | V12-fix/V26 abs caps skipped many steps without preventing collapse. |
| `TRAJ_POINT_STRICT_JSON=1` | Necessary | Prevents malformed point JSON from receiving dense point reward. |
| `TRAJ_FORMAT_REJECTION=1` | Keep | Invalid trajectories should not receive full policy reward. |
| `BOK_CLIP=4.0` | Keep | Used by the high-ceiling BoK line while bounding extreme advantages. |
| `BOK_SMART_FILTER_THRESHOLD=0.955` | Keep for V28 | It releases answer-correct but point-low groups; V24 and V27 runtime logs confirm useful activation. |
| Original non-truncated process prompt | Keep | Preserves SFT-compatible wording while ensuring the final answer instruction is complete. |
| Temporary cooldown instead of emergency LR drop | Keep for V28 | A single oversized update should be skipped and cooled down, not permanently slow the entire run. |
| Fail-fast monitor | Add for V28 | If entropy/format/non-finite recurrence appears, stop and branch instead of silently training a damaged run. |

## 5. V28 final recommendation

Use the standalone script:

```bash
cd /data/workspace/hyleochang/EasyR1-latest
bash examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v28_stable_strict_h200.sh
```

Final V28 main settings:

| Parameter | Value | Rationale |
|---|---:|---|
| `ACTOR_LR` | `1e-6` | V23-stable backbone; avoids V27 high-LR entropy drift. |
| `worker.actor.ppo_epochs` | `1` | Avoids ppo=2 repeated update amplification. |
| `KL_COEF` | `0.03` | Best validated anti-entropy value. |
| `BOK_SMART_FILTER_THRESHOLD` | `0.955` | Keeps point-quality gradients for answer-correct / point-low groups. |
| `TRAJ_POINT_STRICT_JSON` | `1` | Removes malformed-JSON point reward hacking. |
| `INTERLEAVED_PROCESS_PROMPT_FILE` | original non-truncated process prompt | Keeps SFT-style wording and complete answer instruction. |
| `GRAD_SPIKE_ABSOLUTE_CAP` | `0` | Avoids excessive skipped-step artifacts from hard caps. |
| `GRAD_SPIKE_THRESHOLD` | `3.0` | Skips isolated bad updates. |
| `GRAD_SPIKE_COOLDOWN` | `3` | Temporary recovery after a spike. |
| `GRAD_NONFINITE_COOLDOWN` | `10` | Temporary recovery after a non-finite skip. |
| `GRAD_SPIKE_BRAKE_MAX` / `GRAD_NONFINITE_BRAKE_MAX` | `999` / `999` | Emergency LR-drop brakes disabled by default; use skip + temporary cooldown, and fail fast on repeated instability. |
| `V28_FAILFAST_ENABLE` | `1` | Stop damaged runs rather than permanently lowering LR. |

## 6. V28 success / fail-fast criteria

### Expected healthy band

1. Train `format_fail_rate` stays below `0.10` and does not trend toward V26's `0.9+` cliff.
2. Entropy stays around the V23 band (`~0.5-0.7`) and should not sustain above `0.85`.
3. Non-finite gradients should be `0`; one isolated skip is tolerable, but repeated skips mean the run is already damaged.
4. Validation must exceed V12 direct baseline `0.7750`; the first target is to reproduce/exceed V23 `0.7769`, then surpass V12-fix's invalid-format peak without format collapse.
5. `ac_released` should be non-zero but not explosive; a practical target band is roughly `3-8%` of rollouts.

### Automatic fail-fast stop

The V28 script now launches `tools/monitor_v28_failfast.py`, which creates the stop sentinel if any of the following repeats:

- `entropy_loss > 0.85` for 3 consecutive actor reports;
- `format_fail_rate > 0.30` for 3 consecutive reward-health reports;
- `Gradient norm is not finite` reaches 3 total events;
- `low_var_rate > 45%` and `zero_reward_rate > 30%` for 2 warnings.

The launch script watches the sentinel and terminates the trainer, so a damaged run is not allowed to continue into a V27-style NaN burst or a V26-style format cliff.

## 7. Decision summary

V28 should not be another high-LR rescue attempt. It should be a controlled test of the real breakthrough hypothesis:

```text
V23-stable optimizer backbone
+ strict JSON reward support
+ original non-truncated process prompt
+ smart_filter=0.955 point-gradient recovery
+ temporary spike/non-finite cooldown
+ fail-fast stop if entropy/format/non-finite recurrence appears
```

This is the shortest path to beating V12 while preserving V23-like stability.
