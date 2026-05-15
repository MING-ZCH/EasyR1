# V28 First-Principles Diagnosis and Stable Training Design

Date: 2026-04-25

## Executive conclusion

V12-fix and V26 did not achieve the expected optimization. Both runs preserved or temporarily improved answer accuracy, but they failed the actual trajectory-training objective because the policy learned trajectories that still answered correctly while violating the required point/format protocol.

The common mechanism is not simply `mixed data` or `actor_lr`; it is the interaction of four factors:

1. Loose point parsing in older reward runs allowed malformed point text to still receive point credit.
2. Format rejection then turned those malformed trajectories into zero overall reward, causing low-variance / zero-reward batches.
3. Aggressive or poorly placed gradient guards (`GRAD_SPIKE_ABSOLUTE_CAP`) skipped many optimizer steps instead of fixing the reward loophole.
4. V12-fix additionally used `ppo_epochs=2`, doubling policy reuse and increasing non-finite risk.

The correct fix is therefore not prompt-only and not an absolute grad cap. V28 uses strict JSON reward parsing, ppo=1, KL=0.03, GradSpike threshold=3.0, no absolute cap, and an immediate non-finite brake.

## Direct evidence from logs

### V12-fix

Observed configuration from script/log:

- `actor_lr=1e-6`
- `train_files=StepCountQA-RL-Traj_0_10_easy_plus_hard`
- `ppo_epochs=2` (the log echo says ppo=1, but the Hydra config uses `worker.actor.ppo_epochs=2`)
- `max_grad_norm=0.5`
- `kl_coef=0.03`
- `BOK_CLIP=4.0`
- `BOK_SMART_FILTER_THRESHOLD=0.955`
- `GRAD_SPIKE_THRESHOLD=3.5`
- `GRAD_SPIKE_ABSOLUTE_CAP=4.0`

Critical log facts:

- Validation: `val/reward_score=0.7750` at final validation.
- End-of-run format failure: `val/format_fail_reward=1.0`.
- RewardHealth final calls: `format_fail_rate=1.0000`.
- Format failure appeared before the final collapse: first detected `format_fail_rate>=0.5` around RewardHealth call 200; `format_fail_rate>=0.9` around call 213.
- Gradient guard: 22 absolute-cap skipped steps, 32 cooldown lines, 1 non-finite emergency brake.

Interpretation:

V12-fix did not solve the original V12 instability. It converted the run into a high-answer / invalid-format regime. Because `ppo_epochs=2` reuses stale rollouts twice, every reward loophole is amplified. The absolute cap reduced some extreme updates but did not address why those updates occurred.

### V26

Observed configuration from script/log:

- `actor_lr=1.5e-6`
- `train_files=StepCountQA-RL-Traj_0_10_easy_plus_hard`
- `ppo_epochs=1`
- `max_grad_norm=1.0`
- `kl_coef=0.03`
- `BOK_CLIP=4.0`
- `BOK_SMART_FILTER_THRESHOLD=0.955`
- `GRAD_SPIKE_THRESHOLD=3.5`
- `GRAD_SPIKE_ABSOLUTE_CAP=2.5`

Critical log facts:

- No useful validation before collapse in the captured log.
- RewardHealth early: `format_fail_rate=0.0156-0.0312`.
- Format failure cliff: `format_fail_rate>=0.5` around RewardHealth call 76 and `format_fail_rate>=0.9` around call 79.
- Final captured RewardHealth: `overall_mean=0.0711`, `answer_mean=0.7485`, `point_mean=0.7831`, `format_fail_rate=0.9111`.
- Gradient guard: 9 absolute-cap skipped steps, 3 cooldown lines, 1 non-finite emergency brake.

Interpretation:

V26 removed `ppo_epochs=2`, which was good, but it kept the loose parser and added a very low absolute cap. The log pattern is decisive: answer and point scores remain high while overall reward collapses due to format failure. This means the model learned to optimize the wrong objective representation, not that mixed data alone is inherently bad.

### V27 / V27b status

The latest V27 log is now decisive rather than merely early. It separates the failure modes clearly: strict JSON prevents the old format cliff, but high-LR optimization still causes entropy drift, non-finite gradients, and validation degradation.

Observed latest facts:

- V27 latest (`20260425_041701`): easy+hard, `actor_lr=1.5e-6`, `ppo_epochs=1`, `kl=0.03`, `gpu_util=0.65`, strict JSON active, `smart_filter_th=0.955`.
- V27 validation: step 0 `0.7561`, step 20 `0.7448`, step 40 `0.7429`, step 60 `0.7524`, step 80 `0.7316`, step 100 `0.7108`, step 120 `0.7335`, step 140 `0.7316`. It never beats its step-0 baseline.
- V27 format: no V26-style hard cliff. Recent train `format_fail_rate` is about `1-3%`; strict JSON is working.
- V27 entropy: rises past `0.70`, `0.80`, `0.90`, and reaches `>1.00`, far outside the V23 stable band.
- V27 gradients: first non-finite gradient appears at `opt_step=92`; latest full-log audit reports `132` non-finite skip lines and `7` EMA spike detections.
- V27 smart filter remains active and useful: latest monitor shows `allcorrect_filtered=144/1024`, `ac_released=48/1024`, `easy_drgrpo=416/1024`.
- V27 also exposed that silently relying on emergency LR brakes is not a good primary strategy. V28 disables emergency LR-drop brakes by default and instead uses bad-update skip, temporary cooldown, and fail-fast termination.
- V27b latest (`20260425_041455`): easy-only, `actor_lr=2e-6`, 2 GPUs, only step-0 validation is available (`0.7429`), so it is not useful for the main mixed-data conclusion.

Interpretation:

V27 proves the strict-JSON direction but fails as a training run. The root cause is no longer format reward hacking; it is high-LR entropy drift plus non-finite skip bursts. V28 therefore keeps `smart_filter=0.955` but moves to the V23-stable backbone (`lr=1e-6`, `kl=0.03`, `ppo_epochs=1`) with the original non-truncated process prompt, no absolute cap, temporary spike/non-finite cooldown, and fail-fast termination if entropy/format/non-finite recurrence appears.

## Root-cause attribution

### NaN / non-finite gradients

Primary causes:

1. `ppo_epochs=2` in V12/V12-fix: increases effective update intensity and stale-policy reuse.
2. Late zero-reward / low-variance collapse after format failure: corrupts the advantage distribution.
3. Absolute caps: they skip updates after the fact and can produce uneven training, but they do not prevent the reward loophole.

Evidence:

- V12-fix: 22 absolute cap skips + non-finite emergency brake.
- V26: 9 absolute cap skips + non-finite emergency brake despite `ppo_epochs=1`.
- V23: `ppo_epochs=1`, threshold=3.0, no absolute cap, stable captured direct log.

### Format failure

Primary cause:

Older runs allowed malformed point outputs to keep point reward through fallback parsing. This created a Goodhart loop: the model could save tokens and still receive point credit while violating the strict trajectory format.

Evidence:

- V26 final captured state: `answer_mean=0.7485`, `point_mean=0.7831`, but `format_fail_rate=0.9111` and `overall_mean=0.0711`.
- V12-fix final captured state: `format_fail_rate=1.0000`.
- Current reward code now has `TRAJ_POINT_STRICT_JSON=1` default, returning `None` on JSON parse failure instead of falling back to regex coordinates.

### Entropy drift

Primary causes:

1. Too low KL (`kl_coef=0.02` in some runs) combined with stronger update intensity.
2. Format collapse reducing valid reward support, making advantages noisy.
3. Overly sharp BoK routing after the reward surface is corrupted.

V28 keeps KL at 0.03 and does not enable winner boost or quality bonus.

## V28 parameter design

| Parameter | V28 value | Reason |
|---|---:|---|
| Data | `easy_plus_hard` | Highest historical ceiling; aligned with pixmo-test 0-10 distribution. |
| `actor_lr` | `1e-6` | Latest V27 (`1.5e-6`) never beat step-0 validation by step 180, entropy stayed around 0.94-1.02, and non-finite skip lines reached 132; V23 `1e-6` is the stable accuracy backbone. |
| `ppo_epochs` | `1` | Proven stability backbone; avoids V12/V12-fix stale-policy amplification. |
| `max_grad_norm` | `1.0` | V23 stable baseline; avoids V12 sign-GD clipping side effects. |
| `kl_coef` | `0.03` | Stable anti-entropy value from V23/V26 line; safer than 0.02. |
| `BOK_CLIP` | `4.0` | Preserves rare-correct trajectory amplification. |
| `BOK_EASY_THRESHOLD` | `0.50` | V23/V25 standard routing; less aggressive than 0.75. |
| `BOK_SMART_FILTER_THRESHOLD` | `0.955` | Deeper log review shows this is the intended point-quality gradient recovery mechanism; V24/V27 activate it and release 3-5% of rollouts. Entropy risk is confounded with high LR/weak KL, so V28 pairs it with V23-stable LR=1e-6 and KL=0.03. |
| `TRAJ_POINT_STRICT_JSON` | `1` | Core format-failure fix: malformed JSON receives zero point credit. |
| `TRAJ_FORMAT_REJECTION` | `1` | Invalid trajectories do not contribute policy reward. |
| `GRAD_SPIKE_THRESHOLD` | `3.0` | More conservative than V26's 3.5 and historically stable. |
| `GRAD_SPIKE_ABSOLUTE_CAP` | `0` | Absolute caps caused skipped-step bursts in V12-fix/V26. |
| `GRAD_SPIKE_BRAKE_MAX` / `GRAD_NONFINITE_BRAKE_MAX` | `999` / `999` | Emergency LR-drop brakes are disabled by default; bad updates are skipped, temporary cooldown is applied, and repeated failures are handled by fail-fast termination. |
| `gpu_memory_utilization` | `0.65` | H200 140GB safe point: large KV cache while leaving actor-update headroom. |
| `val_freq/save_freq` | `10/10` | Captures early peaks and failure onset. |

## Expected behavior

V28 should be judged successful only if all three conditions hold:

1. `val/reward_score` exceeds the V12 sparse direct benchmark (`0.7750`) and ideally approaches/exceeds V25's historical high-ceiling region.
2. `format_fail_rate` remains near the step-0 baseline and does not enter the V12-fix/V26 cliff pattern.
3. Logs show no repeated non-finite emergency brakes and no absolute-cap skips.



## V28 revision after deeper V27 analysis

The latest V27 full-log audit reached validation step 180 with no hard format cliff but with severe optimization failure: validation fell to `0.7164@180`, entropy stayed around `0.94-1.02`, and non-finite skip lines reached `132`. Therefore the final V28 main script is revised to the V23-stable backbone with the original non-truncated process prompt, temporary cooldown-only gradient recovery, fail-fast monitoring, `ACTOR_LR=1e-6`, `BOK_SMART_FILTER_THRESHOLD=0.955`, `ppo_epochs=1`, `kl=0.03`, `BOK_CLIP=4.0`, `GRAD_SPIKE_THRESHOLD=3.0`, `GRAD_SPIKE_ABSOLUTE_CAP=0`, and `TRAJ_POINT_STRICT_JSON=1`.
