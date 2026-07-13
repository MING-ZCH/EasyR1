# Grad Norm Spike Detection & Adaptive LR Protection

> Date: 2025-03-26
> Files Modified: `verl/workers/actor/dp_actor.py`, `verl/workers/fsdp_workers.py`
> Script Updated: `examples/..._v17.sh`

---

## Problem

NaN occurs when cumulative policy drift exceeds bf16 tolerance (~1.2e-4 for Qwen2.5-VL-7B). Grad_norm spikes >5x normal appear 5-20 steps before NaN onset, serving as an early warning signal.

**V12 example:** S73: grad_norm=9.38 (spike) → S79: 10.14 (spike) → S85: NaN

## Solution: GradSpikeProtect

A **non-invasive, environment-variable-controlled** mechanism that:

1. **Detects spikes** using Exponential Moving Average (EMA) of grad_norm
2. **Skips spike updates** — the dangerous gradient is NOT applied
3. **Enters cooldown** — LR reduced by 10x for 3 subsequent optimizer steps
4. **Automatically recovers** — LR restored to scheduler value after cooldown

## Configuration (Environment Variables)

| Variable | Default | Description |
|---------|---------|-------------|
| `GRAD_SPIKE_PROTECT` | `1` | Enable (1) or disable (0) spike protection |
| `GRAD_SPIKE_THRESHOLD` | `5.0` | Spike threshold: grad_norm > EMA × threshold |
| `GRAD_SPIKE_COOLDOWN` | `3` | Number of optimizer steps with reduced LR after spike |
| `GRAD_SPIKE_LR_FACTOR` | `0.1` | LR multiplier during cooldown (0.1 = 10x reduction) |

## Architecture

```
  update_policy() called per training step
    ├── for _ in range(ppo_epochs):          # 2
    │   └── for mini_batch in mini_batches:  # 1
    │       ├── for micro_batch: loss.backward()  # gradient accumulation
    │       └── _optimizer_step()            # ← SPIKE DETECTION HERE
    │           ├── clip_grad_norm_()
    │           ├── Check: is grad_norm finite?
    │           │   └── No → skip step (original behavior)
    │           ├── Check: is grad_norm > EMA × threshold?
    │           │   └── Yes → SPIKE: skip step, enter cooldown, reduce LR
    │           ├── Check: in cooldown?
    │           │   └── Yes → apply step with reduced LR, decrement counter
    │           └── Normal → apply step, update EMA
    │
  fsdp_workers.py: lr_scheduler.step()
    └── If cooldown active → override scheduled LR with reduced LR
```

## Key Design Decisions

1. **EMA-based detection** (not absolute threshold): Adapts to different model/data combinations. EMA only updated with non-spike values to keep it stable.

2. **Skip spike gradient entirely** (not just reduce LR): The spike gradient is noisy and harmful. Zero update is better than 0.1× of a bad gradient.

3. **Consecutive spike handling**: If spike occurs during cooldown, counter resets but LR isn't reduced further (already at reduced level).

4. **LR scheduler integration**: After `lr_scheduler.step()` overwrites param_groups, `fsdp_workers.py` re-applies reduced LR if cooldown is still active.

5. **Zero overhead when disabled**: All spike logic gated by `if self._spike_enabled`.

## Metrics Logged

| Metric | Description |
|--------|-------------|
| `actor/grad_norm` | Current gradient norm (unchanged) |
| `actor/grad_norm_ema` | EMA of grad_norm |
| `actor/spike_count` | Cumulative spike count |
| `actor/spike_cooldown` | Remaining cooldown steps |

## Expected Impact (Based on V12 Simulation)

With default settings (threshold=5x, cooldown=3, lr_factor=0.1):

- **V12 scenario**: 2 spikes blocked (S73, S79), 6 steps at 10x reduced LR
- **Drift reduction**: ~70% less cumulative drift from spike-affected region
- **Expected NaN delay**: 30-50+ additional safe training steps
- **Potential outcome**: NaN eliminated entirely (drift stays below threshold)

## How to Disable

```bash
export GRAD_SPIKE_PROTECT=0  # Completely disable spike protection
```

## Files Changed

1. **`verl/workers/actor/dp_actor.py`**:
   - `__init__`: Added spike detection state variables (EMA, counters, config from env vars)
   - `_optimizer_step()`: Rewrote with spike detection → skip → cooldown → LR reduction logic
   - Added `_reduce_lr()` and `_restore_lr()` helper methods
   - Added spike metrics to training log output

2. **`verl/workers/fsdp_workers.py`**:
   - `update_actor()`: After `lr_scheduler.step()`, check if cooldown active and re-apply reduced LR

3. **`examples/..._v17.sh`**:
   - Added `GRAD_SPIKE_PROTECT/THRESHOLD/COOLDOWN/LR_FACTOR` env vars in bok_grpo section
   - Added echo for startup config logging

## Backup

Original `dp_actor.py` saved as `dp_actor.py.bak`.
