# V20 Training Monitoring Infrastructure

## Overview

V20 introduces a comprehensive training monitoring system with three layers:
1. **Structured log lines** in training code (real-time, per-step)
2. **Cross-version analysis tool** (`tools/analyze_training.py`)
3. **Real-time monitor** (`tools/monitor_training_v2.py`)

---

## 1. Structured Log Lines (Training Code)

### [TrainHealth] — Gradient health per optimizer step
**Source**: `verl/workers/actor/dp_actor.py`  
**Frequency**: Every optimizer step (i.e., 2x per training step with ppo_epochs=2)  
**Format**:
```
[TrainHealth] opt_step=42 grad_norm=1.8234 ema=1.7102 ratio=1.07x spike_count=0 cooldown=0 brake=off lr=1.00e-06
```
**Fields**:
| Field | Description |
|-------|-------------|
| opt_step | Optimizer step counter (monotonic) |
| grad_norm | Pre-clipped gradient norm |
| ema | Exponential moving average of grad_norm (α=0.1) |
| ratio | grad_norm / ema (spike threshold is 3.5x) |
| spike_count | Cumulative spike count since training start |
| cooldown | Remaining cooldown steps (0 = normal) |
| brake | Emergency brake status: "off" or "ON" |
| lr | Current learning rate |

**Key diagnostic use**:
- EMA drift detection (V19 issue: 1.87 → 3.44 due to cooldown contamination)
- Cooldown state visibility (previously invisible)
- Emergency brake activation tracking

### [ValSummary] — Structured validation results
**Source**: `verl/trainer/ray_trainer.py` (inside `_validate()`)  
**Frequency**: Every val_freq steps  
**Format**:
```
[ValSummary] step=15 val/answer_reward=0.7540 val/format_reward=0.9680 val/point_reward=0.9260 val/reward_score=0.7540
```

### [GradSpikeProtect] — Spike/cooldown/brake events
**Source**: `verl/workers/actor/dp_actor.py`  
**Events**:
```
[GradSpikeProtect] SPIKE DETECTED #1: grad_norm=7.0388, EMA=1.8690, ratio=3.77x. Skipping update & entering cooldown for 6 steps.
[GradSpikeProtect] Cooldown step (remaining=5), grad_norm=2.1234, LR=1.00e-07
[GradSpikeProtect] Cooldown ended. LR restored to 1.00e-06
[GradSpikeProtect] EMERGENCY BRAKE: 6 spikes in last 40 steps. Base LR permanently halved.
```

### [RewardHealth] — Aggregate reward health
**Source**: `verl/workers/reward/function.py`  
**Format**:
```
[RewardHealth] calls=256 overall_mean=0.5200 answer_mean=0.4100 point_mean=0.6500 format_fail_rate=0.0600 stop_violation_rate=0.0300 stopped_by_answer_rate=0.0200 turns_exceeded_rate=0.0300 no_point_pred_rate=0.0010
```

### [BoK-GRPO] — Routing distribution
**Source**: `verl/trainer/core_algos.py`  
**Format**:
```
[BoK-GRPO] batch=1024 tau=0.700 step=1/213 low_var=32/1024 collapsed=2 ... easy_drgrpo=112/1024 all_correct_filtered=0/1024 allwrong_capped=16/1024 adv_mean=-0.096 adv_std=1.058
```

---

## 2. Cross-Version Analysis Tool

**Script**: `tools/analyze_training.py`

### Single experiment analysis
```bash
python tools/analyze_training.py \
  --log logs/train/training_*_v20_*.log
```
Output: Config, summary stats, spike events, val progression, FP16 monitor, reward trends.

### Cross-version comparison
```bash
python tools/analyze_training.py \
  --log logs/train/training_*_v12_*.log \
       logs/train/training_*_v19_*.log \
       logs/train/training_*_v20_*.log
```
Output: Side-by-side comparison tables for config, stability, reward, validation, BoK routing.

### Per-step detail
```bash
python tools/analyze_training.py \
  --log logs/train/training_*_v20_*.log \
  --detailed --steps 0-50
```

### JSON export for plotting
```bash
python tools/analyze_training.py \
  --log logs/train/training_*_v20_*.log \
  --json v20_analysis.json
```

**Backward compatibility**: Handles both old YAML format (V12-V19) and new structured logs (V20+). Legacy validation data is automatically extracted from YAML step blocks.

---

## 3. Real-Time Monitor V2

**Script**: `tools/monitor_training_v2.py`  
**Auto-started by V20 launch script as background process.**

### Manual launch
```bash
python tools/monitor_training_v2.py \
  --log logs/train/training_*_v20_*.log \
  --out logs/monitor/v20_monitor.log \
  --interval 60 \
  --every 5 \
  --summary-every 20 \
  --json-metrics logs/monitor/v20_metrics.jsonl
```

### Features
- **Dashboard line** every 5 steps: grad_norm, EMA, ratio, spike/cooldown/brake status, reward, val
- **Immediate alerts** for spikes, brakes, NaN, val results
- **Health checks** with WARNING/CRITICAL thresholds
- **Periodic summaries** every 20 steps
- **JSON Lines export** for per-step metrics
- **Auto early stop** via stop file mechanism

### Health Thresholds
| Metric | WARNING | CRITICAL |
|--------|---------|----------|
| answer_mean | < 0.72 | < 0.68 |
| point_mean | < 0.85 | < 0.80 |
| grad_norm | > 5.0 | > 15.0 |
| EMA drift | > 0.5 | > 2.0 |
| Spike freq (20 steps) | >= 3 | >= 5 |
| KL per token | > 0.15 | > 0.30 |

---

## 4. V20 Changes Summary

### GradSpikeProtect v2 (dp_actor.py)
1. **EMA cooldown fix**: EMA only updated with non-spike AND non-cooldown grad_norms (prevents V19 EMA drift)
2. **Emergency brake**: If ≥6 spikes in 40 optimizer steps → permanently halves base LR
3. **Cooldown extended**: 3 → 6 steps
4. **Spike history**: Tracked per optimizer step for brake window calculation

### bf16 rollback
- V19 used fp16 (caused 87% subnormal gradients, escalating spikes)
- V20 returns to bf16 (default param_dtype, default rollout dtype)

### Monitoring additions
- [TrainHealth] per-optimizer-step log
- [ValSummary] structured validation log
- monitor_training_v2.py (auto-started)
- analyze_training.py (cross-version analysis)

---

## 5. Quick Recipes

### Check if training is healthy
```bash
# Real-time: tail the monitor log
tail -f logs/monitor/monitor_v20_*.log

# Post-hoc: run analysis
python tools/analyze_training.py --log logs/train/training_*_v20_*.log
```

### Compare V20 with V12 (best baseline)
```bash
python tools/analyze_training.py \
  --log logs/train/training_*_v12_*.log \
       logs/train/training_*_v20_*.log
```

### Extract metrics for plotting
```bash
# From monitor's JSON Lines
cat logs/monitor/metrics_v20_*.jsonl | python -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    print(f'{d[\"opt_step\"]},{d[\"grad_norm\"]},{d[\"ema\"]},{d.get(\"answer_mean\",\"\")}')
"
```

### Check spike frequency
```bash
grep "SPIKE DETECTED" logs/train/training_*_v20_*.log | wc -l
```

### Emergency stop
```bash
touch /tmp/v20_stop_*   # Creates stop file, monitor will detect it
```
