#!/usr/bin/env python3
"""
Training Monitor V2 — Enhanced for V20+ GradSpikeProtect v2
============================================================
Real-time log monitoring with:
  - [TrainHealth] parsing for gradient health tracking
  - [ValSummary] structured validation parsing 
  - GradSpikeProtect v2 awareness (EMA fix, emergency brake, cooldown=6)
  - [RewardHealth] aggregate reward monitoring
  - [BoK-GRPO] routing distribution tracking
  - [FP16Monitor] gradient underflow monitoring
  - Auto-generated periodic summary tables
  - JSON Lines per-step metric export

Usage:
  python tools/monitor_training_v2.py \\
    --log logs/train/training_*_v20_*.log \\
    --out logs/monitor/v20_monitor.log \\
    --interval 30 \\
    --json-metrics logs/monitor/v20_metrics.jsonl

  # With early stop:
  python tools/monitor_training_v2.py \\
    --log logs/train/training_*_v20_*.log \\
    --out logs/monitor/v20_monitor.log \\
    --stop-file /tmp/v20_stop \\
    --auto-stop
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

# ──────── Regexes (same as analyze_training.py) ────────

STEP_RE = re.compile(r'\(Runner[^)]*\)\s+Step\s+(\d+)')

TRAIN_HEALTH_RE = re.compile(
    r'\[TrainHealth\]\s+opt_step=(\d+)\s+grad_norm=([\d.]+)\s+ema=([\d.]+)\s+'
    r'ratio=([\d.]+)x\s+spike_count=(\d+)\s+cooldown=(\d+)\s+'
    r'brake=(\w+)\s+lr=([\d.eE+-]+)'
)

VAL_SUMMARY_RE = re.compile(r'\[ValSummary\]\s+step=(-?\d+)\s+(.*)')

SPIKE_DETECT_RE = re.compile(
    r'\[GradSpikeProtect\]\s+SPIKE DETECTED #(\d+):\s+'
    r'grad_norm=([\d.]+),\s+EMA=([\d.]+),\s+ratio=([\d.]+)x'
)
COOLDOWN_RE = re.compile(
    r'\[GradSpikeProtect\]\s+Cooldown step \(remaining=(\d+)\),\s+'
    r'grad_norm=([\d.]+),\s+LR=([\d.eE+-]+)'
)
BRAKE_RE = re.compile(r'\[GradSpikeProtect\]\s+EMERGENCY BRAKE')

REWARD_HEALTH_RE = re.compile(
    r'\[RewardHealth\].*?'
    r'overall_mean=([\d.na]+)\s+answer_mean=([\d.na]+)\s+point_mean=([\d.na]+)\s+'
    r'format_fail_rate=([\d.na]+)\s+stop_violation_rate=([\d.na]+)'
)

BOK_GRPO_RE = re.compile(
    r'\[BoK-GRPO\]\s+batch=(\d+)\s+tau=([\d.]+)\s+step=(\d+)/(\d+)'
)

FP16_MONITOR_RE = re.compile(
    r'\[FP16Monitor\]\s+step=(\d+):\s+'
    r'zero_grad=([\d.]+)%.*?subnormal=([\d.]+)%'
)

NAN_RE = re.compile(r'Gradient norm is not finite')

KV_RE = re.compile(r'\(Runner[^)]*\)\s{1,3}([\w_]+):\s+([\d.eE+-]+)$')
SECTION_RE = re.compile(r'\(Runner[^)]*\)\s+([\w_]+):$')


# ──────── Thresholds ────────

class Thresholds:
    """Health check thresholds for V20+"""
    ANSWER_MEAN_MIN = 0.72
    ANSWER_MEAN_CRITICAL = 0.68
    POINT_MEAN_MIN = 0.85
    POINT_MEAN_CRITICAL = 0.80
    GRAD_NORM_WARNING = 5.0
    GRAD_NORM_CRITICAL = 15.0
    EMA_DRIFT_WARNING = 0.5   # EMA increased by 0.5 from baseline
    EMA_DRIFT_CRITICAL = 2.0
    SPIKE_FREQ_WARNING = 3    # Spikes in last 20 steps
    SPIKE_FREQ_CRITICAL = 5
    KL_WARNING = 0.15
    KL_CRITICAL = 0.30
    CLIPFRAC_WARNING = 0.40
    EARLY_STOP_CONSECUTIVE = 5


# ──────── State ────────

class MonitorState:
    def __init__(self, window: int = 20):
        self.cur_step: Optional[int] = None
        self.section: Optional[str] = None
        self.steps: Dict[int, Dict] = {}
        self.last_reported_step: int = -1
        
        # TrainHealth history
        self.train_health: List[Dict] = []
        self.grad_norm_window: deque = deque(maxlen=window)
        self.ema_window: deque = deque(maxlen=window)
        
        # Spike tracking
        self.spike_steps: List[int] = []
        self.brake_activated: bool = False
        self.total_spikes: int = 0
        
        # Reward tracking
        self.answer_window: deque = deque(maxlen=window)
        self.point_window: deque = deque(maxlen=window)
        
        # Val tracking
        self.val_results: List[Dict] = []
        
        # Counters
        self.nan_count: int = 0
        self.total_lines: int = 0
        self.consecutive_critical: int = 0


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _sf(v, fmt=".4f"):
    if v is None:
        return "--"
    if isinstance(v, str):
        return v
    return f"{v:{fmt}}"


# ──────── Trend ────────

def get_trend(values: deque) -> Optional[float]:
    if len(values) < 4:
        return None
    mid = len(values) // 2
    first = list(values)[:mid]
    second = list(values)[mid:]
    return sum(second)/len(second) - sum(first)/len(first)


# ──────── Line Parser ────────

def parse_line(line: str, state: MonitorState) -> Dict[str, Any]:
    """Parse one log line, update state, return events."""
    line = ANSI_RE.sub('', line).rstrip()
    events = {}
    
    # Step marker
    m = STEP_RE.search(line)
    if m and 'Step' in line and not any(x in line for x in ['opt_step', 'InterleavedDebug']):
        # Only capture real Step markers from ConsoleLogger
        new_step = int(m.group(1))
        if state.cur_step is not None and new_step != state.cur_step:
            state.last_reported_step = state.cur_step
        state.cur_step = new_step
        state.section = None
        state.steps.setdefault(new_step, {})
        return events
    
    # [TrainHealth]
    m = TRAIN_HEALTH_RE.search(line)
    if m:
        th = {
            "opt_step": int(m.group(1)), "grad_norm": float(m.group(2)),
            "ema": float(m.group(3)), "ratio": float(m.group(4)),
            "spike_count": int(m.group(5)), "cooldown": int(m.group(6)),
            "brake": m.group(7), "lr": float(m.group(8)),
            "step": state.cur_step,
        }
        state.train_health.append(th)
        state.grad_norm_window.append(th["grad_norm"])
        state.ema_window.append(th["ema"])
        events["train_health"] = th
        return events
    
    # [ValSummary]
    m = VAL_SUMMARY_RE.search(line)
    if m:
        vs = {"step": int(m.group(1))}
        for kv in m.group(2).split():
            if '=' in kv:
                k, v = kv.split('=', 1)
                try:
                    vs[k] = float(v)
                except:
                    vs[k] = v
        state.val_results.append(vs)
        events["val"] = vs
        return events
    
    # [GradSpikeProtect] SPIKE
    m = SPIKE_DETECT_RE.search(line)
    if m:
        state.total_spikes += 1
        if state.cur_step is not None:
            state.spike_steps.append(state.cur_step)
        events["spike"] = {
            "num": int(m.group(1)), "grad_norm": float(m.group(2)),
            "ema": float(m.group(3)), "ratio": float(m.group(4)),
        }
        return events
    
    # [GradSpikeProtect] BRAKE
    if BRAKE_RE.search(line):
        state.brake_activated = True
        events["brake"] = True
        return events
    
    # [GradSpikeProtect] Cooldown
    m = COOLDOWN_RE.search(line)
    if m:
        events["cooldown"] = {
            "remaining": int(m.group(1)), "grad_norm": float(m.group(2)),
            "lr": float(m.group(3)),
        }
        return events
    
    # [RewardHealth]
    m = REWARD_HEALTH_RE.search(line)
    if m:
        def _sf2(s):
            return float(s) if s != 'na' else None
        rh = {
            "overall": _sf2(m.group(1)), "answer": _sf2(m.group(2)),
            "point": _sf2(m.group(3)), "format_fail": _sf2(m.group(4)),
            "stop_violation": _sf2(m.group(5)), "step": state.cur_step,
        }
        if rh["answer"] is not None:
            state.answer_window.append(rh["answer"])
        if rh["point"] is not None:
            state.point_window.append(rh["point"])
        events["reward_health"] = rh
        return events
    
    # [FP16Monitor]
    m = FP16_MONITOR_RE.search(line)
    if m:
        events["fp16"] = {
            "step": int(m.group(1)), "zero_grad": float(m.group(2)),
            "subnormal": float(m.group(3)),
        }
        return events
    
    # NaN
    if NAN_RE.search(line):
        state.nan_count += 1
        events["nan"] = {"count": state.nan_count}
        return events
    
    # YAML section/kv (for legacy logs)
    m = SECTION_RE.match(line)
    if m:
        state.section = m.group(1)
        return events
    
    m = KV_RE.match(line)
    if m and state.cur_step is not None and state.section:
        key = f"{state.section}/{m.group(1)}"
        try:
            state.steps[state.cur_step][key] = float(m.group(2))
        except:
            pass
        # Check for val data (legacy format)
        if state.section == "val":
            k = m.group(1)
            try:
                v = float(m.group(2))
                if k == "answer_reward" and state.cur_step is not None:
                    events["val_legacy"] = {"step": state.cur_step, "answer": v}
            except:
                pass
    
    return events


# ──────── Health Check ────────

def check_health(state: MonitorState, step: int) -> List[Tuple[str, str, str]]:
    alerts = []
    T = Thresholds
    
    # Grad health from TrainHealth
    if state.train_health:
        latest = state.train_health[-1]
        gn = latest["grad_norm"]
        ema = latest["ema"]
        
        if gn > T.GRAD_NORM_CRITICAL:
            alerts.append(("CRITICAL", "grad_norm", f"grad_norm={gn:.3f} > {T.GRAD_NORM_CRITICAL}"))
        elif gn > T.GRAD_NORM_WARNING:
            alerts.append(("WARNING", "grad_norm", f"grad_norm={gn:.3f} > {T.GRAD_NORM_WARNING}"))
        
        # EMA drift
        if len(state.ema_window) >= 10:
            ema_first = list(state.ema_window)[0]
            drift = ema - ema_first
            if drift > T.EMA_DRIFT_CRITICAL:
                alerts.append(("CRITICAL", "ema_drift", f"EMA drift={drift:.3f} (baseline→now)"))
            elif drift > T.EMA_DRIFT_WARNING:
                alerts.append(("WARNING", "ema_drift", f"EMA drift={drift:.3f} (baseline→now)"))
        
        # Brake check
        if latest["brake"] == "ON":
            alerts.append(("WARNING", "brake", "Emergency brake is ACTIVE"))
    
    # Spike frequency
    if state.spike_steps:
        recent = [s for s in state.spike_steps if s > (step - 20)]
        if len(recent) >= T.SPIKE_FREQ_CRITICAL:
            alerts.append(("CRITICAL", "spike_freq", f"{len(recent)} spikes in last 20 steps"))
        elif len(recent) >= T.SPIKE_FREQ_WARNING:
            alerts.append(("WARNING", "spike_freq", f"{len(recent)} spikes in last 20 steps"))
    
    # Answer mean
    if state.answer_window:
        am = list(state.answer_window)[-1]
        if am < T.ANSWER_MEAN_CRITICAL:
            alerts.append(("CRITICAL", "answer_mean", f"answer_mean={am:.4f} < {T.ANSWER_MEAN_CRITICAL}"))
        elif am < T.ANSWER_MEAN_MIN:
            alerts.append(("WARNING", "answer_mean", f"answer_mean={am:.4f} < {T.ANSWER_MEAN_MIN}"))
        
        trend = get_trend(state.answer_window)
        if trend is not None and trend < -0.03:
            alerts.append(("WARNING", "answer_trend", f"declining trend={trend:+.4f}"))
    
    # Point mean
    if state.point_window:
        pm = list(state.point_window)[-1]
        if pm < T.POINT_MEAN_CRITICAL:
            alerts.append(("CRITICAL", "point_mean", f"point_mean={pm:.4f} < {T.POINT_MEAN_CRITICAL}"))
        elif pm < T.POINT_MEAN_MIN:
            alerts.append(("WARNING", "point_mean", f"point_mean={pm:.4f} < {T.POINT_MEAN_MIN}"))
    
    return alerts


# ──────── Dashboard ────────

def format_dashboard(step: int, state: MonitorState) -> str:
    """One-line dashboard for the checkpoint."""
    th = state.train_health[-1] if state.train_health else {}
    
    gn = th.get("grad_norm")
    ema = th.get("ema")
    ratio = th.get("ratio")
    spk = th.get("spike_count", "-")
    cd = th.get("cooldown", "-")
    brake = th.get("brake", "-")
    lr = th.get("lr")
    
    # RewardHealth
    am = list(state.answer_window)[-1] if state.answer_window else None
    pm = list(state.point_window)[-1] if state.point_window else None
    a_trend = get_trend(state.answer_window)
    
    parts = [
        f"step={step:>4d}",
        f"gn={_sf(gn,'.3f')}",
        f"ema={_sf(ema,'.3f')}",
        f"r={_sf(ratio,'.2f')}x",
        f"spk={spk}",
        f"cd={cd}",
        f"brk={brake}",
        f"lr={_sf(lr,'.1e')}",
        f"am={_sf(am,'.3f')}",
        f"pm={_sf(pm,'.3f')}",
    ]
    if a_trend is not None:
        parts.append(f"a_t={a_trend:+.4f}")
    
    return " | ".join(parts)


# ──────── Periodic Summary ────────

def print_summary(state: MonitorState, out):
    """Print a periodic summary table."""
    out.write(f"\n{'─'*70}\n")
    out.write(f"[{_ts()}] PERIODIC SUMMARY (step={state.cur_step})\n")
    
    # Gradient
    if state.grad_norm_window:
        gn_list = list(state.grad_norm_window)
        out.write(f"  Grad: mean={sum(gn_list)/len(gn_list):.3f} max={max(gn_list):.3f} min={min(gn_list):.3f}\n")
    if state.ema_window:
        ema_list = list(state.ema_window)
        out.write(f"  EMA: current={ema_list[-1]:.4f} first={ema_list[0]:.4f} drift={ema_list[-1]-ema_list[0]:+.4f}\n")
    
    # Spikes
    out.write(f"  Spikes: total={state.total_spikes} brake={'ON' if state.brake_activated else 'off'}\n")
    out.write(f"  NaN: {state.nan_count}\n")
    
    # Val
    if state.val_results:
        latest = state.val_results[-1]
        ans = latest.get("val/answer_reward", "--")
        out.write(f"  Val latest: step={latest['step']} answer={_sf(ans)}\n")
    
    out.write(f"{'─'*70}\n\n")
    out.flush()


# ──────── Main ────────

def main():
    ap = argparse.ArgumentParser(description="Training Monitor V2")
    ap.add_argument("--log", required=True, help="Training log path")
    ap.add_argument("--out", required=True, help="Monitor output log path")
    ap.add_argument("--interval", type=float, default=30.0, help="Poll interval (sec)")
    ap.add_argument("--every", type=int, default=5, help="Dashboard every N steps")
    ap.add_argument("--summary-every", type=int, default=20, help="Periodic summary every N steps")
    ap.add_argument("--stop-file", default=None, help="Early stop sentinel file")
    ap.add_argument("--auto-stop", action="store_true", help="Enable auto early stop")
    ap.add_argument("--json-metrics", default=None, help="JSON Lines metrics output")
    args = ap.parse_args()
    
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    state = MonitorState(window=20)
    pos = 0
    last_summary_step = -1
    
    json_f = None
    if args.json_metrics:
        os.makedirs(os.path.dirname(args.json_metrics) or ".", exist_ok=True)
        json_f = open(args.json_metrics, "a", encoding="utf-8")
    
    try:
        with open(args.out, "a", encoding="utf-8") as out:
            out.write(f"\n{'='*70}\n")
            out.write(f"[{_ts()}] Training Monitor V2 started\n")
            out.write(f"[{_ts()}] Log: {args.log}\n")
            out.write(f"[{_ts()}] Poll: {args.interval}s  Dashboard: every {args.every} steps\n")
            out.write(f"[{_ts()}] Features: TrainHealth, ValSummary, GradSpikeProtect v2\n")
            out.write(f"{'='*70}\n")
            out.flush()
            
            while True:
                # Check stop
                if args.stop_file and os.path.exists(args.stop_file):
                    out.write(f"[{_ts()}] STOP file detected\n")
                    break
                
                # Read log
                try:
                    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
                        f.seek(pos)
                        chunk = f.read(2 * 1024 * 1024)
                        pos = f.tell()
                except FileNotFoundError:
                    time.sleep(args.interval)
                    continue
                
                if not chunk:
                    time.sleep(args.interval)
                    continue
                
                for raw_line in chunk.splitlines():
                    state.total_lines += 1
                    events = parse_line(raw_line, state)
                    
                    # Immediate alerts for critical events
                    if "spike" in events:
                        sp = events["spike"]
                        out.write(f"[{_ts()}] [SPIKE] #{sp['num']} grad_norm={sp['grad_norm']:.3f} "
                                  f"EMA={sp['ema']:.4f} ratio={sp['ratio']:.1f}x\n")
                        out.flush()
                    
                    if "brake" in events:
                        out.write(f"[{_ts()}] [EMERGENCY BRAKE] Activated! Base LR halved.\n")
                        out.flush()
                    
                    if "nan" in events:
                        out.write(f"[{_ts()}] [NaN] #{events['nan']['count']} at step {state.cur_step}\n")
                        out.flush()
                    
                    if "val" in events:
                        vs = events["val"]
                        ans = vs.get("val/answer_reward", "--")
                        pt = vs.get("val/point_reward", "--")
                        out.write(f"[{_ts()}] [VAL] step={vs['step']} answer={_sf(ans)} point={_sf(pt)}\n")
                        out.flush()
                    
                    if "val_legacy" in events:
                        vl = events["val_legacy"]
                        out.write(f"[{_ts()}] [VAL] step={vl['step']} answer={vl['answer']:.4f}\n")
                        out.flush()
                    
                    if "fp16" in events:
                        fp = events["fp16"]
                        out.write(f"[{_ts()}] [FP16] step={fp['step']} zero={fp['zero_grad']:.1f}% subnormal={fp['subnormal']:.1f}%\n")
                        out.flush()
                    
                    # JSON metrics export
                    if json_f and "train_health" in events:
                        th = events["train_health"]
                        # Add reward health context
                        if state.answer_window:
                            th["answer_mean"] = list(state.answer_window)[-1]
                        if state.point_window:
                            th["point_mean"] = list(state.point_window)[-1]
                        json_f.write(json.dumps(th) + "\n")
                        json_f.flush()
                
                # Dashboard for completed steps
                if state.cur_step is not None and state.cur_step > state.last_reported_step:
                    for s in range(state.last_reported_step + 1, state.cur_step + 1):
                        if s % args.every == 0 or s <= 3:
                            dashboard = format_dashboard(s, state)
                            out.write(f"[{_ts()}] {dashboard}\n")
                            
                            # Health check
                            alerts = check_health(state, s)
                            has_critical = False
                            for severity, metric, msg in alerts:
                                icon = "[X]" if severity == "CRITICAL" else "[!]"
                                out.write(f"[{_ts()}] {icon} [{severity}] {msg}\n")
                                if severity == "CRITICAL":
                                    has_critical = True
                            
                            if has_critical:
                                state.consecutive_critical += 1
                            else:
                                state.consecutive_critical = 0
                            
                            # Auto stop
                            if args.auto_stop and state.consecutive_critical >= Thresholds.EARLY_STOP_CONSECUTIVE:
                                out.write(f"[{_ts()}] [AUTO-STOP] {state.consecutive_critical} consecutive critical alerts\n")
                                if args.stop_file:
                                    with open(args.stop_file, "w") as sf:
                                        sf.write(f"Auto-stopped: {state.consecutive_critical} consecutive critical alerts at step {s}")
                                out.flush()
                            
                            out.flush()
                        
                        # Periodic summary
                        if s > 0 and s % args.summary_every == 0 and s > last_summary_step:
                            print_summary(state, out)
                            last_summary_step = s
                    
                    state.last_reported_step = state.cur_step
                
                time.sleep(args.interval)
    
    finally:
        if json_f:
            json_f.close()


if __name__ == "__main__":
    main()
