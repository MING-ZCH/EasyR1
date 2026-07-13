#!/usr/bin/env python3
"""
Comprehensive Training Analysis Tool for EasyR1 BoK-GRPO / StepCount RL
=========================================================================
Parses training logs (old + new structured formats) and produces:
  1. Per-step structured JSON (for programmatic access)
  2. Comparison tables across experiments
  3. Stability metrics (spike frequency, EMA drift, reward trends)
  4. Key anomaly detection with context

Supports log formats:
  - Console YAML metrics (Step N \\n yaml.dump)
  - [BoK-GRPO] routing lines
  - [BoK-GRPO-Detail] per-group stats
  - [RewardHealth] aggregate lines
  - [GradSpikeProtect] spike/cooldown/brake lines
  - [TrainHealth] per-optimizer-step gradient health (V20+)
  - [ValSummary] structured validation results (V20+)
  - [FP16Monitor] gradient underflow stats
  - [ValTaskMix] validation task distribution
  - NaN detection lines

Usage:
  # Single log analysis
  python tools/analyze_training.py --log logs/train/training_*_v20_*.log

  # Cross-version comparison (multiple logs)
  python tools/analyze_training.py --log logs/train/training_*_v12_*.log logs/train/training_*_v19_*.log logs/train/training_*_v20_*.log

  # Export JSON for programmatic use
  python tools/analyze_training.py --log logs/train/training_*_v20_*.log --json output.json

  # Quick summary (no per-step detail)
  python tools/analyze_training.py --log logs/train/training_*_v20_*.log --summary

  # Filter by step range
  python tools/analyze_training.py --log logs/train/training_*_v20_*.log --steps 0-50
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

# ══════════════════════════ Regex Patterns ══════════════════════════

# Step marker from ConsoleLogger
STEP_RE = re.compile(r'\(Runner[^)]*\)\s+Step\s+(\d+)$')

# [TrainHealth] structured line (V20+)
TRAIN_HEALTH_RE = re.compile(
    r'\[TrainHealth\]\s+opt_step=(\d+)\s+grad_norm=([\d.]+)\s+ema=([\d.]+)\s+'
    r'ratio=([\d.]+)x\s+spike_count=(\d+)\s+cooldown=(\d+)\s+'
    r'brake=(\w+)\s+lr=([\d.eE+-]+)'
)

# [ValSummary] structured line (V20+)
VAL_SUMMARY_RE = re.compile(r'\[ValSummary\]\s+step=(-?\d+)\s+(.*)')

# [GradSpikeProtect] events
SPIKE_DETECT_RE = re.compile(
    r'\[GradSpikeProtect\]\s+SPIKE DETECTED #(\d+):\s+'
    r'grad_norm=([\d.]+),\s+EMA=([\d.]+),\s+ratio=([\d.]+)x'
)
COOLDOWN_RE = re.compile(
    r'\[GradSpikeProtect\]\s+Cooldown step \(remaining=(\d+)\),\s+'
    r'grad_norm=([\d.]+),\s+LR=([\d.eE+-]+)'
)
COOLDOWN_END_RE = re.compile(r'\[GradSpikeProtect\]\s+Cooldown ended\.\s+LR restored to ([\d.eE+-]+)')
BRAKE_RE = re.compile(r'\[GradSpikeProtect\]\s+EMERGENCY BRAKE:\s+(\d+) spikes')

# [RewardHealth]
REWARD_HEALTH_RE = re.compile(
    r'\[RewardHealth\].*?'
    r'overall_mean=([\d.na]+)\s+answer_mean=([\d.na]+)\s+point_mean=([\d.na]+)\s+'
    r'format_fail_rate=([\d.na]+)\s+stop_violation_rate=([\d.na]+)\s+'
    r'stopped_by_answer_rate=([\d.na]+)\s+turns_exceeded_rate=([\d.na]+)'
)

# [BoK-GRPO] routing
BOK_GRPO_RE = re.compile(
    r'\[BoK-GRPO\]\s+batch=(?P<batch>\d+)\s+tau=(?P<tau>[\d.]+)\s+step=(?P<step>\d+)/(?P<total>\d+)\s+'
    r'low_var=(?P<low_var>\d+)/\d+.*?collapsed=(?P<collapsed>\d+).*?'
    r'easy_drgrpo=(?P<easy_drgrpo>\d+)/\d+.*?all_correct_filtered=(?P<all_correct>\d+)/\d+.*?'
    r'allwrong_capped=(?P<allwrong>\d+)/\d+.*?'
    r'(?:winner_boosted=(?P<winner_boosted>\d+)/\d+.*?)?'
    r'adv_mean=(?P<adv_mean>[\d.eE+-]+)\s+adv_std=(?P<adv_std>[\d.eE+-]+)'
)

# [FP16Monitor]
FP16_MONITOR_RE = re.compile(
    r'\[FP16Monitor\]\s+step=(\d+):\s+'
    r'zero_grad=([\d.]+)%.*?subnormal=([\d.]+)%.*?'
    r'max_grad=([\d.eE+-]+).*?min_nonzero=([\d.eE+-]+).*?'
    r'grad_norm=([\d.]+)'
)

# NaN event
NAN_RE = re.compile(r'Gradient norm is not finite')

# YAML-style key-value from ConsoleLogger  
KV_RE = re.compile(r'\(Runner[^)]*\)\s{1,3}([\w_]+):\s+([\d.eE+-]+|!!float\s+[\'"]?\.?nan[\'"]?)$')
SECTION_RE = re.compile(r'\(Runner[^)]*\)\s+([\w_]+):$')


# ══════════════════════════ Parser ══════════════════════════

def _sfloat(s: str) -> Optional[float]:
    """Safe float parse, returns None for 'na' or NaN."""
    if s == 'na' or 'nan' in s.lower():
        return None
    try:
        return float(s)
    except:
        return None


def parse_log(log_path: str, step_range: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """Parse a training log into a structured dict with per-step data.
    
    Returns:
        {
            "log_file": str,
            "version": str (extracted from filename),
            "steps": {step_num: {metrics_dict}},
            "train_health": [{per-optimizer-step health}],
            "val_summaries": [{step, metrics}],
            "spikes": [{step_approx, spike_num, grad_norm, ema, ratio}],
            "cooldowns": [...],
            "brakes": [...],
            "reward_health": [...],
            "bok_routing": [...],
            "fp16_monitor": [...],
            "nan_events": [line_nums],
            "config": {extracted config},
            "summary": {computed summary stats},
        }
    """
    result = {
        "log_file": os.path.basename(log_path),
        "version": _extract_version(log_path),
        "steps": OrderedDict(),
        "train_health": [],
        "val_summaries": [],
        "spikes": [],
        "cooldowns": [],
        "brakes": [],
        "reward_health": [],
        "bok_routing": [],
        "fp16_monitor": [],
        "nan_events": [],
        "config": {},
        "summary": {},
    }
    
    current_step = None
    section = None
    line_num = 0
    config_done = False
    
    # Config patterns
    config_pats = {
        'lr': (re.compile(r'actor_rollout_ref_optim_lr:\s*([\d.eE+-]+)'), float),
        'ppo_epochs': (re.compile(r'ppo_epochs:\s*(\d+)'), int),
        'max_grad_norm': (re.compile(r'max_grad_norm:\s*([\d.]+)'), float),
        'total_epochs': (re.compile(r'total_epochs:\s*(\d+)'), int),
        'rollout_n': (re.compile(r'rollout_n:\s*(\d+)'), int),
    }
    env_keys = [
        'BOK_CLIP', 'BOK_EASY_THRESHOLD', 'BOK_TAU_INIT', 'BOK_TAU_FINAL',
        'BOK_TOTAL_STEPS', 'BOK_LOW_VAR_THRESHOLD', 'BOK_LOGIT_CAP',
        'GRAD_SPIKE_PROTECT', 'GRAD_SPIKE_THRESHOLD', 'GRAD_SPIKE_COOLDOWN',
        'GRAD_SPIKE_BRAKE_WINDOW', 'GRAD_SPIKE_BRAKE_MAX',
        'FP16_GRAD_UNDERFLOW_MONITOR',
    ]
    config_found = set()
    
    with open(log_path, 'r', errors='replace') as f:
        for raw in f:
            line_num += 1
            line = ANSI_RE.sub('', raw).rstrip()
            
            # ── Config extraction (first 5000 lines) ──
            if line_num <= 5000 and not config_done:
                for key, (pat, conv) in config_pats.items():
                    if key not in config_found:
                        m = pat.search(line)
                        if m:
                            result["config"][key] = conv(m.group(1))
                            config_found.add(key)
                
                for ek in env_keys:
                    m = re.search(rf"'{ek}':\s*'([^']+)'", line)
                    if m:
                        try:
                            result["config"][ek] = float(m.group(1))
                        except ValueError:
                            result["config"][ek] = m.group(1)
                
                # dtype detection
                if 'mp_param_dtype' in line:
                    m = re.search(r'mp_param_dtype:\s*(\w+)', line)
                    if m:
                        result["config"]["param_dtype"] = m.group(1)
                
                if line_num == 5000:
                    config_done = True
            
            # ── Step marker ──
            m = STEP_RE.search(line)
            if m:
                current_step = int(m.group(1))
                section = None
                if step_range and (current_step < step_range[0] or current_step > step_range[1]):
                    current_step = None  # Skip this step
                    continue
                result["steps"].setdefault(current_step, {})
                continue
            
            # ── [TrainHealth] (V20+) ──
            m = TRAIN_HEALTH_RE.search(line)
            if m:
                th = {
                    "opt_step": int(m.group(1)),
                    "grad_norm": float(m.group(2)),
                    "ema": float(m.group(3)),
                    "ratio": float(m.group(4)),
                    "spike_count": int(m.group(5)),
                    "cooldown": int(m.group(6)),
                    "brake": m.group(7),
                    "lr": float(m.group(8)),
                    "step": current_step,
                }
                result["train_health"].append(th)
                # Also update current step
                if current_step is not None and current_step in result["steps"]:
                    result["steps"][current_step]["grad_norm"] = th["grad_norm"]
                    result["steps"][current_step]["ema"] = th["ema"]
                    result["steps"][current_step]["spike_ratio"] = th["ratio"]
                    result["steps"][current_step]["lr"] = th["lr"]
                continue
            
            # ── [ValSummary] (V20+) ──
            m = VAL_SUMMARY_RE.search(line)
            if m:
                vs = {"step": int(m.group(1))}
                for kv in m.group(2).split():
                    if '=' in kv:
                        k, v = kv.split('=', 1)
                        vs[k] = _sfloat(v)
                result["val_summaries"].append(vs)
                continue
            
            # ── [GradSpikeProtect] events ──
            m = SPIKE_DETECT_RE.search(line)
            if m:
                result["spikes"].append({
                    "step": current_step, "line": line_num,
                    "spike_num": int(m.group(1)),
                    "grad_norm": float(m.group(2)),
                    "ema": float(m.group(3)),
                    "ratio": float(m.group(4)),
                })
                continue
            
            m = COOLDOWN_RE.search(line)
            if m:
                result["cooldowns"].append({
                    "step": current_step, "remaining": int(m.group(1)),
                    "grad_norm": float(m.group(2)), "lr": float(m.group(3)),
                })
                continue
            
            m = BRAKE_RE.search(line)
            if m:
                result["brakes"].append({
                    "step": current_step, "spikes_in_window": int(m.group(1)),
                })
                continue
            
            # ── [RewardHealth] ──
            m = REWARD_HEALTH_RE.search(line)
            if m:
                rh = {
                    "step": current_step,
                    "overall_mean": _sfloat(m.group(1)),
                    "answer_mean": _sfloat(m.group(2)),
                    "point_mean": _sfloat(m.group(3)),
                    "format_fail_rate": _sfloat(m.group(4)),
                    "stop_violation_rate": _sfloat(m.group(5)),
                    "stopped_by_answer_rate": _sfloat(m.group(6)),
                    "turns_exceeded_rate": _sfloat(m.group(7)),
                }
                result["reward_health"].append(rh)
                continue
            
            # ── [BoK-GRPO] routing ──
            m = BOK_GRPO_RE.search(line)
            if m:
                batch = int(m.group("batch"))
                bok_entry = {
                    "step": int(m.group("step")),
                    "tau": float(m.group("tau")),
                    "total_steps": int(m.group("total")),
                    "batch": batch,
                    "low_var": int(m.group("low_var")),
                    "collapsed": int(m.group("collapsed")),
                    "easy_drgrpo": int(m.group("easy_drgrpo")),
                    "all_correct": int(m.group("all_correct")),
                    "allwrong_capped": int(m.group("allwrong")),
                    "adv_mean": float(m.group("adv_mean")),
                    "adv_std": float(m.group("adv_std")),
                    "winner_boosted": int(m.group("winner_boosted")) if m.group("winner_boosted") else 0,
                }
                bok_entry["bok_softmax"] = batch - bok_entry["low_var"] - bok_entry["easy_drgrpo"] - bok_entry["all_correct"]
                result["bok_routing"].append(bok_entry)
                continue
            
            # ── [FP16Monitor] ──
            m = FP16_MONITOR_RE.search(line)
            if m:
                result["fp16_monitor"].append({
                    "step": int(m.group(1)),
                    "zero_grad_pct": float(m.group(2)),
                    "subnormal_pct": float(m.group(3)),
                    "max_grad": float(m.group(4)),
                    "min_nonzero": float(m.group(5)),
                    "grad_norm": float(m.group(6)),
                })
                continue
            
            # ── NaN events ──
            if NAN_RE.search(line):
                result["nan_events"].append({"line": line_num, "step": current_step})
                continue
            
            # ── YAML key-value metrics ──
            m = SECTION_RE.match(line)
            if m:
                section = m.group(1)
                continue
            
            m = KV_RE.match(line)
            if m and current_step is not None and current_step in result["steps"]:
                key = m.group(1)
                val_str = m.group(2)
                if 'nan' in val_str.lower():
                    val = None
                else:
                    val = _sfloat(val_str)
                
                if section:
                    full_key = f"{section}/{key}"
                else:
                    full_key = key
                result["steps"][current_step][full_key] = val
    
    # ── Compute summary ──
    result["summary"] = _compute_summary(result)
    return result


def _extract_version(path: str) -> str:
    """Extract version tag (e.g., 'v19', 'v20') from log filename."""
    basename = os.path.basename(path)
    m = re.search(r'_v(\d+)_', basename)
    if m:
        return f"v{m.group(1)}"
    m = re.search(r'_v(\d+)', basename)
    if m:
        return f"v{m.group(1)}"
    return basename[:20]


def _compute_summary(data: Dict) -> Dict:
    """Compute aggregate summary statistics."""
    summary = {}
    
    steps = data["steps"]
    step_nums = sorted(steps.keys())
    summary["total_steps"] = len(step_nums)
    summary["step_range"] = f"{step_nums[0]}-{step_nums[-1]}" if step_nums else "none"
    
    # Gradient health
    grad_norms = []
    for s in step_nums:
        gn = steps[s].get("grad_norm")
        if gn is None:
            gn = steps[s].get("actor/grad_norm")
        if gn is not None and isinstance(gn, (int, float)) and math.isfinite(gn):
            grad_norms.append(gn)
    
    if grad_norms:
        summary["grad_norm_mean"] = sum(grad_norms) / len(grad_norms)
        summary["grad_norm_max"] = max(grad_norms)
        summary["grad_norm_min"] = min(grad_norms)
        n10 = min(10, len(grad_norms))
        summary["grad_norm_last10_mean"] = sum(grad_norms[-n10:]) / n10
    
    # Spike stats
    summary["total_spikes"] = len(data["spikes"])
    summary["total_brakes"] = len(data["brakes"])
    summary["total_nans"] = len(data["nan_events"])
    
    if data["spikes"]:
        spike_ratios = [s["ratio"] for s in data["spikes"]]
        summary["spike_ratio_max"] = max(spike_ratios)
        summary["spike_ratio_mean"] = sum(spike_ratios) / len(spike_ratios)
        spike_steps = [s["step"] for s in data["spikes"] if s["step"] is not None]
        if len(spike_steps) >= 2:
            intervals = [spike_steps[i+1] - spike_steps[i] for i in range(len(spike_steps)-1)]
            summary["spike_interval_mean"] = sum(intervals) / len(intervals)
            summary["spike_interval_min"] = min(intervals)
    
    # EMA drift (from TrainHealth)
    if data["train_health"]:
        emas = [th["ema"] for th in data["train_health"] if th["ema"] > 0]
        if len(emas) >= 10:
            n10 = min(10, len(emas))
            summary["ema_first10"] = sum(emas[:n10]) / n10
            summary["ema_last10"] = sum(emas[-n10:]) / n10
            summary["ema_drift"] = summary["ema_last10"] - summary["ema_first10"]
    
    # Reward health trends
    if data["reward_health"]:
        answers = [rh["answer_mean"] for rh in data["reward_health"] if rh["answer_mean"] is not None]
        if answers:
            n10 = min(10, len(answers))
            summary["answer_first10"] = sum(answers[:n10]) / n10
            summary["answer_last10"] = sum(answers[-n10:]) / n10
            summary["answer_trend"] = summary["answer_last10"] - summary["answer_first10"]
    
    # Val progression (from [ValSummary] or legacy YAML steps)
    if data["val_summaries"]:
        vals = data["val_summaries"]
    else:
        # Backward compat: extract from YAML steps dict
        vals = []
        for step_num in sorted(steps.keys()):
            sd = steps[step_num]
            val_ans = sd.get("val/answer_reward")
            if val_ans is not None:
                entry = {"step": step_num, "val/answer_reward": val_ans}
                for vk in ["val/point_reward", "val/format_reward", "val/format_fail_reward",
                           "val/consistency_violation_reward", "val/overall_reward"]:
                    if vk in sd:
                        entry[vk] = sd[vk]
                vals.append(entry)
    
    if vals:
        summary["val_count"] = len(vals)
        ans_key = "val/answer_reward"
        val_answers = [(v["step"], v.get(ans_key)) for v in vals if v.get(ans_key) is not None]
        if val_answers:
            summary["val_first_answer"] = val_answers[0][1]
            summary["val_best_answer"] = max(v[1] for v in val_answers)
            summary["val_last_answer"] = val_answers[-1][1]
            summary["val_best_step"] = max(val_answers, key=lambda x: x[1])[0]
        # Store extracted vals for display
        if not data["val_summaries"]:
            data["val_summaries"] = vals
    
    # BoK routing summary
    if data["bok_routing"]:
        total_batch = sum(r["batch"] for r in data["bok_routing"])
        if total_batch > 0:
            summary["bok_softmax_pct"] = sum(r["bok_softmax"] for r in data["bok_routing"]) / total_batch * 100
            summary["drgrpo_pct"] = sum(r["easy_drgrpo"] for r in data["bok_routing"]) / total_batch * 100
            summary["all_correct_pct"] = sum(r["all_correct"] for r in data["bok_routing"]) / total_batch * 100
            summary["allwrong_capped_pct"] = sum(r["allwrong_capped"] for r in data["bok_routing"]) / total_batch * 100
            summary["winner_boosted_pct"] = sum(r.get("winner_boosted", 0) for r in data["bok_routing"]) / total_batch * 100
            summary["tau_range"] = f"{data['bok_routing'][0]['tau']:.3f}->{data['bok_routing'][-1]['tau']:.3f}"
    
    # FP16 monitor summary
    if data["fp16_monitor"]:
        summary["fp16_subnormal_first"] = data["fp16_monitor"][0]["subnormal_pct"]
        summary["fp16_subnormal_last"] = data["fp16_monitor"][-1]["subnormal_pct"]
        summary["fp16_min_nonzero_first"] = data["fp16_monitor"][0]["min_nonzero"]
        summary["fp16_min_nonzero_last"] = data["fp16_monitor"][-1]["min_nonzero"]
    
    return summary


# ══════════════════════════ Comparison ══════════════════════════

def print_comparison(results: List[Dict], detailed: bool = False):
    """Print side-by-side comparison table of multiple experiments."""
    
    if not results:
        print("No results to compare.")
        return
    
    versions = [r["version"] for r in results]
    max_vlen = max(len(v) for v in versions)
    
    # Header
    print("\n" + "=" * 100)
    print("CROSS-VERSION TRAINING COMPARISON")
    print("=" * 100)
    
    # ── Config ──
    print("\n--- CONFIGURATION ---")
    config_keys = ['lr', 'ppo_epochs', 'max_grad_norm', 'param_dtype', 'BOK_TOTAL_STEPS',
                   'GRAD_SPIKE_PROTECT', 'GRAD_SPIKE_THRESHOLD', 'GRAD_SPIKE_COOLDOWN',
                   'GRAD_SPIKE_BRAKE_WINDOW', 'GRAD_SPIKE_BRAKE_MAX']
    header = f"{'Key':<30}"
    for v in versions:
        header += f" | {v:>{max(max_vlen, 12)}}"
    print(header)
    print("-" * len(header))
    for key in config_keys:
        row = f"{key:<30}"
        for r in results:
            val = r["config"].get(key, "--")
            row += f" | {str(val):>{max(max_vlen, 12)}}"
        print(row)
    
    # ── Summary ──
    print("\n--- TRAINING SUMMARY ---")
    sum_keys = [
        ("total_steps", "Steps completed"),
        ("total_spikes", "Spike events"),
        ("total_brakes", "Emergency brakes"),
        ("total_nans", "NaN events"),
        ("grad_norm_mean", "Grad norm (mean)"),
        ("grad_norm_max", "Grad norm (max)"),
        ("grad_norm_last10_mean", "Grad norm (last 10 mean)"),
        ("spike_ratio_max", "Max spike ratio"),
        ("spike_interval_mean", "Mean spike interval"),
        ("ema_first10", "EMA (first 10)"),
        ("ema_last10", "EMA (last 10)"),
        ("ema_drift", "EMA drift"),
    ]
    header = f"{'Metric':<25}"
    for v in versions:
        header += f" | {v:>{max(max_vlen, 12)}}"
    print(header)
    print("-" * len(header))
    for key, label in sum_keys:
        row = f"{label:<25}"
        for r in results:
            val = r["summary"].get(key)
            if val is None:
                row += f" | {'--':>{max(max_vlen, 12)}}"
            elif isinstance(val, float):
                row += f" | {val:>{max(max_vlen, 12)}.4f}"
            else:
                row += f" | {str(val):>{max(max_vlen, 12)}}"
        print(row)
    
    # ── Reward Health ──
    print("\n--- REWARD HEALTH TREND ---")
    rh_keys = [
        ("answer_first10", "Answer (first 10)"),
        ("answer_last10", "Answer (last 10)"),
        ("answer_trend", "Answer trend"),
    ]
    header = f"{'Metric':<25}"
    for v in versions:
        header += f" | {v:>{max(max_vlen, 12)}}"
    print(header)
    print("-" * len(header))
    for key, label in rh_keys:
        row = f"{label:<25}"
        for r in results:
            val = r["summary"].get(key)
            if val is None:
                row += f" | {'--':>{max(max_vlen, 12)}}"
            else:
                row += f" | {val:>{max(max_vlen, 12)}.4f}"
        print(row)
    
    # ── Validation ──
    print("\n--- VALIDATION (pixmo-test) ---")
    val_keys = [
        ("val_count", "Val checkpoints"),
        ("val_first_answer", "First val answer"),
        ("val_best_answer", "Best val answer"),
        ("val_best_step", "Best val step"),
        ("val_last_answer", "Last val answer"),
    ]
    header = f"{'Metric':<25}"
    for v in versions:
        header += f" | {v:>{max(max_vlen, 12)}}"
    print(header)
    print("-" * len(header))
    for key, label in val_keys:
        row = f"{label:<25}"
        for r in results:
            val = r["summary"].get(key)
            if val is None:
                row += f" | {'--':>{max(max_vlen, 12)}}"
            elif isinstance(val, float):
                row += f" | {val:>{max(max_vlen, 12)}.4f}"
            else:
                row += f" | {str(val):>{max(max_vlen, 12)}}"
        print(row)
    
    # ── BoK Routing ──
    print("\n--- BOK-GRPO ROUTING ---")
    bok_keys = [
        ("bok_softmax_pct", "BoK softmax %"),
        ("drgrpo_pct", "DrGRPO easy %"),
        ("all_correct_pct", "AllCorrect filtered %"),
        ("allwrong_capped_pct", "AllWrong capped %"),
        ("winner_boosted_pct", "V3 Winner boosted %"),
        ("tau_range", "Tau range"),
    ]
    header = f"{'Metric':<25}"
    for v in versions:
        header += f" | {v:>{max(max_vlen, 12)}}"
    print(header)
    print("-" * len(header))
    for key, label in bok_keys:
        row = f"{label:<25}"
        for r in results:
            val = r["summary"].get(key)
            if val is None:
                row += f" | {'--':>{max(max_vlen, 12)}}"
            elif isinstance(val, float):
                row += f" | {f'{val:.1f}%':>{max(max_vlen, 12)}}"
            else:
                row += f" | {str(val):>{max(max_vlen, 12)}}"
        print(row)
    
    # ── Stability Assessment ──
    print("\n--- STABILITY ASSESSMENT ---")
    for r in results:
        v = r["version"]
        s = r["summary"]
        issues = []
        
        if s.get("total_nans", 0) > 0:
            issues.append(f"NaN x{s['total_nans']}")
        if s.get("total_spikes", 0) > 5:
            issues.append(f"Spikes x{s['total_spikes']} (high)")
        elif s.get("total_spikes", 0) > 0:
            issues.append(f"Spikes x{s['total_spikes']}")
        if s.get("total_brakes", 0) > 0:
            issues.append(f"BRAKE activated")
        if s.get("ema_drift") is not None and s["ema_drift"] > 1.0:
            issues.append(f"EMA drift={s['ema_drift']:.2f} (unstable)")
        if s.get("answer_trend") is not None and s["answer_trend"] < -0.03:
            issues.append(f"Answer declining ({s['answer_trend']:+.4f})")
        if s.get("spike_interval_mean") is not None and s["spike_interval_mean"] < 10:
            issues.append(f"Frequent spikes (interval={s['spike_interval_mean']:.0f})")
        
        status = "HEALTHY" if not issues else "UNSTABLE" if len(issues) >= 3 else "WARNING"
        icon = {"HEALTHY": "[OK]", "WARNING": "[!]", "UNSTABLE": "[X]"}[status]
        print(f"  {icon} {v}: {status}", end="")
        if issues:
            print(f" — {'; '.join(issues)}")
        else:
            print()
    
    # ── Detailed per-step (if requested) ──
    if detailed:
        for r in results:
            _print_per_step(r)


def _print_per_step(data: Dict):
    """Print per-step metrics table for a single experiment."""
    v = data["version"]
    print(f"\n{'=' * 80}")
    print(f"PER-STEP DETAIL: {v}")
    print(f"{'=' * 80}")
    
    # Merge train_health into steps for display
    th_by_step = {}
    for th in data["train_health"]:
        s = th.get("step")
        if s is not None:
            if s not in th_by_step:
                th_by_step[s] = th
            else:
                # Keep last (second ppo_epoch)
                th_by_step[s] = th
    
    # Build RewardHealth by step
    rh_by_step = {}
    for rh in data["reward_health"]:
        s = rh.get("step")
        if s is not None:
            rh_by_step[s] = rh
    
    # Val by step
    val_by_step = {}
    for vs in data["val_summaries"]:
        val_by_step[vs["step"]] = vs
    
    # BoK by step
    bok_by_step = {}
    for b in data["bok_routing"]:
        bok_by_step[b["step"]] = b
    
    # Spike steps
    spike_steps = {s["step"] for s in data["spikes"]}
    
    header = (f"{'Step':>4} | {'grad_norm':>9} | {'EMA':>7} | {'ratio':>6} | "
              f"{'spk':>3} | {'cd':>2} | {'LR':>8} | "
              f"{'ans_rh':>6} | {'pt_rh':>5} | "
              f"{'tau':>5} | {'bok%':>4} | {'drg%':>4} | "
              f"{'val_ans':>7} | {'flag':>6}")
    print(header)
    print("-" * len(header))
    
    all_steps = sorted(set(list(data["steps"].keys()) + list(th_by_step.keys())))
    for s in all_steps:
        th = th_by_step.get(s, {})
        rh = rh_by_step.get(s, {})
        v_data = val_by_step.get(s, {})
        bok = bok_by_step.get(s, {})
        step_data = data["steps"].get(s, {})
        
        gn = th.get("grad_norm") or step_data.get("grad_norm") or step_data.get("actor/grad_norm")
        ema = th.get("ema")
        ratio = th.get("ratio")
        spk = th.get("spike_count", "")
        cd = th.get("cooldown", "")
        lr = th.get("lr")
        
        ans = rh.get("answer_mean")
        pt = rh.get("point_mean")
        
        tau = bok.get("tau")
        bok_pct = ""
        drg_pct = ""
        if bok:
            batch = bok.get("batch", 1)
            if batch > 0:
                bok_pct = f"{bok.get('bok_softmax', 0)/batch*100:.0f}"
                drg_pct = f"{bok.get('easy_drgrpo', 0)/batch*100:.0f}"
        
        val_ans = v_data.get("val/answer_reward")
        
        # Flag
        flags = []
        if s in spike_steps:
            flags.append("SPIKE")
        if any(b["step"] == s for b in data["brakes"]):
            flags.append("BRAKE")
        if any(n["step"] == s for n in data["nan_events"]):
            flags.append("NaN")
        flag = ",".join(flags) if flags else ""
        
        def _f(v, fmt=".4f"):
            return f"{v:{fmt}}" if v is not None else "--"
        
        print(
            f"{s:>4} | {_f(gn,'.4f'):>9} | {_f(ema,'.3f'):>7} | {_f(ratio,'.2f'):>6} | "
            f"{str(spk):>3} | {str(cd):>2} | {_f(lr,'.1e'):>8} | "
            f"{_f(ans,'.3f'):>6} | {_f(pt,'.3f'):>5} | "
            f"{_f(tau,'.3f'):>5} | {str(bok_pct):>4} | {str(drg_pct):>4} | "
            f"{_f(val_ans,'.4f'):>7} | {flag:>6}"
        )


def print_single_analysis(data: Dict):
    """Print detailed analysis for a single experiment."""
    v = data["version"]
    s = data["summary"]
    
    print(f"\n{'='*80}")
    print(f"TRAINING ANALYSIS: {v} ({data['log_file']})")
    print(f"{'='*80}")
    
    # Config
    print("\n[Config]")
    for k, v_val in sorted(data["config"].items()):
        print(f"  {k}: {v_val}")
    
    # Summary stats
    print(f"\n[Summary]")
    for k, v_val in sorted(s.items()):
        if isinstance(v_val, float):
            print(f"  {k}: {v_val:.4f}")
        else:
            print(f"  {k}: {v_val}")
    
    # Spike events
    if data["spikes"]:
        print(f"\n[Spike Events ({len(data['spikes'])})]")
        for sp in data["spikes"]:
            print(f"  Step {sp['step']}: #{sp['spike_num']} grad_norm={sp['grad_norm']:.4f} "
                  f"EMA={sp['ema']:.4f} ratio={sp['ratio']:.2f}x")
    
    # Brakes
    if data["brakes"]:
        print(f"\n[Emergency Brakes ({len(data['brakes'])})]")
        for br in data["brakes"]:
            print(f"  Step {br['step']}: {br['spikes_in_window']} spikes in window")
    
    # Val progression
    if data["val_summaries"]:
        print(f"\n[Validation Progression ({len(data['val_summaries'])} checkpoints)]")
        for vs in data["val_summaries"]:
            ans = vs.get("val/answer_reward")
            pt = vs.get("val/point_reward")
            fmt = vs.get("val/format_reward")
            _a = f"{ans:.4f}" if ans is not None else "--"
            _p = f"{pt:.4f}" if pt is not None else "--"
            _f = f"{fmt:.4f}" if fmt is not None else "--"
            print(f"  Step {vs['step']:>4}: answer={_a:<7}  point={_p:<7}  format={_f:<7}")
    
    # FP16 monitor
    if data["fp16_monitor"]:
        print(f"\n[FP16 Monitor ({len(data['fp16_monitor'])} checkpoints)]")
        for fp in data["fp16_monitor"]:
            print(f"  Step {fp['step']:>4}: zero={fp['zero_grad_pct']:.1f}%  "
                  f"subnormal={fp['subnormal_pct']:.1f}%  "
                  f"min_nonzero={fp['min_nonzero']:.2e}  grad_norm={fp['grad_norm']:.4f}")


# ══════════════════════════ Main ══════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EasyR1 Training Analysis Tool")
    parser.add_argument("--log", nargs="+", required=True, help="Training log file(s)")
    parser.add_argument("--json", default=None, help="Export results as JSON")
    parser.add_argument("--summary", action="store_true", help="Quick summary only")
    parser.add_argument("--detailed", action="store_true", help="Include per-step tables")
    parser.add_argument("--steps", default=None, help="Filter step range (e.g., '0-50')")
    args = parser.parse_args()
    
    step_range = None
    if args.steps:
        parts = args.steps.split("-")
        step_range = (int(parts[0]), int(parts[1]))
    
    results = []
    for log_path in args.log:
        if not os.path.exists(log_path):
            print(f"WARNING: Log file not found: {log_path}", file=sys.stderr)
            continue
        print(f"Parsing: {log_path}...", file=sys.stderr)
        data = parse_log(log_path, step_range)
        results.append(data)
        print(f"  -> {data['version']}: {data['summary'].get('total_steps', 0)} steps, "
              f"{len(data['spikes'])} spikes, {len(data['val_summaries'])} val checks", 
              file=sys.stderr)
    
    if not results:
        print("No valid log files found.", file=sys.stderr)
        sys.exit(1)
    
    # Output
    if args.json:
        # JSON export
        def _serialize(obj):
            if isinstance(obj, OrderedDict):
                return dict(obj)
            raise TypeError(f"Not serializable: {type(obj)}")
        
        with open(args.json, 'w') as f:
            json.dump([r for r in results], f, indent=2, default=_serialize)
        print(f"JSON exported to: {args.json}", file=sys.stderr)
    
    if len(results) == 1:
        print_single_analysis(results[0])
        if args.detailed:
            _print_per_step(results[0])
    else:
        print_comparison(results, detailed=args.detailed)


if __name__ == "__main__":
    main()
