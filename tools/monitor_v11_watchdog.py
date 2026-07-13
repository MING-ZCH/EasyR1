#!/usr/bin/env python3
"""
V11 Training Watchdog — 自动监控 + 异常预警 + 早停
==================================================
基于第一性原理审计报告的监控指标实现，实时解析训练日志，
检测异常并在指标持续恶化时创建 stop file 触发早停。

用法:
    python tools/monitor_v11_watchdog.py \
        --log logs/train/training_*.log \
        --out logs/monitor/v11_watchdog.log \
        --stop-file /tmp/v11_stop \
        --interval 30

监控指标和阈值 (基于第一性原理审计报告):
    answer_mean   : >= 0.74 WARNING, >= 0.70 CRITICAL
    point_mean    : >= 0.85 WARNING, >= 0.80 CRITICAL
    clipfrac      : (0.005, 0.40)
    grad_norm     : <= 5.0, no NaN
    kl            : < 0.15 WARNING, < 0.30 CRITICAL
    easy_drgrpo   : 40-90% ratio
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

# ──────────────────────── ANSI strip ────────────────────────
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# ──────────────────────── Log line parsers ────────────────────────
STEP_RE = re.compile(r"Runner.*\)\s+Step\s+(\d+)")
KV_RE = re.compile(r"\(Runner[^)]+\)\s+([\w_/]+):\s+([0-9.eE+-]+|nan|inf)$")
SECTION_RE = re.compile(r"\(Runner[^)]+\)\s+(\w+):$")
SUBSECTION_RE = re.compile(r"\(Runner[^)]+\)\s{3}(\w+):$")

REWARD_HEALTH_RE = re.compile(
    r"\[RewardHealth\].*"
    r"overall_mean=([0-9.]+)\s+answer_mean=([0-9.]+)\s+point_mean=([0-9.]+)\s+"
    r"format_fail_rate=([0-9.]+)\s+stop_violation_rate=([0-9.]+)\s+"
    r"stopped_by_answer_rate=([0-9.]+)\s+turns_exceeded_rate=([0-9.]+)"
)

BOK_GRPO_RE = re.compile(
    r"\[BoK-GRPO\]\s+batch=(\d+).*tau=([0-9.]+)\s+step=(\d+)/(\d+)\s+"
    r"low_var=(\d+)/(\d+).*collapsed=(\d+).*"
    r"easy_drgrpo=(\d+)/(\d+)\s+all_correct_filtered=(\d+)/(\d+)\s+"
    r"adv_mean=([0-9.eE+-]+)\s+adv_std=([0-9.eE+-]+)"
)

BOK_DETAIL_RE = re.compile(
    r"\[BoK-GRPO-Detail\].*"
    r"batch_mean=([0-9.]+)\s+batch_std=([0-9.]+).*"
    r"easy_drgrpo=(\d+)/(\d+)\s+all_correct_filtered=(\d+)/(\d+)\s+"
    r"easy_threshold=([0-9.]+)"
)


# ──────────────────────── Alert Thresholds ────────────────────────

class AlertThresholds:
    """V11 审计报告定义的健康阈值"""
    ANSWER_MEAN_MIN = 0.74
    ANSWER_MEAN_CRITICAL = 0.70
    ANSWER_MEAN_DECLINE_THRESHOLD = 0.03

    POINT_MEAN_MIN = 0.85
    POINT_MEAN_CRITICAL = 0.80
    POINT_MEAN_DECLINE_THRESHOLD = 0.05
    POINT_TREND_MIN_STEPS = 6

    CLIPFRAC_MIN = 0.0005
    CLIPFRAC_MAX = 0.40
    CLIPFRAC_CHECK_AFTER_STEP = 3

    GRAD_NORM_MAX = 5.0
    GRAD_NORM_NAN_LIMIT = 3

    KL_MAX = 0.15
    KL_CRITICAL = 0.30

    EASY_DRGRPO_MIN = 0.40
    EASY_DRGRPO_MAX = 0.90

    EARLY_STOP_CONSECUTIVE_CRITICAL = 5


# ──────────────────────── Watchdog State ────────────────────────

class WatchdogState:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.steps = {}
        self.reward_health_history = []
        self.bok_history = []
        self.alert_counts = {}
        self.consecutive_critical = 0
        self.nan_grad_count = 0
        self.total_steps_seen = 0
        self.training_started = False
        self.answer_trend = deque(maxlen=window_size)
        self.point_trend = deque(maxlen=window_size)
        self.cur_step = None
        self.section = None
        self.sub_section = None
        self.last_completed_step = None
        self.reported_until = -1

    def get_trend(self, values):
        if len(values) < 4:
            return None
        mid = len(values) // 2
        first_half = list(values)[:mid]
        second_half = list(values)[mid:]
        return sum(second_half) / len(second_half) - sum(first_half) / len(first_half)


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _sf(v, fmt=".4f"):
    if v is None:
        return "--"
    return f"{v:{fmt}}"


# ──────────────────────── Line Parser ────────────────────────

def parse_line(raw_line, state):
    line = ANSI_RE.sub("", raw_line).rstrip("\n")
    events = {}

    m = STEP_RE.search(line)
    if m:
        new_step = int(m.group(1))
        if state.cur_step is not None and state.cur_step != new_step:
            state.last_completed_step = state.cur_step
        state.cur_step = new_step
        state.section = None
        state.sub_section = None
        state.steps.setdefault(new_step, {})
        state.total_steps_seen = max(state.total_steps_seen, new_step)
        if not state.training_started:
            state.training_started = True
            state.answer_trend.clear()
            state.point_trend.clear()
        return events

    m = SECTION_RE.match(line)
    if m:
        state.section = m.group(1)
        state.sub_section = None
        return events

    m = SUBSECTION_RE.match(line)
    if m and state.section:
        state.sub_section = m.group(1)
        return events

    m = KV_RE.match(line)
    if m and state.section and state.cur_step is not None:
        key_name = m.group(1)
        val_str = m.group(2)
        val = float("nan") if val_str in ("nan", "inf") else float(val_str)
        sub = state.sub_section
        full_key = f"{state.section}/{sub}/{key_name}" if sub else f"{state.section}/{key_name}"
        state.steps[state.cur_step][full_key] = val
        return events

    m = REWARD_HEALTH_RE.search(line)
    if m:
        rh = {
            "overall_mean": float(m.group(1)),
            "answer_mean": float(m.group(2)),
            "point_mean": float(m.group(3)),
            "format_fail_rate": float(m.group(4)),
            "stop_violation_rate": float(m.group(5)),
            "stopped_by_answer_rate": float(m.group(6)),
            "turns_exceeded_rate": float(m.group(7)),
        }
        state.reward_health_history.append(rh)
        state.answer_trend.append(rh["answer_mean"])
        state.point_trend.append(rh["point_mean"])
        events["reward_health"] = rh
        return events

    m = BOK_GRPO_RE.search(line)
    if m:
        bok = {
            "batch": int(m.group(1)),
            "tau": float(m.group(2)),
            "step": int(m.group(3)),
            "total_steps": int(m.group(4)),
            "low_var": int(m.group(5)),
            "low_var_total": int(m.group(6)),
            "collapsed": int(m.group(7)),
            "easy_drgrpo": int(m.group(8)),
            "easy_total": int(m.group(9)),
            "all_correct_filtered": int(m.group(10)),
            "all_correct_total": int(m.group(11)),
            "adv_mean": float(m.group(12)),
            "adv_std": float(m.group(13)),
        }
        state.bok_history.append(bok)
        events["bok_grpo"] = bok
        return events

    m = BOK_DETAIL_RE.search(line)
    if m:
        events["bok_detail"] = {
            "batch_mean": float(m.group(1)),
            "batch_std": float(m.group(2)),
            "easy_drgrpo": int(m.group(3)),
            "easy_total": int(m.group(4)),
            "all_correct_filtered": int(m.group(5)),
            "all_correct_total": int(m.group(6)),
            "easy_threshold": float(m.group(7)),
        }
        return events

    return events


# ──────────────────────── Health Check ────────────────────────

def check_health(state, step):
    alerts = []
    metrics = state.steps.get(step, {})
    T = AlertThresholds

    # grad_norm
    gn = metrics.get("actor/grad_norm")
    if gn is not None:
        if math.isnan(gn) or math.isinf(gn):
            state.nan_grad_count += 1
            alerts.append(("CRITICAL", "grad_norm",
                           f"step={step} grad_norm=NaN/Inf (累计{state.nan_grad_count}次)"))
            if state.nan_grad_count >= T.GRAD_NORM_NAN_LIMIT:
                alerts.append(("CRITICAL", "grad_norm_accumulated",
                               f"累计{state.nan_grad_count}次NaN, 建议停止训练"))
        elif gn > T.GRAD_NORM_MAX:
            alerts.append(("WARNING", "grad_norm",
                           f"step={step} grad_norm={gn:.3f} > {T.GRAD_NORM_MAX}"))

    # clipfrac
    if step >= T.CLIPFRAC_CHECK_AFTER_STEP:
        cf_h = metrics.get("actor/pg_clipfrac_higher")
        cf_l = metrics.get("actor/pg_clipfrac_lower")
        if cf_h is not None and cf_l is not None:
            cf_total = cf_h + cf_l
            if cf_total < T.CLIPFRAC_MIN:
                alerts.append(("WARNING", "clipfrac",
                               f"step={step} clipfrac={cf_total:.4f} < {T.CLIPFRAC_MIN} "
                               f"(PPO clip可能未激活)"))
            elif cf_total > T.CLIPFRAC_MAX:
                alerts.append(("WARNING", "clipfrac",
                               f"step={step} clipfrac={cf_total:.4f} > {T.CLIPFRAC_MAX} "
                               f"(policy变化过大)"))

    # KL
    kl = metrics.get("actor/kl_loss") or metrics.get("actor/ppo_kl")
    if kl is not None:
        if kl > T.KL_CRITICAL:
            alerts.append(("CRITICAL", "kl",
                           f"step={step} kl={kl:.4f} > {T.KL_CRITICAL}"))
        elif kl > T.KL_MAX:
            alerts.append(("WARNING", "kl",
                           f"step={step} kl={kl:.4f} > {T.KL_MAX}"))

    # answer_mean
    if state.reward_health_history:
        rh = state.reward_health_history[-1]
        am = rh["answer_mean"]
        if am < T.ANSWER_MEAN_CRITICAL:
            alerts.append(("CRITICAL", "answer_mean",
                           f"step={step} answer_mean={am:.4f} < {T.ANSWER_MEAN_CRITICAL}"))
        elif am < T.ANSWER_MEAN_MIN:
            alerts.append(("WARNING", "answer_mean",
                           f"step={step} answer_mean={am:.4f} < {T.ANSWER_MEAN_MIN}"))
        trend = state.get_trend(state.answer_trend)
        if trend is not None and trend < -T.ANSWER_MEAN_DECLINE_THRESHOLD:
            alerts.append(("WARNING", "answer_trend",
                           f"step={step} answer_mean下降={trend:+.4f}"))

    # point_mean
    if state.reward_health_history:
        rh = state.reward_health_history[-1]
        pm = rh["point_mean"]
        if pm < T.POINT_MEAN_CRITICAL:
            alerts.append(("CRITICAL", "point_mean",
                           f"step={step} point_mean={pm:.4f} < {T.POINT_MEAN_CRITICAL}"))
        elif pm < T.POINT_MEAN_MIN:
            alerts.append(("WARNING", "point_mean",
                           f"step={step} point_mean={pm:.4f} < {T.POINT_MEAN_MIN}"))
        trend = state.get_trend(state.point_trend)
        if (trend is not None and trend < -T.POINT_MEAN_DECLINE_THRESHOLD
                and step >= T.POINT_TREND_MIN_STEPS):
            alerts.append(("CRITICAL", "point_trend",
                           f"step={step} point_mean持续下降={trend:+.4f} (V10退化重现!)"))

    # easy_drgrpo
    if state.bok_history:
        bok = state.bok_history[-1]
        easy_ratio = bok["easy_drgrpo"] / max(bok["easy_total"], 1)
        if easy_ratio > T.EASY_DRGRPO_MAX:
            alerts.append(("WARNING", "easy_drgrpo",
                           f"step={step} easy_drgrpo={easy_ratio:.1%} > {T.EASY_DRGRPO_MAX:.0%}"))
        elif easy_ratio < T.EASY_DRGRPO_MIN:
            alerts.append(("WARNING", "easy_drgrpo",
                           f"step={step} easy_drgrpo={easy_ratio:.1%} < {T.EASY_DRGRPO_MIN:.0%}"))

    return alerts


# ──────────────────────── Dashboard Line ────────────────────────

def format_dashboard(step, state):
    m = state.steps.get(step, {})
    rh = state.reward_health_history[-1] if state.reward_health_history else {}
    bok = state.bok_history[-1] if state.bok_history else {}

    gn = m.get("actor/grad_norm")
    cf_h = m.get("actor/pg_clipfrac_higher", 0)
    cf_l = m.get("actor/pg_clipfrac_lower", 0)
    kl = m.get("actor/kl_loss") or m.get("actor/ppo_kl")
    ov = m.get("reward/overall")
    ans_r = m.get("reward/answer")
    pt_r = m.get("reward/point")

    am = rh.get("answer_mean")
    pm = rh.get("point_mean")
    ff = rh.get("format_fail_rate")

    if bok:
        easy_r = bok.get("easy_drgrpo", 0) / max(bok.get("easy_total", 1), 1)
        ac_r = bok.get("all_correct_filtered", 0) / max(bok.get("all_correct_total", 1), 1)
        tau = bok.get("tau")
    else:
        easy_r = ac_r = tau = None

    a_trend = state.get_trend(state.answer_trend)
    p_trend = state.get_trend(state.point_trend)

    parts = [
        f"step={step:>4d}",
        f"gn={_sf(gn, '.3f')}",
        f"cf={cf_h + cf_l:.4f}" if cf_h is not None else "cf=--",
        f"kl={_sf(kl)}",
        f"ov={_sf(ov)}",
        f"ans_r={_sf(ans_r)}",
        f"pt_r={_sf(pt_r)}",
        f"am={_sf(am)}",
        f"pm={_sf(pm)}",
        f"ff={_sf(ff)}",
        f"easy={easy_r:.0%}" if easy_r is not None else "easy=--",
        f"ac={ac_r:.0%}" if ac_r is not None else "ac=--",
        f"tau={_sf(tau, '.3f')}" if tau is not None else "tau=--",
        f"a_tr={a_trend:+.4f}" if a_trend is not None else "",
        f"p_tr={p_trend:+.4f}" if p_trend is not None else "",
    ]
    return " | ".join(p for p in parts if p)


def format_val_summary(step, metrics):
    val_answer = metrics.get("val/answer_reward")
    if val_answer is None:
        return None
    val_overall = metrics.get("val/overall_reward")
    val_fmtf = metrics.get("val/format_fail_reward")
    val_stopv = metrics.get("val/stop_violation_reward")
    return (
        f"[VAL] step={step} answer={_sf(val_answer)} overall={_sf(val_overall)} "
        f"format_fail={_sf(val_fmtf)} stop_violation={_sf(val_stopv)}"
    )


# ──────────────────────── Main Loop ────────────────────────

def main():
    ap = argparse.ArgumentParser(description="V11 Training Watchdog")
    ap.add_argument("--log", required=True, help="训练日志路径")
    ap.add_argument("--out", required=True, help="监控输出日志路径")
    ap.add_argument("--interval", type=float, default=30.0, help="轮询间隔(秒)")
    ap.add_argument("--every", type=int, default=5, help="每N步输出仪表盘")
    ap.add_argument("--stop-file", default=None, help="早停sentinel文件路径")
    ap.add_argument("--auto-stop", action="store_true", help="启用自动早停")
    ap.add_argument("--json-metrics", default=None, help="JSON Lines指标输出")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.json_metrics:
        os.makedirs(os.path.dirname(args.json_metrics) or ".", exist_ok=True)

    state = WatchdogState(window_size=20)
    pos = 0
    last_val_reported = -1

    json_f = open(args.json_metrics, "a", encoding="utf-8") if args.json_metrics else None

    try:
        with open(args.out, "a", encoding="utf-8") as out:
            out.write(f"\n{'='*80}\n")
            out.write(f"[{_ts()}] V11 Watchdog 启动\n")
            out.write(f"[{_ts()}] 日志: {args.log}\n")
            out.write(f"[{_ts()}] 轮询: {args.interval}s  仪表盘每: {args.every}步\n")
            out.write(f"[{_ts()}] 自动早停: {'启用' if args.auto_stop else '禁用'}\n")
            out.write(f"[{_ts()}] 阈值: answer>={AlertThresholds.ANSWER_MEAN_MIN} "
                       f"point>={AlertThresholds.POINT_MEAN_MIN} "
                       f"clip=[{AlertThresholds.CLIPFRAC_MIN},{AlertThresholds.CLIPFRAC_MAX}] "
                       f"kl<{AlertThresholds.KL_MAX}\n")
            out.write(f"{'='*80}\n")
            out.flush()

            while True:
                if args.stop_file and os.path.exists(args.stop_file):
                    out.write(f"[{_ts()}] [STOP] 检测到stop file: {args.stop_file}\n")
                    out.flush()
                    return 0

                try:
                    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
                        f.seek(pos)
                        chunk = f.read(2 * 1024 * 1024)
                        pos = f.tell()
                except FileNotFoundError:
                    out.write(f"[{_ts()}] 日志未找到，等待: {args.log}\n")
                    out.flush()
                    time.sleep(args.interval)
                    continue

                try:
                    if not chunk:
                        time.sleep(args.interval)
                        continue

                    for raw_line in chunk.splitlines(True):
                        parse_line(raw_line, state)

                    # Validation results
                    for s in sorted(state.steps.keys()):
                        if s > last_val_reported and "val/answer_reward" in state.steps[s]:
                            val_msg = format_val_summary(s, state.steps[s])
                            if val_msg:
                                out.write(f"[{_ts()}] {val_msg}\n")
                                out.flush()
                            last_val_reported = s

                    # Completed steps — iterate ALL unreported steps
                    completed = state.last_completed_step
                    if completed is not None and completed > state.reported_until:
                        for report_step in range(state.reported_until + 1, completed + 1):
                            # Health check on every step
                            alerts = check_health(state, report_step)
                            has_critical = False
                            for severity, metric, msg in alerts:
                                state.alert_counts[metric] = state.alert_counts.get(metric, 0) + 1
                                if severity == "CRITICAL":
                                    has_critical = True
                                if report_step % args.every == 0 or report_step <= 5:
                                    pfx = "🔴" if severity == "CRITICAL" else "🟡"
                                    out.write(f"[{_ts()}] {pfx} [{severity}] {msg}\n")

                            if has_critical:
                                state.consecutive_critical += 1
                            else:
                                state.consecutive_critical = 0

                            # Dashboard only on --every interval or first 5 steps
                            if report_step % args.every == 0 or report_step <= 5:
                                dashboard = format_dashboard(report_step, state)
                                out.write(f"[{_ts()}] {dashboard}\n")

                                # Auto early stop
                                if (args.auto_stop and args.stop_file and
                                        state.consecutive_critical >= AlertThresholds.EARLY_STOP_CONSECUTIVE_CRITICAL):
                                    out.write(f"\n[{_ts()}] {'!'*60}\n")
                                    out.write(f"[{_ts()}] [EARLY_STOP] 连续{state.consecutive_critical}次CRITICAL\n")
                                    out.write(f"[{_ts()}] [EARLY_STOP] 创建stop file: {args.stop_file}\n")
                                    out.write(f"[{_ts()}] [EARLY_STOP] 告警统计: {json.dumps(state.alert_counts, ensure_ascii=False)}\n")
                                    out.write(f"[{_ts()}] {'!'*60}\n")
                                    out.flush()

                                    with open(args.stop_file, "w") as sf:
                                        sf.write(json.dumps({
                                            "reason": "consecutive_critical",
                                            "count": state.consecutive_critical,
                                            "step": report_step,
                                            "alerts": state.alert_counts,
                                            "timestamp": _ts(),
                                        }, ensure_ascii=False))
                                    return 1

                                # JSON export
                                if json_f is not None:
                                    m = state.steps.get(report_step, {})
                                    rh = state.reward_health_history[-1] if state.reward_health_history else {}
                                    bok = state.bok_history[-1] if state.bok_history else {}
                                    record = {
                                        "step": report_step,
                                        "timestamp": _ts(),
                                        "grad_norm": m.get("actor/grad_norm"),
                                        "clipfrac_h": m.get("actor/pg_clipfrac_higher"),
                                        "clipfrac_l": m.get("actor/pg_clipfrac_lower"),
                                        "kl": m.get("actor/kl_loss") or m.get("actor/ppo_kl"),
                                        "answer_mean": rh.get("answer_mean"),
                                        "point_mean": rh.get("point_mean"),
                                        "format_fail": rh.get("format_fail_rate"),
                                        "answer_trend": state.get_trend(state.answer_trend),
                                        "point_trend": state.get_trend(state.point_trend),
                                    }
                                    if bok:
                                        record["easy_drgrpo_ratio"] = bok.get("easy_drgrpo", 0) / max(bok.get("easy_total", 1), 1)
                                        record["all_correct_ratio"] = bok.get("all_correct_filtered", 0) / max(bok.get("all_correct_total", 1), 1)
                                        record["tau"] = bok.get("tau")
                                        record["adv_std"] = bok.get("adv_std")
                                    json_f.write(json.dumps(record) + "\n")
                                    json_f.flush()

                                out.flush()

                        state.reported_until = completed

                    time.sleep(args.interval)

                except Exception as exc:
                    import traceback as _tb
                    out.write(f"[{_ts()}] 🔴 [ERROR] watchdog解析异常: {exc}\n")
                    out.write(_tb.format_exc() + "\n")
                    out.flush()
                    time.sleep(args.interval)
                    continue

    except KeyboardInterrupt:
        print(f"\n[{_ts()}] Watchdog 中断")
        return 0
    finally:
        if json_f is not None:
            json_f.close()


if __name__ == "__main__":
    raise SystemExit(main())
