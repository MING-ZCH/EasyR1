#!/usr/bin/env python3
"""V34 dense RL live monitor — metrics, rollout trajectories, fail-fast alerts.

Alerts:
  - no_point dead-gradient rising (no_point_pred_rate / no_point_rows)
  - entropy_loss >= 1.5
  - not finite / NONFINITE gradient events
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

STEP_RE = re.compile(r"\(Runner[^)]*\)\s+Step\s+(\d+)")
TRAIN_HEALTH_RE = re.compile(
    r"\[TrainHealth\]\s+opt_step=(\d+)\s+grad_norm=([\d.]+)\s+ema=([\d.]+)\s+"
    r"ratio=([\d.]+)x\s+spike_count=(\d+)\s+cooldown=(\d+)\s+"
    r"brake=(\w+)\s+lr=([\d.eE+-]+)"
)
REWARD_HEALTH_RE = re.compile(
    r"\[RewardHealth\].*?"
    r"overall_mean=([\d.na]+)\s+answer_mean=([\d.na]+)\s+point_mean=([\d.na]+)\s+"
    r"format_fail_rate=([\d.na]+)\s+stop_violation_rate=([\d.na]+)\s+"
    r"stopped_by_answer_rate=([\d.na]+)\s+turns_exceeded_rate=([\d.na]+)\s+"
    r"no_point_pred_rate=([\d.na]+)"
    r"(?:\s+point_key_typo_mean=([\d.na]+)\s+point_key_typo_rate=([\d.na]+))?"
)
BOK_GRPO_RE = re.compile(
    r"\[BoK-GRPO(?:-Step)?\]\s+batch=(\d+).*?step=(\d+)/(\d+).*?"
    r"(?:no_point_rows=(\d+)/(\d+))?"
)
BOK_GRPO_NO_POINT_RE = re.compile(r"no_point_rows=(\d+)/(\d+)")
INTERLEAVED_SUMMARY_RE = re.compile(
    r"\[InterleavedSummary\]\s+step=(\d+)\s+total_points=(\d+)\s+total_point_tag_seen=(\d+)"
)
INTERLEAVED_WARN_RE = re.compile(
    r"\[InterleavedWarning\].*?stalled_no_points|no_points_across_all_turns"
)
VAL_SUMMARY_RE = re.compile(r"\[ValSummary\]\s+step=(-?\d+)\s+(.*)")
SECTION_RE = re.compile(r"\(Runner[^)]*\)\s+([\w_]+):\s*$")
KV_RE = re.compile(r"\(Runner[^)]*\)\s{1,3}([\w_]+):\s+([\d.eE+-]+|nan|inf|-inf)$")
NONFINITE_RE = re.compile(
    r"Gradient norm is not finite|\[GradSpikeProtect\]\s+NONFINITE|not finite",
    re.IGNORECASE,
)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _sf(v: Any, fmt: str = ".4f") -> str:
    if v is None:
        return "--"
    if isinstance(v, str):
        return v
    return f"{v:{fmt}}"


def _to_float(v: str) -> Optional[float]:
    if v in ("na", "", "nan", "inf", "-inf"):
        return None
    try:
        x = float(v)
        if x != x:  # NaN
            return None
        return x
    except ValueError:
        return None


class State:
    def __init__(self) -> None:
        self.cur_step: Optional[int] = None
        self.section: Optional[str] = None
        self.no_point_rates: Deque[float] = deque(maxlen=8)
        self.point_key_typo_rates: Deque[float] = deque(maxlen=8)
        self.entropy_values: Deque[float] = deque(maxlen=8)
        self.adaptive_kl_values: Deque[float] = deque(maxlen=8)
        self.nonfinite_count = 0
        self.alert_keys: Dict[str, float] = {}
        self.last_dashboard_step = -1
        self.last_summary_step = -1


def _emit(out, level: str, tag: str, msg: str) -> None:
    icon = {"ALERT": "[!!!]", "WARN": "[!]", "INFO": "[i]"}.get(level, "[ ]")
    out.write(f"[{_ts()}] {icon} [{level}] {tag}: {msg}\n")
    out.flush()
    print(f"[{_ts()}] {icon} [{level}] {tag}: {msg}", flush=True)


def _maybe_alert(out, state: State, key: str, level: str, tag: str, msg: str, cooldown_s: float = 120.0) -> None:
    now = time.time()
    last = state.alert_keys.get(key, 0.0)
    if now - last < cooldown_s:
        return
    state.alert_keys[key] = now
    _emit(out, level, tag, msg)


def parse_line(line: str, state: State, out, args) -> None:
    line = ANSI_RE.sub("", line).rstrip()
    if not line:
        return

    m = STEP_RE.search(line)
    if m and "InterleavedDebug" not in line:
        state.cur_step = int(m.group(1))
        state.section = None
        return

    if NONFINITE_RE.search(line):
        state.nonfinite_count += 1
        _maybe_alert(
            out,
            state,
            "nonfinite",
            "ALERT",
            "NONFINITE",
            f"non-finite gradient event #{state.nonfinite_count} at step={state.cur_step}: {line[-200:]}",
            cooldown_s=30.0,
        )
        return

    m = TRAIN_HEALTH_RE.search(line)
    if m:
        gn = float(m.group(2))
        _emit(
            out,
            "INFO",
            "TrainHealth",
            f"step={state.cur_step} opt={m.group(1)} grad_norm={gn:.3f} ema={m.group(3)} "
            f"ratio={m.group(4)}x spike={m.group(5)} brake={m.group(7)} lr={m.group(8)}",
        )
        return

    m = REWARD_HEALTH_RE.search(line)
    if m:
        rh = {
            "overall": _to_float(m.group(1)),
            "answer": _to_float(m.group(2)),
            "point": _to_float(m.group(3)),
            "format_fail": _to_float(m.group(4)),
            "stop_violation": _to_float(m.group(5)),
            "stopped_by_answer": _to_float(m.group(6)),
            "turns_exceeded": _to_float(m.group(7)),
            "no_point_pred": _to_float(m.group(8)),
            "point_key_typo": _to_float(m.group(9)) if m.group(9) is not None else None,
            "point_key_typo_rate": _to_float(m.group(10)) if m.group(10) is not None else None,
        }
        typo_rate = rh["point_key_typo_rate"]
        if typo_rate is not None:
            state.point_key_typo_rates.append(typo_rate)
        np_rate = rh["no_point_pred"]
        if np_rate is not None:
            state.no_point_rates.append(np_rate)
            if np_rate >= args.no_point_critical:
                _maybe_alert(
                    out,
                    state,
                    "no_point_critical",
                    "ALERT",
                    "NO_POINT",
                    f"no_point_pred_rate={np_rate:.4f} >= {args.no_point_critical} (dead-gradient risk)",
                )
            elif np_rate >= args.no_point_warn:
                _maybe_alert(
                    out,
                    state,
                    "no_point_warn",
                    "WARN",
                    "NO_POINT",
                    f"no_point_pred_rate={np_rate:.4f} >= {args.no_point_warn}",
                )
            if len(state.no_point_rates) >= 3:
                delta = state.no_point_rates[-1] - state.no_point_rates[-3]
                if delta >= args.no_point_rise:
                    _maybe_alert(
                        out,
                        state,
                        "no_point_rise",
                        "ALERT",
                        "NO_POINT_RISE",
                        f"no_point_pred_rate rising: {state.no_point_rates[-3]:.4f} -> {state.no_point_rates[-1]:.4f} "
                        f"(+{delta:.4f} over 3 samples)",
                    )
        _emit(
            out,
            "INFO",
            "RewardHealth",
            f"step={state.cur_step} overall={_sf(rh['overall'])} answer={_sf(rh['answer'])} "
            f"point={_sf(rh['point'])} format_fail={_sf(rh['format_fail'])} "
            f"no_point_pred={_sf(rh['no_point_pred'])} "
            f"point_key_typo={_sf(rh['point_key_typo'])} typo_rate={_sf(rh['point_key_typo_rate'])}",
        )
        return

    if "[BoK-GRPO" in line:
        m = BOK_GRPO_NO_POINT_RE.search(line)
        if m:
            n_no, bsz = int(m.group(1)), int(m.group(2))
            ratio = n_no / max(bsz, 1)
            if ratio >= args.no_point_rows_critical:
                _maybe_alert(
                    out,
                    state,
                    "no_point_rows",
                    "ALERT",
                    "NO_POINT_ROWS",
                    f"no_point_rows={n_no}/{bsz} ({ratio:.1%}) at step={state.cur_step}",
                )
            elif ratio >= args.no_point_rows_warn:
                _maybe_alert(
                    out,
                    state,
                    "no_point_rows_warn",
                    "WARN",
                    "NO_POINT_ROWS",
                    f"no_point_rows={n_no}/{bsz} ({ratio:.1%})",
                )
        if "adv_std=" in line or "adv_mean=" in line:
            _emit(out, "INFO", "BoK-GRPO", line.split("] ", 1)[-1][:240])
        return

    m = INTERLEAVED_SUMMARY_RE.search(line)
    if m:
        step, pts, tags = int(m.group(1)), int(m.group(2)), int(m.group(3))
        parse_ratio = pts / max(tags, 1)
        _emit(
            out,
            "INFO",
            "Rollout",
            f"step={step} total_points={pts} point_tag_seen={tags} parse_ratio={parse_ratio:.2f}",
        )
        if tags >= 10 and parse_ratio < 0.3:
            _maybe_alert(
                out,
                state,
                f"rollout_parse_{step}",
                "WARN",
                "ROLLOUT_PARSE",
                f"step={step} low point parse ratio {parse_ratio:.2f} ({pts}/{tags})",
            )
        return

    if INTERLEAVED_WARN_RE.search(line):
        _emit(out, "WARN", "RolloutWarn", line.split("] ", 1)[-1][:240])
        return

    if "[InterleavedDebug]" in line and args.verbose_rollout:
        _emit(out, "INFO", "RolloutDebug", line.split("] ", 1)[-1][:240])
        return

    m = VAL_SUMMARY_RE.search(line)
    if m:
        vs: Dict[str, Any] = {"step": int(m.group(1))}
        for kv in m.group(2).split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                vs[k] = _to_float(v) if _to_float(v) is not None else v
        _emit(
            out,
            "INFO",
            "Val",
            f"step={vs['step']} answer={_sf(vs.get('val/answer_reward'))} "
            f"point={_sf(vs.get('val/point_reward'))} overall={_sf(vs.get('val/overall_reward'))}",
        )
        return

    m = SECTION_RE.match(line)
    if m:
        state.section = m.group(1)
        return

    m = KV_RE.match(line)
    if m and state.section:
        key, raw = m.group(1), m.group(2).lower()
        if key == "entropy_loss":
            ent = _to_float(raw)
            if ent is None or raw in ("nan", "inf", "-inf"):
                _maybe_alert(
                    out,
                    state,
                    "entropy_nonfinite",
                    "ALERT",
                    "ENTROPY_NONFINITE",
                    f"entropy_loss={raw} at step={state.cur_step}",
                    cooldown_s=30.0,
                )
            elif ent is not None:
                state.entropy_values.append(ent)
                if ent >= args.entropy_alert:
                    _maybe_alert(
                        out,
                        state,
                        "entropy_high",
                        "ALERT",
                        "ENTROPY",
                        f"entropy_loss={ent:.4f} >= {args.entropy_alert} at step={state.cur_step}",
                    )
                if state.cur_step is not None and state.cur_step % args.every == 0:
                    _emit(out, "INFO", "Actor", f"step={state.cur_step} entropy_loss={ent:.4f}")
        elif state.section in ("adaptive_kl", "critic") and key in ("kl_loss", "kl"):
            kl = _to_float(raw)
            if kl is not None:
                state.adaptive_kl_values.append(kl)
                _emit(out, "INFO", "AdaptiveKL", f"step={state.cur_step} {state.section}/{key}={kl:.4f}")
        elif key == "grad_norm" and raw in ("nan", "inf", "-inf"):
            _maybe_alert(
                out,
                state,
                "grad_nonfinite",
                "ALERT",
                "GRAD_NONFINITE",
                f"grad_norm={raw} at step={state.cur_step}",
                cooldown_s=30.0,
            )


def print_dashboard(out, state: State, step: int) -> None:
    np_rate = state.no_point_rates[-1] if state.no_point_rates else None
    typo_rate = state.point_key_typo_rates[-1] if state.point_key_typo_rates else None
    ent = state.entropy_values[-1] if state.entropy_values else None
    adaptive_kl = state.adaptive_kl_values[-1] if state.adaptive_kl_values else None
    out.write(
        f"[{_ts()}] DASH | step={step:>4d} | no_point_pred={_sf(np_rate)} | "
        f"typo_rate={_sf(typo_rate)} | entropy_loss={_sf(ent)} | "
        f"adaptive_kl={_sf(adaptive_kl)} | nonfinite={state.nonfinite_count}\n"
    )
    out.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description="V34 live training monitor")
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--every", type=int, default=5, help="Dashboard every N steps")
    ap.add_argument("--summary-every", type=int, default=20)
    ap.add_argument("--entropy-alert", type=float, default=1.5)
    ap.add_argument("--no-point-warn", type=float, default=0.15)
    ap.add_argument("--no-point-critical", type=float, default=0.30)
    ap.add_argument("--no-point-rise", type=float, default=0.10, help="Alert if rate rises this much over 3 samples")
    ap.add_argument("--no-point-rows-warn", type=float, default=0.40)
    ap.add_argument("--no-point-rows-critical", type=float, default=0.60)
    ap.add_argument("--json-metrics", default=None)
    ap.add_argument("--verbose-rollout", action="store_true")
    ap.add_argument("--stop-file", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    state = State()
    pos = 0
    json_f = open(args.json_metrics, "a", encoding="utf-8") if args.json_metrics else None

    with open(args.out, "a", encoding="utf-8") as out:
        out.write(f"\n{'='*72}\n")
        out.write(f"[{_ts()}] V34 live monitor started\n")
        out.write(f"[{_ts()}] log={args.log}\n")
        out.write(
            f"[{_ts()}] alerts: entropy>={args.entropy_alert} | "
            f"no_point>={args.no_point_warn}/{args.no_point_critical} | "
            f"no_point_rise>={args.no_point_rise} | nonfinite/NONFINITE\n"
        )
        out.write(f"{'='*72}\n")
        out.flush()

        try:
            while True:
                if args.stop_file and os.path.exists(args.stop_file):
                    _emit(out, "INFO", "STOP", f"stop file detected: {args.stop_file}")
                    break

                try:
                    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
                        f.seek(pos)
                        chunk = f.read(4 * 1024 * 1024)
                        pos = f.tell()
                except FileNotFoundError:
                    time.sleep(args.interval)
                    continue

                if chunk:
                    for raw in chunk.splitlines():
                        parse_line(raw, state, out, args)

                    if state.cur_step is not None and state.cur_step > state.last_dashboard_step:
                        for s in range(state.last_dashboard_step + 1, state.cur_step + 1):
                            if s % args.every == 0 or s <= 2:
                                print_dashboard(out, state, s)
                            if s % args.summary_every == 0 and s > state.last_summary_step:
                                out.write(
                                    f"[{_ts()}] SUMMARY step={s} no_point_samples="
                                    f"{list(state.no_point_rates)} entropy_samples={list(state.entropy_values)}\n"
                                )
                                out.flush()
                                state.last_summary_step = s
                        state.last_dashboard_step = state.cur_step

                        if json_f and state.no_point_rates:
                            json_f.write(
                                json.dumps(
                                    {
                                        "ts": _ts(),
                                        "step": state.cur_step,
                                        "no_point_pred_rate": state.no_point_rates[-1],
                                        "entropy_loss": state.entropy_values[-1] if state.entropy_values else None,
                                        "nonfinite_count": state.nonfinite_count,
                                    }
                                )
                                + "\n"
                            )
                            json_f.flush()

                time.sleep(args.interval)
        finally:
            if json_f:
                json_f.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
