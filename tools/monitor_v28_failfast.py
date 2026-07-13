#!/usr/bin/env python3
"""V28 fail-fast monitor for StepCount trajectory RL.

This monitor creates a stop sentinel only for recurring failure modes:
non-finite gradients, sustained entropy drift, sustained format-fail spikes,
and severe low-variance / zero-reward batches.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime
from typing import Optional

STEP_RE = re.compile(r"\(Runner[^)]*\)\s+Step\s+(\d+)")
SECTION_RE = re.compile(r"\(Runner[^)]*\)\s+([\w_]+):\s*$")
KV_RE = re.compile(r"\(Runner[^)]*\)\s{1,3}([\w_]+):\s+([\d.eE+-]+)$")
REWARD_HEALTH_RE = re.compile(
    r"\[RewardHealth\].*?format_fail_rate=([\d.na]+)\s+stop_violation_rate=([\d.na]+)"
)
BOK_WARNING_RE = re.compile(r"low_var_rate=([\d.]+)%\s+zero_reward_rate=([\d.]+)%")
NAN_RE = re.compile(r"Gradient norm is not finite")
SPIKE_RE = re.compile(r"\[GradSpikeProtect\]\s+SPIKE DETECTED")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sf(value: str) -> Optional[float]:
    if value == "na":
        return None
    return float(value)


def write_stop(path: str, reason: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(reason + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="StepCount V28 fail-fast monitor")
    ap.add_argument("--log", required=True, help="Training log to tail")
    ap.add_argument("--out", required=True, help="Monitor log output")
    ap.add_argument("--stop-file", required=True, help="Sentinel file to create on fail-fast")
    ap.add_argument("--run-name", default=os.getenv("STEPCOUNT_FAILFAST_RUN_NAME", "V28"), help="Run label used in monitor messages")
    ap.add_argument("--interval", type=float, default=float(os.getenv("V28_FAILFAST_INTERVAL", "20")))
    ap.add_argument("--nan-limit", type=int, default=int(os.getenv("V28_FAILFAST_NAN_LIMIT", "3")))
    ap.add_argument("--entropy-critical", type=float, default=float(os.getenv("V28_FAILFAST_ENTROPY_CRITICAL", "0.85")))
    ap.add_argument("--entropy-consec", type=int, default=int(os.getenv("V28_FAILFAST_ENTROPY_CONSEC", "3")))
    ap.add_argument("--format-critical", type=float, default=float(os.getenv("V28_FAILFAST_FORMAT_CRITICAL", "0.30")))
    ap.add_argument("--format-consec", type=int, default=int(os.getenv("V28_FAILFAST_FORMAT_CONSEC", "3")))
    ap.add_argument("--lowvar-critical", type=float, default=float(os.getenv("V28_FAILFAST_LOWVAR_CRITICAL", "45")))
    ap.add_argument("--zeroreward-critical", type=float, default=float(os.getenv("V28_FAILFAST_ZEROREWARD_CRITICAL", "30")))
    ap.add_argument("--bok-consec", type=int, default=int(os.getenv("V28_FAILFAST_BOK_CONSEC", "2")))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pos = 0
    current_step: Optional[int] = None
    current_section: Optional[str] = None
    nan_count = 0
    spike_count = 0
    entropy_bad = 0
    format_bad = 0
    bok_bad = 0

    with open(args.out, "a", encoding="utf-8") as out:
        out.write(f"[{now()}] {args.run_name} fail-fast monitor started\n")
        out.write(f"[{now()}] log={args.log}\n")
        out.write(
            f"[{now()}] thresholds: nan_limit={args.nan_limit}, "
            f"entropy>{args.entropy_critical} x{args.entropy_consec}, "
            f"format>{args.format_critical} x{args.format_consec}, "
            f"lowvar>{args.lowvar_critical}% and zero_reward>{args.zeroreward_critical}% x{args.bok_consec}\n"
        )
        out.flush()

        while True:
            if os.path.exists(args.stop_file):
                out.write(f"[{now()}] stop file already exists; exiting monitor\n")
                out.flush()
                return 0

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

            for line in chunk.splitlines():
                m = STEP_RE.search(line)
                if m:
                    current_step = int(m.group(1))

                m = SECTION_RE.match(line)
                if m:
                    current_section = m.group(1)
                    continue

                m = KV_RE.match(line)
                if m and current_section == "actor" and m.group(1) == "entropy_loss":
                    entropy = float(m.group(2))
                    entropy_bad = entropy_bad + 1 if entropy > args.entropy_critical else 0
                    out.write(f"[{now()}] entropy={entropy:.4f} step={current_step} bad={entropy_bad}\n")
                    if entropy_bad >= args.entropy_consec:
                        reason = (
                            f"{args.run_name} fail-fast: entropy_loss>{args.entropy_critical} "
                            f"for {entropy_bad} consecutive actor reports at step {current_step}"
                        )
                        out.write(f"[{now()}] STOP {reason}\n")
                        out.flush()
                        write_stop(args.stop_file, reason)
                        return 0

                m = REWARD_HEALTH_RE.search(line)
                if m:
                    fmt = sf(m.group(1))
                    format_bad = format_bad + 1 if fmt is not None and fmt > args.format_critical else 0
                    if fmt is not None:
                        out.write(f"[{now()}] format_fail={fmt:.4f} step={current_step} bad={format_bad}\n")
                    if format_bad >= args.format_consec:
                        reason = (
                            f"{args.run_name} fail-fast: format_fail_rate>{args.format_critical} "
                            f"for {format_bad} consecutive RewardHealth reports at step {current_step}"
                        )
                        out.write(f"[{now()}] STOP {reason}\n")
                        out.flush()
                        write_stop(args.stop_file, reason)
                        return 0

                m = BOK_WARNING_RE.search(line)
                if m:
                    lowvar = float(m.group(1))
                    zeroreward = float(m.group(2))
                    bok_bad = bok_bad + 1 if lowvar > args.lowvar_critical and zeroreward > args.zeroreward_critical else 0
                    out.write(
                        f"[{now()}] bok_warning lowvar={lowvar:.1f}% zero_reward={zeroreward:.1f}% "
                        f"step={current_step} bad={bok_bad}\n"
                    )
                    if bok_bad >= args.bok_consec:
                        reason = (
                            f"{args.run_name} fail-fast: low_var_rate>{args.lowvar_critical}% and "
                            f"zero_reward_rate>{args.zeroreward_critical}% for {bok_bad} warnings at step {current_step}"
                        )
                        out.write(f"[{now()}] STOP {reason}\n")
                        out.flush()
                        write_stop(args.stop_file, reason)
                        return 0

                if NAN_RE.search(line):
                    nan_count += 1
                    out.write(f"[{now()}] nonfinite_grad count={nan_count} step={current_step}\n")
                    if nan_count >= args.nan_limit:
                        reason = f"{args.run_name} fail-fast: non-finite gradient count reached {nan_count} at step {current_step}"
                        out.write(f"[{now()}] STOP {reason}\n")
                        out.flush()
                        write_stop(args.stop_file, reason)
                        return 0

                if SPIKE_RE.search(line):
                    spike_count += 1
                    out.write(f"[{now()}] spike count={spike_count} step={current_step}\n")

            out.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
