#!/usr/bin/env python3
"""Monitor a long EasyR1 training log and periodically sync an offline W&B run."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RAY_PREFIX_RE = re.compile(r"^\([^)]*(?:pid|Runner|WorkerDict)[^)]*\)\s*")
VAL_SUMMARY_RE = re.compile(r"\[ValSummary\]\s+step=(\d+)\s+suite=([^\s]+)\s+(.*)$")
VAL_MIX_RE = re.compile(r"\[ValTaskMix\]\s+suite=([^\s]+)\s+samples=(\d+)\s+(.*)$")
ROLLOUT_RE = re.compile(r"\[Rollout\]\s+Step\s+(\d+):\s+(Start|Finish)\s+generating sequences")
STEP_RE = re.compile(r"^Step\s+(\d+)\s*$")
KV_FLOAT_RE = re.compile(r"([A-Za-z0-9_./-]+)=(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
YAML_KV_RE = re.compile(
    r"^(\s*)([A-Za-z0-9_./-]+):(?:\s+(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|true|false|null))?\s*$"
)


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def clean_line(line: str) -> str:
    line = ANSI_RE.sub("", line).rstrip("\n")
    if "\r" in line:
        line = line.split("\r")[-1]
    line = RAY_PREFIX_RE.sub("", line)
    return line.rstrip()


def parse_scalar(raw: str | None) -> float | bool | None:
    if raw is None:
        return None
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw == "null":
        return None
    try:
        return float(raw)
    except Exception:
        return None


def parse_metrics_blob(blob: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in KV_FLOAT_RE.findall(blob):
        try:
            out[key] = float(val)
        except Exception:
            pass
    return out


def parse_log(log_path: Path, previous_event_keys: set[str]) -> tuple[dict[str, Any], set[str]]:
    parsed: dict[str, Any] = {
        "errors": [],
        "val": {},
        "val_mix": {},
        "rollout": {},
        "step_metrics": {},
        "new_events": [],
        "max_rollout_step": None,
        "max_completed_step": None,
    }
    if not log_path.exists():
        parsed["errors"].append(f"log_missing:{log_path}")
        return parsed, previous_event_keys

    lines = log_path.read_text(errors="replace").splitlines()
    current_step: int | None = None
    stack: list[tuple[int, str]] = []

    for lineno, raw in enumerate(lines, start=1):
        line = clean_line(raw)
        if re.search(r"Traceback|ERROR|Exception|Killed|OutOfMemory|CUDA out|invalid device|RuntimeError", line):
            parsed["errors"].append(f"{lineno}:{line[:500]}")

        m = VAL_MIX_RE.search(line)
        if m:
            suite = m.group(1)
            parsed["val_mix"][suite] = {
                "line": lineno,
                "samples": int(m.group(2)),
                "metrics": parse_metrics_blob(m.group(3)),
            }

        m = VAL_SUMMARY_RE.search(line)
        if m:
            step = int(m.group(1))
            suite = m.group(2)
            metrics = parse_metrics_blob(m.group(3))
            parsed["val"][suite] = {"line": lineno, "step": step, "metrics": metrics}
            event_key = f"val:{step}:{suite}:{lineno}"
            if event_key not in previous_event_keys:
                parsed["new_events"].append({"type": "val_summary", "step": step, "suite": suite, "line": lineno})
                previous_event_keys.add(event_key)

        m = ROLLOUT_RE.search(line)
        if m:
            step = int(m.group(1))
            phase = m.group(2).lower()
            parsed["rollout"].setdefault(str(step), {})[phase] = lineno
            parsed["max_rollout_step"] = max(parsed["max_rollout_step"] or step, step)
            event_key = f"rollout:{step}:{phase}:{lineno}"
            if event_key not in previous_event_keys:
                parsed["new_events"].append({"type": f"rollout_{phase}", "step": step, "line": lineno})
                previous_event_keys.add(event_key)

        m = STEP_RE.match(line)
        if m:
            current_step = int(m.group(1))
            stack = []
            parsed["step_metrics"].setdefault(str(current_step), {})
            parsed["max_completed_step"] = max(parsed["max_completed_step"] or current_step, current_step)
            event_key = f"step:{current_step}:{lineno}"
            if event_key not in previous_event_keys:
                parsed["new_events"].append({"type": "step_block", "step": current_step, "line": lineno})
                previous_event_keys.add(event_key)
            continue

        if current_step is not None:
            y = YAML_KV_RE.match(line)
            if y:
                indent = len(y.group(1))
                key = y.group(2)
                value = parse_scalar(y.group(3))
                while stack and stack[-1][0] >= indent:
                    stack.pop()
                if value is None:
                    stack.append((indent, key))
                elif isinstance(value, float):
                    path = "/".join([item[1] for item in stack] + [key])
                    parsed["step_metrics"].setdefault(str(current_step), {})[path] = value

    parsed["errors"] = parsed["errors"][-20:]
    return parsed, previous_event_keys


def selected_metric_summary(metrics: dict[str, float]) -> dict[str, float]:
    keep = {}
    patterns = (
        "reward",
        "score",
        "loss",
        "kl",
        "entropy",
        "grad",
        "lr",
        "clip",
        "length",
        "turn",
        "time",
        "throughput",
    )
    for key, val in metrics.items():
        low = key.lower()
        if any(p in low for p in patterns):
            keep[key] = val
    return keep


def trend_summary(step_metrics: dict[str, dict[str, float]], end_step: int, window: int) -> str:
    steps = [s for s in range(max(0, end_step - window + 1), end_step + 1) if str(s) in step_metrics]
    if not steps:
        return f"No step metrics found for steps <= {end_step}."
    keys = sorted({k for s in steps for k in selected_metric_summary(step_metrics[str(s)]).keys()})
    lines = [f"## Step {end_step} Trend ({steps[0]}-{steps[-1]})", ""]
    if not keys:
        lines.append("Step blocks were found, but no numeric train metrics matched the trend key set yet.")
        return "\n".join(lines)
    for key in keys:
        vals = [step_metrics[str(s)][key] for s in steps if key in step_metrics[str(s)]]
        if not vals:
            continue
        first = vals[0]
        last = vals[-1]
        avg = statistics.fmean(vals)
        delta = last - first
        lines.append(f"- {key}: last={last:.6g}, avg={avg:.6g}, delta={delta:+.6g}, n={len(vals)}")
    return "\n".join(lines)


def estimate_eta(
    state: dict[str, Any],
    total_steps: int,
    step_metrics: dict[str, dict[str, float]] | None = None,
) -> str:
    reported_times: list[tuple[int, float]] = []
    for raw_step, metrics in (step_metrics or {}).items():
        try:
            step = int(raw_step)
            seconds = float(metrics.get("time_per_step", 0.0))
        except (TypeError, ValueError):
            continue
        if step > 0 and math.isfinite(seconds) and seconds > 0:
            reported_times.append((step, seconds))

    if reported_times:
        reported_times.sort()
        recent_seconds = [seconds for _, seconds in reported_times[-5:]]
        sec_per_step = statistics.median(recent_seconds)
        done = reported_times[-1][0]
        remain = max(total_steps - done, 0)
        eta = dt.datetime.now() + dt.timedelta(seconds=remain * sec_per_step)
        return (
            f"median_step={sec_per_step / 60:.1f} min, done={done}/{total_steps}, "
            f"remaining={remain}, eta={eta.isoformat(timespec='minutes')}, "
            "source=log_time_per_step"
        )

    completed_times = state.get("completed_step_times", {})
    pairs = sorted((int(k), v) for k, v in completed_times.items() if int(k) > 0)
    if len(pairs) < 2:
        return "ETA pending: need at least two completed training steps."
    durations = []
    for (s0, t0), (s1, t1) in zip(pairs, pairs[1:]):
        if s1 > s0:
            duration = (t1 - t0) / (s1 - s0)
            if duration > 0:
                durations.append(duration)
    if not durations:
        return "ETA pending: completed step intervals unavailable."
    sec_per_step = statistics.median(durations[-5:])
    done = pairs[-1][0]
    remain = max(total_steps - done, 0)
    eta = dt.datetime.now() + dt.timedelta(seconds=remain * sec_per_step)
    return (
        f"median_step={sec_per_step / 60:.1f} min, done={done}/{total_steps}, "
        f"remaining={remain}, eta={eta.isoformat(timespec='minutes')}"
    )


def check_process_alive(patterns: list[str]) -> tuple[bool | None, list[str]]:
    if not patterns:
        return None, []
    try:
        proc = subprocess.run(["ps", "-eo", "pid,cmd"], text=True, capture_output=True, check=False, timeout=20)
    except Exception:
        return None, []
    matches = []
    skip_fragments = (
        "monitor_v36_training.py",
        "run_v36_hourly_watchdog.sh",
        "wandb sync",
        "grep",
    )
    for raw in proc.stdout.splitlines():
        if any(fragment in raw for fragment in skip_fragments):
            continue
        if any(pattern in raw for pattern in patterns):
            matches.append(raw.strip()[:500])
    return bool(matches), matches[:20]


def run_wandb_sync(run_dir: Path, project: str | None, sync_log: Path, timeout_s: int) -> dict[str, Any]:
    cmd = [
        "python3",
        "-m",
        "wandb",
        "sync",
        "--include-offline",
        "--no-mark-synced",
        "--no-sync-tensorboard",
    ]
    if project:
        cmd += ["--project", project]
    cmd.append(str(run_dir))
    lock_path = sync_log.with_name(sync_log.stem + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            with sync_log.open("a", encoding="utf-8") as f:
                f.write(f"\n[{now_iso()}] SKIP sync: lock busy ({lock_path})\n")
            return {"status": "skipped_lock_busy", "cmd": cmd}
        with sync_log.open("a", encoding="utf-8") as f:
            f.write(f"\n[{now_iso()}] RUN {' '.join(cmd)}\n")
            f.flush()
            try:
                proc = subprocess.run(cmd, stdout=f, stderr=f, timeout=timeout_s, check=False)
                f.write(f"[{now_iso()}] exit={proc.returncode}\n")
                return {"status": "completed", "returncode": proc.returncode, "cmd": cmd}
            except subprocess.TimeoutExpired:
                f.write(f"[{now_iso()}] timeout after {timeout_s}s\n")
                return {"status": "timeout", "timeout_s": timeout_s, "cmd": cmd}
            except Exception as exc:
                f.write(f"[{now_iso()}] exception={type(exc).__name__}: {exc}\n")
                return {"status": "exception", "exception": f"{type(exc).__name__}: {exc}", "cmd": cmd}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--wandb-run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--project", default="easy_r1")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--sync-interval", type=int, default=600)
    ap.add_argument("--sync-timeout", type=int, default=240)
    ap.add_argument("--summary-every", type=int, default=10)
    ap.add_argument("--total-steps", type=int, default=149)
    ap.add_argument("--stale-after-seconds", type=int, default=7200)
    ap.add_argument("--process-pattern", action="append", default=[])
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    log_path = Path(args.log)
    run_dir = Path(args.wandb_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / "v36_training_monitor_state.json"
    events_path = out_dir / "v36_training_monitor_events.jsonl"
    summary_path = out_dir / "v36_training_monitor_summary.md"
    trends_path = out_dir / "v36_training_monitor_trends.md"
    sync_log = out_dir / "v36_wandb_sync.log"

    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            state = {}
    state.setdefault("seen_events", [])
    state.setdefault("completed_step_times", {})
    state.setdefault("summarized_steps", [])
    seen_events = set(state["seen_events"])
    last_sync = float(state.get("last_sync_ts", 0.0) or 0.0)
    process_patterns = args.process_pattern or [
        "ray::Runner",
        "ray::WorkerDict",
        "WorkerDict",
        "verl.trainer.main",
        "main_ppo",
        "vllm",
    ]

    while True:
        parsed, seen_events = parse_log(log_path, seen_events)
        now = time.time()
        for event in parsed["new_events"]:
            event["seen_at"] = now_iso()
            if event["type"] == "step_block":
                state["completed_step_times"].setdefault(str(event["step"]), now)
            with events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=True) + "\n")

        max_completed = parsed.get("max_completed_step") or 0
        new_trends = []
        for step in range(args.summary_every, max_completed + 1, args.summary_every):
            if step not in state["summarized_steps"]:
                new_trends.append(trend_summary(parsed["step_metrics"], step, args.summary_every))
                state["summarized_steps"].append(step)

        eta = estimate_eta(state, args.total_steps, parsed["step_metrics"])
        log_mtime = log_path.stat().st_mtime if log_path.exists() else None
        log_stale_seconds = (now - log_mtime) if log_mtime else None
        process_alive, process_matches = check_process_alive(process_patterns)
        health = "ok"
        health_reasons: list[str] = []
        if parsed["errors"]:
            health = "errors_detected"
            health_reasons.append("recent error-like lines were found in the training log")
        if process_alive is False:
            health_reasons.append(
                "no matching training/Ray/vLLM process is visible on this host; "
                "fresh log activity is used as the primary liveness signal"
            )
        if log_stale_seconds is not None and log_stale_seconds > args.stale_after_seconds:
            if (parsed.get("max_rollout_step") or 0) > max_completed:
                health = "stalled_or_dead"
                health_reasons.append(
                    f"log has been stale for {log_stale_seconds:.0f}s while rollout step "
                    f"{parsed.get('max_rollout_step')} has not completed"
                )
            else:
                health_reasons.append(f"log has been stale for {log_stale_seconds:.0f}s")
                if process_alive is False:
                    health = "training_process_missing"

        summary_lines = [
            f"# V36 Training Monitor",
            "",
            f"- updated_at: {now_iso()}",
            f"- log: `{log_path}`",
            f"- wandb_run_dir: `{run_dir}`",
            f"- max_completed_step: {max_completed}",
            f"- max_rollout_step: {parsed.get('max_rollout_step')}",
            f"- eta: {eta}",
            f"- health: {health}",
            f"- health_reasons: {health_reasons or ['none']}",
            f"- log_stale_seconds: {log_stale_seconds:.0f}" if log_stale_seconds is not None else "- log_stale_seconds: n/a",
            f"- process_alive: {process_alive}",
            f"- process_matches: {len(process_matches)}",
            f"- errors_recent: {len(parsed['errors'])}",
            "",
            "## Validation",
        ]
        for suite, item in sorted(parsed["val"].items()):
            metrics = item["metrics"]
            summary_lines.append(
                f"- {suite}: step={item['step']}, overall={metrics.get('val/' + suite + '/overall_reward', metrics.get('overall_reward', 'n/a'))}, "
                f"answer={metrics.get('val/' + suite + '/answer_reward', metrics.get('answer_reward', 'n/a'))}, "
                f"format_fail={metrics.get('val/' + suite + '/format_fail_reward', metrics.get('format_fail_reward', 'n/a'))}, "
                f"turns_exceeded={metrics.get('val/' + suite + '/turns_exceeded_reward', metrics.get('turns_exceeded_reward', 'n/a'))}"
            )
        summary_lines.append("")
        summary_lines.append("## Recent Errors")
        summary_lines.extend([f"- {e}" for e in parsed["errors"][-10:]] or ["- none"])
        if new_trends:
            with trends_path.open("a", encoding="utf-8") as f:
                f.write(f"\n# {now_iso()}\n\n")
                f.write("\n\n".join(new_trends))
                f.write("\n")
            summary_lines.append("")
            summary_lines.append("## New 10-Step Trend Summaries")
            summary_lines.extend(new_trends)
        elif trends_path.exists():
            trend_lines = trends_path.read_text(errors="replace").splitlines()[-80:]
            summary_lines.append("")
            summary_lines.append("## Recent 10-Step Trend Summaries")
            summary_lines.extend(trend_lines or ["- none yet"])
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        state["seen_events"] = sorted(seen_events)[-5000:]
        state["last_parse"] = now_iso()
        state["last_max_completed_step"] = max_completed
        state["last_max_rollout_step"] = parsed.get("max_rollout_step")
        state["health"] = health
        state["health_reasons"] = health_reasons
        state["log_stale_seconds"] = log_stale_seconds
        state["process_alive"] = process_alive
        state["process_matches"] = process_matches
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

        if run_dir.exists() and now - last_sync >= args.sync_interval:
            sync_result = run_wandb_sync(run_dir, args.project, sync_log, args.sync_timeout)
            last_sync = now
            state["last_sync_ts"] = last_sync
            state["last_sync"] = now_iso()
            state["last_sync_result"] = sync_result
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

        if args.once:
            return 0
        time.sleep(max(args.interval, 10))


if __name__ == "__main__":
    raise SystemExit(main())
