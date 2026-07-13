#!/usr/bin/env python3
"""V27 Training Monitor — Monitors v27 (4GPU) and v27b (2GPU) simultaneously.

Key monitoring targets:
  - grad_norm NaN detection
  - format_fail_rate (STRICT_JSON=1 should keep <3%)
  - reward trends (overall, answer, point)
  - entropy stability
  - smart filter: ac_released / all_correct_filtered ratio
  - BoK collapse / allwrong / allcorrect stats

Usage: python3 scripts/monitor_logs.py [--full] [--last N] [--pattern GLOB]
nohup python3 scripts/monitor_logs.py --last 5 > /data/workspace/hyleochang/EasyR1-latest/logs/monitor/monitor_v27.log 2>&1
"""
import re, sys, os, glob
import numpy as np

LOG_DIR = "/data/workspace/hyleochang/EasyR1-latest/logs/train"
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

# Auto-detect TOTAL_STEPS from log filename
TOTAL_STEPS_BY_TAG = {'v27b': 358, 'v27': 213}
DEFAULT_TOTAL_STEPS = 213


def _total_steps_for_log(log_path):
    name = os.path.basename(log_path)
    for tag, n in TOTAL_STEPS_BY_TAG.items():
        if tag in name:
            return n
    return DEFAULT_TOTAL_STEPS


def find_all_active_logs(pattern=None):
    """Return latest log per experiment prefix (grouped by everything before timestamp)."""
    if pattern:
        candidates = sorted(glob.glob(os.path.join(LOG_DIR, pattern)))
        return candidates[-1:] if candidates else []
    ts_re = re.compile(r'_\d{8}_\d{6}\.log$')
    groups = {}
    for p in sorted(glob.glob(os.path.join(LOG_DIR, "training_interleaved_traj_v27*.log"))):
        prefix = ts_re.sub('', os.path.basename(p))
        groups.setdefault(prefix, []).append(p)
    return sorted([sorted(v)[-1] for v in groups.values()])


def find_latest_log():
    logs = find_all_active_logs()
    return logs[-1] if logs else None


def parse_log(log_path):
    step_data = {}
    bok_data = {}
    traj_mask_last = {}
    cur_step = None
    section = None
    nan_events = []
    nan_count = 0
    format_fail_history = []
    train_health = {}

    with open(log_path, 'r') as f:
        for line_no, raw in enumerate(f, 1):
            line = ANSI_RE.sub('', raw).rstrip()

            if 'Gradient norm is not finite' in line:
                nan_count += 1
                nan_events.append(line_no)
                continue

            m = re.search(
                r'\[TrainHealth\] opt_step=(\d+)\s+'
                r'grad_norm=([0-9.]+)\s+ema=([0-9.]+)\s+'
                r'ratio=([0-9.]+)x\s+spike_count=(\d+)', line)
            if m:
                train_health[int(m.group(1))] = {
                    'grad_norm': float(m.group(2)),
                    'ema': float(m.group(3)),
                    'ratio': float(m.group(4)),
                    'spike_count': int(m.group(5)),
                }
                continue

            m = re.search(r'\[RewardHealth\] calls=(\d+).*format_fail_rate=([0-9.]+)', line)
            if m:
                format_fail_history.append((int(m.group(1)), float(m.group(2))))
                continue

            m = re.search(
                r'\[BoK-GRPO\].*step=(\d+)/\d+.*collapsed=(\d+).*allwrong_capped=(\d+)/(\d+)', line)
            if m:
                s = int(m.group(1))
                bok_data[s] = {'collapsed': int(m.group(2)),
                               'allwrong': f"{m.group(3)}/{m.group(4)}"}
                mc = re.search(r'all_correct_filtered=(\d+)/(\d+)', line)
                if mc:
                    bok_data[s]['allcorrect'] = f"{mc.group(1)}/{mc.group(2)}"
                mc2 = re.search(r'ac_released=(\d+)/(\d+)', line)
                if mc2:
                    bok_data[s]['ac_released'] = f"{mc2.group(1)}/{mc2.group(2)}"
                mc3 = re.search(r'easy_drgrpo=(\d+)/(\d+)', line)
                if mc3:
                    bok_data[s]['easy_drgrpo'] = f"{mc3.group(1)}/{mc3.group(2)}"
                continue

            m = re.search(
                r'\[traj_mask_reward\].*calls=(\d+).*hit_unused_rate=([0-9.]+).*miss_rate=([0-9.]+).*dup_rate=([0-9.]+)', line)
            if m:
                traj_mask_last = {'calls': int(m.group(1)),
                                  'hit_rate': float(m.group(2)),
                                  'miss_rate': float(m.group(3)),
                                  'dup_rate': float(m.group(4))}
                m_ns = re.search(r'traj_no_sequence.*?:\s*(\d+)', line)
                if m_ns:
                    traj_mask_last['no_sequence'] = int(m_ns.group(1))
                continue

            m = re.search(r'\(Runner[^)]*\)\s+Step\s+(\d+)\s*$', line)
            if m:
                cur_step = int(m.group(1))
                step_data.setdefault(cur_step, {})
                section = None
                continue

            if cur_step is None:
                continue

            m = re.match(r'.*\)\s+(actor|critic|reward|global_seqlen|perf|prompt_length|response_length):\s*$', line)
            if m:
                section = m.group(1)
                continue

            if section in ('reward', 'actor', 'perf', 'response_length'):
                m = re.search(r'\)\s{2,}([\w.]+):\s+([0-9e.\-+]+)\s*$', line)
                if m:
                    prefix = {'reward': 'r_', 'actor': 'a_', 'perf': 'p_', 'response_length': 'rl_'}[section]
                    step_data[cur_step][prefix + m.group(1)] = float(m.group(2))

    for s, d in bok_data.items():
        step_data.setdefault(s, {}).update(d)

    return step_data, traj_mask_last, {
        'nan_count': nan_count, 'nan_events': nan_events,
        'format_fail_history': format_fail_history, 'train_health': train_health,
    }


def main():
    full = '--full' in sys.argv
    show_last = 10
    custom_pattern = None
    for i, a in enumerate(sys.argv):
        if a == '--last' and i + 1 < len(sys.argv):
            show_last = int(sys.argv[i + 1])
        if a == '--pattern' and i + 1 < len(sys.argv):
            custom_pattern = sys.argv[i + 1]

    log_paths = find_all_active_logs(custom_pattern)
    if not log_paths:
        print(f"No v27* logs found in {LOG_DIR}")
        sys.exit(1)

    for log_path in log_paths:
        _monitor_one(log_path, full=full, show_last=show_last)

def _monitor_one(log_path, full=False, show_last=10):
    TOTAL_STEPS = _total_steps_for_log(log_path)
    print(f"\nLog: {os.path.basename(log_path)}")
    step_data, traj_mask_last, extra = parse_log(log_path)
    steps = sorted([s for s in step_data if step_data[s].get('r_overall') is not None])

    if not steps:
        print("No training steps found yet.")
        return

    latest = steps[-1]
    d = step_data[latest]

    print("=" * 100)
    print(f"  V27 TRAINING MONITOR — Step {latest}/{TOTAL_STEPS} ({latest / TOTAL_STEPS * 100:.1f}%)")
    print("=" * 100)

    # === ALERTS ===
    alerts = []
    nan_count = extra['nan_count']
    if nan_count > 0:
        alerts.append(f"NaN GRAD x{nan_count}")
    latest_grad = d.get('a_grad_norm')
    if latest_grad is not None:
        try:
            if np.isnan(latest_grad) or np.isinf(latest_grad):
                alerts.append(f"LATEST GRAD NaN/INF")
        except (TypeError, ValueError):
            pass

    ff_history = extra['format_fail_history']
    if ff_history:
        recent_ff = [r for _, r in ff_history[-10:]]
        avg_ff = np.mean(recent_ff)
        if avg_ff > 0.10:
            alerts.append(f"FORMAT_FAIL HIGH (avg={avg_ff:.1%})")
        elif avg_ff > 0.05:
            alerts.append(f"FORMAT_FAIL WARN (avg={avg_ff:.1%})")

    if d.get('a_entropy_loss', 999) < 0.3:
        alerts.append(f"ENTROPY LOW ({d['a_entropy_loss']:.3f})")
    if d.get('a_entropy_loss', 0) > 0.9:
        alerts.append(f"ENTROPY HIGH ({d['a_entropy_loss']:.3f})")
    if d.get('a_kl_loss', 0) > 0.5:
        alerts.append(f"KL HIGH ({d['a_kl_loss']:.3f})")
    if latest_grad is not None and not np.isnan(latest_grad) and latest_grad > 5.0:
        alerts.append(f"GRAD SPIKE ({latest_grad:.3f})")
    if d.get('r_overall', 1) < 0.5:
        alerts.append(f"REWARD LOW ({d['r_overall']:.3f})")
    if traj_mask_last.get('no_sequence', 0) > 0:
        alerts.append(f"MASK NO_SEQ ({traj_mask_last['no_sequence']})")

    if alerts:
        print(f"\n  ALERTS: {', '.join(alerts)}")
    else:
        print(f"\n  No alerts — training healthy")

    # === Key Metrics ===
    print(f"\n  Reward:  overall={d.get('r_overall', 0):.3f}  answer={d.get('r_answer', 0):.3f}  "
          f"point={d.get('r_point', 0):.3f}  format={d.get('r_format', 0):.3f}")
    print(f"  Actor:   entropy={d.get('a_entropy_loss', 0):.3f}  kl={d.get('a_kl_loss', 0):.4f}  "
          f"grad_norm={d.get('a_grad_norm', 0):.3f}  pg_loss={d.get('a_pg_loss', 0):.4f}")

    # === Grad Norm & NaN ===
    th = extra['train_health']
    if th:
        th_steps = sorted(th.keys())
        lth = th[th_steps[-1]]
        print(f"\n  GradHealth: opt_step={th_steps[-1]}  grad={lth['grad_norm']:.4f}  "
              f"ema={lth['ema']:.4f}  ratio={lth['ratio']:.2f}x  spikes={lth['spike_count']}")
    print(f"  NaN Skip: {nan_count} events", end='')
    if nan_count > 0:
        total_opt = len(th) + nan_count
        print(f" ({nan_count}/{total_opt}={nan_count/total_opt*100:.1f}%)", end='')
        last5 = extra['nan_events'][-5:]
        print(f"  [last at lines: {', '.join(str(l) for l in last5)}]", end='')
    print()

    # === Format Fail ===
    if ff_history:
        recent = ff_history[-20:]
        rates = [r for _, r in recent]
        print(f"  FormatFail: avg={np.mean(rates):.1%}  max={max(rates):.1%}  min={min(rates):.1%}  "
              f"(last {len(recent)} calls)")

    # === Mask Stats ===
    if traj_mask_last:
        print(f"  Mask: calls={traj_mask_last.get('calls', 0):,}  "
              f"hit={traj_mask_last.get('hit_rate', 0):.1%}  "
              f"miss={traj_mask_last.get('miss_rate', 0):.1%}  "
              f"no_seq={traj_mask_last.get('no_sequence', 0)}")

    # === Smart Filter Stats ===
    ac_rel_steps = [s for s in steps if 'ac_released' in step_data[s]]
    if ac_rel_steps:
        last_ac = step_data[ac_rel_steps[-1]]
        print(f"  SmartFilter: allcorrect_filtered={last_ac.get('allcorrect', 'N/A')}  "
              f"ac_released={last_ac.get('ac_released', 'N/A')}  "
              f"easy_drgrpo={last_ac.get('easy_drgrpo', 'N/A')}")

    # === Trend ===
    if len(steps) >= 10:
        r5, p5 = steps[-5:], steps[-10:-5]
        print(f"\n  Trend (step {p5[0]}-{p5[-1]} -> {r5[0]}-{r5[-1]}):")
        for name, key in [('overall', 'r_overall'), ('answer', 'r_answer'),
                          ('point', 'r_point'), ('format', 'r_format'),
                          ('entropy', 'a_entropy_loss'), ('kl', 'a_kl_loss'),
                          ('grad_norm', 'a_grad_norm')]:
            p = np.mean([step_data[s].get(key, 0) for s in p5])
            r = np.mean([step_data[s].get(key, 0) for s in r5])
            arrow = "+" if r > p + 0.005 else "-" if r < p - 0.005 else "="
            flag = " NaN!" if key == 'a_grad_norm' and any(
                np.isnan(step_data[s].get(key, 0)) for s in r5) else ""
            print(f"    {name:<12} {p:.4f} -> {r:.4f}  {arrow}{r-p:+.4f}{flag}")

    # === Table ===
    display = steps if full else steps[-show_last:]
    print(f"\n{'Step':>5} {'overall':>7} {'answer':>7} {'point':>7} {'format':>6} "
          f"{'entropy':>7} {'kl':>7} {'grad':>6} {'fmtfail':>8} "
          f"{'coll':>4} {'allwrong':>10} {'ac_filt':>8} {'ac_rel':>8} {'easy_dr':>8}")
    print('-' * 120)
    for s in display:
        sd = step_data[s]
        def g(k, f='.3f'):
            v = sd.get(k)
            if v is None: return ''
            try:
                if np.isnan(v): return '  NaN!'
            except: pass
            return format(v, f)

        ff_val = ''
        if ff_history:
            closest = min(ff_history, key=lambda x: abs(x[0] - s), default=None)
            if closest: ff_val = f"{closest[1]:.1%}"

        print(f"{s:>5} {g('r_overall'):>7} {g('r_answer'):>7} {g('r_point'):>7} "
              f"{g('r_format'):>6} {g('a_entropy_loss'):>7} {g('a_kl_loss'):>7} "
              f"{g('a_grad_norm'):>6} {ff_val:>8} "
              f"{sd.get('collapsed', ''):>4} {sd.get('allwrong', ''):>10} "
              f"{sd.get('allcorrect', ''):>8} {sd.get('ac_released', ''):>8} {sd.get('easy_drgrpo', ''):>8}")

    # === ETA ===
    tp = step_data[latest].get('p_time_per_step')
    if tp:
        rem = TOTAL_STEPS - latest
        print(f"\n  ETA: ~{rem * tp / 3600:.1f}h ({rem} steps x {tp:.0f}s/step)")


if __name__ == '__main__':
    main()
