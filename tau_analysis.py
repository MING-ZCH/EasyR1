#!/usr/bin/env python3
"""Tau evolution + NaN correlation analysis for V23 vs V24."""
import re

ANSI = re.compile(r'\x1b\[[0-9;]*m')

def parse_log(logpath):
    """Extract tau, grad_norm, entropy from log."""
    bok_pat = re.compile(r'\[BoK-GRPO\]\s+batch=\d+\s+tau=([\d.]+)\s+step=(\d+)/\d+\s+low_var=(\d+)/\d+\s+collapsed=(\d+).*?tau_bumped=(\d+)/\d+.*?easy_drgrpo=(\d+)/\d+.*?all_correct_filtered=(\d+)/\d+')
    
    tau_data = []  # (step, tau, low_var, collapsed, tau_bumped)
    train_data = {}  # step -> dict
    current_step, current_section = None, None
    
    with open(logpath, 'r', errors='replace') as f:
        for raw in f:
            line = ANSI.sub('', raw).strip()
            
            m = bok_pat.search(line)
            if m:
                tau_data.append({
                    'step': int(m.group(2)), 'tau': float(m.group(1)),
                    'lv': int(m.group(3)), 'collapsed': int(m.group(4)),
                    'tau_bumped': int(m.group(5)), 'easy': int(m.group(6)),
                    'ac_f': int(m.group(7))
                })
                continue
            
            clean = re.sub(r'^\([^)]+\)\s*', '', line)
            if clean.startswith('Step ') and len(clean.split()) == 2:
                try:
                    current_step = int(clean.split()[1])
                    if current_step not in train_data: train_data[current_step] = {}
                    current_section = None
                except: pass
                continue
            if clean == 'actor:': current_section = 'actor'; continue
            if clean == 'critic:': current_section = 'critic'; continue
            if current_step and current_section and ':' in clean and not clean.endswith(':'):
                parts = clean.split(':', 1)
                key, val_str = parts[0].strip(), parts[1].strip()
                if key in ('entropy_loss', 'grad_norm', 'kl_loss', 'grad_norm_ema'):
                    try:
                        train_data[current_step][f'{current_section}/{key}'] = float(val_str)
                    except:
                        if 'nan' in val_str.lower():
                            train_data[current_step][f'{current_section}/{key}'] = float('nan')
    
    return tau_data, train_data

# Parse both
LOG_V23 = "/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest/logs/train/training_interleaved_traj_v23_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej1_bok_grpo_20260410_003702.log"
LOG_V24 = "/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest/logs/train/training_interleaved_traj_v24_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej1_bok_grpo_20260412_045606.log"

print("Parsing V23...")
tau23, train23 = parse_log(LOG_V23)
print("Parsing V24...")
tau24, train24 = parse_log(LOG_V24)

import math

# === TAU EVOLUTION ===
print("\n" + "=" * 80)
print("TAU EVOLUTION COMPARISON")
print("=" * 80)

for label, tau_data, train_data in [("V23", tau23, train23), ("V24", tau24, train24)]:
    print(f"\n### {label} ###")
    print(f"{'Step':>6} {'Tau':>6} {'LV':>5} {'Coll':>5} {'TauBump':>8} {'Entropy':>8} {'GradNorm':>9} {'KL':>8}")
    print("-" * 65)
    
    prev_tau = None
    for d in tau_data:
        s = d['step']
        if s % 10 == 0 or s == tau_data[0]['step'] or s == tau_data[-1]['step'] or d['tau'] < 0.4:
            td = train_data.get(s, {})
            ent = td.get('actor/entropy_loss', None)
            gn = td.get('actor/grad_norm', None)
            kl = td.get('actor/kl_loss', None)
            
            ent_str = f"{ent:.4f}" if ent else "-"
            gn_str = "NaN" if gn is not None and math.isnan(gn) else (f"{gn:.4f}" if gn else "-")
            kl_str = f"{kl:.4f}" if kl else "-"
            
            marker = ""
            if prev_tau and abs(d['tau'] - prev_tau) > 0.05: marker = " <<"
            if gn is not None and (math.isnan(gn) or gn > 5): marker = " !!!"
            
            print(f"{s:>6} {d['tau']:>6.3f} {d['lv']:>5} {d['collapsed']:>5} {d['tau_bumped']:>8} {ent_str:>8} {gn_str:>9} {kl_str:>8}{marker}")
            prev_tau = d['tau']
    
    # Summary
    taus = [d['tau'] for d in tau_data]
    print(f"\nTau range: {min(taus):.3f} - {max(taus):.3f}")
    print(f"Tau first: {taus[0]:.3f}, last: {taus[-1]:.3f}")
    # Find when tau drops below 0.5, 0.4, 0.3
    for threshold in [0.5, 0.4, 0.3]:
        below = [d for d in tau_data if d['tau'] < threshold]
        if below:
            print(f"Tau first < {threshold}: step {below[0]['step']} (tau={below[0]['tau']:.3f})")
        else:
            print(f"Tau never drops below {threshold}")

# === NaN CORRELATION ===
print("\n" + "=" * 80)
print("NaN CORRELATION ANALYSIS (V24)")
print("=" * 80)

nan_steps_v24 = [s for s in sorted(train24.keys()) if 'actor/grad_norm' in train24[s] and math.isnan(train24[s]['actor/grad_norm'])]
print(f"\nNaN grad_norm steps: {nan_steps_v24}")

# For each NaN step, find the corresponding tau and surrounding metrics
if nan_steps_v24:
    tau_by_step = {d['step']: d for d in tau24}
    print(f"\n{'Step':>6} {'Tau':>6} {'Entropy':>8} {'KL':>8} {'GN_EMA':>8}")
    print("-" * 42)
    # Also show 5 steps before first NaN
    start = max(nan_steps_v24[0] - 5, 61)
    for s in range(start, nan_steps_v24[-1] + 1):
        if s in train24:
            td = train24[s]
            ent = td.get('actor/entropy_loss', None)
            kl = td.get('actor/kl_loss', None)
            gn = td.get('actor/grad_norm', None)
            gn_ema = td.get('actor/grad_norm_ema', None)
            tau_d = tau_by_step.get(s, {})
            tau_v = tau_d.get('tau', None) if tau_d else None
            
            marker = " <<< NaN" if gn is not None and math.isnan(gn) else ""
            tau_str = f"{tau_v:.3f}" if tau_v else "-"
            ent_str = f"{ent:.4f}" if ent else "-"
            kl_str = f"{kl:.4f}" if kl else "-"
            gn_ema_str = f"{gn_ema:.4f}" if gn_ema else "-"
            
            print(f"{s:>6} {tau_str:>6} {ent_str:>8} {kl_str:>8} {gn_ema_str:>8}{marker}")

# === TAU DROP TIMELINE ===
print("\n" + "=" * 80)
print("V24 TAU DROP DETAILED TIMELINE")
print("=" * 80)
# Show every step where tau changes significantly
prev_tau = None
for d in tau24:
    if prev_tau is None or abs(d['tau'] - prev_tau) > 0.01 or d['step'] >= 185:
        td = train24.get(d['step'], {})
        gn = td.get('actor/grad_norm', None)
        gn_str = "NaN" if gn and math.isnan(gn) else (f"{gn:.3f}" if gn else "-")
        ent = td.get('actor/entropy_loss', None)
        ent_str = f"{ent:.3f}" if ent else "-"
        print(f"step={d['step']:>3} tau={d['tau']:.3f} lv={d['lv']:>3} collapsed={d['collapsed']:>2} entropy={ent_str} gn={gn_str}")
    prev_tau = d['tau']

print("\n" + "=" * 80)
