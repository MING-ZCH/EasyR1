#!/usr/bin/env python3
"""Comprehensive v22 NaN grad_norm analysis."""
import re

log_path = '/data/workspace/hyleochang/EasyR1-latest/logs/train/training_interleaved_traj_v22_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej1_bok_grpo_20260408_024614.log'
ANSI = re.compile(r'\x1b\[[0-9;]*m')

steps = {}
current_step = None
section = None

with open(log_path, 'r', errors='replace') as f:
    for raw_line in f:
        line = ANSI.sub('', raw_line).strip()
        m = re.match(r'\([^)]+\)\s*(.*)', line)
        content = m.group(1) if m else line
        
        m = re.match(r'^Step (\d+)$', content)
        if m:
            current_step = int(m.group(1))
            if current_step not in steps:
                steps[current_step] = {}
            section = None
            continue
        
        if current_step is None:
            continue
        
        m_sec = re.match(r'^(actor|critic|reward|perf|response_length|prompt_length|global_seqlen):$', content)
        if m_sec:
            section = m_sec.group(1)
            continue
        
        m_kv = re.match(r'^\s*([\w/.]+):\s+(.+)', content)
        if m_kv and section:
            key = m_kv.group(1)
            val = m_kv.group(2).strip()
            steps[current_step][f"{section}/{key}"] = val

print(f"{'Step':>5} {'grad':>10} {'ema':>8} {'entropy':>8} {'kl':>8} {'pg_loss':>8} {'overall':>8} {'answer':>8} {'point':>8} {'format':>6} {'resp_mn':>8}")
print("-" * 110)
for s in sorted(steps.keys()):
    d = steps[s]
    gn = d.get('actor/grad_norm', '?')
    is_nan = 'nan' in str(gn).lower()
    gn_disp = 'NaN' if is_nan else gn
    ema = d.get('actor/grad_norm_ema', '?')
    en = d.get('actor/entropy_loss', '?')
    kl = d.get('actor/kl_loss', '?')
    pg = d.get('actor/pg_loss', '?')
    ov = d.get('reward/overall', '?')
    ans = d.get('reward/answer', '?')
    pt = d.get('reward/point', '?')
    fm = d.get('reward/format', '?')
    resp = d.get('response_length/mean', '?')
    flag = ' <NaN' if is_nan else ''
    print(f'{s:>5} {gn_disp:>10} {ema:>8} {en:>8} {kl:>8} {pg:>8} {ov:>8} {ans:>8} {pt:>8} {fm:>6} {resp:>8}{flag}')
