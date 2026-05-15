#!/usr/bin/env python3
import re, math
ANSI = re.compile(r'\x1b\[[0-9;]*m')
LOG = "/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest/logs/train/training_interleaved_traj_v12_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej0_bok_grpo_20260315_013620.log"

val_data, bok_data, train_blocks = [], [], {}
val_pat = re.compile(r'\[ValSummary\]\s+step=(\d+)\s+(.*)')
bok_pat = re.compile(r'\[BoK-GRPO\]\s+batch=\d+\s+tau=([\d.]+)\s+step=(\d+)/\d+\s+low_var=(\d+)/\d+.*?easy_drgrpo=(\d+)/\d+.*?all_correct_filtered=(\d+)/\d+')
current_step, current_section = None, None
nan_steps = []

with open(LOG, 'r', errors='replace') as f:
    for raw in f:
        line = ANSI.sub('', raw).strip()
        m = val_pat.search(line)
        if m:
            step = int(m.group(1))
            d = {}
            for kv in m.group(2).split():
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    try: d[k] = float(v)
                    except: pass
            val_data.append((step, d))
            continue
        m = bok_pat.search(line)
        if m:
            bok_data.append({'tau': float(m.group(1)), 'step': int(m.group(2)),
                'lv': int(m.group(3)), 'easy': int(m.group(4)), 'ac_f': int(m.group(5))})
            continue
        clean = re.sub(r'^\([^)]+\)\s*', '', line)
        if clean.startswith('Step ') and len(clean.split()) == 2:
            try:
                current_step = int(clean.split()[1])
                if current_step not in train_blocks: train_blocks[current_step] = {}
                current_section = None
            except: pass
            continue
        if clean == 'actor:': current_section = 'actor'; continue
        if clean == 'critic:': current_section = 'critic'; continue
        if current_step and current_section and ':' in clean and not clean.endswith(':'):
            parts = clean.split(':', 1)
            key, val_str = parts[0].strip(), parts[1].strip()
            if key in ('entropy_loss', 'grad_norm', 'kl_loss', 'pg_loss'):
                try:
                    train_blocks[current_step][f'{current_section}/{key}'] = float(val_str)
                except:
                    if 'nan' in val_str.lower():
                        train_blocks[current_step][f'{current_section}/{key}'] = float('nan')
                        if key == 'grad_norm': nan_steps.append(current_step)

print("V12 ANALYSIS")
print("=" * 60)
print(f"\n### VAL ({len(val_data)} checkpoints) ###")
best_s, best_v = 0, 0
for step, d in val_data:
    ans = d.get('val/answer_reward', 0)
    if ans > best_v: best_v, best_s = ans, step
    print(f"  step {step}: answer={ans:.4f}" + (" ***" if step == best_s and ans == best_v else ""))
print(f"Peak: {best_v:.4f} @ step {best_s}")

filled = sorted([s for s in train_blocks if train_blocks[s]])
if filled:
    print(f"\n### TRAINING ({len(filled)} steps) ###")
    ent = [(s, train_blocks[s].get('actor/entropy_loss')) for s in filled if 'actor/entropy_loss' in train_blocks[s]]
    kl = [(s, train_blocks[s].get('actor/kl_loss')) for s in filled if 'actor/kl_loss' in train_blocks[s]]
    gn = [(s, train_blocks[s].get('actor/grad_norm')) for s in filled if 'actor/grad_norm' in train_blocks[s]]
    if ent: print(f"Entropy: {ent[0][1]:.4f} → {ent[-1][1]:.4f}")
    if kl: print(f"KL: {kl[0][1]:.4f} → {kl[-1][1]:.4f}")
    if gn:
        nan_c = sum(1 for _,v in gn if math.isnan(v))
        non_nan = [(s,v) for s,v in gn if not math.isnan(v)]
        if non_nan:
            max_gn = max(v for _,v in non_nan)
            print(f"GradNorm: {non_nan[0][1]:.4f} → {non_nan[-1][1]:.4f}, max={max_gn:.4f}, NaN={nan_c}/{len(gn)}")
    print(f"NaN steps: {nan_steps[:20]}")

    # Sample
    print(f"\n{'Step':>6} {'Entropy':>8} {'KL':>8} {'GradNorm':>9}")
    for s in filled:
        if s % 20 == 0 or s == filled[0] or s == filled[-1]:
            d = train_blocks[s]
            e = d.get('actor/entropy_loss', None)
            k = d.get('actor/kl_loss', None)
            g = d.get('actor/grad_norm', None)
            gs = "NaN" if g and math.isnan(g) else (f"{g:.4f}" if g else "-")
            print(f"{s:>6} {e if e else '-':>8} {k if k else '-':>8} {gs:>9}")

if bok_data:
    bins = {}
    for d in bok_data:
        b = (d['step'] // 30) * 30
        if b not in bins: bins[b] = []
        bins[b].append(d)
    print(f"\n### BOK ({len(bok_data)} steps) ###")
    print(f"{'Bin':>6} {'Tau':>6} {'BoK%':>6} {'Easy%':>6} {'AC_f%':>7}")
    for b in sorted(bins.keys()):
        items = bins[b]; n = len(items)
        avg = lambda k: sum(x[k] for x in items) / n
        print(f"{b:>6} {avg('tau'):>6.3f} {avg('lv')/1024*100:>5.1f}% {avg('easy')/1024*100:>5.1f}% {avg('ac_f')/1024*100:>6.1f}%")
