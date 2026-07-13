#!/usr/bin/env python3
"""V23 quick metrics for comparison."""
import re, math
ANSI = re.compile(r'\x1b\[[0-9;]*m')
LOG = "/data/workspace/hyleochang/EasyR1-latest/logs/train/training_interleaved_traj_v23_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej1_bok_grpo_20260410_003702.log"

val_data, train_blocks, bok_data, nan_steps = [], {}, [], []
val_pat = re.compile(r'\[ValSummary\]\s+step=(\d+)\s+(.*)')
bok_pat = re.compile(r'\[BoK-GRPO\]\s+batch=\d+.*?step=(\d+)/\d+\s+low_var=(\d+)/\d+.*?easy_drgrpo=(\d+)/\d+.*?all_correct_filtered=(\d+)/\d+.*?adv_mean=([\d.eE+-]+)\s+adv_std=([\d.eE+-]+)')
current_step, current_section = None, None

with open(LOG, 'r', errors='replace') as f:
    for raw in f:
        line = ANSI.sub('', raw).strip()
        m = val_pat.search(line)
        if m:
            step, rest = int(m.group(1)), m.group(2)
            d = {}
            for kv in rest.split():
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    try: d[k] = float(v)
                    except: pass
            val_data.append((step, d))
            continue
        m = bok_pat.search(line)
        if m:
            bok_data.append({'step': int(m.group(1)), 'lv': int(m.group(2)),
                'easy': int(m.group(3)), 'ac_f': int(m.group(4)),
                'adv_std': float(m.group(6))})
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
            if key in ('entropy_loss', 'grad_norm', 'kl_loss', 'pg_loss', 'grad_norm_ema'):
                try:
                    v = float(val_str)
                    train_blocks[current_step][f'{current_section}/{key}'] = v
                except:
                    if 'nan' in val_str.lower():
                        train_blocks[current_step][f'{current_section}/{key}'] = float('nan')
                        if key == 'grad_norm': nan_steps.append(current_step)

print("V23 SUMMARY")
print("=" * 60)
# Val
print("\n### VAL ###")
best_step, best_score = 0, 0
for step, d in val_data:
    ans = d.get('val/answer_reward', 0)
    if ans > best_score: best_score, best_step = ans, step
    print(f"  step {step}: answer={ans:.4f}")
print(f"Peak: {best_score:.4f} @ step {best_step}")

# Training
filled = sorted([s for s in train_blocks if train_blocks[s]])
if filled:
    ent = [(s, train_blocks[s].get('actor/entropy_loss')) for s in filled if 'actor/entropy_loss' in train_blocks[s]]
    kl = [(s, train_blocks[s].get('actor/kl_loss')) for s in filled if 'actor/kl_loss' in train_blocks[s]]
    gn = [(s, train_blocks[s].get('actor/grad_norm')) for s in filled if 'actor/grad_norm' in train_blocks[s]]
    print(f"\n### TRAINING ###")
    if ent: print(f"Entropy: {ent[0][1]:.4f} → {ent[-1][1]:.4f}")
    if kl: print(f"KL: {kl[0][1]:.4f} → {kl[-1][1]:.4f}")
    if gn:
        nan_c = sum(1 for _,v in gn if math.isnan(v))
        non_nan = [(s,v) for s,v in gn if not math.isnan(v)]
        if non_nan: print(f"GradNorm: {non_nan[0][1]:.4f} → {non_nan[-1][1]:.4f}, NaN={nan_c}/{len(gn)}")
    print(f"NaN steps: {nan_steps}")

# BoK
if bok_data:
    bins = {}
    for d in bok_data:
        b = (d['step'] // 50) * 50
        if b not in bins: bins[b] = []
        bins[b].append(d)
    print(f"\n### BOK ({len(bok_data)} steps) ###")
    for b in sorted(bins.keys()):
        items = bins[b]; n = len(items)
        avg = lambda k: sum(x[k] for x in items) / n
        print(f"  bin {b}: BoK={avg('lv')/1024*100:.1f}% Easy={avg('easy')/1024*100:.1f}% AC_f={avg('ac_f')/1024*100:.1f}%")
