#!/usr/bin/env python3
"""V24 full analysis - handles ANSI escape codes."""
import re, math

LOG = "/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest/logs/train/training_interleaved_traj_v24_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej1_bok_grpo_20260412_045606.log"

ANSI = re.compile(r'\x1b\[[0-9;]*m')

val_data = []
bok_data = []
health_data = []
train_blocks = {}
nan_steps = []

val_pat = re.compile(r'\[ValSummary\]\s+step=(\d+)\s+(.*)')
bok_pat = re.compile(r'\[BoK-GRPO\]\s+batch=\d+.*?step=(\d+)/\d+\s+low_var=(\d+)/\d+.*?easy_drgrpo=(\d+)/\d+.*?all_correct_filtered=(\d+)/\d+.*?ac_released=(\d+)/\d+.*?adv_mean=([\d.eE+-]+)\s+adv_std=([\d.eE+-]+)')
health_pat = re.compile(r'\[RewardHealth\]\s+calls=\d+\s+overall_mean=([\d.]+)\s+answer_mean=([\d.]+)\s+point_mean=([\d.]+)\s+format_fail_rate=([\d.]+)')

current_step = None
current_section = None

with open(LOG, 'r', errors='replace') as f:
    for raw_line in f:
        line = ANSI.sub('', raw_line).strip()
        
        # Val
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
        
        # BoK
        m = bok_pat.search(line)
        if m:
            bok_data.append({'step': int(m.group(1)), 'lv': int(m.group(2)), 
                'easy': int(m.group(3)), 'ac_f': int(m.group(4)), 'ac_r': int(m.group(5)),
                'adv_mean': float(m.group(6)), 'adv_std': float(m.group(7))})
            continue
        
        # Health
        m = health_pat.search(line)
        if m:
            health_data.append({'overall': float(m.group(1)), 'answer': float(m.group(2)),
                'point': float(m.group(3)), 'fmt_fail': float(m.group(4))})
            continue
        
        # Strip pid prefix
        clean = re.sub(r'^\([^)]+\)\s*', '', line)
        
        # Step header: "Step 61"
        if clean.startswith('Step ') and len(clean.split()) == 2:
            try:
                current_step = int(clean.split()[1])
                if current_step not in train_blocks:
                    train_blocks[current_step] = {}
                current_section = None
            except: pass
            continue
        
        # Section
        if clean == 'actor:':
            current_section = 'actor'
            continue
        if clean == 'critic:':
            current_section = 'critic'
            continue
        
        # Metrics
        if current_step and current_section and ':' in clean and not clean.endswith(':'):
            parts = clean.split(':', 1)
            key = parts[0].strip()
            val_str = parts[1].strip()
            if key in ('entropy_loss', 'grad_norm', 'kl_loss', 'pg_loss', 'kl_coef',
                       'pg_clipfrac_higher', 'pg_clipfrac_lower', 'lr', 'ppo_kl',
                       'reward_score', 'spike_count', 'spike_cooldown', 'grad_norm_ema'):
                try:
                    v = float(val_str)
                    train_blocks[current_step][f'{current_section}/{key}'] = v
                except:
                    if 'nan' in val_str.lower():
                        train_blocks[current_step][f'{current_section}/{key}'] = float('nan')
                        if key == 'grad_norm': nan_steps.append(current_step)

# ==================== OUTPUT ====================
print("=" * 80)
print("V24 COMPLETE TRAINING ANALYSIS")
print("=" * 80)

# VAL
print("\n### VAL SCORE TRAJECTORY ###")
print(f"{'Step':>6} {'Answer':>8} {'Point':>8} {'Format':>8} {'Overall':>8}")
print("-" * 45)
best_step, best_score = 0, 0
for step, d in val_data:
    ans = d.get('val/answer_reward', 0)
    pt = d.get('val/point_reward', 0)
    fmt = d.get('val/format_reward', 0)
    ovr = d.get('val/overall_reward', 0)
    if ans > best_score: best_score, best_step = ans, step
    mark = " ***" if ans == best_score and step == best_step else ""
    print(f"{step:>6} {ans:>8.4f} {pt:>8.4f} {fmt:>8.4f} {ovr:>8.4f}{mark}")
print(f"Peak: {best_score:.4f} @ step {best_step}")

# TRAINING
print(f"\n### TRAINING METRICS ###")
sorted_steps = sorted(train_blocks.keys())
filled = [s for s in sorted_steps if train_blocks[s]]
print(f"Steps: {len(sorted_steps)}, with metrics: {len(filled)}")
if filled:
    sample = [s for s in filled if s % 15 == 0 or s == filled[0] or s == filled[-1]]
    print(f"{'Step':>6} {'Entropy':>8} {'PG_Loss':>9} {'KL_Loss':>8} {'GradNorm':>9} {'GN_EMA':>8}")
    print("-" * 55)
    for s in sample:
        d = train_blocks[s]
        def fmt(k):
            v = d.get(k)
            if v is None: return '-'
            if isinstance(v, float) and math.isnan(v): return 'NaN'
            return f'{v:.4f}' if abs(v) >= 0.001 else f'{v:.2e}'
        print(f"{s:>6} {fmt('actor/entropy_loss'):>8} {fmt('actor/pg_loss'):>9} {fmt('actor/kl_loss'):>8} {fmt('actor/grad_norm'):>9} {fmt('actor/grad_norm_ema'):>8}")
    
    # Trends
    ent = [(s, train_blocks[s]['actor/entropy_loss']) for s in filled if 'actor/entropy_loss' in train_blocks[s]]
    kl = [(s, train_blocks[s]['actor/kl_loss']) for s in filled if 'actor/kl_loss' in train_blocks[s]]
    gn = [(s, train_blocks[s]['actor/grad_norm']) for s in filled if 'actor/grad_norm' in train_blocks[s]]
    pg = [(s, train_blocks[s]['actor/pg_loss']) for s in filled if 'actor/pg_loss' in train_blocks[s]]
    
    if ent: print(f"\nEntropy: {ent[0][1]:.4f} → {ent[-1][1]:.4f} (delta {ent[-1][1]-ent[0][1]:+.4f})")
    if kl: print(f"KL Loss: {kl[0][1]:.4f} → {kl[-1][1]:.4f} (delta {kl[-1][1]-kl[0][1]:+.4f})")
    if pg: print(f"PG Loss: {pg[0][1]:.4f} → {pg[-1][1]:.4f}")
    if gn:
        nan_c = sum(1 for _, v in gn if math.isnan(v))
        non_nan = [(s, v) for s, v in gn if not math.isnan(v)]
        if non_nan:
            max_gn = max(v for _, v in non_nan)
            max_s = [s for s, v in non_nan if v == max_gn][0]
            print(f"GradNorm: {non_nan[0][1]:.4f} → {non_nan[-1][1]:.4f}, max={max_gn:.4f}@{max_s}, NaN={nan_c}/{len(gn)}")

# NaN
print(f"\n### NaN GRAD NORM: {len(nan_steps)} steps ###")
if nan_steps: print(f"Steps: {nan_steps}")

# BOK
print(f"\n### BOK ROUTING ({len(bok_data)} steps) ###")
if bok_data:
    bins = {}
    for d in bok_data:
        b = (d['step'] // 25) * 25
        if b not in bins: bins[b] = []
        bins[b].append(d)
    print(f"{'Bin':>6} {'BoK%':>6} {'Easy%':>6} {'AC_f%':>7} {'AC_r%':>7} {'Useful%':>7} {'AdvStd':>8}")
    print("-" * 55)
    for b in sorted(bins.keys()):
        items = bins[b]; n = len(items)
        avg = lambda k: sum(x[k] for x in items) / n
        lv, easy, acf, acr = avg('lv'), avg('easy'), avg('ac_f'), avg('ac_r')
        print(f"{b:>6} {lv/1024*100:>5.1f}% {easy/1024*100:>5.1f}% {acf/1024*100:>6.1f}% {acr/1024*100:>6.1f}% {(lv+easy+acr)/1024*100:>6.1f}% {avg('adv_std'):>8.4f}")

# HEALTH
print(f"\n### REWARD HEALTH ({len(health_data)} points) ###")
if health_data:
    n = len(health_data)
    step_interval = max(1, n // 8)
    idxs = list(range(0, n, step_interval)) + [n-1]
    print(f"{'~Step':>6} {'Overall':>8} {'Answer':>8} {'Point':>8} {'FmtFail':>8}")
    print("-" * 45)
    for i in idxs:
        approx_step = 60 + i  # rough approximation
        d = health_data[i]
        print(f"{approx_step:>6} {d['overall']:>8.4f} {d['answer']:>8.4f} {d['point']:>8.4f} {d['fmt_fail']:>8.4f}")

# VAL detail
print(f"\n### VAL DETAIL ###")
print(f"{'Metric':>35} {'First':>8} {'Last':>8} {'Delta':>8}")
for key in ['val/consistency_violation_reward', 'val/format_reward',
            'val/turns_exceeded_reward', 'val/point_reward', 'val/point_eval_steps_reward']:
    fv = val_data[0][1].get(key, 0)
    lv = val_data[-1][1].get(key, 0)
    print(f"{key.replace('val/',''):>35} {fv:>8.4f} {lv:>8.4f} {lv-fv:>+8.4f}")

print("\n" + "=" * 80)
