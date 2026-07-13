#!/usr/bin/env python3
"""V24 full analysis with correct format parsing."""
import re

LOG = "/data/workspace/hyleochang/EasyR1-latest/logs/train/training_interleaved_traj_v24_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej1_bok_grpo_20260412_045606.log"

val_data = []
bok_data = []
health_data = []
train_blocks = {}  # step -> {metric: value}

val_pat = re.compile(r'\[ValSummary\]\s+step=(\d+)\s+(.*)')
bok_pat = re.compile(r'\[BoK-GRPO\]\s+batch=\d+.*?step=(\d+)/\d+\s+low_var=(\d+)/\d+.*?easy_drgrpo=(\d+)/\d+.*?all_correct_filtered=(\d+)/\d+.*?ac_released=(\d+)/\d+.*?adv_mean=([\d.eE+-]+)\s+adv_std=([\d.eE+-]+)')
health_pat = re.compile(r'\[RewardHealth\]\s+calls=\d+\s+overall_mean=([\d.]+)\s+answer_mean=([\d.]+)\s+point_mean=([\d.]+)\s+format_fail_rate=([\d.]+)')
step_header_pat = re.compile(r'^(?:\(Runner pid=\d+\)\s+)?Step\s+(\d+)\s*$')
detail_pat = re.compile(r'\[BoK-GRPO-Detail\].*?easy_scale=([\d.]+)')
train_health_pat = re.compile(r'\[TrainHealth\]\s+opt_step=\d+\s+grad_norm=([\d.eE+-]+|nan)\s+ema=([\d.eE+-]+|nan)')

current_step = None
current_section = None  # 'actor' or 'critic'
nan_steps = []

with open(LOG, 'r', errors='replace') as f:
    for line in f:
        stripped = line.strip()
        
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
        
        # BoK routing
        m = bok_pat.search(line)
        if m:
            step = int(m.group(1))
            es_m = detail_pat.search(line)
            bok_data.append({
                'step': step, 'lv': int(m.group(2)), 'easy': int(m.group(3)),
                'ac_f': int(m.group(4)), 'ac_r': int(m.group(5)),
                'adv_mean': float(m.group(6)), 'adv_std': float(m.group(7))
            })
            continue
        
        # RewardHealth
        m = health_pat.search(line)
        if m:
            health_data.append({'overall': float(m.group(1)), 'answer': float(m.group(2)),
                                'point': float(m.group(3)), 'fmt_fail': float(m.group(4))})
            continue
        
        # TrainHealth (grad norm)
        m = train_health_pat.search(line)
        if m:
            gn = m.group(1)
            if gn == 'nan':
                nan_steps.append(len(health_data))  # approximate step
            continue
        
        # Step header
        m = step_header_pat.search(stripped.replace('(Runner pid=1228830) ', ''))
        if m or ('Step ' in line and len(stripped.split()) <= 4):
            try:
                # extract step num
                parts = stripped.split()
                for p in parts:
                    if p.isdigit():
                        current_step = int(p)
                        if current_step not in train_blocks:
                            train_blocks[current_step] = {}
                        current_section = None
                        break
            except:
                pass
            continue
        
        # Section headers
        if stripped.endswith('actor:') or stripped == 'actor:':
            current_section = 'actor'
            continue
        if stripped.endswith('critic:') or stripped == 'critic:':
            current_section = 'critic'
            continue
        
        # Metric values like "  entropy_loss: 0.525"
        if current_step and current_section:
            clean = stripped
            if clean.startswith('(Runner'):
                clean = re.sub(r'^\(Runner pid=\d+\)\s*', '', clean)
            if ':' in clean and not clean.endswith(':'):
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
                            if key == 'grad_norm':
                                nan_steps.append(current_step)

print("=" * 80)
print("V24 COMPLETE TRAINING ANALYSIS")
print("=" * 80)

# === Val ===
print("\n### VAL SCORE TRAJECTORY ###")
print(f"Checkpoints: {len(val_data)}")
if val_data:
    print(f"{'Step':>6} {'Answer':>8} {'Point':>8} {'Format':>8} {'Overall':>8}")
    print("-" * 45)
    best_step, best_score = 0, 0
    for step, d in val_data:
        ans = d.get('val/answer_reward', 0)
        pt = d.get('val/point_reward', 0)
        fmt = d.get('val/format_reward', 0)
        ovr = d.get('val/overall_reward', 0)
        marker = " ***" if ans > best_score else ""
        if ans > best_score: best_score, best_step = ans, step
        print(f"{step:>6} {ans:>8.4f} {pt:>8.4f} {fmt:>8.4f} {ovr:>8.4f}{marker}")
    print(f"\nPeak: {best_score:.4f} @ step {best_step}")
    print(f"First→Last: {val_data[0][1].get('val/answer_reward',0):.4f} → {val_data[-1][1].get('val/answer_reward',0):.4f}")

# === Training Metrics ===
print(f"\n### TRAINING METRICS EVOLUTION ###")
print(f"Steps with data: {len(train_blocks)}")
if train_blocks:
    sorted_steps = sorted(train_blocks.keys())
    sample = [s for s in sorted_steps if s % 15 == 0 or s == sorted_steps[0] or s == sorted_steps[-1]]
    print(f"{'Step':>6} {'Entropy':>8} {'PG_Loss':>8} {'KL_Loss':>8} {'GradNorm':>9} {'GN_EMA':>8}")
    print("-" * 55)
    for s in sample:
        d = train_blocks[s]
        ent = d.get('actor/entropy_loss', None)
        pg = d.get('actor/pg_loss', None)
        kl = d.get('actor/kl_loss', None)
        gn = d.get('actor/grad_norm', None)
        gn_ema = d.get('actor/grad_norm_ema', None)
        print(f"{s:>6} {ent if ent is not None else '-':>8} {pg if pg is not None else '-':>8} {kl if kl is not None else '-':>8} {gn if gn is not None else '-':>9} {gn_ema if gn_ema is not None else '-':>8}")
    
    # Entropy trend
    ent_vals = [(s, train_blocks[s].get('actor/entropy_loss')) for s in sorted_steps if 'actor/entropy_loss' in train_blocks[s]]
    if ent_vals:
        print(f"\nEntropy: {ent_vals[0][1]:.4f} (step {ent_vals[0][0]}) → {ent_vals[-1][1]:.4f} (step {ent_vals[-1][0]})")
    kl_vals = [(s, train_blocks[s].get('actor/kl_loss')) for s in sorted_steps if 'actor/kl_loss' in train_blocks[s]]
    if kl_vals:
        print(f"KL Loss: {kl_vals[0][1]:.4f} (step {kl_vals[0][0]}) → {kl_vals[-1][1]:.4f} (step {kl_vals[-1][0]})")
    gn_vals = [(s, train_blocks[s].get('actor/grad_norm')) for s in sorted_steps if 'actor/grad_norm' in train_blocks[s]]
    if gn_vals:
        import math
        nan_count = sum(1 for _, v in gn_vals if math.isnan(v))
        non_nan = [(s, v) for s, v in gn_vals if not math.isnan(v)]
        if non_nan:
            print(f"GradNorm: {non_nan[0][1]:.4f} → {non_nan[-1][1]:.4f}, NaN count: {nan_count}/{len(gn_vals)}")

# === NaN ===
print(f"\n### NaN GRAD NORM ###")
print(f"NaN steps: {len(nan_steps)}")
if nan_steps:
    print(f"Steps: {nan_steps[:50]}")

# === BoK Routing ===
print(f"\n### BOK ROUTING ###")
print(f"Data points: {len(bok_data)}")
if bok_data:
    bins = {}
    for d in bok_data:
        b = (d['step'] // 25) * 25
        if b not in bins: bins[b] = []
        bins[b].append(d)
    print(f"{'Bin':>6} {'BoK%':>6} {'Easy%':>6} {'AC_f%':>7} {'AC_r%':>7} {'Useful%':>7} {'AdvMean':>8} {'AdvStd':>8}")
    print("-" * 65)
    for b in sorted(bins.keys()):
        items = bins[b]
        n = len(items)
        avg = lambda k: sum(x[k] for x in items) / n
        lv, easy, acf, acr = avg('lv'), avg('easy'), avg('ac_f'), avg('ac_r')
        useful = lv + easy + acr
        print(f"{b:>6} {lv/1024*100:>5.1f}% {easy/1024*100:>5.1f}% {acf/1024*100:>6.1f}% {acr/1024*100:>6.1f}% {useful/1024*100:>6.1f}% {avg('adv_mean'):>+8.4f} {avg('adv_std'):>8.4f}")

# === RewardHealth ===
print(f"\n### REWARD HEALTH ###")
print(f"Data points: {len(health_data)}")
if health_data:
    # Sample every ~25
    interval = max(1, len(health_data) // 8)
    print(f"{'#':>4} {'Overall':>8} {'Answer':>8} {'Point':>8} {'FmtFail':>8}")
    print("-" * 40)
    for i in range(0, len(health_data), interval):
        d = health_data[i]
        print(f"{i:>4} {d['overall']:>8.4f} {d['answer']:>8.4f} {d['point']:>8.4f} {d['fmt_fail']:>8.4f}")
    d = health_data[-1]
    print(f"LAST {d['overall']:>8.4f} {d['answer']:>8.4f} {d['point']:>8.4f} {d['fmt_fail']:>8.4f}")

# === Val Detailed ===
print(f"\n### VAL DETAILED ###")
if val_data:
    print(f"{'Metric':>35} {'First':>8} {'Last':>8} {'Delta':>8}")
    for key in ['val/consistency_violation_reward', 'val/format_reward',
                'val/stop_violation_reward', 'val/turns_exceeded_reward',
                'val/point_eval_steps_reward', 'val/point_reward']:
        fv = val_data[0][1].get(key, 0)
        lv = val_data[-1][1].get(key, 0)
        print(f"{key.replace('val/',''):>35} {fv:>8.4f} {lv:>8.4f} {lv-fv:>+8.4f}")

print("\n" + "=" * 80)
