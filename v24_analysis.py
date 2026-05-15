#!/usr/bin/env python3
"""V24 complete training analysis - streaming parse."""
import re, sys

LOG = "/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest/logs/train/training_interleaved_traj_v24_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej1_bok_grpo_20260412_045606.log"

val_data = []
bok_data = []
train_metrics = []
nan_steps = []

val_pat = re.compile(r'\[ValSummary\]\s+step=(\d+)\s+(.*)')
bok_pat = re.compile(r'\[BoK-GRPO\]\s+step=(\d+).*?lv=(\d+).*?easy=(\d+).*?ac_filtered=(\d+).*?ac_released=(\d+).*?easy_scale=([\d.]+)')
train_pat = re.compile(r"'(actor/entropy_loss|actor/pg_loss|actor/pg_clipfrac|critic/reward_score|actor/kl_loss|actor/kl_coef|actor/grad_norm|critic/grad_norm)':\s*([\d.eE+-]+|nan)")
step_pat = re.compile(r"'global_step':\s*(\d+)")

with open(LOG, 'r', errors='replace') as f:
    for line in f:
        m = val_pat.search(line)
        if m:
            step = int(m.group(1))
            rest = m.group(2)
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
            bok_data.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                            int(m.group(4)), int(m.group(5)), float(m.group(6))))
            continue
        if 'actor/' in line or 'critic/' in line:
            metrics = dict(train_pat.findall(line))
            step_m = step_pat.search(line)
            if metrics and step_m:
                step = int(step_m.group(1))
                train_metrics.append((step, metrics))
                if 'nan' in str(metrics.get('actor/grad_norm', '')):
                    nan_steps.append(step)

print("=" * 80)
print("V24 COMPLETE TRAINING ANALYSIS")
print("=" * 80)

print("\n### VAL SCORE TRAJECTORY ###")
print(f"Total val checkpoints: {len(val_data)}")
if val_data:
    print(f"{'Step':>6} {'Answer':>8} {'Point':>8} {'Format':>8} {'Overall':>8}")
    print("-" * 45)
    best_step, best_score = 0, 0
    for step, d in val_data:
        ans = d.get('val/answer_reward', 0)
        pt = d.get('val/point_reward', 0)
        fmt = d.get('val/format_reward', 0)
        ovr = d.get('val/overall_reward', 0)
        marker = ""
        if ans > best_score:
            best_score = ans
            best_step = step
            marker = " <-- BEST"
        print(f"{step:>6} {ans:>8.4f} {pt:>8.4f} {fmt:>8.4f} {ovr:>8.4f}{marker}")
    print(f"\nBest answer_reward: {best_score:.4f} at step {best_step}")
    first_ans = val_data[0][1].get('val/answer_reward', 0)
    last_ans = val_data[-1][1].get('val/answer_reward', 0)
    print(f"First (step {val_data[0][0]}): {first_ans:.4f}")
    print(f"Last  (step {val_data[-1][0]}): {last_ans:.4f}")
    print(f"Delta: {last_ans - first_ans:+.4f}")

print(f"\n### TRAINING METRICS ###")
print(f"Total snapshots: {len(train_metrics)}")
if train_metrics:
    sample_steps = set()
    for s, _ in train_metrics:
        if s % 15 == 0 or s <= 65 or s >= 210:
            sample_steps.add(s)
    print(f"{'Step':>6} {'Entropy':>9} {'PG_Loss':>9} {'KL_Loss':>9} {'GradNorm':>10}")
    print("-" * 50)
    for step, d in train_metrics:
        if step in sample_steps:
            ent = d.get('actor/entropy_loss', '-')
            pg = d.get('actor/pg_loss', '-')
            kl = d.get('actor/kl_loss', '-')
            gn = d.get('actor/grad_norm', '-')
            print(f"{step:>6} {ent:>9} {pg:>9} {kl:>9} {gn:>10}")
    first_s, first_d = train_metrics[0]
    last_s, last_d = train_metrics[-1]
    print(f"\nFirst step {first_s}: {first_d}")
    print(f"Last  step {last_s}: {last_d}")

print(f"\n### NaN GRAD NORM ###")
print(f"Steps with NaN: {len(nan_steps)}")
if nan_steps:
    print(f"NaN steps: {nan_steps[:30]}{'...' if len(nan_steps) > 30 else ''}")

print(f"\n### BOK ROUTING ###")
print(f"Total BoK steps: {len(bok_data)}")
if bok_data:
    bins = {}
    for step, lv, easy, ac_f, ac_r, es in bok_data:
        b = (step // 25) * 25
        if b not in bins:
            bins[b] = {'lv':[], 'easy':[], 'ac_f':[], 'ac_r':[], 'es':[]}
        bins[b]['lv'].append(lv)
        bins[b]['easy'].append(easy)
        bins[b]['ac_f'].append(ac_f)
        bins[b]['ac_r'].append(ac_r)
        bins[b]['es'].append(es)
    print(f"{'Bin':>6} {'BoK%':>6} {'Easy%':>6} {'AC_f%':>7} {'AC_r%':>7} {'Useful%':>7}")
    print("-" * 48)
    for b in sorted(bins.keys()):
        d = bins[b]
        n = len(d['lv'])
        avg_lv = sum(d['lv'])/n; avg_easy = sum(d['easy'])/n
        avg_acf = sum(d['ac_f'])/n; avg_acr = sum(d['ac_r'])/n
        useful = avg_lv + avg_easy + avg_acr
        print(f"{b:>6} {avg_lv/1024*100:>5.1f}% {avg_easy/1024*100:>5.1f}% {avg_acf/1024*100:>6.1f}% {avg_acr/1024*100:>6.1f}% {useful/1024*100:>6.1f}%")

print(f"\n### VAL DETAILED METRICS ###")
if val_data:
    print(f"{'Metric':>35} {'First':>8} {'Last':>8} {'Delta':>8}")
    for key in ['val/consistency_violation_reward', 'val/format_reward',
                'val/stop_violation_reward', 'val/stopped_by_answer_reward',
                'val/turns_exceeded_reward', 'val/point_eval_steps_reward',
                'val/point_reward', 'val/no_point_pred_reward']:
        fv = val_data[0][1].get(key, 0)
        lv = val_data[-1][1].get(key, 0)
        sk = key.replace('val/','')
        print(f"{sk:>35} {fv:>8.4f} {lv:>8.4f} {lv-fv:>+8.4f}")

print("\n" + "=" * 80)
