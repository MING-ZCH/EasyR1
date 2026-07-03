import re

log_path = '/data/workspace/hyleochang/EasyR1-latest/logs/train/training_interleaved_traj_v6_StepCount_mask_reward_v4_bok_grpo_hm0_gatesoft_fmtrej1_bok_grpo_20260305_171146.log'
ansi_re = re.compile(r'\x1b\[[^m]*m')

steps = {}
current_step = None
section = None

with open(log_path, 'r') as f:
    for line in f:
        clean = ansi_re.sub('', line).strip()
        m = re.search(r'\) Step (\d+)$', clean)
        if m:
            current_step = int(m.group(1))
            if current_step not in steps:
                steps[current_step] = {}
            section = None
            continue
        if current_step is not None:
            if 'actor:' in clean: section = 'actor'
            elif 'critic:' in clean: section = 'critic'
            elif 'advantages:' in clean: section = 'advantages'
            elif clean.endswith('rewards:'): section = 'rewards'
            elif clean.endswith('score:'): section = 'score'
            for key in ['pg_loss', 'ppo_kl', 'entropy_loss', 'grad_norm', 'kl_loss', 'format_fail']:
                m2 = re.search(key + r':\s*([\-\d\.e\+]+)', clean)
                if m2:
                    steps[current_step][key] = float(m2.group(1))
            if section == 'rewards':
                m3 = re.search(r'mean:\s*([\-\d\.e\+]+)', clean)
                if m3:
                    steps[current_step]['reward_mean'] = float(m3.group(1))
                    section = None
            if section == 'advantages':
                m4 = re.search(r'mean:\s*([\-\d\.e\+]+)', clean)
                if m4:
                    steps[current_step]['adv_mean'] = float(m4.group(1))
                    section = None

print(f"{'Step':>4} | {'pg_loss':>8} | {'ppo_kl':>10} | {'entropy':>7} | {'grad':>6} | {'kl_loss':>7} | {'rew_mn':>7} | {'adv_mn':>7} | {'fmt_fl':>6}")
print('-' * 88)
for s in sorted(steps.keys()):
    d = steps[s]
    if 'pg_loss' not in d: continue
    ff = d.get('format_fail', -1)
    ff_str = f'{ff:>6.3f}' if ff >= 0 else '   N/A'
    print(f"{s:>4d} | {d.get('pg_loss',0):>8.4f} | {d.get('ppo_kl',0):>10.7f} | {d.get('entropy_loss',0):>7.3f} | {d.get('grad_norm',0):>6.2f} | {d.get('kl_loss',0):>7.4f} | {d.get('reward_mean',0):>7.3f} | {d.get('adv_mean',0):>7.4f} | {ff_str}")

pg_losses = [steps[s]['pg_loss'] for s in sorted(steps.keys()) if 'pg_loss' in steps[s]]
rewards = [steps[s]['reward_mean'] for s in sorted(steps.keys()) if 'reward_mean' in steps[s]]
ffs = [steps[s]['format_fail'] for s in sorted(steps.keys()) if 'format_fail' in steps[s]]
print(f"\n=== Summary (Steps 1-{max(steps.keys())}) ===")
print(f"pg_loss range: [{min(pg_losses):.4f}, {max(pg_losses):.4f}], avg={sum(pg_losses)/len(pg_losses):.4f}")
print(f"reward_mean range: [{min(rewards):.3f}, {max(rewards):.3f}], avg={sum(rewards)/len(rewards):.3f}")
if ffs:
    print(f"format_fail range: [{min(ffs):.3f}, {max(ffs):.3f}], avg={sum(ffs)/len(ffs):.3f}")
    mid = len(ffs)//2
    print(f"format_fail early(1-{mid}) avg: {sum(ffs[:mid])/max(1,len(ffs[:mid])):.3f}, late({mid+1}-{len(ffs)}) avg: {sum(ffs[mid:])/max(1,len(ffs[mid:])):.3f}")
mid_r = len(rewards)//2
print(f"reward early avg: {sum(rewards[:mid_r])/max(1,mid_r):.3f}, late avg: {sum(rewards[mid_r:])/max(1,len(rewards)-mid_r):.3f}")
