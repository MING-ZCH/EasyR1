import re

log_path = '/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest/logs/train/training_interleaved_traj_v6_StepCount_mask_reward_v4_bok_grpo_hm0_gatesoft_fmtrej1_bok_grpo_20260305_171146.log'
ansi_re = re.compile(r'\x1b\[[^m]*m')

# Collect reward values grouped by step (each step has n=16 rollouts per prompt, 64 prompts = 1024 total)
rewards_by_batch = {}
batch_num = 0
current_rewards = []

with open(log_path, 'r') as f:
    for line in f:
        clean = ansi_re.sub('', line).strip()
        
        # RewardHealth marks end of a batch's reward computation
        if '[RewardHealth]' in clean:
            m = re.search(r'calls=(\d+)', clean)
            if m:
                batch_num = int(m.group(1))
        
        # Extract individual reward details
        m = re.search(r'trajectory_reward.*overall=([\d\.]+).*format=([\d\.]+).*point=([\d\.]+).*answer=([\d\.]+)', clean)
        if m:
            overall = float(m.group(1))
            fmt = float(m.group(2))
            point = float(m.group(3))
            answer = float(m.group(4))
            current_rewards.append({'overall': overall, 'format': fmt, 'point': point, 'answer': answer})

# Analyze reward value distribution
overalls = [r['overall'] for r in current_rewards]
answers = [r['answer'] for r in current_rewards]
points = [r['point'] for r in current_rewards]

print(f"=== Reward Value Distribution (n={len(current_rewards)} detail samples) ===")

# Bin overall rewards
bins = {}
for v in overalls:
    key = round(v, 2)
    bins[key] = bins.get(key, 0) + 1
print("\nOverall reward distribution:")
for k in sorted(bins.keys()):
    pct = bins[k]/len(overalls)*100
    bar = '#' * int(pct/2)
    print(f"  {k:>6.2f}: {bins[k]:>4d} ({pct:>5.1f}%) {bar}")

# Bin answer rewards
bins_a = {}
for v in answers:
    key = round(v, 2)
    bins_a[key] = bins_a.get(key, 0) + 1
print("\nAnswer reward distribution:")
for k in sorted(bins_a.keys()):
    pct = bins_a[k]/len(answers)*100
    bar = '#' * int(pct/2)
    print(f"  {k:>6.2f}: {bins_a[k]:>4d} ({pct:>5.1f}%) {bar}")

# Check how many are essentially binary (0 or ~1)
binary_count = sum(1 for v in overalls if v < 0.01 or v > 0.95)
print(f"\nBinary-like rewards (0 or >0.95): {binary_count}/{len(overalls)} = {binary_count/len(overalls)*100:.1f}%")

# Intermediate values
mid_count = sum(1 for v in overalls if 0.01 <= v <= 0.95)
print(f"Intermediate rewards (0.01-0.95): {mid_count}/{len(overalls)} = {mid_count/len(overalls)*100:.1f}%")

# Check format_fail=0 cases specifically  
ff_zero = sum(1 for r in current_rewards if r['format'] < 0.01)
print(f"\nFormat fail (format=0): {ff_zero}/{len(current_rewards)} = {ff_zero/len(current_rewards)*100:.1f}%")
