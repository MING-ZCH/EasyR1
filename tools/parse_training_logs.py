#!/usr/bin/env python3
"""Unified log parser for EasyR1 training logs."""
import re, json, os, sys

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

def parse_log(log_path):
    result = {
        "config": {}, "validation": [], "training_rewards": [],
        "nan_events": {"total_events": 0, "nan_steps": [], "first_nan_step": None},
        "routing": [], "actor_metrics": [], "total_lines": 0, "total_steps": 0,
    }
    
    current_step = None
    in_reward_block = False
    current_rewards = {}
    current_actor = {}
    in_val_block = False
    val_data = {}
    config_found = set()
    line_count = 0
    
    config_patterns = {
        'lr': (r'actor_rollout_ref_optim_lr:\s*([\d.eE+\-]+)', float),
        'ppo_epochs': (r'ppo_epochs:\s*(\d+)', int),
        'max_grad_norm': (r'max_grad_norm:\s*([\d.]+)', float),
        'kl_coef': (r'kl_coef:\s*([\d.]+)', float),
        'total_epochs': (r'total_epochs:\s*(\d+)', int),
        'rollout_n': (r'rollout_n:\s*(\d+)', int),
    }
    
    with open(log_path, 'r', errors='replace') as f:
        for raw in f:
            line_count += 1
            line = ANSI_RE.sub('', raw).rstrip()
            
            # Config (first 5000 lines)
            if line_count <= 5000:
                for key, (pattern, conv) in config_patterns.items():
                    if key not in config_found:
                        m = re.search(pattern, line)
                        if m:
                            result["config"][key] = conv(m.group(1))
                            config_found.add(key)
                
                for env_key in ['BOK_CLIP', 'BOK_EASY_THRESHOLD', 'BOK_TAU_INIT', 
                                'BOK_TAU_FINAL', 'BOK_LOW_VAR_THRESHOLD', 'BOK_LOGIT_CAP']:
                    m = re.search(rf"'{env_key}':\s*'([^']+)'", line)
                    if m:
                        try: result["config"][env_key] = float(m.group(1))
                        except: result["config"][env_key] = m.group(1)
                
                for ck in [('clip_ratio:', 'clip_ratio'), ('clip_ratio_high:', 'clip_ratio_high'), 
                           ('clip_ratio_dual_high:', 'clip_ratio_dual_high')]:
                    m = re.search(rf'{ck[0]}\s*([\d.]+)', line)
                    if m and ck[1] not in result["config"]:
                        result["config"][ck[1]] = float(m.group(1))
                
                m = re.search(r'adv_estimator:\s*(\w+)', line)
                if m and 'algorithm' not in result["config"]:
                    result["config"]['algorithm'] = m.group(1)
            
            # NaN
            if 'Gradient norm is not finite' in line:
                result["nan_events"]["total_events"] += 1
            
            # BoK-GRPO routing
            m_bok = re.search(r'\[BoK-GRPO\] batch=(\d+) tau=([\d.]+) step=(\d+)/(\d+)\s+'
                              r'low_var=(\d+)/\d+.*?easy_drgrpo=(\d+)/\d+.*?all_correct_filtered=(\d+)/\d+', line)
            if m_bok:
                batch = int(m_bok.group(1))
                step = int(m_bok.group(3))
                result["total_steps"] = int(m_bok.group(4))
                low_var = int(m_bok.group(5))
                easy_drgrpo = int(m_bok.group(6))
                all_correct = int(m_bok.group(7))
                result["routing"].append({
                    "step": step, "tau": float(m_bok.group(2)),
                    "bok": batch - low_var - easy_drgrpo - all_correct,
                    "drgrpo": easy_drgrpo, "all_correct": all_correct, 
                    "low_var": low_var, "batch": batch
                })
            
            # Step marker
            sm = re.search(r'Runner.*Step (\d+)$', line)
            if sm:
                if current_step is not None:
                    if current_rewards:
                        result["training_rewards"].append({"step": current_step, **current_rewards})
                    if current_actor:
                        result["actor_metrics"].append({"step": current_step, **current_actor})
                current_step = int(sm.group(1))
                in_reward_block = False
                current_rewards = {}
                current_actor = {}
                continue
            
            if 'Runner' not in line:
                continue
            
            # Actor metrics
            if current_step is not None:
                for key in ['grad_norm', 'ppo_kl', 'pg_loss', 'pg_clipfrac_higher', 'pg_clipfrac_lower']:
                    m = re.search(rf'\b{key}:\s+([\d.eE+\-]+)', line)
                    if m:
                        try: current_actor[key] = float(m.group(1))
                        except: pass
                    elif key in line and 'nan' in line.lower():
                        current_actor[key] = None
            
            # Reward block
            if line.endswith(') reward:'):
                in_reward_block = True
                continue
            
            if in_reward_block:
                parts = line.split(')', 1)
                if len(parts) > 1:
                    content = parts[1]
                    if content and content[0] != ' ':
                        in_reward_block = False
                    else:
                        m = re.match(r'\s+(\w+):\s+([\d.eE+\-]+)$', content)
                        if m and not m.group(1).startswith('point_step'):
                            current_rewards[m.group(1)] = float(m.group(2))
    
    # Save last
    if current_step is not None:
        if current_rewards:
            result["training_rewards"].append({"step": current_step, **current_rewards})
        if current_actor:
            result["actor_metrics"].append({"step": current_step, **current_actor})
    
    result["total_lines"] = line_count
    
    # Map NaN to steps
    if result["routing"] and result["nan_events"]["total_events"] > 0:
        nan_step_set = set()
        step_lines = []
        nan_lines = []
        with open(log_path, 'r', errors='replace') as f:
            for i, raw in enumerate(f, 1):
                cleaned = ANSI_RE.sub('', raw)
                m = re.search(r'\[BoK-GRPO\].*step=(\d+)/', cleaned)
                if m: step_lines.append((i, int(m.group(1))))
                if 'Gradient norm is not finite' in cleaned:
                    nan_lines.append(i)
        for nl in nan_lines:
            best = None
            for sl, step in step_lines:
                if sl < nl: best = step
                else: break
            if best: nan_step_set.add(best)
        result["nan_events"]["nan_steps"] = sorted(nan_step_set)
        if nan_step_set:
            result["nan_events"]["first_nan_step"] = min(nan_step_set)
    
    # Summaries
    if result["training_rewards"]:
        ans = [r.get('answer', 0) for r in result["training_rewards"]]
        fmt = [r.get('format', 0) for r in result["training_rewards"]]
        pts = [r.get('point', 0) for r in result["training_rewards"]]
        n = len(ans); n10 = min(10, n)
        result["reward_summary"] = {
            "answer_min": min(ans), "answer_max": max(ans),
            "answer_first10": sum(ans[:n10])/n10, "answer_last10": sum(ans[-n10:])/n10,
            "format_first10": sum(fmt[:n10])/n10, "format_last10": sum(fmt[-n10:])/n10,
            "point_first10": sum(pts[:n10])/n10, "point_last10": sum(pts[-n10:])/n10,
            "total_train_steps": n,
        }
    
    if result["routing"]:
        tb = sum(r["batch"] for r in result["routing"])
        result["routing_summary"] = {
            "bok_pct": sum(r["bok"] for r in result["routing"]) / tb * 100,
            "drgrpo_pct": sum(r["drgrpo"] for r in result["routing"]) / tb * 100,
            "all_correct_pct": sum(r["all_correct"] for r in result["routing"]) / tb * 100,
            "low_var_pct": sum(r["low_var"] for r in result["routing"]) / tb * 100,
            "tau_range": f"{result['routing'][0]['tau']:.3f}->{result['routing'][-1]['tau']:.3f}",
        }
    
    return result

if __name__ == "__main__":
    experiments = {
        "V7-standard": "training_interleaved_traj_v7_StepCount_mask_reward_v4_grpo_standard_hm0_gatesoft_fmtrej1_grpo_20260307_072046.log",
        "V8-bok": "training_interleaved_traj_v8_StepCount_mask_reward_v4_bok_grpo_hm0_gatesoft_fmtrej1_bok_grpo_20260309_033602.log",
        "V12-bok": "training_interleaved_traj_v12_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej0_bok_grpo_20260315_013620.log",
        "V13-bok": "training_interleaved_traj_v13_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej0_bok_grpo_20260317_155208.log",
        "V14-hard": "training_interleaved_traj_v14_hard_only_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej0_bok_grpo_20260318_162557.log",
        "V14-mixed": "training_interleaved_traj_v14_mixed_StepCount_mask_reward_v4_bok_grpo_hm0_gateoff_fmtrej0_bok_grpo_20260320_211920.log",
    }
    
    results = {}
    for name, fn in experiments.items():
        path = os.path.join("logs/train", fn)
        if os.path.exists(path):
            print(f"Parsing {name}...", end=" ", flush=True)
            try:
                results[name] = parse_log(path)
                ns = len(results[name].get("routing", []))
                nv = len(results[name].get("validation", []))
                nr = len(results[name].get("training_rewards", []))
                print(f"OK ({results[name]['total_lines']} lines, {ns} routing, {nr} rewards, {nv} vals)")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback; traceback.print_exc()
        else:
            print(f"SKIP {name}: not found")
    
    # Print comparison tables
    print("\n" + "=" * 130)
    print("CROSS-EXPERIMENT COMPARISON")
    print("=" * 130)
    
    print("\n--- CONFIGURATION ---")
    hdr = f"{'Exp':<15} | {'Algo':<12} | {'LR':>10} | {'EP':>4} | {'GN':>5} | {'BOK_CLIP':>8} | {'Thresh':>7} | {'tau_i':>6} | {'eff_int':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        c = r["config"]
        algo = c.get('algorithm', '?')
        lr = c.get('lr', 0)
        ep = c.get('ppo_epochs', '?')
        gn = c.get('max_grad_norm', '?')
        bc = c.get('BOK_CLIP', '-')
        bt = c.get('BOK_EASY_THRESHOLD', '-')
        ti = c.get('BOK_TAU_INIT', '-')
        clip = c.get('clip_ratio_dual_high', c.get('clip_ratio_high', c.get('clip_ratio', 0)))
        try: eff = f"{float(clip)*float(lr)*float(ep):.2e}"
        except: eff = "?"
        print(f"{name:<15} | {algo:<12} | {lr:>10} | {str(ep):>4} | {str(gn):>5} | {str(bc):>8} | {str(bt):>7} | {str(ti):>6} | {eff:>10}")
    
    print("\n--- NaN STABILITY ---")
    hdr2 = f"{'Exp':<15} | {'NaN Events':>10} | {'NaN Steps':>9} | {'1st NaN':>8} | {'NaN%':>6} | {'Eff Steps':>10}"
    print(hdr2)
    print("-" * len(hdr2))
    for name, r in results.items():
        ne = r["nan_events"]
        ns = len(ne["nan_steps"])
        actual = len(r.get("routing", []))
        first = ne["first_nan_step"] or "-"
        pct = f"{ns/actual*100:.1f}%" if actual > 0 and ns > 0 else "0.0%"
        eff = actual - ns
        print(f"{name:<15} | {ne['total_events']:>10} | {ns:>9} | {str(first):>8} | {pct:>6} | {eff:>10}")
    
    print("\n--- TRAINING REWARD TRENDS ---")
    hdr3 = f"{'Exp':<15} | {'Ans_f10':>7} | {'Ans_l10':>7} | {'Ans_max':>7} | {'Fmt_f10':>7} | {'Fmt_l10':>7} | {'Pt_f10':>6} | {'Pt_l10':>6} | {'Steps':>5}"
    print(hdr3)
    print("-" * len(hdr3))
    for name, r in results.items():
        rs = r.get("reward_summary", {})
        if rs:
            print(f"{name:<15} | {rs.get('answer_first10',0):>7.3f} | {rs.get('answer_last10',0):>7.3f} | {rs.get('answer_max',0):>7.3f} | {rs.get('format_first10',0):>7.3f} | {rs.get('format_last10',0):>7.3f} | {rs.get('point_first10',0):>6.3f} | {rs.get('point_last10',0):>6.3f} | {rs.get('total_train_steps',0):>5}")
        else:
            print(f"{name:<15} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {'N/A':>6} | {'N/A':>6} | {'N/A':>5}")
    
    print("\n--- BoK-GRPO ROUTING ---")
    hdr4 = f"{'Exp':<15} | {'BoK%':>6} | {'DrGRPO%':>7} | {'AllCorr%':>8} | {'LowVar%':>7} | {'tau_range':>15}"
    print(hdr4)
    print("-" * len(hdr4))
    for name, r in results.items():
        rs = r.get("routing_summary", {})
        if rs:
            print(f"{name:<15} | {rs['bok_pct']:>6.1f} | {rs['drgrpo_pct']:>7.1f} | {rs['all_correct_pct']:>8.1f} | {rs['low_var_pct']:>7.1f} | {rs['tau_range']:>15}")
        else:
            print(f"{name:<15} | {'N/A':>6} | {'N/A':>7} | {'N/A':>8} | {'N/A':>7} | {'N/A':>15}")
    
    # Save JSON
    output = {}
    for name, r in results.items():
        output[name] = {
            "config": r["config"],
            "validation": r.get("validation", []),
            "nan_steps_count": len(r["nan_events"]["nan_steps"]),
            "nan_first_step": r["nan_events"]["first_nan_step"],
            "nan_total_events": r["nan_events"]["total_events"],
            "reward_summary": r.get("reward_summary", {}),
            "routing_summary": r.get("routing_summary", {}),
            "total_steps": r["total_steps"],
            "actual_steps": len(r.get("routing", [])),
        }
    with open("docs/cross_experiment_comparison.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nJSON saved to docs/cross_experiment_comparison.json")
