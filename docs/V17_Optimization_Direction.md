# V17 Report Verification & Optimization Direction
> Generated: 2026-03-27
> Status: V17 NOT yet started training

---

## Part 1: Report Verification Results

### Errors Found & Fixed in `V7_V16_Complete_Version_Analysis.md`

| # | Section | Error | Correction | Source |
|---|---------|-------|------------|--------|
| 1 | V7-GRPO params | KL_coef=1e-2 | **KL_coef=5e-2** (runtime log shows 0.05) | `logs/grpo_standard/` |
| 2 | Section 7 | "V17 LR=1e-6, eff_intensity=5.6e-7" | **V17 LR=1.5e-6, eff_intensity=8.4e-7** (bok_grpo case line 177) | `v17.sh:177` |
| 3 | Section 7 | "1e-6 是最平衡选择(V17已采用)" | **V17实际采用1.5e-6 (与V12相同)** | `v17.sh` |
| 4 | Appendix | "V17 - V12 recipe + V16 safety" | Updated to reflect actual params | — |

### Cross-Verified Correct Parameters

| Version | Parameter | Report Value | Runtime/Script Value | ✓ |
|---------|-----------|-------------|---------------------|---|
| V7-GRPO | clip_h | 0.30 | 0.30 (grpo_standard case) | ✅ |
| V7-GRPO | LR | 1e-6 | 1e-6 | ✅ |
| V12 | KL_coef | 0.02 | 0.02 (runtime log) | ✅ |
| V15 | KL_coef | 0.02 | 0.02 (runtime log) | ✅ |
| V16 | KL_coef | 0.03 | 0.03 (runtime log) | ✅ |
| V17 | ppo_epochs | 2 | 2 | ✅ |
| V17 | grad_norm | 0.5 | 0.5 | ✅ |
| V17 | clip_h | 0.28 | 0.28 | ✅ |
| V17 | KL_coef | 2e-2 | 2e-2 | ✅ |
| V17 | BOK_CLIP | 4.0 | 4.0 | ✅ |
| V17 | EASY_TH | 0.50 | 0.50 | ✅ |
| V17 | save_freq | 15 | 15 | ✅ |
| V17 | data | easy-only 0_10 | StepCountQA-RL-Traj_0_10 | ✅ |

---

## Part 2: V17 Actual Configuration (Verified)

```
LR = 1.5e-6 (bok_grpo case, line 177 — "same as V11 for fair comparison")
ppo_epochs = 2
clip_ratio_high = 0.28
eff_intensity = 1.5e-6 × 0.28 × 2 = 8.4e-7 ← DANGER ZONE (same as V12!)
KL_coef = 2e-2
max_grad_norm = 0.5
BOK_CLIP = 4.0 (V12=3.0)
EASY_TH = 0.50 (V12=0.75)
TAU_INIT = 0.7, TAU_FINAL = 0.3
rollout_n = 16, rollout_temp = 1.0
save_freq = 15, val_freq = 15
total_epochs = 1, ~178 steps
data = StepCountQA-RL-Traj_0_10 (easy-only, 11,455 samples)
```

---

## Part 3: V17 vs V12 Deep Comparison

### What's the SAME (eff_intensity = 8.4e-7)
- LR=1.5e-6, ppo=2, grad_norm=0.5, KL=0.02
- Easy-only data, 178 steps, binary reward
- Predicted NaN rate: ~40-50%

### What's DIFFERENT in V17

| Parameter | V12 | V17 | Expected Impact |
|-----------|-----|-----|-----------------|
| BOK_CLIP | 3.0 | **4.0** | ≈Neutral — only activates at τ<0.3 for 1-2/16 groups (minority of training) |
| EASY_TH | 0.75 | **0.50** | **Positive** — routes 78% fewer groups through BoK → reduces gradient peak variance |
| save_freq | 20 | **15** | **Positive** — 13 checkpoints vs 10, +30% capture opportunities |
| val_freq | — | **15** | **Positive** — real-time monitoring of policy quality |

### Quantitative Analysis: EASY_TH Impact

With `easy data (batch_mean~0.82)`, typical group reward distribution:
- AllCorrect (16/16): ~25% of groups → filtered (adv=0)
- EasyDrGRPO: TH=0.75 routes `score_mean ≥ 0.75` → ~42/64 groups  
  TH=0.50 routes `score_mean ≥ 0.50` → ~56/64 groups (+33%)
- BoK (remaining): TH=0.75 → ~18/64 groups | TH=0.50 → ~4/64 groups (**-78%!**)

**Key**: BoK groups generate 18-24x the gradient of EasyDrGRPO. Reducing BoK groups from 28% to 6% of batch significantly lowers gradient variance → may delay NaN onset by 5-10 steps (S85 → S90-95).

### BOK_CLIP=4.0 vs 3.0 Advantage Analysis (Simulated)

| pass@16 | τ | K-scaling adv_win | Clipped@3.0 | Clipped@4.0 | Δ |
|---------|---|-------------------|-------------|-------------|---|
| 1/16 | 0.7 | 2.23 | 2.23 | 2.23 | 0 |
| 1/16 | 0.5 | 3.85 | 3.00 | 3.85 | **+0.85** |
| 1/16 | 0.3 | 8.48 | 3.00 | 4.00 | **+1.00** |
| 4/16 | 0.7 | 1.19 | 1.19 | 1.19 | 0 |
| 4/16 | 0.3 | 2.35 | 2.35 | 2.35 | 0 |

CLIP=4.0 only matters at τ<0.5 AND pass@16≤2/16. In V12, NaN onset was at S85 where τ≈0.57 — at that τ, CLIP never activates! The CLIP difference only helps in late training (S130+, τ<0.4) which is already NaN-dominated.

**Verdict**: CLIP=4.0 → ≈neutral net effect on V17.

---

## Part 4: V17 Optimization Assessment

### V17 Expected Outcome
- **Probability of beating V12 (82.04%)**: ~35-45%
- **Expected range**: 80.0% - 83.5%
- **Primary improvement source**: EASY_TH=0.50 (gradient variance reduction) + denser checkpointing
- **Primary risk**: same eff_intensity=8.4e-7 → same ~40-50% NaN rate
- **Decision: Run V17 as-is first** — low opportunity cost, clean experimental signal

### Critical V17 Monitoring Targets
1. `grad_norm` in S70-100 (NaN onset zone) — compare with V12's S85
2. `[BoK-GRPO] easy_drgrpo=?/64` — should be >55/64 if EASY_TH working
3. `adv_range` — should be smaller than V12 due to fewer BoK groups
4. If NaN starts before S60 → kill run immediately (EASY_TH increase may have backfired)

---

## Part 5: Optimization Roadmap (Priority Order)

### Priority 1: V17 — Run As-Is (NOW)
```
Config: As designed (LR=1.5e-6, EASY_TH=0.50, CLIP=4.0, save=15)
Expected: 80-83%, training ~4-6 hours
Purpose: Establish whether EASY_TH + denser save improves over V12
```

### Priority 2: V17b — Safe LR Reduction (PREPARE NOW)
```
Change: ACTOR_LR="1e-6" → eff_intensity=5.6e-7 (border zone)
Everything else identical to V17
Expected: 80.5-81.5%, much lower NaN risk (~3-9%)
When: Launch if V17 pixmo-test < 81%
```

**V17b Math**: 
- V17 clean steps ~85-95, V17b clean steps ~165-175
- V17 per-step intensity 1.0x, V17b 0.67x
- V17 effective training = 85×1.0 = **85**, V17b = 170×0.67 = **114** (+34%)
- V17b trades peak potential for reliability

### Priority 3: V18 — sqrt(K) Advantage Scaling (BREAKTHROUGH POTENTIAL)
```
Code change (verl/trainer/core_algos.py L624):
  BEFORE: advantages_1d[global_i] = (w[j].item() - 1.0 / K) * K
  AFTER:  advantages_1d[global_i] = (w[j].item() - 1.0 / K) * math.sqrt(K)

Config: LR=3e-6 (compensate for 4x smaller gradient), BOK_CLIP=2.0
Expected: 81-84%, near-zero NaN rate
```

**Why sqrt(K) works**:
- Current K=16 scaling: advantage peaks at 8.48 → clipped to 4.0 → gradient spike → bf16 overflow
- sqrt(K)=4 scaling: advantage peaks at 2.12 → NO clipping needed → smooth gradients
- With 3x higher LR to compensate: effective gradient comparable to V12 but much more stable
- eff_intensity_effective ≈ 3e-6 × 0.28 × 2 / 4 = 4.2e-7 (SAFE zone!)

**sqrt(K) Advantage Table**:
| pass@16 | τ | sqrt(K) adv_win | sqrt(K) adv_lose |
|---------|---|-----------------|------------------|
| 1/16 | 0.7 | 0.56 | -0.04 |
| 1/16 | 0.5 | 0.96 | -0.06 |
| 1/16 | 0.3 | 2.12 | -0.14 |
| 4/16 | 0.5 | 0.42 | -0.14 |

Clean gradient profile: winner gets moderate positive, losers get small negative. No extreme values.

### Priority 4: V17c — Cosine LR Decay (Quick Win)
```
Change: Add lr_scheduler_type=cosine, decay from 1.5e-6 to 3e-7
Purpose: Full LR during safe zone (S0-S80), reduced LR in danger zone (S85+)
Expected: Fewer NaN in late training, preserves early learning
When: Can combine with V17 or V18
```

---

## Part 6: Decision Matrix

| Priority | Version | Key Change | Expected pixmo-test | NaN Risk | Effort |
|----------|---------|-----------|---------------------|----------|--------|
| **1** | V17 | EASY_TH=0.50, save=15 | 80-83% | High (40-50%) | **Ready** |
| **2** | V17b | V17 + LR=1e-6 | 80.5-81.5% | Low (3-9%) | **1-line change** |
| **3** | V18 | sqrt(K) + LR=3e-6 | 81-84% | Very low | Code change |
| **4** | V17c | V17 + cosine LR decay | 81-83% | Medium | Config change |

---

## Part 7: Recommendation

### Immediate Action
1. **Launch V17 as-is** — the V12 recipe + EASY_TH improvement is the logical next step
2. **Prepare V17b script** — one-line change backup (`ACTOR_LR="1e-6"`)
3. **If V17 < 81%** → launch V17b; **if V17 ≥ 82%** → V17 strategy validated

### Medium-term (after V17 results)
4. **Design V18 with sqrt(K) scaling** — this addresses the fundamental NaN root cause
5. **Combine: sqrt(K) + cosine LR + EASY_TH=0.50** = theoretically optimal configuration

### The V12 Paradox Resolution
V12's 82.04% came from **high-intensity fast learning** in the first 85 clean steps. The question is:
- **V17 strategy**: Same intensity, better routing → catch a slightly better peak
- **V18 strategy**: Fix the root cause → 178 fully clean steps × moderate intensity = more total useful training

V18 with sqrt(K) is the scientifically correct solution. V17/V17b are pragmatic bets on checkpoint timing.

