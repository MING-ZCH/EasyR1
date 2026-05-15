# BoK Negative Advantage Structural Analysis & Winner Boost Assessment

## Executive Summary

Two questions analyzed from first principles:
1. **Is BoK's -0.9 negative floor the REAL reason for easy_drgrpo?** → **YES ✓**
2. **Is Winner Boost truly necessary?** → **Weakly justified, optional**

---

## 1. BoK Negative Advantage: Mathematical Proof

### The Structural Limit

BoK advantage formula: `A_i = (w_i - 1/K) × K`

For a **wrong trajectory** in any group:
```
Step 1: z-normalization    → z_wrong << 0 (very negative)
Step 2: softmax logit      → z_wrong / τ → very negative  
Step 3: softmax weight     → ≈ 0 (near zero, decays exponentially)
Step 4: uniform mixing     → w_wrong = (1-mix) × 0 + mix × (1/K) = mix/K
Step 5: advantage          → (mix/K - 1/K) × K = mix - 1 = -(1-mix)
```

**Result: A_wrong = -(1 - MIX) = -(1 - 0.1) = -0.9**

This is a **STRUCTURAL HARD FLOOR**:
- Independent of τ (temperature)
- Independent of K (rollout count)  
- Independent of score distribution
- Controlled ONLY by `BOK_UNIFORM_MIX` parameter

### Empirical Verification

| n_correct | τ=0.7 | τ=0.5 | τ=0.3 | τ=0.1 | DrGRPO |
|-----------|-------|-------|-------|-------|--------|
| 1/16      | -0.87 | -0.90 | -0.90 | -0.90 | -0.60  |
| 8/16      | -0.82 | -0.88 | -0.90 | -0.90 | -1.14  |
| 14/16     | -0.89 | -0.90 | -0.90 | -0.90 | **-2.50** |
| 15/16     | -0.90 | -0.90 | -0.90 | -0.90 | **-2.50** |

Across ALL τ values and n_correct, BoK negative advantage converges to -0.9.

### Historical Evidence

This was first identified in V7 analysis:
> "BOK-GRPO的advantage范围MUCH窄于Standard GRPO。负方向仅-0.9(因为softmax计算)"
> — v7_training_analysis_report.md, L74

> "adv_min≈-0.9：负advantage较小，意味着对'坏样本'的惩罚信号弱于GRPO(-3.75)"
> — v7_training_analysis_report.md, L152

> "BOK advantage不对称：正向[0, +3.0]，负向[-1.0, 0]；正向强化是负向惩罚的3倍"
> — v7_bok_grpo_parameter_analysis.md, L71-72

---

## 2. Why -0.9 Floor Causes easy_drgrpo to be ESSENTIAL

### Negative Signal Comparison by Group Type

| Group (n correct) | Route | BoK A_wrong | DrGRPO A_wrong | Ratio | Better |
|--------------------|-------|-------------|----------------|-------|--------|
| n=1/16  (hard)     | bok   | **-0.90**   | -0.25          | 0.28x | BoK ★  |
| n=2/16  (hard)     | bok   | **-0.90**   | -0.37          | 0.42x | BoK ★  |
| n=4/16  (hard)     | bok   | **-0.90**   | -0.57          | 0.64x | BoK ★  |
| n=6/16  (hard)     | bok   | **-0.90**   | -0.77          | 0.86x | BoK ★  |
| **n=7-8** | **crossover** | -0.90 | -1.00 | 1.11x | — |
| n=10/16 (easy)     | drgrpo| -0.90       | **-1.29**      | 1.43x | DrGRPO ★ |
| n=12/16 (easy)     | drgrpo| -0.90       | **-1.73**      | 1.92x | DrGRPO ★ |
| n=14/16 (easy)     | drgrpo| -0.90       | **-2.50**      | 2.78x | DrGRPO ★ |

### Key Insight

**Crossover at n≈7/16**: 
- For **hard groups (n≤6)**: BoK gives STRONGER negative signal (-0.9 vs -0.25...-0.77)
- For **easy groups (n≥8)**: DrGRPO gives STRONGER negative signal (-1.0...-2.5 vs -0.9)

**The functional meaning of easy_drgrpo**:

> For easy groups (most trajectories correct), the rare WRONG trajectories are 
> "surprising errors" that the model should learn to avoid. BoK's weak -0.9 
> penalty provides insufficient "avoid" signal. DrGRPO's -2.5 penalty is 
> 2.78x harsher — effectively telling the model "this error on an easy 
> question is UNACCEPTABLE."

This is NOT about gradient budget. It's about **negative sample learning efficiency**.

### Total Negative Signal per Group

| n_correct | BoK |neg_sum| | DrGRPO |neg_sum| | DrGRPO/BoK |
|-----------|-----|---|---|------|---|---|------|
| 8/16      | 6.97 | 7.98 | 1.14x |
| 10/16     | 5.28 | 7.73 | 1.46x |
| 12/16     | 3.56 | 6.91 | 1.94x |
| 14/16     | 1.80 | 4.99 | **2.78x** |

---

## 3. Winner Boost Necessity Assessment

### Under Bimodal Distribution (V19-like)

| Config | Boost-eligible groups | Correct traj boosted |
|--------|----------------------|---------------------|
| Uniform p=0.79 | 0.0% (none!) | 0.00% |
| Bimodal 60/40 (V19) | **17.95% (~12/64)** | 5.19% |
| Bimodal 50/50 | 30.92% (~20/64) | 9.60% |

Under bimodal distribution, boost affects ~18% of groups and ~5% of correct trajectories.

### Gradient Impact

| Metric | Without Boost | With Boost (×3.0) | With Boost (×2.0) |
|--------|--------------|-------------------|-------------------|
| Hard correct adv | 4.00 | 6.93 | 5.66 |
| Per-token gradient | 1.12 | 1.94 (+73%) | 1.58 (+41%) |
| Batch gradient delta | baseline | +22% total | +13% total |

### Directional Correctness

Winner Boost IS directionally correct:
- Differentiates "gold" n=1/16 (rare correct) from "common" n=8/16 correct
- Aligned with pass@32→pass@1 gap bridging objective
- These rare correct trajectories ARE the signal for the 16% capability gap

### Risk Assessment

| Factor | Assessment |
|--------|-----------|
| Gradient spike risk | LOW — GradSpikeProtect v2 mitigates |
| Overfit risk | LOW — boosting correct paths, not noise |
| Code complexity | LOW — 8 lines, easy to disable (set BOOST=1.0) |
| Empirical evidence | NONE — never been tested in actual training |

### Verdict

**WEAKLY JUSTIFIED, KEEP AS OPTIONAL**
- Set `BOK_WINNER_BOOST=2.0` (conservative, +41% for affected tokens)
- If no improvement after 30 steps, disable by setting to 1.0
- NOT the most impactful intervention — easy_drgrpo threshold is far more important

---

## 4. Revised Understanding of BoK-GRPO Hybrid Design

### Why BoK for Hard Groups (n≤7)?

| Advantage Type | BoK | DrGRPO |
|---------------|-----|--------|
| **Positive (correct)** | **+4.0 (concentrated)** | +2.5 (z-norm max) |
| Negative (wrong) | -0.9 (adequate) | -0.5...-1.1 (weaker!) |

BoK excels at hard groups because:
1. **Positive**: Concentrated softmax gives +4.0 (vs DrGRPO +2.5) = 1.6x stronger "learn this path"
2. **Negative**: -0.9 is actually BETTER than DrGRPO's -0.5 for n=1-4 groups

### Why DrGRPO for Easy Groups (n≥8)?

| Advantage Type | BoK | DrGRPO |
|---------------|-----|--------|
| Positive (correct) | +0.14 (diluted) | +0.37 (2.6x better) |
| **Negative (wrong)** | -0.9 (weak) | **-2.5 (2.78x harsher)** |

DrGRPO excels at easy groups because:
1. **Negative**: -2.5 penalty is CRITICAL — wrong answers on easy questions need harsh punishment ★★★
2. **Positive**: 2.6x more positive signal for correct paths (bonus, not the key reason)

### The Complete Picture

```
BoK-GRPO Hybrid = 
  Hard groups → BoK (STRONG positive for rare correct, adequate negative)
  Easy groups → DrGRPO (STRONG negative for rare wrong, bonus positive)
  
  Each algorithm handles the RARE signal in its assigned difficulty tier.
```

---

## 5. Configuration Recommendations for V20

Based on this analysis:

```bash
# Core hybrid routing — most impactful parameter
BOK_EASY_THRESHOLD=0.75          # V12 recipe, proven best (15.45%)
# Raising to 0.75 ensures more groups get DrGRPO negative signal

# Winner Boost — optional, conservative
BOK_WINNER_BOOST=2.0             # Down from 3.0, safer
# Can disable (set to 1.0) if grad spikes occur

# REMOVE V3 over-engineering
BOK_EASY_SCALE=1.0               # Disable dampening (TH=0.75 is the fix)
BOK_QUALITY_BONUS=0.0            # Disable (minimal effect per analysis)
```

### Why TH=0.75 is Sufficient

With TH=0.75, groups with n≥12/16 go to easy_drgrpo:
- These groups get DrGRPO's -2.5 penalty for wrong trajectories
- n=8-11 groups stay in BoK (where BoK's -0.9 ≈ DrGRPO's -1.0)
- The crossover region (n=7-8) has minimal penalty difference either way

V12 (TH=0.75) achieved 15.45% StepCount-500 vs V17 (TH=0.50) at 12.20%.
The threshold affects the routing boundary between the two advantage regimes.

---

## 6. Appendix: Complete First-Principles Q&A (Session 2)

### Q0: TH=0.50 "最大化两者利用" 的真相

**你的记忆部分正确**：TH=0.50在双峰数据(easy+hard)下确实能同时利用BoK和DrGRPO：

| Config | BoK | Easy_DrGRPO | AllCorrect |
|--------|-----|-------------|------------|
| V12: easy-only, TH=0.75 | **14.6%** | 81.3% | 4.1% |
| V17: easy-only, TH=0.50 | 0.1% (!!!) | 95.8% | 4.2% |
| V19: easy+hard, TH=0.50 | **37.0%** | 36.6% | 26.4% |

**关键发现**：
- TH=0.50 + easy-only → BoK=0.1%，相当于**完全禁用BoK**（V17回退到12.20%）
- TH=0.50 + bimodal → BoK=37%，**合理分配**（类似V12的14.6% BoK效果）
- TH=0.75 + easy-only → BoK=14.6%，**V12配方，验证最佳（15.45%）**

**V17选择TH=0.50的原始动机不是"最大化两者"，而是"减少BoK梯度方差防NaN"**（见V17_Optimization_Direction.md L70）。但这在easy-only数据上导致BoK完全失效。

### Q1: Easy_DrGRPO vs BoK 对正确sample的正向奖励

| n_correct | BoK A_correct | DrGRPO A_correct | 谁更强？ |
|-----------|---------------|------------------|----------|
| 2/16 (hard) | **+4.000** | +2.500 | BoK 1.6× |
| 4/16 (hard) | **+2.611** | +1.724 | BoK 1.5× |
| 8/16 (交叉) | +0.872 | +0.997 | ≈平 |
| 12/16 (easy) | +0.297 | **+0.576** | DrGRPO 1.9× |
| 14/16 (easy) | +0.128 | **+0.377** | DrGRPO 2.9× |

**结论**：
- Hard组：BoK正向信号比DrGRPO强1.5-1.6倍（集中信用分配到稀有正确路径）
- Easy组：DrGRPO正向信号比BoK强1.9-4.3倍，但绝对值都很小（<0.6）
- **Easy组的关键优势不是正向信号，而是负向信号（-2.5 vs -0.9 = 2.78x）**

### Q2: BoK公式逐项解释

**公式**: `Aᵢ = (wᵢ - 1/K) × K`，其中 `wᵢ = (1-mix)·softmax(zᵢ/τ) + mix·(1/K)`

#### `× K` 的作用
- 没有×K: advantage范围 [-0.0625, 0.9375]，均值绝对值≈0.06 → **梯度太小无法训练**
- 有×K: advantage范围 [-1.0, 15.0]，均值绝对值≈1.0 → 与DrGRPO [-2.5, 2.5] 量级可比
- **本质是缩放因子**，使BoK和DrGRPO的梯度量级匹配，确保混合训练时信号平衡

#### `BOK_UNIFORM_MIX=0.1` 的作用
- mix=0 (纯softmax): 低τ时最差trajectory权重→0，最好→1.0 → "赢者通吃"
  - 负面advantage理论最大: -(0-1/K)×K = -1.0
  - 正面advantage理论最大: (1-1/K)×K = 15.0 → 极端浓缩
- mix=0.1: 最差trajectory至少有 w_min=MIX/K=0.00625
  - 负面advantage上界: (0.00625-0.0625)×16 = -0.9 (这就是-0.9结构性地板的来源！)
  - 效果1: 防止梯度爆炸（w→0时∇log_π可能巨大）
  - 效果2: 保证所有trajectory都有梯度（类似ε-greedy的exploration机制）
  - 效果3: 与BOK_CLIP=4.0双重保护正向extreme

| MIX | A_max(n=2) | A_min(n=2) | A_max(n=14) | A_min(n=14) |
|-----|-----------|-----------|------------|------------|
| 0.00 | +4.000 | -0.991 | +0.451 | -0.998 |
| 0.10 | +4.000 | -0.892 | +0.420 | -0.898 |
| 0.20 | +4.000 | -0.793 | +0.368 | -0.799 |

### Q3: V3参数最终判定

| Parameter | 值 | 必要性 | 理由 |
|-----------|---|--------|------|
| BOK_WINNER_BOOST | 1.0(禁用) | ❌可选 | 影响5%的正确trajectory；方向正确但量级小；从未实验验证 |
| BOK_EASY_SCALE | 1.0(禁用) | ❌不必要 | 治标不治本；TH=0.75或双峰数据已自然平衡 |
| BOK_QUALITY_BONUS | 0.0(禁用) | ❌不必要 | 与softmax原生排序冗余；BoK质量spread(0.67)已优于DrGRPO(0.32) |
| Asymmetric Clip | 保留 | ✅仅当BOOST>1 | 如BOOST=1.0则此修改无效果 |
