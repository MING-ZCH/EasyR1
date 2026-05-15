# sqrt(K) Advantage Scaling 深度分析报告
> 分析日期: 2026-03-27
> 目的: 评估将 BoK-GRPO advantage 从 `*K` 改为 `*sqrt(K)` 的必要性和影响

---

## 1. 问题背景

### 1.1 当前代码 (core_algos.py L624)
```python
advantages_1d[global_i] = (w[j].item() - 1.0 / K) * K  # K=16
```

### 1.2 提议修改
```python
advantages_1d[global_i] = (w[j].item() - 1.0 / K) * math.sqrt(K)  # sqrt(16)=4
```

### 1.3 为什么需要这个操作？

**表面原因**: 减小BoK路径的advantage幅度，降低梯度方差，推迟NaN onset。

**数学原理**: softmax权重 `w ∈ [0, 1]`，`w - 1/K ∈ [-1/K, 1-1/K]`。乘以K后：
- advantage ∈ [-1, K-1] = [-1, 15] (理论最大)
- 实际由于uniform mixing(0.1)和clipping(4.0)，范围为 [-0.9, 4.0]
- 乘以sqrt(K)后：advantage ∈ [-0.25, 3.75] (理论)，实际 [-0.22, 3.37]

**本质目的**: K scaling的设计意图是让 `E[advantage]=0`（因为 `E[w]=1/K`），同时给予"赢家"足够的正向梯度信号。但K=16的scale factor过大，导致梯度方差过高。

---

## 2. 量化分析

### 2.1 Advantage 幅度对比

| 场景 | τ | K-scale adv_win | √K-scale adv_win | 差异倍数 |
|------|---|-----------------|-------------------|----------|
| 1/16正确 | 0.7 | +4.000 (clipped) | +3.205 | 1.25x |
| 1/16正确 | 0.5 | +4.000 (clipped) | +3.357 | 1.19x |
| 1/16正确 | 0.3 | +4.000 (clipped) | +3.375 | 1.19x |
| 2/16正确 | 0.7 | +4.000 (clipped) | +1.401 | **2.85x** |
| 2/16正确 | 0.5 | +4.000 (clipped) | +1.540 | **2.60x** |
| 3/16正确 | 0.7 | +3.366 | +0.841 | **4.00x** |
| 4/16正确 | 0.7 | +2.306 | +0.576 | **4.00x** |
| 8/16正确 | 0.7 | +0.794 | +0.198 | **4.00x** |

**关键发现**:
- **1/16正确**: BOK_CLIP=4.0 已经在保护！K-scale的8.48被clip到4.0，√K-scale的3.37不需要clip → 差异仅1.2x
- **2-3/16正确**: 真正受益的场景，差异2.6-4.0x
- **4/16+正确**: 固定4.0x差异，但这些组通常走EasyDrGRPO路径而非BoK

### 2.2 与 EasyDrGRPO 的 advantage 比较

EasyDrGRPO (12/16正确): winner_adv = **0.559**, range = [-1.677, 0.559]

| BoK场景 | K-scale winner | vs DrGRPO | √K-scale winner | vs DrGRPO |
|---------|---------------|-----------|-----------------|-----------|
| 1/16@τ=0.5 | 4.000 | **7.2x** | 3.357 | **6.0x** |
| 2/16@τ=0.5 | 4.000 | **7.2x** | 1.540 | **2.8x** |
| 4/16@τ=0.7 | 2.306 | **4.1x** | 0.576 | **1.0x** |

**解读**: 
- K-scale让BoK winner获得DrGRPO winner的4-7倍梯度 → 过度集中
- √K-scale让BoK winner获得DrGRPO winner的1-6倍梯度 → 适度amplification
- 对于2/16场景，√K-scale将比例从7.2x降到2.8x，非常合理

### 2.3 batch-level 梯度方差分析

模拟batch: 64 samples (4 groups × 16), EASY_TH=0.50配置

| Config | adv range | adv std | max/mean ratio |
|--------|-----------|---------|----------------|
| K-scale | [-1.677, 4.000] | **1.073** | **5.3x** |
| √K-scale | [-1.677, 1.540] | **0.750** | **3.1x** |

Variance reduction: 30% (1.073 → 0.750)

---

## 3. NaN Root Cause 修正分析

### 3.1 关键发现：NaN不止来自BoK路径

Monte Carlo模拟揭示了一个**颠覆性结论**：

| Batch配置 | adv_std | NaN onset (est.) |
|-----------|---------|------------------|
| V12 (TH=0.75, 28%BoK) | 0.982 | **S85** (观测值) |
| Pure DrGRPO (无BoK) | **0.866** | **S111** |
| V17 (TH=0.50, 6%BoK, K) | 0.897 | S102 |
| V17 (TH=0.50, 6%BoK, √K) | 0.838 | S118 |

**颠覆性结论**: 即使完全没有BoK路径（纯DrGRPO），NaN也会在S111出现！
EasyDrGRPO的adv range [-1.68, 0.56] 本身就贡献了87%的batch方差。

### 3.2 各因素对N_critical的贡献

```
N_critical ∝ 1 / (LR × ppo_epochs × adv_std)²
```

**LR是最主要的杠杆**（二次方关系），advantage scaling是次要的：

| Config | adv_std | LR | N_critical | Status |
|--------|---------|-----|------------|--------|
| V12 (baseline) | 0.99 | 1.5e-6 | 85 | ❌ |
| V17-A (原V17, K-scale) | 0.90 | 1.5e-6 | 102 | ❌ |
| V17-B (√K + 1.5e-6) | 0.84 | 1.5e-6 | 118 | ❌ |
| **V17-C (K + 1e-6)** | 0.90 | **1e-6** | **231** | ⚠️ |
| **V17-D (√K + 1e-6)** | 0.84 | **1e-6** | **265** | ⚠️ |
| V17-E (K + 7.5e-7) | 0.90 | 7.5e-7 | 411 | ✅ |
| V17-F (√K + 7.5e-7) | 0.84 | 7.5e-7 | 472 | ✅ |

**杠杆效力对比**:
- √K scaling (固定LR=1.5e-6): S102 → S118 (+16步, **+15.7%**)
- LR 1.5e-6→1e-6 (固定K-scale): S102 → S231 (+129步, **+126%**)
- 两者组合: S102 → S265 (+163步, **+160%**)

---

## 4. 正面影响

### 4.1 梯度方差降低
- batch adv_std: 1.07 → 0.75 (−30%)
- max/mean ratio: 5.3x → 3.1x (−42%)
- BoK winner gradient dominance: 22.9% → 10.0% (−56%)
- 梯度更加均匀，优化轨迹更平滑

### 4.2 NaN安全边际提升
- 配合LR=1e-6: N_crit从231→265 (+34步, +15%)
- 不是决定性的，但提供有意义的额外安全边际

### 4.3 BoK amplification保留
- √K-scale让BoK winner获得DrGRPO winner的2-3x梯度（vs K-scale的7x）
- "Best-of-K"选择性仍然存在，只是更温和
- 2.7x的amplification在理论上更健康（多数RL论文推荐advantage在[-2, 2]范围内）

---

## 5. 负面影响 / 风险

### 5.1 BoK路径学习信号减弱
- BoK winner advantage: 4.0 → 1.5（对于2/16场景）
- 对于最困难的group（1/16正确），由于clip=4.0保护，减弱有限（仅1.2x）
- **但这些困难group本来就很少出现**（EASY_TH=0.50下仅6%的batch有BoK组）

### 5.2 可能需要配套的BOK_CLIP调整
- 当前BOK_CLIP=4.0是为K-scale设计的
- √K-scale下advantage最大值只有3.37，CLIP=4.0永远不会触发
- 建议：降低BOK_CLIP到2.0（让clip继续发挥安全网作用）

### 5.3 学习速度影响 — 微乎其微
- BoK组仅占6% batch (EASY_TH=0.50)
- 94%的学习来自EasyDrGRPO（不受影响）
- 总学习速度影响: 0.06 × 0.25 + 0.94 × 1.0 = 0.955 → **仅慢4.5%**

### 5.4 需要修改代码（增加代码复杂度）
- 改动1行代码，但需要新增环境变量控制
- 需要足够的实验验证

---

## 6. 对训练/模型性能的影响

### 6.1 总学习量对比

关键公式: `total_learning = clean_steps × LR × ppo_epochs`

| Config | Clean Steps | LR | Total Learning | vs V12 |
|--------|------------|-----|---------------|--------|
| V12 | 85 | 1.5e-6 | 2.55e-4 | 1.00x |
| V17-A (K, 1.5e-6) | 102 | 1.5e-6 | 3.06e-4 | 1.20x |
| V17-B (√K, 1.5e-6) | 118 | 1.5e-6 | 3.54e-4 | 1.39x |
| **V17-C (K, 1e-6)** | **~178** | **1e-6** | **3.56e-4** | **1.40x** |
| **V17-D (√K, 1e-6)** | **~178** | **1e-6** | **3.56e-4** | **1.40x** |

**关键发现**: V17-C 和 V17-D 的总学习量几乎相同（3.56e-4），都比V12多40%！
差异在于 V17-D 有更大的安全边际(N_crit=265 vs 231)。

### 6.2 预期对pixmo-test的影响

```
V12: 82.04% (85 clean steps, high intensity → fast but risky)
V17-C: 81-82.5% (预估, 178 clean steps, moderate intensity → reliable)
V17-D: 81-82.5% (预估, 178 clean steps, gentler gradients → most reliable)
```

V17-C和V17-D预期性能接近甚至可能超过V12，因为：
1. 40%更多总学习量（更多梯度更新）
2. 更平滑的优化轨迹（不受NaN步骤干扰）
3. 更密的checkpoint保存(save=15)可以捕捉peak

---

## 7. 结论与建议

### 7.1 sqrt(K)的价值判断

**结论: sqrt(K)值得做，但不是最重要的改动**。

| 操作 | NaN onset改善 | 学习速度影响 | 代码复杂度 |
|------|-------------|------------|-----------|
| LR 1.5e-6→1e-6 | +126% (最大) | -33%每步, +40%总量 | 0行 |
| EASY_TH 0.75→0.50 | +20% (中等) | 无影响 | 0行 |
| √K scaling | +15% (额外) | -4.5% | 1行+环境变量 |

### 7.2 推荐的V17配置

**最优方案**: V17-C (EASY_TH=0.50 + LR=1e-6 + K-scale保持不变)

理由:
- N_crit=231步 >> 178步总训练量 → 充足安全边际
- 无需修改advantage计算代码 → 零风险引入bug
- 总学习量3.56e-4 = V12的1.4x → 更充分的训练
- 如果V17-C仍然出现NaN(说明估算偏乐观)，再加√K作为V17-D

**备选方案**: V17-D (√K + LR=1e-6)
- 仅当V17-C出现NaN时才需要
- N_crit=265步，额外15%安全边际
- 需要1行代码改动

### 7.3 具体脚本修改

V17-C (推荐):
```bash
# v17.sh, line 177:
# 原: [[ -z "${ACTOR_LR}" ]] && ACTOR_LR="1.5e-6"
# 改: [[ -z "${ACTOR_LR}" ]] && ACTOR_LR="1e-6"
```

V17-D (备选, 如V17-C失败):
```bash
# v17.sh, line 177:
[[ -z "${ACTOR_LR}" ]] && ACTOR_LR="1e-6"
# 加: 
export BOK_ADV_SCALE=${BOK_ADV_SCALE:-"sqrt_k"}  # or "k" for original
```
```python
# core_algos.py L624:
_bok_adv_scale = os.environ.get("BOK_ADV_SCALE", "k")
scale_factor = math.sqrt(K) if _bok_adv_scale == "sqrt_k" else K
advantages_1d[global_i] = (w[j].item() - 1.0 / K) * scale_factor
```

---

## 8. 举例对比：现有配置 vs 推荐配置

### 典型训练batch (easy data, 4 groups × 16 samples)

**Group分布**:
- Group A: 16/16正确 → AllCorrect (adv=0, 过滤)
- Group B: 12/16正确 → EasyDrGRPO (pass_rate=0.75 > TH=0.50)
- Group C: 10/16正确 → EasyDrGRPO (pass_rate=0.625 > TH=0.50)  
- Group D: 2/16正确 → BoK路径 (pass_rate=0.125 < TH=0.50)

**现有V17脚本 (K-scale, LR=1.5e-6)**:

| Group | Path | adv_winner | adv_loser | 梯度贡献 |
|-------|------|-----------|-----------|---------|
| A | Filtered | 0 | 0 | 0 |
| B | DrGRPO | +0.559 | -1.677 | 中等 |
| C | DrGRPO | +0.791 | -1.186 | 中等 |
| D | BoK | **+4.000** | **-0.880** | **极大** |

→ Group D的winner **1个sample** 贡献了batch中 22.9% 的正向梯度!
→ adv_std = 1.07, N_crit ≈ 102步 → ❌ NaN

**推荐V17-C (K-scale, LR=1e-6)**:

| Group | Path | adv_winner | adv_loser | 梯度贡献 |
|-------|------|-----------|-----------|---------|
| A | Filtered | 0 | 0 | 0 |
| B | DrGRPO | +0.559 | -1.677 | 中等 |
| C | DrGRPO | +0.791 | -1.186 | 中等 |
| D | BoK | **+4.000** | **-0.880** | **极大**(同上) |

→ advantage完全相同! 但LR从1.5e-6降到1e-6 → 每步参数更新量减少33%
→ N_crit ≈ 231步 → ⚠️ 够用(178步训练)
→ 总学习量: 178 × 1e-6 × 2 = 3.56e-4 ≈ V12的 85 × 1.5e-6 × 2 = 2.55e-4 的 1.4倍

**备选V17-D (√K-scale, LR=1e-6)**:

| Group | Path | adv_winner | adv_loser | 梯度贡献 |
|-------|------|-----------|-----------|---------|
| A | Filtered | 0 | 0 | 0 |
| B | DrGRPO | +0.559 | -1.677 | 中等 |
| C | DrGRPO | +0.791 | -1.186 | 中等 |
| D | BoK | **+1.540** | **-0.220** | **适度** |

→ Group D的winner梯度从4.0降到1.5，但仍是DrGRPO winner的2.8x
→ adv_std = 0.75 (-30%), N_crit ≈ 265步 → ⚠️ 更安全
→ "Best-of-K"选择性保留(2.8x > 1x)，但不再dominates batch梯度

