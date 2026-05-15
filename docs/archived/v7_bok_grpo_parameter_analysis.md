# V7 BOK-GRPO 参数深入分析：DAPO机制、KL系数、训练优化

## 一、DAPO Auto-disable Safety 机制分析

### 1.1 机制描述
```python
# 当 DAPO_FILTER=1 且 low_var_rate > threshold(50%) 时：
# 自动将被DAPO过滤(零梯度)的group重新用fallback(drgrpo)计算advantage
```

### 1.2 潜在问题：硬阈值跳变
- low_var_rate=49.9% → 所有low-var组 **零梯度**
- low_var_rate=50.1% → 所有low-var组 **drgrpo梯度**
- 理论上可能导致振荡，但实际数据显示一旦越过50%几乎不可恢复

### 1.3 三种处理模式对比

| 模式 | DAPO_FILTER | 行为 | 推荐 |
|------|-------------|------|------|
| A: 始终fallback | 0 | 低方差组用drgrpo baseline | ✅ **最优** |
| B: DAPO过滤 | 1 | 低方差组零梯度 | ❌ 已证明死螺旋 |
| C: DAPO+Safety | 1+auto-disable | 正常过滤，危机降级 | ⚠️ 仅作保险 |

### 1.4 结论
- **BOK_DAPO_FILTER=0 是最优选择**
- Dr.GRPO论文论证：零方差组在全局baseline下仍有信息
- Safety机制保留作为最后防线，正常训练不应触发
- 阈值0.5无需调整（因为DAPO_FILTER应关闭）

---

## 二、KL系数分析：0.01 vs 0.03 vs 0.05

### 2.1 KL约束力量化对比

loss = pg_loss + kl_loss × kl_coef，KL贡献占pg_loss比例：

| BOK Step | kl_loss | pg_loss | kl%@0.01 | kl%@0.03 | kl%@0.05 |
|----------|---------|---------|----------|----------|----------|
| 1        | 0.006   | 0.013   | 0.5%     | 1.4%     | 2.3%     |
| 20       | 0.039   | 0.020   | 1.9%     | 5.9%     | 9.8%     |
| **50**   | **0.122**| 0.020  | **6.1%** | **18.3%**| **30.5%**|
| **65**   | **0.448**| 0.007  | 64%      | 192%     | 320%     |

### 2.2 关键洞察

**Step 50是转折点**：
- kl_coef=0.01：KL仅占pg_loss的6.1% → 无约束 → 继续漂移
- kl_coef=0.03：KL占18.3% → 有约束力，可能阻止漂移
- kl_coef=0.05：KL占30.5% → 强约束，显著减缓策略偏移

### 2.3 为什么推荐0.05而非0.03

1. **BOK softmax已经提供隐式学习率放大**：top sample advantage=3.0 vs GRPO的~1-2，BOK梯度强度是GRPO的1.5-3x
2. **kl_coef支付的不是"学习速度降低"的代价**：0.05通过约束策略偏移来保护format，同时BOK softmax仍提供加速
3. **Standard GRPO的kl_coef=0.05在167步全程稳定**，kl_loss始终<0.045
4. **Steps 1-50数据证明**：BOK的优势不来自弱KL，而来自集中梯度（kl_coef=0.01时BOK≈GRPO）

### 2.4 实验策略
| 优先级 | kl_coef | 目标 |
|--------|---------|------|
| **第一轮** | **0.05** | 验证稳定性 |
| 第二轮 | 0.03 | 如第一轮稳定，寻求加速 |
| ❌ 禁止 | 0.01 | 已证明不稳定 |

---

## 三、其他参数分析

### 3.1 BOK_CLIP (当前3.0)
BOK advantage不对称：正向[0, +3.0]，负向[-1.0, 0]
- 正向强化是负向惩罚的3倍
- 对format fail这种需要惩罚的行为，信号较弱
- **建议**：保持3.0，如kl_coef=0.05仍不稳定可降至2.0

### 3.2 BOK_UNIFORM_MIX (0.1)
- 10% uniform + 90% softmax，防止过度集中
- **建议**：保持0.1

### 3.3 TAU Schedule (0.8→0.5, cosine)
| Step | tau    | 含义 |
|------|--------|------|
| 0    | 0.800  | 接近uniform |
| 50   | 0.745  | 温和集中 |
| 100  | 0.621  | 中等集中 |
| 178  | 0.500  | 最终集中度 |
- **建议**：0.5合理。如entropy仍升高过快可考虑0.6

### 3.4 temperature (1.0)
- 提高到1.05-1.1可增加rollout多样性→减少零方差组
- **建议**：可选升到1.05，非必须

### 3.5 无需修改的参数
- ppo_epochs=1：BOK已有集中梯度，不需要多次更新
- lr_warmup_ratio=0.05：9步warmup足够
- max_grad_norm=1.0：防止gradient explosion
- BOK_ADV_NORMALIZE=0：开启会抵消softmax集中效果
- BOK_FALLBACK_MODE=drgrpo：Dr.GRPO论文核心方法

---

## 四、推荐配置

```bash
# 关键修复
algorithm.kl_coef=5e-2          # 匹配GRPO正则化强度
BOK_DAPO_FILTER=0               # 关闭DAPO过滤

# 已优化
BOK_TAU_FINAL=0.5               # 防止过度集中
BOK_DAPO_AUTO_DISABLE_THRESHOLD=0.5  # 保险机制

# 保持
BOK_TAU_INIT=0.8
BOK_CLIP=3.0
BOK_UNIFORM_MIX=0.1
BOK_FALLBACK_MODE=drgrpo
BOK_ADV_NORMALIZE=0
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_DUAL=3.0
```

### 预期指标范围
- kl_loss: [0.003, 0.050]
- entropy: [0.48, 0.55]
- format_fail: <0.03
- 如果稳定运行50+步且reward≈GRPO，第二轮可尝试kl_coef=0.03

---

*分析时间：2025-03-09*
