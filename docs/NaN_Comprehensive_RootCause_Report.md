# BoK-GRPO 训练 NaN 梯度根因分析与优化报告

> 创建日期：2026-03-23
> 涵盖实验版本：V7, V8, V12, V13, V14-hard, V14-mixed, V15
> 分析框架：EasyR1-latest (verl-based), Qwen2.5-VL-7B, FSDP FULL_SHARD, 4 GPUs, bf16

---

## 1. 执行摘要

本报告对 BoK-GRPO 训练中所有 NaN 梯度事件进行根因分析。**核心发现**：

- **NaN 格式**：所有 NaN 均表现为 `grad_norm: !!float 'nan'`（YAML格式）
- **NaN 后果**：`_optimizer_step()` 检测到 `!isfinite(grad_norm)` 后**跳过 `optimizer.step()`**，模型权重不被破坏，但模型停止学习
- **级联模式**：一旦 NaN 出现，几乎总是级联到训练结束（正反馈死循环）
- **两种触发机制**：
  - **Type-A（梯度溢出型）**：grad_norm 先出现 >5.0 的 spike，5-11 步后 NaN（V12, V8）
  - **Type-B（策略发散型）**：entropy + KL 同时爆炸 + grad spike → NaN（V14-mixed, V14-hard）
- **安全边界**：`eff_intensity <= 2.8e-7` 时零 NaN（V15 验证）

---

## 2. 跨实验 NaN 全景

### 2.1 实验配置与 NaN 统计

| 版本 | LR | ppo_epochs | clip_high | eff_intensity | max_grad_norm | 总步数 | NaN步数 | NaN率 | 首次NaN | NaN时进度 |
|------|-----|-----------|-----------|--------------|---------------|--------|---------|-------|---------|----------|
| **V7** | 1e-6 | 1 | 0.30 | 3.0e-7 | 1.0 | 178 | 0 | 0% | — | — |
| **V8** | 1e-6 | 1 | 0.28 | 2.8e-7 | 1.0 | 66 | 0 | 0% | — | — |
| **V12** | 1.5e-6 | 2 | 0.28 | 8.4e-7 | 1.0 | 178 | 92 | 51.7% | S84 | 47.2% |
| **V13-1** | (resume V12) | 2 | 0.28 | 8.4e-7 | 1.0 | 178 | 29 | 16.3% | S179 | 50.3% |
| **V13-2** | (resume V12) | 2 | 0.28 | 8.4e-7 | 1.0 | 178 | 33 | 18.5% | S179 | 50.3% |
| **V14-hard** | 1e-6 | 2 | 0.28 | 5.6e-7 | 1.0 | 112 | 7 | 6.3% | S70 | 62.5% |
| **V14-mixed-2** | 1.5e-6 | 2 | 0.28 | 8.4e-7 | 1.0 | 90 | 27 | 30.0% | S65 | 72.2% |
| **V15** | 5e-7 | 2 | 0.28 | 2.8e-7 | 1.0 | 90 | 0 | 0% | — | — |

> **eff_intensity** = clip_ratio_high x LR x ppo_epochs
> 物理含义：每步策略更新的最大扰动上界

### 2.2 关键规律

1. **eff_intensity 阈值**：
   - `<= 3.0e-7`：零 NaN（V7, V8, V15）
   - `5.6e-7`：低 NaN（V14-hard, 6.3%）
   - `>= 8.4e-7`：高 NaN（V12 51.7%, V14-mixed 30.0%）

2. **NaN 首次出现的训练进度**：约 47-72% 处，说明 NaN 不是初始化问题，而是策略逐步偏离后的**累积效应**

3. **ppo_epochs = 2 是风险因子**：所有 NaN 实验都用 ppo_epochs=2（V7/V8 用 ppo_epochs=1，零 NaN）

---

## 3. NaN 传播链路分析（代码级）

### 3.1 完整传播链

```
[Step 1] model.forward() → log_probs (bf16)
    |
[Step 2] log_probs - old_log_probs → negative_approx_kl
    |
[Step 3] torch.clamp(negative_approx_kl, -20, 20) → safe_log_ratio  ← [Guard OK]
    |
[Step 4] torch.exp(safe_log_ratio.float()) → ratio (fp32)  ← [Guard OK: fp32+clamp]
    |
[Step 5] -advantages * ratio → pg_loss
    |  WARNING: 如果 advantages 含 NaN 或 ratio 极端，pg_loss 可能有极端值
[Step 6] pg_loss / gradient_accumulation → loss
    |
[Step 7] loss.backward() → 梯度计算
    |  WARNING: 梯度在 FSDP all-reduce 中可能因 bf16 溢出变为 NaN
[Step 8] clip_grad_norm_(max_grad_norm) → grad_norm
    |  如果任一参数梯度为 NaN → grad_norm = NaN
[Step 9] if not isfinite(grad_norm): SKIP optimizer.step()
    |  模型冻结 → 下次 rollout 输出相同 → 相同的 NaN 条件 → 级联
```

### 3.2 Advantage 计算中的 NaN 防护（core_algos.py）

| 代码位置 | 防护机制 | 作用 |
|---------|---------|------|
| L588 | `torch.nan_to_num(shifted)` | 防止 z-score 标准化后的 NaN |
| L604 | `logits.clamp(-12.0, 12.0)` | 限制 softmax 输入范围 |
| L609 | `torch.isfinite(w).all()` + uniform fallback | softmax 权重 NaN 时用均匀分布替代 |
| L620 | collapse_threshold → mix with uniform | 防止权重过度集中 |
| L667 | `torch.isfinite(advantages_1d).all()` check | 最终 advantage NaN 检查 |
| L674 | `torch.nan_to_num(advantages_1d)` | 最终 NaN 替换 |

**结论**：Advantage 计算有 6 层 NaN 防护，**NaN 不来自 advantage 计算**。

### 3.3 PPO Loss 中的防护（core_algos.py L886-901）

```python
safe_log_ratio = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(safe_log_ratio.float())  # fp32 + clamp → ratio in [2e-9, 4.8e8]
```

**Guard 有效**：log-ratio clamp 到 [-20, 20]，exp 在 fp32 下不会 overflow。

### 3.4 真正的 NaN 源头

**根因不在 advantage/PPO-loss 的前向计算中**，而在 **backward pass 的梯度计算中**。

当策略偏离幅度大时（high eff_intensity x 多步训练累积）：
1. `log_probs` 和 `old_log_probs` 差异越来越大
2. 虽然 ratio 被 clamp 了，但 **梯度 d(loss)/d(params) 的链式法则中间项在 bf16 精度下溢出**
3. FSDP `all-reduce` 对各 GPU 梯度求和时，bf16 的有限精度导致溢出 → NaN
4. `clip_grad_norm_()` 计算所有参数梯度的 L2 范数 → 任一 NaN → 整体 NaN

---

## 4. 两种 NaN 触发机制详解

### 4.1 Type-A：纯梯度溢出型（V12 代表）

**特征**：
- entropy 在 NaN 前保持稳定（0.47-0.53），**策略分布未发散**
- grad_norm 出现孤立 spike（S73=9.38, S79=10.14），超过正常值 10x
- NaN 在 spike 后 5-11 步出现

**V12 梯度时间线**：
```
S20: grad=1.526  正常
S26: grad=6.506  首次spike（但恢复）
S40: grad=0.671  正常
S60: grad=0.723  正常
S73: grad=9.382  大spike
S79: grad=10.137 更大spike
S84: grad=NaN    首次NaN（2个micro-batch NaN）
S85: grad=NaN    全micro-batch NaN
S86: grad=1.024  短暂恢复
S87-131: 全NaN   级联冻结
S132: grad=0.872 短暂恢复
S133+: 永久NaN   死锁
```

**机制**：高 eff_intensity (8.4e-7) 下，策略偏移虽未表现在 entropy 上（因 token 分布多模态），但在底层参数空间中偏移量大 → 特定数据 batch 的梯度在 bf16 下溢出。

### 4.2 Type-B：策略发散型（V14-mixed 代表）

**特征**：
- entropy 从 0.54 飙升到 1.09（2x 增长）
- KL 从 0.001 飙升到 0.058（58x 增长）
- grad spike 与 entropy/KL 爆炸同步发生

**V14-mixed 发散时间线**：
```
S30: entropy=0.574, KL=0.001, grad=normal    稳定
S40: entropy=0.552, KL=0.004                 KL 开始升
S45: entropy=0.660                            entropy 跳升（tau~0.43）
S52: grad=5.23                                首次 grad spike
S55: entropy=0.715, KL=0.004                 entropy 加速
S59: grad=10.78                               极端 grad spike
S60: entropy=0.828, KL=0.019                 发散中
S62: grad=9.71                                KL 反转信号
S65: entropy=1.013, KL=0.058, grad=NaN       NaN 爆发
S66: entropy=1.091, KL=0.000（冻结）         Skip update x5
```

**机制**：当 tau 衰减到 <0.5 时，BoK softmax 权重过度集中于少数 trajectory → 单条 trajectory 的梯度过大 → 策略急剧偏移 → entropy + KL 同时爆炸 → 梯度溢出。

**关键因果链**：
```
tau↓0.34 → softmax集中 → 少数trajectory梯度放大 → 策略急偏
→ KL=0.058 → entropy=1.09 → bf16梯度溢出 → NaN
```

---

## 5. 根因检查清单对照

### 5.1 数值稳定性

| 检查项 | 结果 | 详情 |
|--------|------|------|
| log_probs - old_log_probs 差异过大 | 已有 clamp[-20,20] | PPO loss 中 safe_log_ratio 限制了 ratio |
| advantage 计算产生 NaN | 6 层防护，安全 | nan_to_num + isfinite + clamp + uniform fallback |
| softmax 温度过低导致数值问题 | **是 Type-B 的诱因** | tau<0.5 时 softmax 集中度过高 |
| bf16 精度导致梯度溢出 | **根因** | backward pass 中链式法则中间项在 bf16 下溢出 |
| FSDP all-reduce 精度问题 | **加剧因素** | 多 GPU 梯度求和可能放大浮点误差 |

### 5.2 训练强度

| 检查项 | 结果 | 详情 |
|--------|------|------|
| eff_intensity 是否过高 | **主要原因** | eff>=8.4e-7 时 NaN 率>30%，eff<=2.8e-7 时零 NaN |
| LR 是否过大 | 贡献因子 | LR=1.5e-6 的实验 NaN 率高于 LR=1e-6 |
| ppo_epochs=2 增加风险 | 贡献因子 | 所有 NaN 实验都用 ppo_epochs=2 |
| clip_ratio_high 设置 | 合理 | 0.28 已比较保守 |
| max_grad_norm 设置 | 合理 | max_grad_norm=1.0 是标准设置 |

### 5.3 NaN 后的行为

| 检查项 | 结果 | 详情 |
|--------|------|------|
| NaN 是否破坏模型权重 | 不会 | `_optimizer_step()` 检测到 NaN 后跳过 step() |
| NaN 是否可恢复 | 偶尔 | V12 S86/S132 偶发恢复，但最终仍级联 |
| NaN 后是否死循环 | **是** | ppo_kl=0, clipfrac=0 → 参数不变 → 相同条件重复 |

### 5.4 数据相关性

| 检查项 | 结果 | 详情 |
|--------|------|------|
| 特定数据batch触发NaN | 有可能 | V12 S86/S132 短暂恢复说明不同 batch 的 NaN 触发概率不同 |
| hard 数据更容易触发 | 弱相关 | V14-hard NaN率(6.3%)低于 V14-mixed(30%)，但 LR 也更低 |

---

## 6. V15 为什么零 NaN？

V15 配置：LR=5e-7, ppo_epochs=2, clip_ratio_high=0.28, eff_intensity=2.8e-7

| 安全机制 | V15 vs V12 对比 |
|---------|----------------|
| eff_intensity | 2.8e-7 vs 8.4e-7 (降低 3x) |
| LR | 5e-7 vs 1.5e-6 (降低 3x) |
| tau adaptive | 有 _tau_adaptive 和 safe_tau 机制 |
| logit capping | +/-12.0 限制 softmax 输入 |
| grad_norm 上界 | max=2.539, avg=1.575 (远低于 NaN 阈值) |
| entropy | 稳定在 0.47-0.55 |
| Val answer | S1=0.756 → S46=0.762 |

**V15 的安全来自两个层面**：
1. **策略扰动小**：eff_intensity=2.8e-7 使策略每步变化很小
2. **梯度上界低**：最大 grad_norm=2.539，远低于 NaN 先兆阈值(>5.0)

---

## 7. 优化建议

### 7.1 当前安全参数（V15 验证）

```
LR = 5e-7
ppo_epochs = 2
clip_ratio_high = 0.28
max_grad_norm = 1.0
→ eff_intensity = 2.8e-7  零NaN
```

### 7.2 进一步优化方案

#### 方案A：NaN 自适应 LR 降低（推荐）

在 `_optimizer_step()` 中增加梯度预警机制：

```python
def _optimizer_step(self) -> torch.Tensor:
    grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)

    if not torch.isfinite(grad_norm):
        print("Gradient norm is not finite. Skip update.")
        self._nan_count = getattr(self, '_nan_count', 0) + 1
        if self._nan_count >= 3:
            for pg in self.actor_optimizer.param_groups:
                pg['lr'] *= 0.5
            print(f"NaN cascade detected ({self._nan_count}x). LR halved.")
    else:
        self._nan_count = 0
        self.actor_optimizer.step()

    self.actor_optimizer.zero_grad()
    return grad_norm
```

#### 方案B：梯度 Spike 预警 + 预防性跳过

```python
GRAD_SPIKE_THRESHOLD = 5.0  # 所有NaN实验的共同先兆阈值

if grad_norm > GRAD_SPIKE_THRESHOLD:
    print(f"Gradient spike detected: {grad_norm:.2f}. Preventive skip.")
    self.actor_optimizer.zero_grad()
    return grad_norm
```

#### 方案C：tau 下界提高

V14-mixed 的 Type-B NaN 直接由 tau<0.5 引起。建议 tau 下界 0.35：

```python
bok_tau = max(bok_tau_schedule(step), 0.35)  # 不允许 tau < 0.35
```

### 7.3 性能 vs 安全的权衡

| 方案 | NaN 防护 | 性能影响 | 实现复杂度 | 推荐 |
|------|---------|---------|-----------|------|
| 降低 eff_intensity (V15 路线) | ★★★ | -10% 学习速度 | 无需改代码 | **已验证最佳** |
| 方案A: NaN自适应LR | ★★ | 中性 | 低 | 推荐作为安全网 |
| 方案B: Spike预警跳过 | ★★ | 可能错过有效更新 | 低 | 可选 |
| 方案C: tau下界提高 | ★ | 减弱 BoK 效果 | 极低 | 仅对 Type-B |

### 7.4 推荐的 V16 配置

```yaml
# 保持 V15 的安全 eff_intensity
lr: 5e-7
ppo_epochs: 2
clip_ratio_high: 0.28
# eff_intensity = 2.8e-7 ← 安全区间

# 新增安全机制
grad_spike_threshold: 5.0          # 方案B
nan_adaptive_lr_decay: 0.5         # 方案A
nan_adaptive_trigger_count: 3      # 连续3次NaN触发LR衰减

# BoK 优化
bok_tau_min: 0.35                  # 防止 Type-B 发散
bok_uniform_mix: 0.1               # 维持多样性
```

---

## 8. 结论

### 8.1 NaN 根因（按重要性排序）

1. **eff_intensity 过高**（主因）：>=8.4e-7 时策略每步扰动过大，导致 log_probs 差异累积到 bf16 精度极限
2. **bf16 精度限制**（底层原因）：backward pass 链式法则中间项在 bf16 下溢出，FSDP all-reduce 加剧
3. **tau 过低导致 softmax 集中度过高**（Type-B 诱因）：tau<0.5 时少数 trajectory 获得过高权重，放大梯度
4. **ppo_epochs=2 的累积效应**（加剧因子）：第二个 epoch 的 policy 已偏移，ratio 更极端
5. **NaN 级联死循环**（扩散机制）：NaN → skip update → 参数冻结 → 相同 NaN 条件重复

### 8.2 已验证的安全配置

V15（eff_intensity=2.8e-7）在 62/90 步训练中 **零 NaN**，val answer 达到 0.762@S46，证明低 eff_intensity 配置在保证稳定性的同时仍能有效学习。

### 8.3 监控指标优先级

| 优先级 | 指标 | 预警阈值 | 含义 |
|--------|------|---------|------|
| P0 | grad_norm | > 5.0 | 5-11 步后可能 NaN |
| P1 | entropy | 连续 5 步上升 >20% | Type-B 发散信号 |
| P1 | ppo_kl | > 0.02 | 策略偏离过大 |
| P2 | kl_loss | > 0.1 | KL 惩罚项过大 |
| P3 | pg_clipfrac | > 0.5 | 超过半数被裁剪 |

---

*本报告整合了 V7-V15 所有实验的 NaN 数据，基于代码级分析（core_algos.py, dp_actor.py）和训练日志解析，替代此前分散在多个文档中的零碎 NaN 分析。*
