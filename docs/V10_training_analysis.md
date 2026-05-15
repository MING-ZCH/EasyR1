# V10 训练深度分析报告

> 日志: `training_interleaved_traj_v10_..._20260313_062317.log`
> 时间: 2026-03-13 06:23 ~ 17:15 (约11小时)
> 规模: 18MB, 59571行, 45个训练步

---

## 1. V10 配置

| 参数 | V10 值 | V11 改进值 | 变化原因 |
|------|--------|-----------|---------|
| ppo_epochs | **1** | 2 | P0: ppo_epochs=1导致ratio≡1，PPO完全失效 |
| max_grad_norm | 1.0 | 0.5 | 减小梯度爆炸风险 |
| kl_coef | 0.02 | 0.02 | 不变 |
| lr | 1.5e-6 | 1.5e-6 | 不变 |
| ANSWER_WEIGHT | 0.7 | 0.6 | 修复point退化 |
| POINT_WEIGHT | 0.2 | 0.3 | 增强point学习信号 |
| FORMAT_WEIGHT | 0.1 | 0.1 | 不变 |
| BOK_EASY_THRESHOLD | 0.6 | 0.75 | 0.6在batch_mean=0.84时99.2%走DrGRPO |
| BOK_FILTER_ALL_CORRECT | 0 | 1 | P2: 避免全正确组浪费梯度 |
| BOK_DAPO_FILTER | 1 | 0 | P2替代dapo_filter |
| TRAJ_FORMAT_REJECT | 1 | 0 | 文件名fmtrej0 |

---

## 2. V10 完整训练指标

### 2.1 RewardHealth 轨迹 (45步)

| 阶段 | answer_mean | point_mean | grad_norm |
|------|------------|------------|-----------|
| Phase 1 (0-10) | 0.7938 | 0.8659 | 0.6985 |
| Phase 2 (11-20) | 0.7618 | 0.8580 | 1.1134 |
| Phase 3 (21-30) | 0.8137 | 0.8685 | 0.7497 |
| Phase 4 (31-44) | 0.8042 | 0.8763 | 0.8446 |

**整体趋势:**
- answer_mean: 首5步avg=0.7849 → 末5步avg=0.7915, **Δ=+0.0067 (STAGNANT, +0.67%)**
- point_mean: 首5步avg=0.8724 → 末5步avg=0.8860, **Δ=+0.0136 (STABLE)**
- answer极值: min=0.6471 (Step 20) max=0.8582 (Step 25)

### 2.2 BoK-GRPO 统计 (每10步采样)

| BoK Step | batch_mean | tau | low_var | n_zero_std | dapo_filtered | adv_range |
|----------|-----------|-----|---------|------------|---------------|-----------|
| 1 | 0.8713 | 0.700 | 80/1024 | 5/64 | 80/1024 | [-0.897, 3.000] |
| 11 | 0.8032 | 0.699 | 64/1024 | 4/64 | 64/1024 | [-0.897, 3.000] |
| 21 | 0.8372 | 0.697 | 192/1024 | 11/64 | 192/1024 | [-0.897, 3.000] |
| 31 | 0.8775 | 0.693 | 192/1024 | 11/64 | 192/1024 | [-0.897, 3.000] |
| 41 | 0.8132 | 0.687 | 64/1024 | 4/64 | 64/1024 | [-0.897, 3.000] |

**BoK分析:**
- batch_mean在0.80-0.88间波动，无明显上升趋势
- low_var (dapo_filtered) 在 64~192/1024 (6.3%~18.8%) 间波动
- n_zero_std在4~11/64 (6.3%~17.2%) — 较多全正确组被dapo过滤
- adv_range固定在 [-0.897, 3.000] — 被BOK_CLIP=3.0截断

### 2.3 PPO 健康指标

| 指标 | V10值 | 诊断 |
|------|-------|------|
| pg_clipfrac_higher | max=0.000094, non_zero=9/44 (20.5%) | **PPO几乎完全失效** |
| pg_clipfrac_lower | max=0.000007 | 同上 |
| ppo_kl | avg=0.000001, max=0.000067 | **KL≈0 — 策略几乎未偏移** |
| grad_norm | avg=0.8476, max=4.656, spikes(>2.0)=2/44 | 健康但有偶发尖峰 |
| pg_loss | 范围 [-0.01, 0.045] | 波动较大 |
| entropy | 范围 [0.489, 0.528] | 稳定 |

---

## 3. V10 核心问题诊断

### 🔴 问题1: PPO完全死亡 (ppo_epochs=1)
**根因**: `ppo_epochs=1` 导致 `old_log_probs = new_log_probs` → `ratio ≡ 1.0` → PPO clipping永远不触发。

**证据**:
- clipfrac_higher max = 0.000094 (仅千分之一的 token 发生 clipping)
- ppo_kl avg = 0.000001 (和 0 无区别)
- 44步中仅9步有任何非零 clipfrac

**影响**: 失去PPO的信赖域约束，训练完全退化为 vanilla policy gradient，无法有效控制探索范围。

### 🟡 问题2: answer_mean停滞不前
**数据**: 45步训练仅提升 +0.67% (0.7849 → 0.7915)

**原因分析**:
1. PPO死亡 → 无法有效从good trajectory学习
2. BOK_EASY_THRESHOLD=0.6 太低 → 99.2%的组走DrGRPO → BOK的best-of-K优化几乎被禁用
3. 没有P2全正确过滤 → all-correct组浪费梯度 (这些组的advantage≈0但仍计算)

### 🟡 问题3: 偶发gradient spikes
- Step 15: grad_norm = 4.656 (正常值的5.5倍)
- Step 33: grad_norm = 2.723 (正常值的3.2倍)
- max_grad_norm=1.0 应该能 clip 住（但说明某些批次 advantage 方差过大）

---

## 4. V10→V11 改进效果评估 (前10步对比)

### 4.1 指标对比

| 指标 | V10 (前10步) | V11 (前10步) | Delta | 评价 |
|------|------------|------------|-------|------|
| answer_mean | avg=0.7921 | avg=0.7938 | +0.0018 | 起步相当 |
| point_mean | avg=0.8646 | avg=0.8634 | -0.0012 | 起步相当 |
| grad_norm | avg=0.7091 | avg=1.0040 | **+0.2949** | V11梯度更强 |
| clipfrac_higher | 2/10非零, max=1.7e-5 | **8/9非零, max=0.001** | **×59倍** | ✅ PPO复活 |
| ppo_kl | avg≈0 | avg=-1.1e-5 | 微小 | 安全 |

### 4.2 P0 (ppo_epochs=2) 效果 — ✅ 达到预期
- **V10**: clipfrac_higher max = 0.000094 → **PPO 完全死亡**
- **V11**: clipfrac_higher max = 0.001000, 88.9%步有非零clipping → **PPO 复活！**
- V11的clipping频率是V10的 **59倍**，ratio不再≡1，PPO信赖域约束开始工作

### 4.3 P1 (Conditional-Advantage Routing) + BOK_EASY_THRESHOLD=0.75 效果 — ✅ 达到预期
- V11日志显示: `easy_drgrpo=416/1024` → 40.6% 走 easy DrGRPO path
- V11日志显示: `all_correct_filtered=288/1024` → 28.1% 被 P2 过滤
- 对比 V10: `dapo_filtered=80/1024` (7.8%) → V11的路由更智能

**P1路由分析 (V11 Step 1 batch=1024)**:
- 288 tokens → P2全正确过滤 (28.1%) — 零梯度
- 416 tokens → P1 DrGRPO路由 (40.6%) — 得到DrGRPO advantage + adv_clip=2.5
- 320 tokens → BOK softmax路由 (31.3%) — hard BOK advantage
- ^^ 比V10的"99.2% DrGRPO + 0.8% BOK"合理得多

### 4.4 梯度信号强度 — ✅ 显著增强
- V11 grad_norm avg=1.004 vs V10 avg=0.709 → **+41.6% 更强梯度**
- V11 max_grad_norm=0.5 的 clip 应该能在后期避免 V10 的 4.656 级别 spike

### 4.5 仍需观察的指标
- V11 才运行了 10 步 (V10 运行了 45 步)
- 需要更多步数观察 answer_mean 的**上升斜率**是否优于 V10
- 需要等 V11 到 30+ 步后看 point_mean 是否保持稳定

---

## 5. 核心结论

### V10 诊断总结
| 编号 | 问题 | 严重度 | V11修复 |
|------|------|--------|---------|
| P0 | ppo_epochs=1 → PPO死亡 | 🔴 致命 | ✅ ppo_epochs=2 |
| P1 | threshold=0.6 → BOK被禁用 | 🟡 严重 | ✅ threshold=0.75 + Conditional Routing |
| P2 | 无全正确过滤 → 梯度浪费 | 🟡 中等 | ✅ BOK_FILTER_ALL_CORRECT=1 |
| R | point_weight=0.2 不足 | 🟡 中等 | ✅ point_weight=0.3 |
| G | max_grad_norm=1.0 过大 | 🟢 轻微 | ✅ 0.5 |

### V11 前期验证结论
1. **PPO 复活**: clipfrac 从 ~0 提升到 0.001 (59倍)，ratio 不再恒等于1 ✅
2. **路由生效**: P1 (40.6%) + P2 (28.1%) + BOK (31.3%) 三路分流正常 ✅
3. **梯度增强**: 平均 grad_norm +41.6%，说明学习信号更强 ✅
4. **安全稳定**: ppo_kl ≈ 0，entropy 稳定 ~0.51，无异常 ✅
5. **需持续观察**: answer_mean 上升趋势需到 30+ 步后才能判断最终效果

---

## 6. V10→V11 修改清单

| 文件 | 修改内容 |
|------|---------|
| `verl/trainer/core_algos.py` | P1: Conditional-Advantage routing, P2: all-correct filter, threshold=0.75, adv_clip=2.5 |
| `verl/workers/actor/dp_actor.py` | 无修改（ppo_epochs由脚本传入） |
| `examples/...v11.sh` | P0: ppo_epochs=2, max_grad_norm=0.5, reward weights, watchdog集成 |
| `tools/monitor_v11_watchdog.py` | 新建: 530行自动监控+早停系统 |
| `docs/V11_modifications.md` | 完整V11改进文档 |
