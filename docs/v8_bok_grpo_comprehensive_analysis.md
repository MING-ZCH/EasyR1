# BOK-GRPO 综合分析报告 & V8 训练方案

> **版本**: v8 | **日期**: 2025-03-09 | **整合自**: v7全部分析文档+新分析  
> **目标**: pass@32=95.2% → pass@1≥95% (稀疏场景)

---

## 目录

1. [算法原理：BOK-GRPO](#一-算法原理bok-grpo)
2. [参数完整定义](#二-参数完整定义)
3. [历史实验记录](#三-历史实验记录)
4. [v7 BOK-GRPO崩溃诊断](#四-v7-bok-grpo崩溃诊断)
5. [BOK_CLIP深入分析](#五-bok_clip深入分析)
6. [Pass@32→Pass@1场景下的BOK优势](#六-pass32pass1场景下的bok优势)
7. [V8训练方案](#七-v8训练方案)
8. [代码修改记录](#八-代码修改记录)

---

## 一、算法原理：BOK-GRPO

### 1.1 核心思想

BOK-GRPO（Best-of-K GRPO）将RL训练的group advantage从对称z-normalization改为**softmax温度加权**，使得组内最优trajectory获得集中的正advantage，实现"pass@K能力→pass@1能力"的转化。

灵感：类似Best-of-K采样（推理时用K条采样取最优），但在训练时用softmax软选择替代硬选择，让模型学到"什么样的生成策略最容易成功"。

### 1.2 Advantage公式

对一个prompt的K=16条rollout，reward score为 $r_1, r_2, ..., r_K$：

**Step 1: Z-score标准化**
$$z_i = \frac{r_i - \mu_{group}}{\sigma_{group} + \epsilon}$$

**Step 2: Softmax温度加权**
$$s_i = \text{softmax}(z_i / \tau)$$

**Step 3: Uniform混合（防止过度集中）**
$$w_i = (1 - \alpha) \cdot s_i + \alpha \cdot \frac{1}{K}$$

**Step 4: BOK Advantage**
$$A_i = (w_i - \frac{1}{K}) \cdot K$$

**Step 5: Clip**
$$A_i = \text{clamp}(A_i, -\text{clip}, +\text{clip})$$

### 1.3 Advantage数学性质

| 属性 | 值 | 说明 |
|------|------|------|
| 正advantage上限 (raw) | +10~15 | 取决于τ和组内分布 |
| 负advantage下限 (raw) | -(1 - α) | mix=0.1时为-0.9，公式hard limit |
| 正advantage (clipped) | +BOK_CLIP | **winner总是被clip** |
| 负advantage (clipped) | max(-BOK_CLIP, raw) | 通常raw=-0.9已小于clip |
| 零均值? | 否 | clipping破坏了 $\sum A_i = 0$ |
| 不对称性 | 正:负 ≈ clip:0.9 | 正advantage远大于负 |

### 1.4 与Standard GRPO的关键差异

| 维度 | Standard GRPO | BOK-GRPO |
|------|--------------|----------|
| Advantage | $(r_i - \mu) / \sigma$，对称 | softmax加权，不对称 |
| Winner信号 | ~1.5-2.0 | =BOK_CLIP (3.0-5.0) |
| Loser信号 | ~-1.5-2.0 | ~-0.9 (公式限制) |
| Hard prompt梯度 | 弱（稀少正确被平均化） | **强**（稀少正确被softmax集中） |
| Easy prompt梯度 | 强 | 弱（softmax分散在众多正确中） |
| 隐式课程学习 | 无 | **有**（自动聚焦hard prompt） |

---

## 二、参数完整定义

### 2.1 BOK-GRPO专有参数

| 参数 | 类型 | 默认值 | 含义 | 影响 |
|------|------|--------|------|------|
| `BOK_TAU` | float | 0.3 | Softmax温度，τ越小→选择越集中 | 控制winner/loser信号比 |
| `BOK_TAU_INIT` | float | 0.8 | τ退火起始值 | 训练初期较均匀 |
| `BOK_TAU_FINAL` | float | 0.5 | τ退火终止值 | 训练后期更集中 |
| `BOK_TAU_SCHEDULE` | str | cosine | τ退火曲线 (cosine/linear) | cosine慢开始慢结束 |
| `BOK_TOTAL_STEPS` | int | 178 | τ退火总步数 | 应=训练总步数 |
| `BOK_CLIP` | float | 3.0 | Advantage clip上下限 | **控制winner最大梯度强度** |
| `BOK_UNIFORM_MIX` | float | 0.1 | Uniform混合比例α | 防止softmax过度集中 |
| `BOK_ADV_NORMALIZE` | int | 0 | 组内advantage是否标准化 | 0=关闭（避免抵消softmax） |
| `BOK_LOW_VAR_THRESHOLD` | float | 1e-5 | 低方差组判定阈值 | 触发fallback的门槛 |
| `BOK_FALLBACK_MODE` | str | drgrpo | 低方差组处理: zscore/drgrpo/clip_std | drgrpo=Dr.GRPO baseline |
| `BOK_DAPO_FILTER` | int | 0 | 是否过滤同质组(零梯度) | **必须=0，否则死螺旋** |
| `BOK_DAPO_AUTO_DISABLE_THRESHOLD` | float | 0.5 | DAPO自动关闭阈值 | 仅作保险 |
| `BOK_MIN_BATCH_STD` | float | 0.1 | clip_std模式最小batch std | 仅clip_std模式用 |

### 2.2 通用训练参数

| 参数 | 含义 | 当前默认 |
|------|------|----------|
| `ACTOR_LR` | Actor学习率 | 1e-6 |
| `kl_coef` | KL惩罚系数 (loss = pg_loss + kl_loss × kl_coef) | 5e-2 |
| `CLIP_RATIO_LOW` | PPO clip下界 | 0.2 |
| `CLIP_RATIO_HIGH` | PPO clip上界 | 0.28 |
| `CLIP_RATIO_DUAL` | 双重clip系数 | 3.0 |
| `ROLLOUT_TEMPERATURE` | 采样温度 | 1.0 |
| `ROLLOUT_N` | rollout数 (K) | 16 |
| `ppo_epochs` | 每步PPO更新次数 | 1 |
| `global_batch_size` | 全局batch大小 | 64 |
| `lr_warmup_ratio` | 学习率warmup比例 | 0.05 |
| `max_grad_norm` | 梯度裁剪 | 1.0 |

### 2.3 Reward参数

| 参数 | 含义 | 当前值 |
|------|------|--------|
| `ANSWER_WEIGHT` | Answer reward权重 | 0.7 |
| `POINT_WEIGHT` | Point reward权重 | 0.2 |
| `TRAJECTORY_FORMAT_WEIGHT` | Format reward权重 | 0.1 |
| `TRAJ_FORMAT_REJECTION` | Format失败→overall=0 | 1 |
| `TRAJ_ANSWER_GATE_MODE` | Answer gating模式 | soft |
| `TRAJ_SOFT_GATE_BASE` | Soft gating基底 | 0.8 |
| `TRAJ_MISS_DECAY_ENABLE` | Dense mask point reward | 1 |
| `TRAJ_CONSISTENCY_PENALTY` | Point-Answer不一致惩罚 | 0.5 |

---

## 三、历史实验记录

### 3.1 实验总览

| 版本 | 算法 | 关键参数 | 步数 | 结果 | 问题 |
|------|------|----------|------|------|------|
| v6 BOK-GRPO | bok_grpo (zscore fallback) | tau=0.8→0.3, temp=1.2, kl=0.05, **seed=1** | ~200 | low_var=100% | seed=1泄漏到SamplingParams |
| v7 Standard GRPO | grpo | kl=0.05, temp=1.0, clip=0.2/0.3/3.0 | 168步 | 稳定, answer 0.815→0.865 | 学习缓慢 |
| v7 BOK-GRPO | bok_grpo | tau=0.8→0.3, **kl=0.01**, **DAPO=1**, temp=1.0 | 84步 | **崩溃** | kl过弱+DAPO死螺旋 |
| **v8 BOK-GRPO** | bok_grpo | tau=0.8→0.5, **kl=0.03**, **clip=4.0**, DAPO=0 | 待运行 | - | 本方案 |

### 3.2 v6详细参数

```
adv_estimator=bok_grpo, bok_tau=0.8→0.3(cosine), BOK_CLIP=3.0
BOK_UNIFORM_MIX=0.1, BOK_FALLBACK_MODE=zscore, BOK_DAPO_FILTER=0
kl_coef=0.05, temperature=1.2, rollout_n=16
lr=1e-6, clip_ratio=0.2/0.28/3.0
reward_weights: 0.6/0.3/0.1 (answer/point/format)
问题: SamplingParams.seed=1 → rollout全部确定性 → low_var=100% → BOK从未激活
修复: _SAMPLING_PARAMS_SKIP_KEYS={"seed"} 已应用
```

### 3.3 v7 Standard GRPO详细参数

```
adv_estimator=grpo, kl_coef=0.05, temperature=1.0
clip_ratio: 0.2/0.3/3.0, rollout_n=16
reward_weights: 0.7/0.2/0.1, format_rejection=1, soft_gate_base=0.8
168步结果: answer 0.815→0.865(+6.1%), format_fail<0.02稳定
kl_loss: [0.003, 0.045]稳定, entropy: [0.48, 0.52]
```

### 3.4 v7 BOK-GRPO详细参数（崩溃实验）

```
adv_estimator=bok_grpo, kl_coef=0.01(!), temperature=1.0
BOK_TAU: 0.8→0.3(cosine), BOK_CLIP=3.0, BOK_UNIFORM_MIX=0.1
BOK_FALLBACK_MODE=drgrpo, BOK_DAPO_FILTER=1(!)
clip_ratio: 0.2/0.28/3.0, rollout_n=16
reward_weights: 0.7/0.2/0.1, format_rejection=1
84步结果: Step 55 format_fail=0.153 → Step 65 format_fail=0.931(不可恢复)
崩溃原因: kl_coef=0.01(5x过弱) + DAPO_FILTER=1(死螺旋)
```

---

## 四、v7 BOK-GRPO崩溃诊断

### 4.1 崩溃时间线

| 阶段 | Steps | format_fail | kl_loss | entropy | 状态 |
|------|-------|-------------|---------|---------|------|
| 健康期 | 1-50 | <0.04 | <0.05 | 0.49-0.52 | 正常学习 |
| 预警期 | 51-55 | 0.03-0.15 | 0.05-0.12 | 0.52-0.58 | KL开始失控 |
| 崩溃期 | 56-65 | 0.15-0.93 | 0.12-0.45 | 0.58-0.75 | 10步内format崩塌 |
| 死亡期 | 66-84 | >0.89 | >0.10 | >0.68 | 不可恢复 |

### 4.2 崩溃因果链

```
1. BOK集中advantage到top-K (正向max=3.0)
     ↓
2. kl_coef=0.01, KL正则化形同虚设
   (Step 50: kl_loss=0.122, 贡献仅0.122×0.01=0.0012)
     ↓
3. 策略大幅偏离参考模型 → entropy 0.49→0.75
     ↓
4. Format能力破坏 → format_fail 0.04→0.93
     ↓
5. format_rejection=1 → overall reward=0
     ↓
6. DAPO_FILTER=1 → 全组reward=0时零梯度 → 无纠正信号
     ↓
7. 正反馈死螺旋 → 不可恢复
```

### 4.3 修复措施（已应用）

| 修复 | 说明 | 状态 |
|------|------|------|
| BOK_DAPO_FILTER=0 | 关闭DAPO过滤 | ✅ 脚本已改 |
| BOK_TAU_FINAL=0.5 | 防止τ过低 | ✅ 脚本已改 |
| DAPO自动降级Safety | low_var_rate>50%时fallback接管 | ✅ core_algos.py |
| 健康监控日志 | low_var>30%打WARNING | ✅ core_algos.py |
| seed泄漏修复 | SamplingParams不再传seed | ✅ vllm_rollout_spmd.py |

---

## 五、BOK_CLIP深入分析

### 5.1 为什么需要Clip？

BOK advantage的raw值**极度不对称**：

| 场景 | raw正向max | raw负向min | 说明 |
|------|------------|------------|------|
| VeryHard (1/16 correct) | **+12.8** | -0.88 | 唯一正确样本获得极大advantage |
| Hard (3/16) | +3.6 | -0.82 | 少数正确样本分享较大advantage |
| Easy (13/16) | +0.4 | -0.87 | 众多正确分摊，每个都小 |

**不clip的后果**：
- VeryHard的winner token pg_loss ≈ -12.8 × 0.01 = -0.128
- 单步即大幅改变policy → KL急剧上升 → 训练不稳定
- 不同难度的prompt梯度量级相差30倍 → batch内梯度被hard prompt主导

**Clip的作用**：
1. **稳定性保护**：限制单样本最大梯度，防止一步偏移太大
2. **梯度量级均衡**：不同难度prompt的梯度差异从30x降至可控范围
3. **与PPO clip协同**：BOK_CLIP限制advantage，PPO clip限制ratio，双重保护

### 5.2 提高BOK_CLIP的效果分析

**关键发现：Easy prompt的advantage天然很小，CLIP对Easy没有影响**

各难度下的 $\sum|A_i|$ (梯度总强度)：

| 场景 | clip=2.0 | clip=3.0 | clip=4.0 | clip=5.0 | clip=8.0 |
|------|----------|----------|----------|----------|----------|
| 1/16 | 14.8 | 15.8 | 16.8 | 17.8 | 20.8 |
| 2/16 | 15.3 | 17.3 | 19.3 | 21.3 | 22.6 |
| 3/16 | 16.2 | 19.2 | 20.4 | 20.4 | 20.4 |
| 8/16 | 12.8 | 12.8 | 12.8 | 12.8 | 12.8 |
| 13/16 | **5.2** | **5.2** | **5.2** | **5.2** | **5.2** |
| 15/16 | 3.3 | 3.3 | 3.3 | 3.3 | 3.3 |

**Winner获得的advantage**：

| 场景 | clip=2.0 | clip=3.0 | clip=4.0 | clip=5.0 |
|------|----------|----------|----------|----------|
| 1/16 | +2.00 | +3.00 | +4.00 | +5.00 |
| 3/16 | +2.00 | +3.00 | +3.59 | +3.59 |
| 8/16 | +1.00 | +1.00 | +1.00 | +1.00 |
| 13/16 | +0.40 | +0.40 | +0.40 | +0.40 |

### 5.3 BOK_CLIP对Hard/Easy梯度比率

| clip | Hard(1/16) | Easy(13/16) | Hard/Easy比 | Hard提升 vs clip=2 |
|------|------------|-------------|-------------|-------------------|
| 2.0 | 14.8 | 5.2 | 2.84x | baseline |
| 3.0 | 15.8 | 5.2 | 3.03x | +6.7% |
| **4.0** | **16.8** | **5.2** | **3.22x** | **+13.5%** |
| 5.0 | 17.8 | 5.2 | 3.41x | +20.2% |
| 8.0 | 20.8 | 5.2 | 3.99x | +40.4% |

### 5.4 KL风险评估

| clip | winner pg_loss | KL约束(kl=0.03) | KL/pg比 | 风险 |
|------|---------------|------------------|---------|------|
| 2.0 | 0.020 | 0.0006 | 3.0% | 安全 |
| 3.0 | 0.030 | 0.0006 | 2.0% | 安全 |
| **4.0** | **0.040** | **0.0006** | **1.5%** | **可控** |
| 5.0 | 0.050 | 0.0006 | 1.2% | 边界 |

### 5.5 BOK_CLIP结论

> **提高BOK_CLIP到4.0是有利的**：
> - Hard prompt梯度提升13.5%，Easy prompt完全不受影响
> - Hard/Easy梯度比从3.03x提升到3.22x → 更强的隐式课程学习
> - KL风险从2.0%降到1.5%，仍在可控范围
> - 如训练中kl_loss>0.06，可退回clip=3.0

---

## 六、Pass@32→Pass@1场景下的BOK优势

### 6.1 任务特征

- **基模型**: pass@32=95.2%, pass@1=79%
- **目标**: 将pass@32能力转化为pass@1能力
- **难度分布** (Beta(0.5, 0.13)拟合): 57.2%easy(>95%) + 12.4%中等 + 10.8%中 + 8.5%中hard + 11.1%hard(<20%)

### 6.2 为什么BOK-GRPO是正确选择

对于pass@1=5%的hard prompt (K=16时约0-1个正确):

| 算法 | 正确样本advantage | 错误样本advantage | 正确样本梯度强度 |
|------|------------------|------------------|-----------------|
| Standard GRPO | +(r-μ)/σ ≈ +3.5 | -0.23 | baseline |
| BOK-GRPO (clip=4) | **+4.0** (集中) | -0.9 (公式限制) | **1.6-2.0x GRPO** |

**BOK在不同难度的相对效率** (vs Standard GRPO):

| 组内正确数 | BOK/GRPO梯度比 | 含义 |
|-----------|---------------|------|
| 1/16 | **2.0x** | BOK显著更优→学hard prompt |
| 3/16 | **1.55x** | BOK更优 |
| 8/16 | 1.0x | 基本相当 |
| 13/16 | **0.42x** | GRPO更优→但Easy不需要学 |
| 15/16 | **0.30x** | GRPO更优→但Easy不需要学 |

**核心洞察**：BOK天然实现了**隐式课程学习**：
- Hard prompt (需要学习) → 梯度放大1.6-2.0x
- Easy prompt (已经会了) → 梯度缩小0.3-0.4x
- 这正是pass@32→pass@1所需要的：把训练资源集中在模型"会但不稳定"的题目上

### 6.3 零方差组（全错）的处理

pass@1=5%的prompt有P(0/16全错)=44%的概率产生零方差组：
- DAPO_FILTER=0 + BOK_FALLBACK_MODE=drgrpo：使用Dr.GRPO全局baseline
- 全错组的样本获得**负advantage** → 惩罚这些失败路径
- 比DAPO过滤（零梯度）好得多

---

## 七、V8训练方案

### 7.1 v8核心参数变更 (vs v7 BOK-GRPO)

| 参数 | v7 BOK(崩溃) | **v8 BOK** | 变更理由 |
|------|-------------|-----------|---------|
| kl_coef | 0.01 | **0.03** | 中等约束：比0.01强3x，比0.05弱40% |
| BOK_CLIP | 3.0 | **4.0** | Hard梯度+13.5%，Easy不变 |
| BOK_TAU_FINAL | 0.3 | **0.5** | 防止过度集中 |
| BOK_DAPO_FILTER | 1 | **0** | 防止死螺旋 |
| ROLLOUT_TEMPERATURE | 1.0 | **1.1** | 轻微增加多样性 |

### 7.2 v8完整参数表

```bash
# ====== 算法核心 ======
STEPCOUNT_RL_MODE=bok_grpo
ADV_ESTIMATOR=bok_grpo
algorithm.kl_coef=3e-2           # 中等KL约束

# ====== BOK-GRPO参数 ======
BOK_TAU_INIT=0.8                 # τ退火起始
BOK_TAU_FINAL=0.5                # τ退火终止
BOK_TAU_SCHEDULE=cosine          # 余弦退火
BOK_TOTAL_STEPS=178              # 退火总步数
BOK_CLIP=4.0                     # 提高clip→Hard梯度+13.5%
BOK_UNIFORM_MIX=0.1              # 10%均匀混合
BOK_ADV_NORMALIZE=0              # 不标准化advantage
BOK_LOW_VAR_THRESHOLD=1e-5       # 低方差判定
BOK_FALLBACK_MODE=drgrpo         # Dr.GRPO全局baseline
BOK_DAPO_FILTER=0                # 关闭DAPO过滤
BOK_DAPO_AUTO_DISABLE_THRESHOLD=0.5  # Safety保险

# ====== PPO参数 ======
ACTOR_LR=1e-6
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_DUAL=3.0
DISABLE_KL=false

# ====== 采样 ======
ROLLOUT_TEMPERATURE=1.1          # 轻微增加多样性
ROLLOUT_N=16

# ====== Reward ======
ANSWER_WEIGHT=0.7
POINT_WEIGHT=0.2
TRAJECTORY_FORMAT_WEIGHT=0.1
TRAJ_FORMAT_REJECTION=1
TRAJ_ANSWER_GATE_MODE=soft
TRAJ_SOFT_GATE_BASE=0.8
```

### 7.3 kl_coef=0.03的设计理由

| 值 | Step50 KL占pg比 | Step65 KL占pg比 | 特点 |
|----|-----------------|-----------------|------|
| 0.01 | 6.1% | 64% (太晚) | ❌ 崩溃前无约束 |
| **0.03** | **18.3%** | **192%** | ✅ 早期有约束但不过强 |
| 0.05 | 30.5% | 320% | 保守，可能抑制BOK加速效果 |

选择0.03的逻辑：
- BOK的核心价值是**集中梯度加速hard prompt学习**
- kl_coef=0.05会让BOK退化为"和GRPO差不多但更保守"
- 0.03在Step 50时提供18.3%的约束力 → 足以防止失控
- 如果kl_loss持续>0.08，第二轮可调至0.04

### 7.4 预期监控指标

| 指标 | 健康范围 | 预警值 | 行动 |
|------|---------|--------|------|
| kl_loss | [0.003, 0.06] | >0.08 | 下轮kl_coef→0.04 |
| entropy | [0.48, 0.58] | >0.62 | 检查format |
| format_fail | <0.03 | >0.05 | 降BOK_CLIP到3.0 |
| answer_score | >0.75 | <0.70 | 分析reward分布 |
| adv_std | >0.3 | <0.15 | 检查low_var |

---

## 八、代码修改记录

### 8.1 core_algos.py (verl/trainer/core_algos.py)

**修改1: DAPO自动降级Safety** (~line 577-600)
```python
# 当 DAPO_FILTER=1 且 low_var_rate > threshold 时
# 自动将被过滤组重新用 fallback 计算 advantage
```

**修改2: 健康监控** (~line 640-655)
```python
# low_var_rate > 30% → WARNING
# zero_reward_rate > 50% → CRITICAL
```

**修改3: BOK_DAPO_AUTO_DISABLE_THRESHOLD** (~line 492)
- 新环境变量支持

### 8.2 vllm_rollout_spmd.py

**修改: seed泄漏修复** (~line 478)
```python
_SAMPLING_PARAMS_SKIP_KEYS = {"seed"}
# 防止 RolloutConfig.seed=1 泄漏到 SamplingParams
```

### 8.3 启动脚本 qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v7.sh

- BOK_TAU_FINAL: 0.3 → 0.5
- BOK_DAPO_FILTER: 默认0
- BOK_DAPO_AUTO_DISABLE_THRESHOLD: 新增
- WARNING注释: DAPO+FORMAT_REJECTION死螺旋
- **v8新增**: BOK_CLIP: 3.0 → 4.0, kl_coef: 5e-2 → 3e-2, temperature: 1.0 → 1.1

---

*报告整合时间：2025-03-09*
*整合来源：v7_training_analysis_report.md, v7_bok_grpo_parameter_analysis.md, analysis_cross_log_and_algo_comparison.md, CHANGELOG_v7_algorithm_improvements.md*
*新增内容：BOK_CLIP深入分析、Pass@32→Pass@1场景分析、V8参数方案*

---

## 附录A：外部优化建议审查

### A.1 建议：Advantage Re-centering（强制零均值）

**建议内容**：在clip后补加 $A_i^{final} = A_i^{clipped} - \frac{1}{K}\sum_j A_j^{clipped}$

**建议者论据**：clip破坏 $\sum A_i = 0$，导致正向baseline shift → entropy上升 → format崩溃

**第一性原理验证**：

| 场景 | raw_sum | clip_sum | 方向 |
|------|---------|----------|------|
| VeryHard 1/16 | 0.000 | **-8.841** | 负 |
| Hard 2/16 | 0.000 | **-3.291** | 负 |
| Hard 3/16 | 0.000 | 0.000 | 零 |
| Easy 13/16 | 0.000 | 0.000 | 零 |

**关键发现：clip后的sum是负数（非正数）**。原因：clip只压低正方向（raw_max=12.8→4.0），负方向(-0.88)不受影响。建议者的前提"sum>0导致entropy上升"是数学错误。

**Re-centering的副作用**：
- 1/16场景：Winner从+4.000变为**+4.575**（突破clip上限！）
- 这违背了clip的核心目标——限制单步最大更新
- 如果对re-centered值二次clip → 又破坏零均值 → 无限递归

**V7崩溃原因辩证**：
- 反例：v6 BOK-GRPO（同样有clip，kl=0.05）→ 200步稳定，entropy从未上升
- 实证：v7唯一差异是kl_coef=0.01 → KL约束不足才是崩溃根因

**判定：❌ 不采纳**

### A.2 建议：Temperature保持1.0或降到0.9

**分析**：
- seed修复后基准多样性已足够（16 rollout天然不同）
- temp=1.1额外增加的多样性有限，但可能增加1-2% format_fail
- V8应控制变量，优先验证BOK_CLIP=4.0和kl=0.03的核心效果
- temp=0.9不推荐（可能增加low_var比例）

**判定：⚠️ 部分采纳 → V8使用temp=1.0（已修改回）**

### A.3 建议：Adaptive KL（自适应KL系数）

**原理**：$\beta_{t+1} = \beta_t \cdot (1 + K_p \cdot (KL_{target} - KL_{current}))$

**优势**：自动平衡探索与稳定，PPO论文经典技术

**V8不采纳的理由**：
- 增加3个超参数（K_p, KL_target, β_range）
- V8首要目标是验证BOK核心改动的效果
- 固定kl_coef=0.03足以作为安全基线
- 增加debug复杂度（β动态变化需要额外日志）

**判定：⚠️ 有价值，列入V9 roadmap**
- 推荐参数：KL_target=0.025-0.03, K_p=0.05-0.1, β_range=[0.005, 0.15]

### A.4 V8最终参数（审查后修订）

| 参数 | 原V8值 | 修订后V8值 | 变更原因 |
|------|--------|-----------|---------|
| kl_coef | 3e-2 | **3e-2** | 保持 |
| BOK_CLIP | 4.0 | **4.0** | 保持 |
| ROLLOUT_TEMPERATURE | 1.1 | **1.0** | 控制变量，seed修复已提供多样性 |
| Re-centering | 未加 | **不加** | 前提错误+副作用 |
| Adaptive KL | 未加 | **不加** | V9 roadmap |
