# V8 训练日志深度分析 & V9 优化方案

## 一、V8 实验配置
| 参数 | V8 值 | 备注 |
|---|---|---|
| 算法 | BOK-GRPO | softmax-temperature advantage |
| kl_coef | 3e-2 | 比v7(5e-2)降低 |
| BOK_CLIP | 4.0 | 限制adv_max=4.0 |
| BOK_TAU | 0.8→0.5 cosine | τ退火 |
| ANSWER_WEIGHT | 0.7 | |
| POINT_WEIGHT | 0.2 | |
| FORMAT_REJECTION | 1 | |
| DAPO_FILTER | 0 | 关闭(避免v6死亡螺旋) |
| FALLBACK_MODE | drgrpo | 低方差组fallback |
| ADV_NORMALIZE | 0 | Per-group归一化关闭 |
| rollout_n | 16 | |
| total_steps | 178 (1 epoch) | |

## 二、V8 三阶段训练趋势

### Phase 1: 稳定期 (Step 1-110)
- **entropy**: 0.49 → 0.55 (缓慢上升)
- **kl_loss**: 0.004 → 0.066
- **pg_loss**: -0.003 ~ -0.009 (正常负值)
- **val_answer**: 0.756 → 0.760 (**零提升，关键问题**)
- **format_fail**: 2% → 3%
- BOK τ: 0.800 → 0.620

### Phase 2: 崩塌过渡期 (Step 110-130)
- **entropy**: 0.55 → 0.64 (突破阈值)
- **kl_loss**: 0.066 → 0.104
- **format_fail**: 3% → 23%

### Phase 3: 崩塌期 (Step 130-178)
- **entropy**: 0.71 → 1.35 (失控)
- **kl_loss**: 0.10 → 0.59
- **pg_loss**: -0.010 → +0.012 (翻正=策略退化)
- **val_answer**: 0.758 → 0.737 (下降)
- **format_fail**: 10% → 34%

## 三、Mask Quality 深度分析

### SAM3 Mask IoU 分布
| IoU 区间 | 数量 | 占比 | 累计 |
|---|---|---|---|
| [0.9, 1.0) | 30,699 | 31.6% | 100% |
| [0.8, 0.9) | 30,915 | 31.8% | 68.4% |
| [0.7, 0.8) | 16,563 | 17.1% | 36.5% |
| [0.6, 0.7) | 11,011 | 11.3% | 19.5% |
| < 0.5 | 3,487 | 3.6% | 3.6% |

**结论**: 97,099个mask条目 mean IoU=0.807，80.5%≥0.7。**Mask 质量良好，不是瓶颈。**

### Hit Rate 稳定性分析
| 阶段 | hit_any均值 | point_mean |
|---|---|---|
| 早期 (Step 1-60) | 83.2%-88.3% | 0.883 |
| 中期 (Step 60-120) | 85.8%-88.2% | 0.888 |
| 晚期 (Step 120-178) | 81.5%-84.0% | 0.891 |

**hit_any 不变的根因**: 并非mask质量差，而是：
1. SFT基模型已具备~85% hit rate（模型能力天花板，非数据天花板）
2. point_weight=0.2 梯度信号太弱，无法推动pointing改善
3. mask IoU天花板(0.8+)高于模型pointing能力

**建议**: 不删除数据。mask质量足够支撑训练信号。

## 四、Answer Reward 深度分析

### 错误答案分析 (n=3,333)
| 错误分布 | 数量 | 占比 |
|---|---|---|
| ±1 (差1个) | 2,729 | **81.9%** |
| ±2 | 440 | 13.2% |
| ±3+ | 164 | 4.9% |

| 错误方向 | 数量 | 占比 |
|---|---|---|
| 过数 (幻觉) | 1,787 | **53.6%** |
| 漏数 (遗漏) | 1,546 | 46.4% |

**高point但答案错**: 1,795例 (53.9%的错误) point≥0.8但答案wrong
**Consistency违规**: 879例 (26.4%) pred_pts≠pred_answer

### 是否需要 Soft Answer Decay?

**answer_mean 全程变化**: 0.758 → 0.765 → 0.768 (+0.01，几乎零提升)

**结论**: **是的，需要开启。** 核心理由：
1. 81.9%错误仅差±1 → 二值0/1对"差1个"和"差10个"无区分
2. answer_mean 178步几乎不变 → 二值信号在80%正确率下梯度饱和
3. 1,795例高point但答案错 → 模型知道WHERE但COUNT精度不够
4. v8的2%涨分来自trajectory结构，非answer精度（answer_mean平坦证明）

## 五、V9 优化方案

### 核心策略: BOK-GRPO + 非对称指数衰减answer reward

保留BOK-GRPO作为算法创新，同时解决两个根因：
1. **kl不足 → 提高kl_coef + 降低BOK_CLIP**
2. **answer信号饱和 → 非对称指数衰减**

### V9 参数变更

| 参数 | V8 | V9 | 变更理由 |
|---|---|---|---|
| RL_MODE | bok_grpo | **bok_grpo** | 保留核心算法创新 |
| kl_coef | 3e-2 | **5e-2** | v8崩塌根因：kl不足以约束BOK集中梯度 |
| BOK_CLIP | 4.0 | **3.0** | 降低adv_max 4.0→3.0，减~25%梯度集中 |
| BOK_TAU_INIT | 0.8 | **0.7** | 更强选择性(仍conservative) |
| BOK_TAU_FINAL | 0.5 | **0.5** | 不变(v8证明<0.5损害格式) |
| ANSWER_WEIGHT | 0.7 | **0.8** | answer信号更丰富→提权 |
| POINT_WEIGHT | 0.2 | **0.1** | point plateau→降权 |
| SOFT_ANSWER_DECAY | (无) | **1** | 核心新增 |
| DECAY_ALPHA | (无) | **5.0** | 与accuracy_reward一致 |

### 非对称指数衰减设计 (accuracy_reward style)

```
正确: answer_score = soft_base + soft_bonus * point_quality  [不变]
错误: answer_score = exp(-alpha * |pred-gt| / max(gt,1))
      alpha = alpha_base * 2  if pred > gt (幻觉重罚)
      alpha = alpha_base      if pred < gt (遗漏正常罚)
```

**分数示例 (alpha_base=5.0)**:
| gt | pred | error | alpha | score |
|---|---|---|---|---|
| 10 | 9 | -1(漏) | 5.0 | **0.607** |
| 10 | 11 | +1(幻觉) | 10.0 | **0.368** |
| 10 | 8 | -2 | 5.0 | **0.368** |
| 5 | 4 | -1 | 5.0 | **0.368** |
| 5 | 6 | +1 | 10.0 | **0.135** |
| 3 | 2 | -1 | 5.0 | **0.189** |
| 3 | 4 | +1 | 10.0 | **0.036** |

**设计优势**: 
- 自然适应不同gt规模（gt=10差1=10%误差，gt=3差1=33%误差）
- 非对称惩罚抑制幻觉（overcounting被2x重罚）
- correct(1.0) 始终显著 > wrong-by-1(0.37-0.61) >> wrong-by-2+(<0.37)

### 代码修改记录

**文件1**: `examples/reward_function/StepCount_mask_reward.py`
- 位置: L1858-1879 (在 if pred==gt 的 else 分支)
- 变更: 非对称指数衰减 answer_score = clamp(safe_exp(-alpha * norm_err))
- vars: TRAJ_SOFT_ANSWER_DECAY (默认0), TRAJ_ANSWER_DECAY_ALPHA (默认5.0)
- 向后兼容: 默认关闭

**文件2**: `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v9.sh`
- BOK-GRPO with optimized params (kl=0.05, CLIP=3.0, TAU 0.7→0.5)
- SOFT_ANSWER_DECAY=1, DECAY_ALPHA=5.0
- ANSWER_WEIGHT=0.8, POINT_WEIGHT=0.1

### 监控指标
| 指标 | 预期 | 危险线 |
|---|---|---|
| val_answer | >0.77 (20步内) | <0.73 |
| entropy | 0.48-0.55 | >0.58 |
| format_fail | 2-5% | >8% |
| kl_loss | <0.07 | >0.10 |
| answer_mean | >0.80 (含soft decay) | 下降趋势 |
| adv_mean | ≈0 | 持续<-0.05 |

---

## V9 Opt-C 最终配置审计 & 算法对比实验

### 日期: 2025-01

### V9 Opt-C 参数确认 (已审计通过)

| 参数 | V8 | V9 Opt-C | 变更理由 |
|------|-----|---------|---------|
| kl_coef | 0.03 | **0.05** | V8 step-110 collapse → +67%约束 |
| BOK_CLIP | 4.0 | **3.0** | 降低峰值梯度25% |
| BOK_TAU_INIT | 0.8 | **0.7** | 更早开始集中化 |
| ANSWER_WEIGHT | 0.7 | **0.8** | 增强answer信号(主评测指标) |
| POINT_WEIGHT | 0.2 | **0.1** | 配合answer提权降低 |
| DECAY_ALPHA | N/A | **8.0** | Opt-C: 加大correct-wrong gap |
| DECAY_CAP | N/A | **0.4** | 限制wrong reward上限 |
| GATE_MODE | N/A | **off** | 消除correct_inconsistent reversal |

### 三种算法对比分析

**为什么选Dr.GRPO（而非Standard GRPO）作为对照：**
1. Dr.GRPO解决全correct/全wrong零梯度问题（同BOK-GRPO的fallback机制）
2. Dr.GRPO是BOK-GRPO的"线性化版本" → 消融设计更干净
3. Standard GRPO在全correct组(~2.7%)完全浪费梯度
4. Dr.GRPO的batch-level baseline能给easy/hard prompt提供方向信号

**BOK-GRPO独特优势：**
- 组内质量区分：softmax让higher-quality correct (point_score高) 获得更大weight
- 非对称信号放大：对highest reward过度分配 → correct强化 > wrong抑制
- τ Annealing = 课程学习：training progression控制selectivity

**BOK-GRPO潜在劣势：**
- 训练稳定性差（V8已验证，需kl+clip双保险）
- 6个额外超参（vs Dr.GRPO 2个，GRPO 0个）
- Soft decay下组内方差缩小 → softmax区分能力被削弱

**Dr.GRPO优势：**
- 天然稳定（线性z-norm，梯度方差可控）
- Batch-level baseline处理全correct/全wrong组
- 超参少，调优成本低

### 对照实验设计

| 实验 | 脚本 | 唯一变量 |
|------|------|---------|
| V9-BOK | `v9.sh` | `STEPCOUNT_RL_MODE=bok_grpo` (softmax非线性weighting) |
| V9-DrGRPO | `v9_drgrpo_baseline.sh` | `STEPCOUNT_RL_MODE=drgrpo` (线性z-norm + batch baseline) |

**所有其余参数完全一致**：kl_coef=0.05, lr=1e-6, soft_decay(alpha=8.0,cap=0.4), gate=off, ANSWER_WT=0.8, rollout_n=16

**评测指标**：answer pass@1 accuracy (pixmo-test benchmark)

**预期结果**：
- 如果BOK > DrGRPO: softmax集中化在counting任务中有效 → 保留BOK创新
- 如果BOK ≈ DrGRPO: BOK's 额外complexity无收益 → 切换到Dr.GRPO简化系统
- 如果BOK < DrGRPO: BOK's stability cost > concentration benefit → Dr.GRPO更优
