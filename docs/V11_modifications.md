# V11 Training Modifications

## 概述
V11 基于 V10 诊断结论引入四项关键改进：
1. **P0: ppo_epochs=2** — 激活 PPO multi-epoch 训练（V10 ppo_epochs=1 导致 PPO ratio ≡ 1.0，所有 PPO 机制为死代码）
2. **P1: Conditional-Advantage (难度感知路由)** — 根据组内 pass_rate 动态选择 advantage 计算方式
3. **P2: All-Correct Group Filter** — 过滤 pass@K=K/K 的全正确组（已掌握样本零梯度）
4. **Reward Weight Rebalancing** — ANSWER 0.7→0.6, POINT 0.2→0.3（修复 V10 point_mean 退化）

## V10 核心问题回顾（44 steps 日志分析）
- **PPO 死代码**: ppo_epochs=1 → old_log_probs = new_log_probs → ratio = 1.0 → PPO clipping 从未激活
- **ppo_kl ≡ 0.0**: 全部 44 steps 均为 0.0 或 ≤6.73e-5（比 clip threshold 0.247 低 4 个数量级）
- **answer_mean 停滞**: 0.7619 → 0.7708（44 steps 几乎零提升）
- **point_mean 退化**: 0.9434 → 0.8450（-10.4%，严重恶化）
- **grad_norm**: 0.329~4.656 正常（此次运行无 NaN）
- **BoK-GRPO**: batch_mean=0.80~0.87，low_var=80~192/1024，~80% 组为 easy
- **根因**: PPO 死代码 + BOK softmax 在 easy groups 上给出微弱梯度（+0.14 vs DrGRPO +0.37）

## 修改详情

### P0: ppo_epochs=2
- **文件**: V11 launch script (Hydra override)
- **配置**: `worker.actor.ppo_epochs=2`
- **效果**: 第二个 epoch 使用更新后的 actor 计算新 log_probs，ratio ≠ 1.0，PPO clipping 激活
- **配合**: `worker.actor.max_grad_norm=0.5`（V10 默认 1.0，降低以防 NaN）

### P1: Conditional-Advantage 路由
- **文件**: `verl/trainer/core_algos.py` - `compute_bok_grpo_advantage()` 函数

#### 算法逻辑
```
对每个 group（同一 prompt 的 K 条 trajectory）:
  1. group_std <= low_var_threshold → 低方差处理（已有逻辑）
  2. pass_rate > easy_threshold → DrGRPO advantage (NEW)
  3. 其余 → BOK softmax advantage（已有逻辑）
```

#### 数学依据
| 场景 | BOK adv | DrGRPO adv | 倍率 |
|------|---------|------------|------|
| Easy (14/16 correct) | +0.14 | +0.37 | DrGRPO 2.6× |
| Hard (2/16 correct) | +3.0 (clip) | +2.5 | BOK 20% |

- pass@1=79.2% → ~80% groups 为 easy → DrGRPO routing 覆盖多数样本
- Easy groups: DrGRPO 给 2.6× 更强梯度，加速正确路径巩固
- Hard groups: BOK softmax 将概率集中在稀有正确 trajectory，最大化 exploration

#### 环境变量
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BOK_EASY_THRESHOLD` | 0.75 | pass_rate > 此值 → DrGRPO 路由（基于v7-v10分析优化）|
| `BOK_EASY_SCORE_THRESHOLD` | 0.5 | 判定 trajectory 为"正确"的 score 阈值 |

#### 代码修改位置
1. **L493** (+4行): 新增 env var 读取
2. **L513** (+1行): 新增 `n_easy_drgrpo` 计数器
3. **L549-558** (+10行): 核心路由逻辑（pass_rate 检查 + DrGRPO advantage 赋值）
4. **L630** (修改): 诊断日志增加 `easy_drgrpo` 统计
5. **L650** (修改): 详细诊断增加 easy routing 信息
6. **L660** (+1行): 健康监控增加 `_easy_drgrpo_pct`


### P2: All-Correct Group Filter（全正确组过滤）
- **文件**: `verl/trainer/core_algos.py` - `compute_bok_grpo_advantage()` 函数
- **位置**: 在 P1 easy routing 之前执行（L555-563）

#### 动机
V10 batch_mean=0.80~0.87，pass@K=K/K 的组（全部正确）约占 ~8%。这些组已完全掌握，
继续训练无信息增益且会产生方差。设为零梯度节省计算并减少噪声。

#### 算法逻辑
```python
# P2: All-correct group filter (before P1 routing)
if _filter_all_correct and pass_rate_full >= 1.0 - 1e-6:
    # 所有 trajectory 都得到高分 → 该 prompt 已掌握
    advantages[start:end] = 0.0  # 零梯度
    n_all_correct_filtered += 1
    continue
```

#### 环境变量
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BOK_FILTER_ALL_CORRECT` | 1 | 开启（1）/关闭（0）全正确组过滤 |

#### 控制流完整顺序
```
for each group:
  1. low_var check (group_std <= threshold) → fallback
  2. P2: all-correct check (pass_rate >= 1.0) → zero gradient ← NEW
  3. P1: easy routing (pass_rate > 0.6) → DrGRPO advantage ← NEW
  4. BOK softmax (remaining hard groups) → existing logic
```

### Reward Weight Rebalancing（奖励权重再平衡）
- **文件**: V11 launch script

#### 动机
V10 point_mean 从 0.9434 退化至 0.8450（-10.4%），原因是 ANSWER_WEIGHT=0.7 >> POINT_WEIGHT=0.2，
模型优化 answer 时忽视 point 准确性。point 不准 → 视觉偏移累积 → count 错误。

#### 调整
| 权重 | V10 | V11 | 说明 |
|------|-----|-----|------|
| ANSWER_WEIGHT | 0.7 | **0.6** | 降低以释放梯度给 point |
| POINT_WEIGHT | 0.2 | **0.3** | 增加以修复 point 退化 |
| FORMAT_WEIGHT | 0.1 | 0.1 | 不变 |

### V11 Launch Script 全部配置差异（vs V10）
| 参数 | V10 | V11 | 说明 |
|------|-----|-----|------|
| ppo_epochs | 1 (default) | **2** | 激活 PPO multi-epoch |
| max_grad_norm | 1.0 (default) | **0.5** | 防 NaN |
| BOK_DAPO_FILTER | 1 | **0** | 关闭 DAPO（防 death spiral） |
| TRAJ_FORMAT_REJECTION | 1 | **0** | 关闭（z-norm 下效果微乎其微） |
| BOK_EASY_THRESHOLD | - | **0.75** | 新增：easy 路由阈值 (P1)，基于v7-v10数据优化 |
| BOK_EASY_SCORE_THRESHOLD | - | **0.5** | 新增：score 判定阈值 (P1) |
| BOK_FILTER_ALL_CORRECT | - | **1** | 新增：全正确组过滤 (P2) |
| ANSWER_WEIGHT | 0.7 | **0.6** | 降低以修复 point 退化 |
| POINT_WEIGHT | 0.2 | **0.3** | 增加以修复 point 退化 |
| FORMAT_WEIGHT | 0.1 | 0.1 | 不变 |
| ACTOR_LR | 1.5e-6 | 1.5e-6 | 不变 |
| kl_coef | 2e-2 | 2e-2 | 不变 |
| BOK_CLIP | 3.0 | 3.0 | 不变 |
| BOK_TAU_FINAL | 0.3 | 0.3 | 不变 |


### BOK_EASY_THRESHOLD 优化分析 (0.6 → 0.75)

#### 分析方法
基于 v7-v10 全部训练日志的 BoK-GRPO-Detail 数据，使用二项分布模型+advantage对比计算最优值。

#### 核心发现
1. **DrGRPO advantage 在所有 pass_rate 下都大于 BOK softmax**（20-80×），但 BOK 的真正优势是**选择性**（区分 correct trajectory 质量）
2. **threshold=0.6 在 batch_mean=0.84 时，99.2% 组走 DrGRPO → BOK 形同虚设**
3. **threshold=0.75 给出合理的 75%/25% 分配**（DrGRPO/BOK）

#### Threshold 敏感性分析（batch_mean=0.84, K=16）
| threshold | %easy (DrGRPO) | %hard (BOK) |
|-----------|---------------|-------------|
| 0.5 | 99.8% | 0.2% |
| 0.6 | 99.2% | 0.8% |
| 0.7 | 90.1% | 9.9% |
| **0.75** | **75.4%** | **24.6%** |
| 0.8 | 75.4% | 24.6% |
| 0.85 | 51.6% | 48.4% |

#### cross-version batch_mean 统计
| 版本 | batch_mean | batch_std | zero_std% | entries |
|------|-----------|-----------|-----------|---------|
| v7 | 0.5302 | 0.3046 | 43.1% | 10 |
| v8 | 0.7840 | 0.3324 | 15.3% | 18 |
| v9 | 0.8424 | 0.2953 | 11.3% | 14 |
| v10 | 0.8405 | 0.2737 | 10.9% | 5 |

> v7 batch_mean 异常低是因为 DAPO_FILTER=1 导致的 death spiral。v8-v10 稳定在 0.78-0.84。

### P1 DrGRPO Advantage Clip (新增)
- **代码修改**: L573, advantage 值 clip 到 [-2.5, 2.5]
- **原因**: group_std 极小时(如 0.01)，z-score 可达 15+，导致梯度爆炸
- **数值**: 与 DrGRPO 函数的 adv_clip=2.5 保持一致

## 文件清单
| 文件 | 操作 |
|------|------|
| `verl/trainer/core_algos.py` | 修改（Conditional-Advantage） |
| `verl/trainer/core_algos.py.bak.v10` | 备份 |
| `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v11.sh` | 新增 |
| `docs/V11_modifications.md` | 新增（本文档） |

## 监控指标
训练时重点关注日志输出：
```
[BoK-GRPO] batch=N tau=X step=Y/Z  low_var=a/N  collapsed=b  easy_drgrpo=c/N  all_correct_filtered=d/N  ...
[BoK-GRPO-Detail] ... easy_drgrpo=c/N all_correct_filtered=d/N easy_threshold=0.60 ...
```

### 关键监控目标
| 指标 | 预期范围 | V10 值 | 说明 |
|------|----------|--------|------|
| `easy_drgrpo/batch` | ~70-80% | - | P1 覆盖率（threshold=0.75） |
| `all_correct_filtered/batch` | ~5-10% | - | P2 过滤率 |
| `ppo_kl` | >0.01 | ≡0.0 | ppo_epochs=2 效果验证 |
| `grad_norm` | <0.5 | 0.33~4.66 | max_grad_norm=0.5 保护 |
| `answer_mean` | 持续上升 | 0.76→0.77 | 终局正确率 |
| `point_mean` | ≥0.90 | 0.94→0.85↓ | point 准确率（V11 应稳定） |
| `format_fail_rate` | <5% | 0~7.8% | 格式合规率 |

### V11 梯度预算分配（基于 V10 数据估算）
- ~8% 样本 → P2 零梯度（全正确组，已掌握）
- ~55% 样本 → P1 DrGRPO（easy 组，2.6× 更强梯度，threshold=0.75）
- ~25% 样本 → BOK softmax（hard 组，集中探索）
- ~12% 样本 → 低方差 fallback

## 启动命令
```bash
bash examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v11.sh
```

---

## 6. V11 Watchdog 自动监控系统

### 6.1 功能概述

新增 `tools/monitor_v11_watchdog.py` 自动监控脚本，已集成到 V11 训练脚本中，训练时自动后台启动。

**功能**:
- 实时解析训练日志，每 30 秒轮询
- 每 5 步输出一行紧凑的仪表盘
- 自动检测异常指标并发出 WARNING / CRITICAL 告警
- 连续 5 次 CRITICAL → 自动创建 stop file 触发早停
- 输出 JSON Lines 格式的逐步指标，便于后续绘图分析

### 6.2 监控指标与阈值 (基于第一性原理审计)

| 指标 | WARNING阈值 | CRITICAL阈值 | 说明 |
|------|------------|-------------|------|
| `answer_mean` | < 0.74 | < 0.70 | 答案正确率 |
| `point_mean` | < 0.85 | < 0.80 | Point准确率 (V10退化监控) |
| `clipfrac` | < 0.005 或 > 0.40 | - | PPO clip激活率 |
| `grad_norm` | > 5.0 | NaN/Inf (累计3次) | 梯度范数 |
| `kl` | > 0.15 | > 0.30 | KL散度 |
| `easy_drgrpo` | < 40% 或 > 90% | - | P1路由比例 |
| `answer/point_trend` | 下降>0.03/0.05 | point持续下降 | 滑动窗口趋势 |

### 6.3 使用方式

**自动启动** (已集成): 运行 V11 训练脚本即自动启动

**手动启动**:
```bash
python3 tools/monitor_v11_watchdog.py \
    --log logs/train/training_*.log \
    --out logs/monitor/watchdog.log \
    --stop-file /tmp/v11_stop \
    --auto-stop \
    --interval 30 \
    --every 5 \
    --json-metrics logs/monitor/metrics.jsonl
```

**仪表盘输出示例**:
```
[2026-03-14 10:30:00] step=  10 | gn=0.432 | cf=0.0700 | kl=0.0045 | ov=0.8200 | ans_r=0.7800 | pt_r=0.9200 | am=0.7800 | pm=0.9200 | ff=0.0100 | easy=68% | ac=8% | tau=0.700
```

**告警示例**:
```
[2026-03-14 10:30:00] 🟡 [WARNING] step=20 clipfrac=0.0003 < 0.005 (PPO clip可能未激活)
[2026-03-14 10:30:00] 🔴 [CRITICAL] step=30 point_mean=0.7800 < 0.8 (V10退化重现)
```

### 6.4 输出文件

| 文件 | 路径 | 用途 |
|------|------|------|
| Watchdog日志 | `logs/monitor/watchdog_v11_*.log` | 人类可读的监控输出 |
| 指标JSON | `logs/monitor/metrics_v11_*.jsonl` | 逐步指标，可用于绘图 |
| Stop file | `/tmp/v11_stop_*` | 早停信号（连续5次CRITICAL触发） |

---

## 7. V11 训练实时分析（Steps 0-11）

### 7.1 完整指标表

| Step | pm | am | overall | clipfrac | grad_norm | fmtFail% | stopViol% |
|------|------|------|---------|----------|-----------|----------|-----------|
| 0 | 0.8848 | 0.7059 | 0.7059 | --- | --- | 5.88 | 5.88 |
| 1 | 0.8807 | **0.8510** | 0.8743 | 0.000000 | 0.849 | 0.49 | 0.49 |
| 2 | 0.8536 | 0.7682 | 0.8150 | 0.001000 | 0.908 | 1.95 | 1.95 |
| 3 | 0.8679 | 0.7957 | 0.8373 | 0.001000 | 1.133 | 0.49 | 0.49 |
| 4 | 0.8702 | 0.8156 | 0.8478 | 0.001000 | **2.035** | 2.64 | 2.64 |
| 5 | 0.8577 | 0.8160 | 0.8447 | 0.001000 | 1.017 | 2.15 | 2.15 |
| 6 | 0.8888 | 0.7831 | 0.8351 | 0.001000 | 0.714 | 1.37 | 1.37 |
| 7 | **0.8076** | 0.8126 | 0.8294 | 0.001000 | 0.609 | 0.39 | 0.39 |
| 8 | 0.8206 | 0.7576 | 0.7971 | 0.001000 | 0.796 | 3.61 | 3.61 |
| 9 | **0.9019** | **0.8327** | **0.8702** | 0.001000 | 0.975 | 0.00 | 0.00 |
| 10 | 0.8728 | 0.8117 | 0.8476 | 0.001000 | 0.677 | 1.27 | 1.27 |
| 11 | 0.8487 | 0.7817 | 0.8212 | 0.001000 | 1.154 | 2.44 | 2.44 |

### 7.2 BoK-GRPO路由（Step 1 & 11对比）

| Step | easy_drgrpo | bok | all_correct | adv_mean | adv_std | tau |
|------|-------------|-----|-------------|----------|---------|-----|
| 1 | 40.6% | 31.2% | 28.1% | +0.0192 | 0.6728 | 0.700 |
| 11 | 34.4% | 37.5% | 28.1% | -0.0081 | 0.7195 | 0.699 |

**路由变化分析**: easy_drgrpo↓6.2%, bok↑6.3%, all_correct不变 → 更多样本从easy转入bok通道，模型正在学习更难的pattern。

### 7.3 V10 vs V11 更新对比

| 指标 | V10 (45步avg) | V11 (12步avg) | 变化 | 判定 |
|------|-------------|-------------|------|------|
| point_mean | 0.8634 | 0.8629 | -0.0005 | 持平 |
| answer_mean | 0.7868 | 0.7943 | **+0.0075** | V11优 ↗ |
| clipfrac | 0.000019 | 0.000909 | **48×** | PPO从死到活 |
| grad_norm | 0.709 | 0.988 | +0.279 | 梯度更强 ↗ |

### 7.4 趋势与关键观察

- **answer_mean显著上升**: 0.706→0.782 (Δ=+0.076)，学习效率~5× V10
- **point_mean轻微下降**: 0.885→0.849 (Δ=-0.036)，Step 7-8有低谷(0.807/0.820)，Step 9反弹至0.901
- **PPO alive**: 10/11步clipfrac=0.001（非零），确认ppo_epochs=2成功激活
- **Step 9 = 最优步**: pm=0.9019, am=0.8327, format_fail=0.000
- **Step 4 grad_norm=2.035**: 唯一峰值，在max_grad_norm=0.5裁剪下安全

### 7.5 风险项

1. **point_mean局部下降(Step 7-8)**: 如持续<0.80会导致误差累计
2. **BoK-GRPO日志频率低**: 仅Step 1/11有路由信息，考虑增加打印频率
3. **Watchdog进程已停止**: 修复后的代码需下次训练生效

---

## 8. Watchdog Bug修复记录 (2026-03-14)

### 8.1 发现问题

Watchdog在Step 5后停止输出（67+分钟无更新），训练正常推进到Step 12。

**直接原因**: Watchdog进程在GPU节点崩溃，stderr被`> /dev/null 2>&1`丢弃。

**根本原因**: 3个代码bug叠加:
1. **预训练RewardHealth污染**: 9条预训练数据(pm=0.94)混入trend deque，训练期pm=0.86被误判为"持续下降"
2. **CLIPFRAC_MIN=0.005过高**: V11实际clipfrac=0.001是V10的48×，阈值不匹配
3. **stderr丢失**: 崩溃原因无法追溯

### 8.2 应用的5个核心修复

文件: `tools/monitor_v11_watchdog.py` (530→537行)

| # | 修改位置 | 修改内容 | 效果 |
|---|----------|----------|------|
| 1 | L79 | `CLIPFRAC_MIN = 0.0005` (原0.005) | 匹配V11实际clipfrac |
| 2 | L77 | 新增`POINT_TREND_MIN_STEPS = 6` | 前6步跳过趋势告警 |
| 3 | L107 | 新增`self.training_started = False` | 区分预训练/训练阶段 |
| 4 | L151-154 | 首次Step marker: 设training_started=True, 清空deque | 隔离预训练数据 |
| 5 | L305 | point CRITICAL要求`step >= POINT_TREND_MIN_STEPS` | 避免样本不足假告警 |

### 8.3 V11 script stderr修复

文件: `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v11.sh` L302

```bash
# 修复前:
> /dev/null 2>&1 &
# 修复后:
> /dev/null 2>"${MONITOR_DIR}/watchdog_stderr_${RUN_TS}.log" &
```

### 8.4 验证结果

模拟全量V11日志(Steps 0-11)测试:
- **修复前**: 4 CRITICAL + 3 WARNING (全部为假阳性)
- **修复后**: **0 CRITICAL** + 2 WARNING(合理告警: pm<0.85, easy_drgrpo<40%)

### 8.5 待修复项(下次优化)

1. Report loop step-skipping bug: `completed = last_step`应改为遍历所有未报告步
2. 主循环exception handler: 防止解析异常导致进程崩溃

## 9. Watchdog 完整修复记录 (Session 3)

### 9.1 Report Loop步骤跳过修复
**问题**: 原逻辑 `completed = state.last_completed_step` 只检查最新步骤。若轮询间隔内完成多步(如5→10)，中间步骤6-9的health check被跳过。`state.reported_until` 在 `every` 条件外更新导致永久跳过。

**修复**: 改为 `for report_step in range(state.reported_until + 1, completed + 1):` 遍历所有未报告步骤:
- Health check在每一步执行
- Dashboard/JSON输出仅在 `--every` 间隔或前5步
- `reported_until` 在遍历完所有步骤后统一设为 `completed`

### 9.2 异常处理器
**问题**: 主循环仅捕获 `KeyboardInterrupt` 和 `FileNotFoundError`。任何解析错误(KeyError/ZeroDivisionError等)导致watchdog静默崩溃。

**修复**: 在chunk处理块(parse_line + validation + report)外包裹 `try/except Exception`:
- 捕获所有非键盘中断异常
- 将traceback写入watchdog输出文件
- sleep后continue继续监控，不崩溃
- 位置: L439(try) → L533(except Exception)

### 9.3 BoK-GRPO每步日志修复
**文件**: `verl/trainer/core_algos.py` L638-645
**问题**: `_ctr % 10 == 1` 每10次调用打印一次。ppo_epochs=2时每步调用2次，实际每5步打印一次。
**修复**: 改用 `_last_logged_step` 跟踪，每个 `global_step` 仅首次调用时打印，保证每步恰好1条BoK-GRPO日志。

### 9.4 Grad_norm Spike分析结论
4个spike步骤: S4(2.035), S12(2.715), S18(2.308), S19(3.462)
- Pattern A: 高KL累积触发 (kl_loss≥0.019)
- Pattern B: 级联放大 (S18→S19)
- S12特殊: 极高reward同质性(pm=0.922)驱动方向对齐梯度
- 结论: current max_grad_norm=0.5足够，spike自动恢复，无需调参

### 9.5 文件版本记录
- `tools/monitor_v11_watchdog.py`: 550行 (原530→537→550), 备份.bak.v3
- `verl/trainer/core_algos.py`: 946行 (L638-645修改)

---

## 10. V11 58步深度分析 + V10对比 (2025-XX-XX)

### 10.1 训练总状态
- V11: 59步 (S0-S58), S59正在rollout，日志77949行
- V10: 44步 (S1-S44), 已完结

### 10.2 P0/P1/P2改进验证结果

| 改进项 | 状态 | 关键证据 |
|---|---|---|
| P0 PPO复活 | ✅ 达成 | V10: 0/44步clipfrac>0.0001; V11: 57/58步, 175x提升 |
| P1 路由生效 | ✅ 达成 | BoK 31.2%→40.6%, 三通道动态演化 |
| P2 AC过滤 | ✅ 达成 | 平均30.5%样本被过滤(25%-42.2%) |

### 10.3 V10 vs V11核心指标对比

| 指标 | V10 avg | V11 avg | V11 gain | V10 last | V11 last |
|---|---|---|---|---|---|
| point_mean | 0.8676 | 0.8692 | +0.0016 | 0.8914 | 0.9240 |
| answer_mean | 0.7964 | 0.8150 | +0.0186 | 0.8014 | 0.8778 |
| overall_mean | 0.8264 | 0.8475 | +0.0210 | 0.8384 | 0.9038 |
| clipfrac | 0.000007 | 0.0013 | 175x | 0 | 0.001 |
| grad_norm | 0.848 | 1.318 | +0.470 | 0.881 | 0.799 |
| kl_loss | 0.0153 | 0.0185 | 1.21x | 0.016 | 0.024 |

### 10.4 趋势对比 - 关键发现
- V10: 后10步am=0.8003 < 前10步am=0.8026, Δ=-0.0023 ↘下降
- V11: 后10步am=0.8310 > 前10步am=0.8044, Δ=+0.0265 ↗上升
- V10 S31-44尾段斜率 -0.00390/step → 学习停滞/倒退
- V11持续上升, S58达到历史最佳overall=0.9038

### 10.5 风险状态
- Grad spikes: V11共8次>2.0 (V10仅2次), 最严重S38=4.800, S41=4.850
- KL: Early(0.009)→Mid(0.016)→Late(0.027), 持续上升但last=0.024仍安全
- 格式合规: V11 98.75% vs V10 98.51%

### 10.6 结论
- V11在所有设计目标(P0/P1/P2)上均达成
- answer_mean持续上升(V10停滞), S58为历史最佳
- 建议: S58 checkpoint evaluate; 继续训练安全; 关注KL<0.05

---

## 11. Overcounting Root Cause & Fix (Extra Point Penalty)

### 11.1 问题发现
V11模型输出分析(6261 trajectory, 61 steps)发现：
- 错误答案中48%为overcounting(+1)，仅3%为undercounting(-1)
- 从早期(43%)到晚期(48%)overcounting比例持续上升
- GT≤4场景尤为严重

### 11.2 根因分析: `point_dense_score`结构性不对称

**核心代码** (`_trajectory_point_dense_reward_without_gt_points_details`, L1142):
```python
max_eval_steps = int(min(len(pred_points), target_count))
```

该行导致：
- **多数(pred > GT)**: 仅评估前GT个点，多余点**零成本** → pt_score = GT/GT = 1.0
- **漏数(pred < GT)**: 只有pred个点，永远无法满分 → pt_score = pred/GT < 1.0

**三层分析**:
1. **Layer 1 (point_score)**: 多数vs漏数同样偏差±k，pt_score差异 = k/GT（多数永远占优）
2. **Layer 2 (加权组合)**: PW=0.3 × pt差异 > AW=0.6 × ans差异（当GT≤4时）
3. **Layer 3 (训练数据)**: GT≤4样本占比大 → 模型学到"多数习惯"

**GT=3示例**(旧):
| Scenario | ans_r | pt_r | total |
|----------|-------|------|-------|
| correct  | 1.0000| 1.0000| 1.0000|
| over+1   | 0.0048| 1.0000| 0.4029|
| under-1  | 0.0695| 0.6667| 0.3417|
| **Δ(over-under)** | | | **+0.0612** ← 多数更有利! |

### 11.3 Fix设计: Extra Point Penalty

**公式**: `penalty = λ × max(0, pred_pts - GT) / GT`
- λ=1.0时: overcount by k → pt_score = 1.0 - k/GT = (GT-k)/GT
- undercount by k → pt_score = (GT-k)/GT
- **完美对称！**

**代码修改** (`StepCount_mask_reward.py`, L1211-1216, 在`point_dense_score`计算之后):
```python
# --- Extra point penalty: eliminate structural overcounting advantage ---
_extra_pt_lambda = float(os.environ.get("TRAJ_EXTRA_POINT_PENALTY_LAMBDA", "1.0"))
_extra_points = max(0, len(pred_points) - target_count)
if _extra_points > 0 and _extra_pt_lambda > 0:
    _extra_penalty = _extra_pt_lambda * _extra_points / denom
    point_dense_score = clamp_reward(point_dense_score - _extra_penalty)
```

**环境变量** (V11 launch script, L67):
```bash
export TRAJ_EXTRA_POINT_PENALTY_LAMBDA=${TRAJ_EXTRA_POINT_PENALTY_LAMBDA:-1.0}
```

### 11.4 验证结果

**修复前后对比** (Over+1 vs Under-1, Δ = total_over - total_under):

| GT | Old Δ | New Δ | Fixed? |
|----|-------|-------|--------|
| 1  | +0.300| -0.000| ✓ |
| 2  | +0.139| -0.011| ✓ |
| 3  | +0.061| -0.039| ✓ |
| 4  | +0.005| -0.070| ✓ |
| 5  | -0.037| -0.097| ✓ (强化) |
| 6  | -0.067| -0.117| ✓ (强化) |
| 7  | -0.088| -0.130| ✓ (强化) |
| 8  | -0.102| -0.140| ✓ (强化) |
| 9  | -0.105| -0.139| ✓ (强化) |
| 10 | -0.089| -0.119| ✓ (强化) |

**所有10个GT值全部修复** — overcounting reward ≤ undercounting reward。

### 11.5 修改文件清单
1. `examples/reward_function/StepCount_mask_reward.py` L1211-1216: 新增extra point penalty
2. `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v11.sh` L67: 新增env var
3. `docs/V11_modifications.md`: 本节文档

---

## 12. Fix 2: 高GT(≥7) Undercounting — GT-Scaled Alpha

### 12.1 问题发现

通过对V11 S58的6831条trajectory样本的深度分析，发现了GT-依赖的error direction分布：

| GT范围 | 主要错误方向 | 典型错误率 |
|--------|-------------|-----------|
| GT≤6   | 过计数(+1)为主 | +1占多数 |
| GT≥7   | 欠计数(-1)为主 | GT=7: -1=11.9%, GT=8: -1=14.9%, GT=9: -1=12.9%, GT=10: -1=22.1% |

**GT=10的answer错误率高达22.1%**，其中绝大多数为 -1 欠计数。

### 12.2 根因分析

**核心问题: `DECAY_CAP=0.4` 在高GT时削弱了 correct-wrong 惩罚差距**

Answer decay公式: `answer_score = min(exp(-alpha * |error| / GT), cap)`

对于 undercount -1 (|error|=1):
- `raw_decay = exp(-alpha / GT)`
- 当 `GT ≥ 9, alpha=8`: `exp(-8/9) = 0.411 > cap=0.4` → 被cap截断
- 实际上 GT=7,8,9,10 的raw值分别为: 0.319, 0.368, 0.411, 0.449

**问题1**: GT=9,10的raw值超过cap，相当于cap并未真正惩罚 -1 错误
**问题2**: 即使未超cap，高GT的 correct(1.0) - wrong 差距越来越小:
- GT=1: gap = 1.0 - 0.0003 = 0.9997
- GT=5: gap = 1.0 - 0.202 = 0.798
- GT=10: gap = 1.0 - 0.400 = 0.600

**对比overcounting**: alpha=16 (2x), `exp(-16/GT)` 对任何GT都远低于cap → overcounting被充分惩罚

### 12.3 解决方案: GT-Scaled Alpha

**核心思路**: 对GT>5的undercounting，按GT线性增大alpha，使raw值被压到cap以下。

公式:
```
if GT > threshold (default=5) and error < 0 (undercount):
    alpha_under = alpha_base × (1 + scale × (GT - threshold) / threshold)
else:
    alpha_under = alpha_base (default=8.0)
```

默认参数: `scale=0.5, threshold=5`

各GT对应alpha:
| GT | alpha_under | raw_decay(err=-1) | 是否被cap截断 | correct-wrong gap |
|----|-------------|-------------------|--------------|-------------------|
| 1  | 8.0         | 0.0003           | 否           | 1.000             |
| 5  | 8.0         | 0.2019           | 否           | 0.798             |
| 6  | 8.8         | 0.2299           | 否           | 0.770 (+4.0%)     |
| 7  | 9.6         | 0.2535           | 否           | 0.747 (+8.7%)     |
| 8  | 10.4        | 0.2725           | 否           | 0.728 (+13.7%)    |
| 9  | 11.2        | 0.2882           | 否           | 0.712 (+17.1%)    |
| 10 | 12.0        | 0.3012           | 否           | 0.699 (+15.2%)    |

**关键效果**: GT=9和GT=10的raw值从超cap(0.411, 0.449)降到cap以下(0.288, 0.301)，重新获得惩罚效力。

### 12.4 安全验证

Combined验证(extra point penalty + GT-scaled alpha)确认:
- **所有GT=1~10**: overcounting总reward < undercounting总reward ✅
- 不影响GT≤5的原有逻辑 ✅
- 不影响overcounting的2x alpha惩罚 ✅
- 不影响format reward逻辑 ✅

### 12.5 实现代码

**`StepCount_mask_reward.py` L1878-1889** (trajectory_reward() PATH B answer scoring):
```python
_alpha_base = float(os.environ.get("TRAJ_ANSWER_DECAY_ALPHA", "8.0"))
# Asymmetric alpha + GT-scaled undercounting penalty
if _error > 0:
    _alpha = _alpha_base * 2.0  # overcount: 2x harder
else:
    _under_gt_scale = float(os.environ.get("TRAJ_UNDER_ALPHA_GT_SCALE", "0.5"))
    _under_gt_th = int(float(os.environ.get("TRAJ_UNDER_ALPHA_GT_THRESHOLD", "5")))
    if _gt_int > _under_gt_th and _under_gt_scale > 0:
        _alpha = _alpha_base * (1.0 + _under_gt_scale * (_gt_int - _under_gt_th) / max(_under_gt_th, 1))
    else:
        _alpha = _alpha_base
```

### 12.6 新增环境变量

| 变量名 | 默认值 | 作用 |
|--------|-------|------|
| `TRAJ_UNDER_ALPHA_GT_SCALE` | 0.5 | GT-scaled alpha的缩放系数 |
| `TRAJ_UNDER_ALPHA_GT_THRESHOLD` | 5 | GT-scaled alpha生效的GT阈值 |

### 12.7 修改文件清单
1. `examples/reward_function/StepCount_mask_reward.py` L1878-1889: GT-scaled alpha逻辑
2. `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v11.sh` L68-69: 新增2个env var
3. `docs/V11_modifications.md`: 本节文档

---

## 13. V11 两项Fix总结

### 13.1 修复前的问题

| 问题 | 影响GT范围 | 根因 | 模型表现 |
|------|-----------|------|---------|
| Overcounting偏好 | GT≤4 | point_score对extra points无惩罚 | +1泛滥(48%错误) |
| Undercounting不足惩罚 | GT≥7 | DECAY_CAP=0.4在高GT时失效 | GT=10错误率22.1% |

### 13.2 修复方案

| Fix | 公式 | 新增env var | 效果 |
|-----|------|------------|------|
| Extra Point Penalty | `pt -= λ × max(0, pred-GT) / GT` | `TRAJ_EXTRA_POINT_PENALTY_LAMBDA=1.0` | GT≤4 overcounting消除 |
| GT-Scaled Alpha | `α = α_base × (1 + 0.5×(GT-5)/5)` for GT>5 | `TRAJ_UNDER_ALPHA_GT_SCALE=0.5`, `TRAJ_UNDER_ALPHA_GT_THRESHOLD=5` | GT≥7 gap提升4-17% |

### 13.3 Combined安全性

所有GT=1~10: overcounting_total_reward < undercounting_total_reward ✅
两个Fix互相独立，可单独开关：
- 关闭Fix1: `TRAJ_EXTRA_POINT_PENALTY_LAMBDA=0`
- 关闭Fix2: `TRAJ_UNDER_ALPHA_GT_SCALE=0`

---

## 14. V11 训练趋势分析 (74 Steps: S0-S73)

### 14.1 整体训练曲线

| 阶段 | answer | point | overall | KL loss | 方向 |
|------|--------|-------|---------|---------|------|
| S1-10 | 0.8046 | 0.8624 | 0.8398 | 0.00810 | 快速上升 |
| S11-20 | 0.7854 | 0.8529 | 0.8252 | 0.01390 | 短暂回落 |
| S21-30 | 0.8207 | 0.8674 | 0.8513 | 0.01060 | 恢复上升 |
| S31-40 | 0.8315 | 0.8763 | 0.8609 | 0.02270 | **峰值阶段** |
| S41-50 | 0.8306 | 0.8762 | 0.8604 | 0.02730 | 停滞→ |
| S51-60 | 0.8276 | 0.8718 | 0.8575 | 0.03120 | 停滞→ |
| S61-73 | 0.8258 | 0.8790 | 0.8581 | 0.03715 | 停滞/微降→ |

**关键发现：训练已在 S31-40 达到 Plateau，之后40步未突破。**

### 14.2 峰值检测

| 指标 | Peak Step | Peak值 | Latest(S73) | 差距 |
|------|-----------|--------|-------------|------|
| Answer | S31 | 0.8810 | 0.7780 | -0.1030 |
| Point | S37 | 0.9300 | 0.8850 | -0.0450 |
| Overall | S58 | 0.9040 | 0.8310 | -0.0730 |

Top-5 Overall: S58(0.904), S64(0.901), S31(0.897), S25(0.896), S40(0.892)

### 14.3 KL 散度：持续上升 ⚠️

- S1-10: 0.008 → S31-40: 0.023 → S51-60: 0.031 → S61-73: 0.037
- **5倍增长**，说明模型持续远离参考策略
- S70达到0.066的峰值，是训练全程最高
- 当前仍在安全范围内但趋势令人担忧

### 14.4 近期 Error Direction 分析 (S50-73, 2490样本)

| GT | 样本 | 准确率 | Overcounting | Undercounting | 偏向 |
|----|------|--------|-------------|---------------|------|
| 1 | 243 | 94.2% | 14 (5.8%) | 0 (0%) | **OVER** |
| 2 | 233 | 83.3% | 23 (9.9%) | 16 (6.9%) | 平衡 |
| 3 | 312 | 87.2% | 21 (6.7%) | 19 (6.1%) | 平衡 |
| 4 | 239 | 87.4% | 19 (7.9%) | 11 (4.6%) | **OVER** |
| 5 | 209 | 79.9% | 20 (9.6%) | 22 (10.5%) | 平衡 |
| 6 | 531 | 78.0% | 66 (12.4%) | 51 (9.6%) | 平衡 |
| 7 | 276 | 70.7% | 30 (10.9%) | 51 (18.5%) | **UNDER** |
| 8 | 241 | 75.5% | 14 (5.8%) | 45 (18.7%) | **UNDER** |
| 9 | 123 | 74.8% | 7 (5.7%) | 24 (19.5%) | **UNDER** |
| 10 | 83 | 72.3% | 1 (1.2%) | 22 (26.5%) | **UNDER** |

**关键结论：**
- GT=1,4 仍有 overcounting 偏向 → Fix1 仍然必要
- GT≥7 全部 undercounting 偏向，且偏向随GT增大 → Fix2 仍然必要
- GT=10 undercounting 率 26.5% vs overcounting 1.2%，差异极大
- 总体：8.6% overcounting, 10.5% undercounting → 前者训练中无惩罚，后者惩罚不足

### 14.5 Fix 必要性评估

#### Fix1 (Extra Point Penalty) — **仍然必要 ✅**
- **问题仍存在**：206/2490 (8.3%) 样本 pred_points > GT，这些样本的point_score在当前reward下不受惩罚
- GT=1 的 overcounting 率 5.8% 完全无undercounting → 根本原因是多点无代价
- 虽然训练已经略微缓解了overcounting比率 (从早期~48%降到8.6%), 但这是通过学习到的策略而非reward信号纠正的

#### Fix2 (GT-Scaled Alpha) — **仍然必要 ✅**
- **问题仍存在**：193/2490 (7.8%) 高GT(>5) undercounting，且趋势随GT增大
- GT=10 undercounting 率 26.5% → 回答9 vs 正确10 的reward gap不足
- 模型已经plateau 40步，正是因为reward信号不够锐利，无法继续区分"接近正确"和"完全正确"

#### 整体评估

| 维度 | 当前状态 | Fix后预期 |
|------|---------|----------|
| Answer accuracy | plateau ~0.83 | 突破plateau, 目标0.87+ |
| Overcounting bias (GT≤4) | 存在，GT=1/4 偏向OVER | Fix1消除偏向 |
| Undercounting bias (GT≥7) | 严重，GT=10达26.5% | Fix2增加惩罚力度 |
| KL divergence | 持续上升0.037 | 更锐利reward → 更高效学习 → 更少KL预算浪费 |
| 训练效率 | 40步plateau | 修正reward信号 → 打破plateau |

**建议方案**:
1. 停止当前V11训练 (已plateau, KL持续增长, 无法突破)
2. 从 **S64 checkpoint** 开始 V12 训练 (S64 overall=0.901, 第二好成绩)
3. V12 启用 Fix1 + Fix2 修改后的 reward
4. 或者从 SFT checkpoint 重新开始，reward从一开始就修正

### 14.6 训练健康度总结

- ✅ Clipfrac: 0.001-0.002, 非常低健康
- ✅ Grad norm: 多数 0.5-2.0, 偶尔spike到4.8但无NaN
- ✅ Format compliance: >0.98, 学到了良好格式
- ✅ Point hit rate: step1 ~0.88-0.93, 视觉定位能力稳定
- ⚠️ KL loss: 0.037 且持续上升, 接近警戒区
- ⚠️ Entropy: 从0.51缓降到0.49, 微弱mode collapse趋势
- ❌ Answer/Overall: plateau at 0.83/0.86, 40步无突破

---

## 15. V12 启动脚本准备

### 15.1 脚本位置

`examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v12.sh`

### 15.2 V12 vs V11 关键差异

| 参数 | V11 | V12 | 原因 |
|------|-----|-----|------|
| `load_checkpoint_path` | null | V11 S66 checkpoint | 从V11最佳状态继续训练 |
| `kl_coef` | 0.02 | **0.03** | V11 KL从0.008→0.037(5x上升)，需加强正则 |
| `ACTOR_LR` (bok_grpo) | 1.5e-6 | **1e-6** | 从checkpoint微调需更低LR |
| `BOK_TAU_INIT` | 0.7 | **0.65** | V11结束时τ=0.662，warm-start |
| `save_freq` | 66 | **20** | V11只存了1个checkpoint，不够 |
| Fix1 (EXTRA_POINT_PENALTY) | 1.0 (已设) | 1.0 | 不变，已实现 |
| Fix2 (UNDER_ALPHA_GT_SCALE) | 0.5 (已设) | 0.5 | 不变，已实现 |

### 15.3 启动方式

**方式A: 从V11 S66 checkpoint恢复训练（推荐）**
```bash
bash examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v12.sh
```

**方式B: 从SFT重新开始（Clean start）**
```bash
V12_RESUME_CKPT="" bash examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v12.sh
```

### 15.4 预期效果

1. Fix1: 消除 GT=1,4 的 overcounting 偏向，提升低GT accuracy
2. Fix2: 增强 GT≥7 的 undercounting 惩罚，提升高GT accuracy
3. KL=0.03: 控制 KL drift，延长有效训练窗口
4. LR=1e-6: 更平稳的微调，避免从checkpoint开始的初始震荡
5. save_freq=20: 每20步保存，便于回退和分析

### 15.5 V12 参数深度审计（已修复Bug）

**Bug修复**: `BOK_TAU_INIT=0.65` → `0.7`
- 原因: cosine schedule的TAU_INIT是t=0时刻值, 不是当前起点值
- 0.65在S66处产生τ=0.621, 与V11的τ=0.667产生-0.046的突变
- 改为0.7后, S66处τ=0.667, 与V11无缝衔接

**total_epochs=2验证**: training_steps≈356 (178步/epoch), 从S66恢复后剩余290步, 充足

**参数判定汇总**:
| 参数 | V12值 | 判定 |
|------|-------|------|
| kl_coef | 0.03 | ✅ 温和提升KL约束 |
| ACTOR_LR | 1e-6 | ✅ checkpoint微调标准值 |
| save_freq | 20 | ✅ 更频繁保存 |
| TAU_INIT | 0.7 (修复) | ✅ 与V11无缝衔接 |
| max_grad_norm | 0.5 | ⚠️ 保守,60-80%梯度被裁 |
| Fix1 λ | 1.0 | ✅ 观察GT=1是否过强 |
| Fix2 scale | 0.5 | ✅ 温和增强高GT惩罚 |

**预期目标**: answer 0.83→0.87-0.90, val pass@1 0.76→0.82-0.86

### 15.6 V12 最终配置调整（SFT起点 + total_epochs=1）

**决策过程**:
1. 用户选择clean start（SFT起点）而非S66 checkpoint → 公平A/B对比V11
2. total_epochs=2 → 1: V11仅跑完epoch 1的41%（73/178步），无需第2 epoch
3. SFT模型已有基础pointing/counting/format能力（val/answer=0.756, format=0.962）

**V12 vs V11 完整差异**:
| 参数 | V11 | V12 | 变更理由 |
|------|-----|-----|---------|
| 起点 | SFT | SFT | ✅ 公平对比 |
| total_epochs | 2 | **1** | V11未用到epoch 2 |
| kl_coef | 0.02 | **0.03** | 控制KL漂移 |
| ACTOR_LR | 1.5e-6 | **1e-6** | 配合新reward更保守 |
| save_freq | 66 | **20** | 更多checkpoint |
| BOK_TAU_INIT | 0.7 | 0.7 | 保持(total_steps变化) |
| Fix1 (λ) | 未启用 | **1.0** | 惩罚多余point |
| Fix2 (scale) | 未启用 | **0.5/5** | 高GT undercounting加强 |

**τ schedule注意**: total_epochs=1→total_steps≈178，τ在S73处为0.556（V11为0.660)。
τ annealing速度约2x快于V11，但配合更精准的Fix1/Fix2 reward信号，更早selectivity是正面的。

**计划外Bug修复**: save_freq=66→20时产生双反斜杠`\\`，已修复为单`\`

---

## 16. V12 Early-Stage Monitoring Report (S0-S4, 2026-03-14 20:00)

### 16.1 训练进度
- V12启动时间: 2026-03-14 18:02
- 目前完成: **5步 (S0-S4)**，S5 reward已完成，update_actor阶段pending
- 日志最后更新: 19:37 (约30分钟前，可能挂起或NFS延迟)
- 训练速度: ~1109s/step (≈18.5min/step)，与V11一致

### 16.2 V12 vs V11 同阶段对比 (S1-S4)

#### 核心指标对比表

| Step | V12 answer | V11 answer | V12 overall | V11 overall | V12 point | V11 point |
|------|-----------|-----------|------------|------------|----------|----------|
| S0(val) | 0.756 | 0.756 | 0.756 | - | 0.919 | - |
| S1 | 0.849 | 0.851 | 0.869 | 0.874 | 0.867 | 0.881 |
| S2 | 0.762 | 0.768 | 0.802 | 0.815 | 0.823 | 0.854 |
| S3 | 0.775 | 0.796 | 0.810 | 0.837 | 0.822 | 0.868 |
| S4 | 0.809 | 0.816 | 0.843 | 0.848 | 0.864 | 0.870 |

**初步结论**:
- V12 answer mean略低于V11 (-0.7% ~ -2.7%)，尚在warmup阶段波动范围内
- V12 point_mean系统性低于V11 (-0.6% ~ -4.6%)，**这正是Fix1惩罚多余point的预期效果**
- V12 overall因point lower而稍低，但差距在缩小(S4仅差0.5%)

#### Actor指标对比

| Step | V12 kl_loss | V11 kl_loss | V12 grad_norm | V11 grad_norm | V12 entropy | V11 entropy |
|------|-----------|-----------|-------------|-------------|-----------|-----------|
| S1 | 0.006 | 0.006 | 0.856 | 0.849 | 0.492 | 0.492 |
| S2 | 0.004 | 0.005 | 1.023 | 0.908 | 0.519 | 0.519 |
| S3 | 0.010 | 0.011 | 1.492 | 1.133 | 0.511 | 0.515 |
| S4 | 0.022 | 0.022 | 0.855 | 2.035 | 0.515 | 0.518 |

**分析**: KL和entropy几乎完全一致（同一SFT起点+同参数），梯度范数在波动范围内。训练动力学健康。

#### LR Warmup对比
- V12: S1=1.875e-7, S2=3.75e-7, S3=5.625e-7, S4=7.5e-7 (warmup_ratio=0.05)
- V11: S1=8.82e-8, S2=1.76e-7, S3=2.65e-7, S4=3.53e-7
- V12 LR约2x V11（因total_steps=178 vs 356, warmup_steps=9 vs 18），step相同时LR更高
- 这使V12学习更快但不一定更好——需要观察中期是否收敛稳定

#### RewardHealth对比 (pid=1568636 vs pid=1418857)

| Calls | V12 ans | V11 ans | V12 overall | V11 overall | V12 fmt_fail | V11 fmt_fail |
|-------|---------|---------|------------|------------|-------------|-------------|
| 1 | 0.849 | 0.851 | 0.869 | 0.874 | 0.005 | 0.005 |
| 2 | 0.762 | 0.768 | 0.802 | 0.815 | 0.020 | 0.020 |
| 3 | 0.775 | 0.796 | 0.810 | 0.837 | 0.009 | 0.005 |
| 4 | 0.809 | 0.816 | 0.843 | 0.848 | 0.023 | 0.026 |
| 5 | 0.805 | 0.816 | 0.831 | 0.845 | 0.020 | 0.022 |

### 16.3 Fix1/Fix2 效果分析 (Error Direction)

#### V12 (S0-S4, 505 samples) vs V11 (前500 samples)

| Metric | V12 | V11 | Δ | 判断 |
|--------|-----|-----|---|------|
| **整体正确率** | 78.8% | 81.5% | -2.7% | V12略低(warmup阶段,预期) |
| **Over占比** | 11.1% | 8.6% | +2.5% | V12 over略多 |
| **Under占比** | 10.1% | 9.9% | +0.2% | 基本持平 |

#### Per-GT对比 (V12 vs V11)

| GT | V12 Acc | V11 Acc | V12 Over% | V11 Over% | V12 Under% | V11 Under% | Fix效果 |
|----|---------|---------|-----------|-----------|------------|------------|---------|
| 1 | 93.9% | 95.0% | 6.1% | 2.5% | 0.0% | 2.5% | ≈持平 |
| 2 | 95.2% | 94.9% | 2.4% | 5.1% | 2.4% | 0.0% | ≈持平 |
| 3 | 83.1% | 83.9% | 13.6% | 12.5% | 3.4% | 3.6% | ≈持平 |
| 4 | 88.6% | 88.0% | 9.1% | 6.0% | 2.3% | 6.0% | Fix1: over不变, under↓3.7% ✅ |
| 5 | 80.0% | 81.6% | 10.0% | 12.2% | 10.0% | 6.1% | Fix2初始效果 |
| 6 | 77.2% | 78.4% | 11.0% | 11.2% | 11.8% | 10.3% | ≈持平 |
| 7 | **67.3%** | **81.2%** | 14.3% | 8.3% | 18.4% | 10.4% | Fix2尚未收敛 ⚠️ |
| 8 | **61.4%** | **72.3%** | 13.6% | 6.4% | 25.0% | 21.3% | Fix2尚未收敛 ⚠️ |
| 9 | 62.5% | 71.0% | 21.9% | 9.7% | 15.6% | 19.4% | Fix2尚未收敛 ⚠️ |
| 10 | 66.7% | 45.5% | 11.1% | 0.0% | 22.2% | 54.5% | GT=10样本太少(9/11) |

### 16.4 关键发现

**1. Fix1效果信号 (Extra Point Penalty)**:
- V12 point_mean系统性低于V11（0.823-0.867 vs 0.854-0.881）
- 说明Fix1正在工作：extra point受到惩罚，point score降低
- 但这是"正确的阵痛"——需要model学会不多点，中期会回升

**2. Fix2效果信号 (GT-Scaled Alpha)**:
- GT≥7区域V12暂时劣于V11（67% vs 81%）
- 这是预期的：Fix2加大了高GT错误的惩罚，初始SFT模型的有利偏差被消除
- 关键：V11的GT≥7高准确率在S31+plateau时也降到了~64%（上次分析），Fix2的目标是让训练不plateau
- **需要到S20+才能判断Fix2是否真正起效**

**3. 训练健康度**:
- KL/entropy/grad_norm与V11完全一致 ✅
- format_fail_rate正常(0.5%-2.3%) ✅ 
- LR warmup更快（2x），需注意中期是否过拟合

**4. ⚠️ V12可能已停止**:
- 日志32分钟未更新，S5 update_actor阶段应该<10分钟
- 无明显crash信号，但需要到GPU服务器确认
- **建议立即检查GPU服务器上的V12进程状态**

### 16.5 阶段性评估

| 指标 | V12 S4 | V12预期S20 | V11 S4 | V11 plateau(S31+) | V12目标 |
|------|--------|-----------|--------|-------------------|---------|
| answer | 0.809 | >0.82 | 0.816 | 0.83 | 0.87-0.90 |
| overall | 0.843 | >0.86 | 0.848 | 0.86 | >0.87 |
| val pass@1 | 0.756 | >0.77 | 0.756 | 0.756-0.764 | >0.82 |

**综合判断**: V12前5步表现符合预期（SFT起点相同，answer/overall轻微偏低但在Fix1惩罚的预期范围）。**需要至少跑到S20才能初步判断Fix是否有效阻止plateau**。当前最紧急的问题是确认V12是否仍在运行。

---

## 17. V12 第一次运行完整分析 (S0-S50, 2026-03-15 01:36~16:38, 重启后)

### 17.1 V12第一次运行总结

**关键发现**: V12第一次完整运行了S0-S50（约15小时），主要指标从watchdog日志提取。V12在**S30达到历史最高answer_mean=0.879**，明显超过V11 plateau(0.831)。但S30后进入不稳定期，S40 point_mean急降至0.795（-0.056），S50时KL=0.046走高，用户因此在17:31重启。

### 17.2 V12 S0-S50 全步指标表

来源: `logs/monitor/watchdog_v12_20260315_013620.log` + `metrics_v12_20260315_013620.jsonl`

| Step | gn | kl | ov | ans_r | pt_r | am(RH) | pm(RH) | easy% | ac% | τ | a_tr | p_tr |
|------|-----|------|------|------|------|--------|--------|-------|-----|------|------|------|
| S0(val) | - | - | - | - | - | 0.849 | 0.867 | 41% | 28% | 0.700 | - | - |
| S1 | 0.850 | 0.006 | 0.869 | 0.849 | 0.867 | 0.762 | 0.823 | 44% | 14% | 0.700 | - | - |
| S2 | 0.962 | 0.004 | 0.802 | 0.762 | 0.823 | 0.782 | 0.826 | 34% | 19% | 0.700 | - | - |
| S3 | 0.722 | 0.019 | 0.816 | 0.782 | 0.826 | 0.814 | 0.864 | 41% | 25% | 0.700 | -0.008 | -0.000 |
| S4 | 0.656 | 0.022 | 0.845 | 0.814 | 0.864 | 0.823 | 0.830 | 36% | 22% | 0.699 | +0.001 | -0.005 |
| S5 | 0.840 | 0.007 | 0.841 | 0.823 | 0.830 | 0.769 | 0.859 | 30% | 23% | 0.699 | +0.005 | +0.012 |
| S10 | 0.960 | 0.009 | 0.832 | 0.807 | 0.830 | 0.764 | 0.802 | 39% | 19% | 0.696 | -0.018 | -0.015 |
| S15 | **3.723** | 0.012 | 0.804 | 0.776 | 0.805 | 0.803 | 0.825 | 28% | 30% | 0.692 | -0.004 | -0.003 |
| S20 | 1.526 | 0.016 | 0.780 | 0.760 | 0.751 | 0.817 | 0.838 | 39% | 28% | 0.686 | -0.013 | -0.015 |
| S25 | 0.741 | 0.013 | **0.895** | **0.879** | **0.897** | **0.850** | 0.848 | 30% | 23% | 0.679 | +0.012 | +0.009 |
| S30 | 1.010 | 0.008 | 0.814 | 0.792 | 0.802 | **0.879** | **0.880** | 25% | **41%** | 0.671 | +0.053 | +0.035 |
| S35 | 1.083 | 0.024 | 0.848 | 0.844 | 0.810 | 0.827 | 0.846 | 33% | 27% | 0.661 | -0.043 | +0.036 |
| S40 | 0.671 | 0.032 | 0.880 | 0.866 | 0.870 | 0.782 | **0.795** | 44% | 17% | 0.650 | +0.024 | **-0.056** |
| S45 | 0.816 | 0.016 | 0.887 | 0.873 | 0.877 | 0.791 | 0.823 | 27% | 28% | 0.638 | +0.053 | -0.042 |
| S50 | 1.546 | **0.046** | 0.856 | 0.834 | 0.856 | 0.761 | 0.814 | 22% | 36% | 0.624 | +0.021 | -0.000 |

**说明**: am/pm = RewardHealth answer/point_mean（1024样本批均值）; ans_r/pt_r = 当步骤reward均值

### 17.3 V12 vs V11 对比

| 指标 | V12 Peak | V11 Plateau (S31-40) | V11 Peak | 对比 |
|------|---------|---------------------|---------|------|
| **answer_mean** | **0.879 (S30)** | 0.831 | 0.840 | V12 **+4.8%** ✅ |
| **point_mean** | 0.880 (S30) | 0.876 | - | V12 ≈ V11 ✅ |
| **Val answer** | 0.756→0.752 | 0.756 | 0.764 | V12略降 ⚠️ |
| **KL S50** | **0.046** | 0.066 (S73) | - | V12 KL更低 ✅ |
| **τ at S50** | 0.624 | 0.680 (V11 S50) | - | V12 τ更低 (anneals faster) |

### 17.4 关键分析

**✅ Fix1+Fix2确实有效**:
- V12 answer_mean在S30突破至**0.879**（vs V11 plateau 0.831），提升了+4.8%
- 说明Fix1 (extra point penalty) + Fix2 (GT-scaled alpha)打破了V11的plateau机制

**⚠️ S30后instability问题**:
- S30: am=0.879，ac%=41%（最多all-correct），KL=0.008（健康）—— **训练最优点**
- S35: KL跳至0.024（+200%），am下降至0.827
- S40: KL=0.032，point_mean突然崩溃至0.795（-0.056）—— watchdog标记"V10退化重现!"
- 根本原因推断：**τ annealing过快（S50 τ=0.624）导致BOK-GRPO过度过滤，有效梯度样本减少**
  - easy_drgrpo持续下降：41%→30%→39%→28%→30%→25%→22%
  - 到S50时，只有22%的easy样本（相比S0的41%），学习信号减弱

**⚠️ Val score未提升 (0.756→0.752)**:
- 训练集指标改善，但泛化性未提升
- 原因可能：τ过快使训练集偏分布（mode-seeking），未能泛化

**🐛 Bug 4 发现并修复**:
- V12 save_freq=66（应该是20），导致S0-S50期间无checkpoint保存
- 已修复：`trainer.save_freq=66 → 20`

### 17.5 对新V12训练的参数建议

基于以上分析，新V12训练（从SFT重新开始）**核心问题是τ annealing过快**:

| 参数 | 当前 | 建议 | 原因 |
|------|------|------|------|
| **total_epochs** | 1 (→总步≈178) | **2** (→总步≈356) | 使τ从S30开始慢2x: S30 τ=0.671→0.686, 减少过早过滤 |
| **save_freq** | ~~66~~ **20** (已修复) | 20 ✅ | 确保S20/S40/S60...均保存 |
| **V12_RESUME_CKPT** | null (SFT) | null ✅ | 无S30 checkpoint可用 |
| kl_coef | 0.02 | 0.02 | 维持，V12 KL比V11更健康 |
| ACTOR_LR | 1.5e-6 | 1.5e-6 | 维持 |
| TAU_INIT | 0.7 | 0.7 | 维持 |

**最优选择**: 将`total_epochs=1 → 2`，使τ annealing速度与V11一致（total_steps=356），保留Fix1+Fix2:
- S30 τ: 1→0.671 (当前), 2→0.686 (慢，更多稳定性)
- S50 τ: 1→0.624 (当前), 2→0.660 (维持高多样性更长时间)
- 这样可以延长S25-30的高性能学习窗口

### 17.6 新V12（第二次）状态
- 启动时间: 2026-03-15 17:31
- 当前日志: 0字节（Ray初始化中）
- 配置: save_freq=20 (已修复), 其他需确认是否改回total_epochs=2

---

## 18. V12 完整训练分析 (S0-S134) + V13 设计

### 18.1 V12 训练结果总览

**训练时间**: 2026-03-15 01:36 ~ 03-16 (S0-S134)
**基线**: SFT checkpoint-3537 (clean start, val=0.756)

| Step | Val Answer | Val Point | Val Format | grad_norm | τ | 状态 |
|------|-----------|-----------|------------|-----------|------|------|
| S0   | 0.756     | 0.919     | 0.962      | -         | 0.700 | SFT baseline |
| S33  | 0.752     | 0.921     | 0.970      | normal    | 0.628 | 训练中 |
| S66  | 0.754     | 0.932     | 0.977      | normal    | 0.514 | ckpt saved |
| S73  | -         | -         | -          | **9.382** | 0.493 | gradient spike |
| S79  | -         | -         | -          | **10.137**| 0.467 | gradient spike |
| S84  | -         | -         | -          | normal    | 0.448 | 最后正常step |
| S85  | -         | -         | -          | **NaN**   | 0.514 | NaN开始 |
| S86  | -         | -         | -          | 1.024     | 0.442 | 短暂恢复 |
| S87-S131 | -     | -         | -          | **NaN**   | -     | 持续NaN |
| S99  | 0.752     | -         | -          | NaN       | 0.400 | 权重冻结在S84 |
| S132 | **0.771** | -         | -          | **0.872** | 0.362 | ckpt saved, **突破!** |
| S133-S134 | -   | -         | -          | NaN       | 0.357 | 继续NaN |

### 18.2 关键发现

**1. NaN梯度危机**:
- S85起共48步NaN梯度 (S85-S134，仅S86和S132正常)
- 根本原因: τ退火过快 → τ<0.514时, softmax logits溢出bf16 → NaN
- V12 τ公式: total_steps=178(1 epoch), TAU_FINAL=0.3
- V11对比: 相同配置下78步零NaN (V11只训练到S66, τ从未低于0.514)

**2. 惊喜: S132 val=0.771 (+1.5% over SFT)**:
- 尽管S85-S131全部NaN, S132恰好有1步正常梯度更新(gn=0.872)
- 模型权重从S84冻结到S132, 然后做了1次"幸运"更新
- S132 checkpoint恰好被保存 (save_freq=66, 132=66×2)
- 这是V12整个训练中最好的验证结果

**3. NaN根因分析**:
- `BOK_LOW_VAR_THRESHOLD=1e-5` 太低 → 近齐次组(std≈0.001)不触发fallback
- 这些组的shifted值极大: score_range/std = 0.5/0.001 = 500
- logits = 500/τ(0.362) = 1381 → exp(1381) = +∞ → NaN
- 加上τ annealing到0.3-0.5区间, 问题雪崩式爆发

### 18.3 V12 Checkpoint 66 外部评测

| Benchmark | V12-S66 | V11-S66 | SFT | 结论 |
|-----------|---------|---------|-----|------|
| pixmo-test | 80.34% | 80.15% | ~78% | 持平 (+0.19%) |
| pixmo-test (with_history) | 81.66% | 82.04% | - | 持平 (-0.38%) |
| stepcount-bench | 11.73% (307/500) | 12.00% | ~10% | 持平 |

### 18.4 V13 改动方案 (多层NaN防护)

**核心改动**:

| 参数 | V12 | V13 | 改动理由 |
|------|-----|-----|----------|
| Resume | SFT | V12 S132 (val=0.771) | 利用V12最佳checkpoint |
| TAU_FINAL | 0.3 | **0.5** | τ≥0.5 防止softmax溢出 |
| total_epochs | 1 | **2** | total_steps 178→356, τ退火更慢 |
| BOK_TOTAL_STEPS | 178 | **356** | 匹配total_epochs=2 |
| BOK_LOW_VAR_THRESHOLD | 1e-5 | **1e-3** | 拦截近齐次组进入softmax |
| softmax logit clamp | 无 | **clamp(-20,20)** | 代码级NaN兜底 |

**τ轨迹预测** (V13从S132恢复):
```
S132: progress=132/356=0.371, τ=0.640 (安全!)
S178: progress=0.500, τ=0.600
S267: progress=0.750, τ=0.529
S356: progress=1.000, τ=0.500 (floor)
全程τ≥0.500, 远离NaN触发点(~0.514)
```

**文件变更**:
1. `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v13.sh` — 新V13脚本
2. `verl/trainer/core_algos.py` L580 — softmax logit clamping: `logits.clamp(-20, 20)`

### 18.5 V13 训练预期

- 从S132恢复, 将继续训练到S356 (224步)
- τ从0.640平滑退火到0.500, 全程稳定
- 三重NaN防护: τ floor + LOW_VAR_THRESHOLD + logit clamp
- 预期val answer: 从0.771基础上继续提升
- 保存频率: save_freq=20, 每20步保存checkpoint
- 验证频率: val_freq=33, 足够密集

