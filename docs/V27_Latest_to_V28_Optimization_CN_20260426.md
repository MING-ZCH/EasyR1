# V27 最新训练进展诊断与 V28 优化方案（2026-04-26）

## 0. 结论摘要

V27 最新日志已经从“早期可观察”进入“明确失败”阶段：

- V27 已到 step 159/213，但验证集 answer 从 step 0 的 `0.7561` 下降到 step 140 的 `0.7316`。
- step 92 起出现 non-finite gradient；完整日志复核后累计 `132` 条 non-finite skip，且有 `7` 次 EMA spike。
- entropy 长期停留在 `0.94-1.02`，已经脱离 V23 稳定区间 `0.49-0.55`。
- strict JSON 有效避免了 v26 式 format cliff；最新 format fail 约 `1-3%`，不是主因。
- smart_filter 仍然有效；最新 `ac_released=48/1024`，说明它仍在释放 answer-correct / point-low 组的 point 学习信号。
- 新发现一个训练保护代码问题：non-finite emergency brake 打印“永久降 LR”，但 scheduler/cooldown 后仍会把 LR 恢复到原始 `1.5e-6`，导致 non-finite skip burst 持续发生。

因此：

> V27 不建议继续作为主线；也没有值得保留的优质 checkpoint。V28 应保留 `smart_filter=0.955`，但必须使用 V23-stable LR/KL、原格式非截断 process prompt、strict JSON、临时 cooldown-only 梯度保护，以及 fail-fast 停训，而不是默认永久降 LR。

当前 V28 主线推荐：

```text
ACTOR_LR=1e-6
KL_COEF=0.03
BOK_SMART_FILTER_THRESHOLD=0.955
TRAJ_POINT_STRICT_JSON=1
ppo_epochs=1
max_grad_norm=1.0
GRAD_SPIKE_ABSOLUTE_CAP=0
GRAD_SPIKE_BRAKE_MAX=999
GRAD_NONFINITE_BRAKE_MAX=999
original non-truncated process prompt
```

## 1. V27 最新验证曲线

V27 最新验证结果如下：

| step | answer / reward_score | point diagnostic | format_fail | 判断 |
|---:|---:|---:|---:|---|
| 0 | 0.7561 | 0.7092 | 0.0378 | SFT 起点 |
| 20 | 0.7448 | 0.7130 | 0.0321 | 下降 |
| 40 | 0.7429 | 0.7161 | 0.0359 | 继续低于起点 |
| 60 | 0.7524 | 0.6523 | 0.3554 | format 瞬时尖峰 |
| 80 | 0.7316 | 0.7179 | 0.0284 | 明显下降 |
| 100 | 0.7108 | 0.7238 | 0.0227 | 崩到最低 |
| 120 | 0.7335 | 0.7207 | 0.0284 | 未恢复 |
| 140 | 0.7316 | 0.7239 | 0.0227 | 仍低 |

关键判断：

1. V27 没有任何一次验证超过 step 0 baseline。
2. V27 没有超过 V12/V23，甚至没有保持 SFT 起点。
3. point diagnostic 稍升，但 answer 降，说明训练信号没有转化为泛化 count accuracy。
4. format 不再像 v26 那样变成 `0.9-1.0` cliff；strict JSON 是有效修复。

## 2. V27 最新训练健康状态

监控输出显示：

```text
V27 step 159/213
ALERTS: NaN GRAD x132, ENTROPY HIGH (0.94-1.02)
Reward: overall=0.766 answer=0.722 point=0.778 format=0.988
NaN Skip: 132 full-log non-finite skip lines
FormatFail: avg=1.7%-1.9%, max≈3.1%-4.7%
SmartFilter: allcorrect_filtered=144/1024, ac_released=48/1024, easy_drgrpo=416/1024
```

最近 5 step 趋势：

| metric | step 150-154 | step 155-159 | 变化 |
|---|---:|---:|---:|
| overall | 0.7932 | 0.7752 | -0.0180 |
| answer | 0.7484 | 0.7284 | -0.0200 |
| point | 0.8190 | 0.7992 | -0.0198 |
| format | 0.9824 | 0.9840 | +0.0016 |
| entropy | 0.9584 | 0.9592 | +0.0008 |
| KL | 0.1582 | 0.1722 | +0.0140 |
| grad_norm | 0.1966 | 0.0000 | non-finite/skip 影响 |

解释：

- V27 的 train point 曾经改善，但后期也开始回落。
- answer 与 validation 同向下降，说明不是 eval 噪声。
- format 已稳定，因此主问题是 optimization instability / entropy drift / non-finite skip。

## 3. non-finite 与 entropy 时间线

### 3.1 Entropy 先失控，再出现 non-finite

日志里 entropy 里程碑：

```text
first entropy > 0.70: line 55383, entropy_loss=0.702
first entropy > 0.80: line 58479, entropy_loss=0.817
first entropy > 0.90: line 67206, entropy_loss=0.954
first entropy > 1.00: line 88528, entropy_loss=1.019
```

### 3.2 non-finite 从 opt_step=92 开始

关键日志：

```text
opt_step=92  first non-finite gradient
opt_step=98  nonfinite_count=3 -> emergency brake
opt_step=110+ non-finite 持续出现
NaN Skip total = 132 (full-log audit)
```

这说明：

1. entropy drift 是更早的前兆；
2. non-finite 是后期结果；
3. 仅靠 non-finite 后处理已经太晚；
4. V28 应该通过低 LR、稳定 KL、临时 cooldown、fail-fast 早停来前置控制；默认不应对单个异常 step 永久降 LR。

## 4. strict JSON 与 smart_filter 的归因

### 4.1 strict JSON：有效

V27 没有复现 v26 的 format cliff：

- v26 后期 `format_fail≈0.9-1.0`；
- v27 最新 format fail 大多 `1-3%`；
- validation 除 step 60 一次尖峰外，format 基本低。

所以 strict JSON 的方向正确，应继续保留。

### 4.2 smart_filter：不是主罪魁，应保留

V27 最新仍有：

```text
allcorrect_filtered=144/1024
ac_released=48/1024
easy_drgrpo=416/1024
```

`ac_released` 非零，说明 `smart_filter=0.955` 正在释放 answer-correct / point-low group，让这些样本继续学习 point quality。

第一性原理上：

```text
overall_mean = 0.6 * answer + 0.3 * point + 0.1 * format
当 answer=1 且 format=1 时：overall_mean = 0.7 + 0.3 * point
0.955 = 0.7 + 0.3 * 0.85
```

因此 `smart_filter=0.955` 的含义是：

- point 均值 ≥ 0.85：真正 mastered，过滤；
- point 均值 < 0.85：answer 虽正确但 point 不够好，释放继续训练。

这正好对应 StepCount 的核心瓶颈：answer 对但点不准会导致长期误差累计。

结论：

> V27 失败不是 smart_filter 本身失败，而是 high LR + entropy drift + non-finite skip + LR brake 不持久共同导致。

## 5. 新发现：emergency brake 不持久的问题

V27 日志显示：

```text
[GradSpikeProtect] NONFINITE EMERGENCY BRAKE: ... Base LR permanently halved.
...
Cooldown ended. LR restored to 1.50e-06
```

这意味着 scheduler/cooldown 逻辑会在后续 step 把 LR 恢复到原始 high LR。也就是说，“永久降 LR”在日志语义上成立，但实际 optimizer LR 会被 scheduler 覆盖。

这会导致：

1. non-finite 出现后没有真正降低风险；
2. skip burst 反复发生；
3. v27 后 44% step 被 NaN skip 消耗；
4. 后期训练基本失去有效优化意义。

### 5.1 已完成的代码修复/恢复

已恢复为 cooldown-only 主策略：V28 主脚本默认关闭 emergency LR-drop brake。
单个异常 step 的处理顺序是：

1. skip 当前 bad update；
2. 进入 temporary cooldown，用 `GRAD_SPIKE_LR_FACTOR=0.1` / `GRAD_NONFINITE_LR_FACTOR=0.1` 降低随后少数 step 的更新强度；
3. 若异常重复出现，由 fail-fast monitor 写入 stop sentinel 并终止训练；
4. 不对单个 outlier 永久/全局降 LR。

```text
scheduler.step()
apply _lr_brake_factor
then apply cooldown lr factor if needed
```

这样 scheduler 不会再把 brake 后的 LR 恢复到原始值。

## 6. V28 具体优化决策

### 6.1 保持不变的核心参数

| 参数 | V28 值 | 原因 |
|---|---:|---|
| `ACTOR_LR` | `1e-6` | 回到 V23 稳定区间，避免 v27 high-LR drift |
| `KL_COEF` | `0.03` | V23 稳定值；先不升到 0.05 以免过保守 |
| `BOK_SMART_FILTER_THRESHOLD` | `0.955` | 保留 point-quality 梯度恢复 |
| `ppo_epochs` | `1` | 避免 stale rollout 放大错误 |
| `max_grad_norm` | `1.0` | V23 稳定配置 |
| `TRAJ_POINT_STRICT_JSON` | `1` | 防止旧 parser reward hacking |
| `TRAJ_FORMAT_REJECTION` | `1` | 格式错直接零 reward |
| `GRAD_SPIKE_ABSOLUTE_CAP` | `0` | 避免后验 skip cap 造成训练碎片化 |

### 6.2 新增/强化的 V28 控制

| 改动 | 当前值 | 作用 |
|---|---:|---|
| fixed process prompt | 已修复 | 避免 process turn 缺少最终 answer 指令 |
| emergency LR brake | V28 默认关闭 | 单个异常 step 只 skip + temporary cooldown，不永久/全局降 LR |
| `GRAD_SPIKE_BRAKE_MAX` / `GRAD_NONFINITE_BRAKE_MAX` | `999` / `999` | 默认不触发 emergency LR drop；只使用 skip + cooldown |
| `KL_COEF` 脚本化 | 默认 `0.03` | 便于后续用 `KL_COEF=0.05` 快速对照 |

### 6.3 暂不建议的改动

| 暂不改 | 原因 |
|---|---|
| 关闭 smart_filter | 证据不支持；会损失 point-low all-correct 组的梯度 |
| 立刻把 KL 默认升到 0.05 | 可能过度限制学习；先用 LR=1e-6 + 原格式非截断 prompt + cooldown/fail-fast 验证 |
| 降 rollout temperature | 会削弱 BoK pass@K→pass@1 的核心机制 |
| 开启 point-aware answer gate | 属于下一阶段 reward-shape ablation，不应混入 V28 主线 |
| hard-only 或大比例 wrong replay | 先确认优化稳定性，再做数据强化 |

## 7. V28 运行判据

### 7.1 step 0-40

健康标准：

- entropy 不应快速超过 `0.70`；
- format_fail train/val 均应 `<0.1`；
- `ac_released` 应非零，理想 `3-8%`；
- 无 non-finite；
- validation 至少不应持续低于 step 0。

若 step 40 前 entropy > `0.75`，建议停止并分支：

```text
KL_COEF=0.05 或 BOK_EASY_SCALE=0.7
```

### 7.2 step 40-100

健康标准：

- entropy 仍应大致在 `0.50-0.70`；
- validation 应出现超过 step 0 的趋势；
- train point 可以升，但 answer 不能同步下降；
- non-finite 必须为 0。

若出现一次 non-finite：

- 默认不会因单个异常 step 永久降 LR；
- 但仍应记录该 checkpoint 之后是否继续有效；
- 若 10 step 内连续 non-finite，应停止该 run。

### 7.3 成功标准

V28 不是只看 train reward，而是：

1. `val/reward_score` 超过 `0.7561` step0；
2. 第一目标超过 V23 peak `0.7769`；
3. 第二目标超过 V12 sparse direct `0.7750` 且无 format / NaN 问题；
4. entropy 不超过 `0.80`；
5. 保存有效 checkpoint，而不是等到 drift 后 final。

## 8. 如果 V28 仍失败，下一步 ablation 顺序

### A. entropy 仍涨但无 non-finite

优先：

```text
KL_COEF=0.05
```

或：

```text
BOK_EASY_SCALE=0.7
```

解释：如果 smart_filter 释放的 Easy/Dr.GRPO 梯度太强，先缩放 Easy path，而不是关闭 smart_filter。

### B. ac_released 过高并伴随 entropy 增长

做 dynamic smart threshold：

```text
目标：ac_released 维持在 3-8%
过高则提高 threshold
过低则降低 threshold
```

### C. answer 不升但 point 升

说明 point 信号没有有效转化为 count answer。下一步做：

```text
TRAJ_ANSWER_GATE_MODE=soft
TRAJ_SOFT_GATE_BASE=0.85 或 0.90
```

### D. sparse 仍卡在 0.77-0.78

说明优化稳定但数据没有覆盖真实错误。下一步做：

- mined wrong replay；
- count 7-10 高错误样本低比例混入；
- 优先 mask-backed；
- maskless 样本用 `stat_mask_sim`，避免 `count_iou` 膨胀。

## 9. 最终推荐

V27 最新进展已经证明：

- strict JSON 修好了 format cliff；
- smart_filter 仍提供有价值的 point 梯度；
- high LR 造成 entropy drift；
- non-finite 发生后原 emergency brake 不持久，导致大量 skip；
- v27 没有产生优质 checkpoint，不建议继续作为主线。

V28 应按以下主线启动：

```text
V28 = V23-stable LR/KL
    + smart_filter=0.955
    + strict JSON
    + fixed complete process prompt
    + temporary cooldown + fail-fast monitor
    + dense validation/save
```

这比“关闭 smart_filter”更符合当前证据，也更贴合 StepCount 的第一性原理：将 answer-correct 但 point-low 的轨迹继续转化为 point-quality 学习信号，同时用稳定优化参数防止 entropy 和 non-finite。
