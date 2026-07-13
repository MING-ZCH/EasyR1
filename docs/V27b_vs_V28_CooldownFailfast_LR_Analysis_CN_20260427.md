# V27b 当前保护方案 vs V28 cooldown-only + fail-fast，以及 2e-6 / 1.5e-6 LR 选择分析（2026-04-27）

## 0. 结论

对于主线 V28，`cooldown-only + fail-fast` 是正确方向；V27b easy-data 也已对齐到该保护策略。现在二者的核心差别不再是保护策略，而是数据和 LR：V28 是 easy+hard + `1e-6` 主线，V27b 是 easy-only + `2e-6` stress ablation。原因是 StepCount trajectory RL 的主要风险不是“单个大梯度本身”，而是大梯度背后的状态已经进入坏分布：format fail、zero reward、low variance、entropy drift、non-finite。单个异常 step 应该 skip 并短期 cooldown；如果异常重复出现，应停训分支，而不是继续训练或永久降 LR。

LR 选择上：

```text
主线：1e-6 最优
高 LR ablation：1.5e-6 优于 2e-6
2e-6：只适合极短 smoke / upper-bound ablation，不适合作为主线
```

如果只能在 `2e-6` 和 `1.5e-6` 之间选，选择 `1.5e-6`。但当前总目标是超过 V12 且具备 V23 稳定性，因此主线仍应使用 `1e-6`。

## 1. 两种保护策略的本质区别

### 1.1 V27b 当前方案（已对齐 cooldown-only + fail-fast）

V27b easy-data 当前脚本特点：

- easy-only 数据；
- `ACTOR_LR=2e-6`；
- `TRAJ_POINT_STRICT_JSON=1` 已修正；
- `BOK_SMART_FILTER_THRESHOLD=0.955`；
- `GRAD_SPIKE_THRESHOLD=3.0`；
- `GRAD_SPIKE_LR_FACTOR=0.1`；
- `GRAD_SPIKE_BRAKE_MAX=999`；
- `GRAD_NONFINITE_BRAKE_MAX=999`；
- `V27B_FAILFAST_ENABLE=1`；
- 已加入 fail-fast stop watcher；
- 仍是 high-LR upper-bound ablation，不是主线稳定脚本。

问题：

1. `2e-6` 在 StepCount trajectory RL 中没有足够日志证明可稳定超过 V12/V23；
2. 保护策略已与 V28 对齐，因此剩余主要风险来自 high LR 和 easy-only 数据分布；
3. 它可以回答 high-LR/easy-only 是否短期可稳定，但不能证明主线最优。

### 1.2 V28 cooldown-only + fail-fast

V28 当前主线特点：

- easy+hard 数据；
- `ACTOR_LR=1e-6`；
- `ppo_epochs=1`；
- `KL_COEF=0.03`；
- `TRAJ_POINT_STRICT_JSON=1`；
- 原格式非截断 process prompt；
- `BOK_SMART_FILTER_THRESHOLD=0.955`；
- `GRAD_SPIKE_ABSOLUTE_CAP=0`；
- `GRAD_SPIKE_LR_FACTOR=0.1`；
- `GRAD_NONFINITE_LR_FACTOR=0.1`；
- `GRAD_SPIKE_BRAKE_MAX=999`；
- `GRAD_NONFINITE_BRAKE_MAX=999`；
- `V28_FAILFAST_ENABLE=1`。

本质：

```text
单个 bad update：skip + short cooldown
重复 bad state：fail-fast stop
不因一个 outlier 永久降 LR
不让已经坏掉的 run 继续训练
```

这更符合 StepCount 的失败模式：format / entropy / reward support 一旦进入坏分布，继续训练通常不是恢复，而是加速退化。

## 2. 为什么 cooldown-only + fail-fast 更优

### 2.1 单个 outlier 不代表全局 LR 错误

RL batch 具有高方差，尤其 StepCount trajectory 包含多轮 point、stop、answer，某个 batch 出现大梯度可能只是：

- 该 batch object 数更难；
- 该 group reward 低方差；
- all-wrong / all-correct routing 边界样本较多；
- point 偏移造成 answer 错误累计；
- prompt/format 生成偶然偏移。

如果只因为一个 outlier 就永久降 LR，会降低后面所有健康 batch 的学习效率。

### 2.2 重复异常代表状态已坏，应停训而非继续

V27 的关键教训是：first non-finite 不是孤立事件，而是 entropy 已经先漂移。继续训练只会得到更多 NaN skip 和更差 validation。

因此最优策略不是“永久降低 LR 继续跑”，而是：

```text
发现重复 entropy/format/non-finite 异常
→ 保存日志
→ 停训
→ 分支调整 KL/LR/smart_filter scale/prompt/reward
```

### 2.3 fail-fast 比 silent recovery 更适合研究迭代

当前目标不是稳定跑完任意一个 run，而是找出能超过 V12 且具有 V23 稳定性的配置。fail-fast 能更快判定：

- 当前参数是否进入错误区域；
- 是 LR 问题、KL 问题、format 问题还是 reward routing 问题；
- 是否需要开 `KL_COEF=0.05` / 降 `BOK_EASY_SCALE` / 改 smart filter schedule。

## 3. LR=2e-6 vs 1.5e-6

### 3.1 2e-6 的证据不足且风险更高

当前可见日志里，`2e-6` 的代表是 V12Bp 类实验：

- 有 `38` 条 non-finite / not-finite 相关记录；
- validation peak 约 `0.7694@40`；
- 没有超过 V12 baseline `0.7750`；
- 没有超过 V23 `0.7769`；
- 不能证明 `2e-6` 是健康高效配置。

V27b easy-data 目前没有足够完整训练日志证明 `2e-6` 在 strict JSON + smart filter + cooldown-only/fail-fast 下可稳定提升。因此它只能作为 upper-bound ablation。

### 3.2 1.5e-6 有高 ceiling 迹象，但失败概率仍高

`1.5e-6` 的证据更复杂：

- V25 达到 `0.7826@150`，但发生 format collapse，因此不是健康成功；
- V24 使用 `1.5e-6 + KL=0.02 + smart_filter=0.955`，有一定稳定性但后期 entropy 风险；
- V27 使用 `1.5e-6 + strict JSON + smart_filter=0.955`，format cliff 被避免，但 validation 下降到 `0.7164@180`，non-finite 达到 `132` 条。

所以 `1.5e-6` 比 `2e-6` 更有实验依据，但仍不适合作为 V28 主线。

### 3.3 主线为什么仍选择 1e-6

V23 证明：

- `lr=1e-6`；
- `ppo_epochs=1`；
- `KL_COEF=0.03`；
- no abs cap；
- best parsed val `0.7769@135`；
- `0` non-finite lines。

这正好匹配用户目标：超过 V12 且保持训练稳定。V28 的创新不应放在 high LR 冒险上，而应放在 strict JSON + smart_filter point-quality gradient recovery + fail-fast 保护上。

## 4. 推荐实验优先级

### Priority 1：V28 main

```text
lr=1e-6
cooldown-only + fail-fast
strict JSON
smart_filter=0.955
easy+hard
```

目标：复现/超过 V23 的稳定性，并超过 V12 answer。

### Priority 2：V28-highLR-1.5e-6 ablation

如果 V28 main 前 40-80 step 健康但学习速度偏慢，再开：

```text
ACTOR_LR=1.5e-6
保持 cooldown-only + fail-fast
保持 KL=0.03
保持 strict JSON
保持 smart_filter=0.955
```

这是高 LR 对照的首选，不是 `2e-6`。

### Priority 3：V27b easy-data strict-json ablation

V27b easy-data 已保留为：

```text
easy-only
lr=2e-6
strict JSON
smart_filter=0.955
cooldown-only + fail-fast
```

但它应该被标记为 upper-bound / stress test，只能用于回答：easy-only + high LR 是否能在短期内稳定，而不是主线追分。

## 5. 最终判断

| 方案 | 适合作为主线？ | 原因 |
|---|---:|---|
| V27b guard-aligned 方案 | 否 | 已有 cooldown-only + fail-fast，但仍是 easy-only + lr=2e-6 stress test，证据不足 |
| V28 cooldown-only + fail-fast | 是 | 单个 outlier 不永久降 LR，重复坏状态自动停训，最符合 V27/V26 失败模式 |
| lr=2e-6 | 否 | 证据不足且 non-finite 风险高 |
| lr=1.5e-6 | 可做 ablation | 比 2e-6 更有依据，但仍不能做主线 |
| lr=1e-6 | 是 | V23-stable backbone，最符合“超过 V12 且稳定”的目标 |

最终建议：

```text
先跑 V28 main: lr=1e-6
如果健康但慢，再开 V28-highLR-1.5e-6 ablation
不要把 V27b lr=2e-6 作为主线
```
