# V27 NaN / non-finite 问题第一性原理根因分析（2026-04-26）

## 0. 一句话结论

V27 的 NaN 不是单点原因，而是一条因果链：

```text
高 LR=1.5e-6
+ process prompt 截断导致中期 format/stop 波动
+ strict JSON + format rejection 把格式错样本变成大量 zero-reward
+ BoK/DrGRPO 在低方差/零奖励混合 batch 上产生尖峰梯度
+ entropy 从 0.5 稳定区间漂到 0.9-1.0
+ opt_step=92 起出现 non-finite gradient
+ emergency brake 被 scheduler/cooldown 恢复 LR，导致 NaN skip burst 持续
```

所以，V27 的 NaN 不是 `smart_filter=0.955` 直接造成的，也不是 mixed data 单独造成的。它是 **高学习率下的格式波动—零奖励分布—梯度尖峰—熵漂移—non-finite** 连锁反应。

## 1. 现象层：NaN 从哪里开始

V27 最新日志显示：

```text
first non-finite gradient: opt_step=92
NaN Skip total: 132 full-log non-finite skip lines
NaN Skip ratio: high; previous step-159 monitor snapshot was 88/198=44.4%
first emergency brake: opt_step=98, nonfinite_count=3
latest entropy: about 0.94-1.02
```

关键上下文：

```text
step=92
RewardHealth overall=0.7302 answer=0.6767 point=0.7581 format_fail_rate=0.0322
BoK batch_mean=0.7302 batch_std=0.3240 low_var=144/1024 collapsed=4
Gradient norm is not finite. Skip update. opt_step=92 nonfinite_count=1 lr=1.50e-06
entropy_loss=0.987
kl_loss=0.172
pg_loss=0.028
```

说明 non-finite 发生时：

- format 已经不是主崩溃点；
- entropy 已经接近 `1.0`；
- KL 已经较高；
- reward 分布仍有较大 std，但策略分布已经进入高熵不稳定区。

## 2. 时间线：NaN 前已经有明显前兆

### 2.1 Entropy 先升高

日志里 entropy 里程碑：

```text
first entropy > 0.70: entropy_loss=0.702
first entropy > 0.80: entropy_loss=0.817
first entropy > 0.90: entropy_loss=0.954
first entropy > 1.00: entropy_loss=1.019
```

这说明 NaN 不是突然出现，而是 entropy 先逐步脱离 V23 稳定区间。

V23 稳定区间：

```text
entropy ≈ 0.49-0.55
```

V27 后期：

```text
entropy ≈ 0.94-1.02
```

### 2.2 梯度尖峰早于 non-finite

在 non-finite 之前，V27 已经出现多次 spike：

```text
spike #3: step=45, grad_norm=4.5303, ratio=3.79x
spike #4: step=48, grad_norm=9.0235, ratio=7.55x
spike #5: step=49, grad_norm=4.1661, ratio=3.49x
```

这组 spike 是关键转折点。它发生在 entropy 真正爆高之前，因此更像是后续 entropy drift 的触发器之一。

## 3. 第一性原理：为什么 format 波动会触发梯度尖峰

StepCount 训练 reward 是：

```text
overall = 0.6 * answer + 0.3 * point + 0.1 * format
```

并且开启：

```text
TRAJ_FORMAT_REJECTION=1
TRAJ_POINT_STRICT_JSON=1
```

这意味着：

1. 输出格式正确时，answer/point/format 都有 reward；
2. 输出格式错时，format=0，且 format rejection 可把 overall 直接置零；
3. strict JSON 会阻止旧 parser fallback 给 malformed point 继续发 point reward。

这本身是正确的，但在 prompt 不完整或高 LR 导致格式抖动时，会让一部分样本突然变成 zero-reward。

### 3.1 V27 step 45-49 出现格式/零奖励异常

日志显示：

```text
step=45 format_fail_rate=0.1816, grad spike #3
step=47 format_fail_rate=0.1982
step=48 format_fail_rate=0.1445, grad spike #4
step=49 format_fail_rate=0.2725, zero_reward_rate=27.2%, low_var_rate=35.9%, grad spike #5
```

特别是 step 49：

```text
[BoK-GRPO][WARNING] step=49 low_var_rate=35.9% zero_reward_rate=27.2%
```

这说明 batch 里大量 trajectory 被 format rejection 或停止异常打成零分。

### 3.2 为什么 zero-reward 会让梯度危险

BoK-GRPO/DrGRPO 的 advantage 依赖组内相对差异。若一个 batch 中出现：

- 一部分轨迹 reward=0；
- 一部分轨迹 reward 正常；
- 若干 group 低方差；
- 若干 group 有极端 winner；

则 advantage 分布会突然变尖。即使最终有 `BOK_CLIP=4.0`，大量 token 上的策略梯度仍可能在某些 batch 中变得很大。

第一性原理上，这等价于：

```text
原本平滑的 reward landscape
→ 被 format rejection 切成离散的 0/正常两簇
→ group-relative advantage 对少数 winner 施加强正梯度
→ 高 LR 下参数更新过大
→ policy entropy / KL 进入不稳定区
```

## 4. process prompt 截断是重要放大因素

本轮检查发现 active process prompt 原本被截断，结尾停在：

```text
If all target objects have been marked with red dots and counting i
```

这意味着中间轮次 prompt 缺少完整的最终 answer 指令。对 trajectory 任务来说，这是高影响问题，因为 process prompt 会在每一轮反复使用。

它可能导致：

1. 该输出 `<answer>` 时仍继续 `<point>`；
2. 停止格式不稳定；
3. `stop_violation_rate` / `format_fail_rate` 在中期升高；
4. strict JSON + format rejection 将这些错误转为 zero-reward；
5. zero-reward batch 触发 gradient spike。

V28 已修复该 prompt，明确要求：

```text
如果全部目标已标红，输出 <think>All target objects have been counted.</think><answer>n</answer>
并且 </answer> 后不输出额外文本。
```

## 5. 为什么不是 smart_filter 直接导致 NaN

`smart_filter=0.955` 的作用是：

```text
all-answer-correct 且 point_mean < 0.85 的 group 不过滤，释放到 Easy/DrGRPO 继续学 point
```

V27 最新仍显示：

```text
allcorrect_filtered=144/1024
ac_released=48/1024
easy_drgrpo=416/1024
```

这说明 smart_filter 仍在做正确的事情：给 answer-correct / point-low 组提供 point-quality 梯度。

如果 smart_filter 是直接主因，应看到：

- ac_released 异常高；
- format 不波动但 entropy 直接随 ac_released 爆炸；
- 关闭 smart_filter 的类似配置稳定。

但现有证据不是这样：

1. step 45-49 的直接异常是 format_fail / zero_reward / low_var / spike；
2. smart_filter release 量在正常范围；
3. v24/v27 的共同风险更像高 LR / KL / update dynamics；
4. v23 稳定但没有 smart_filter，问题是它也可能损失 point-gradient ceiling。

因此：

> smart_filter 是需要保留但需要配合稳定 LR/KL 的有效机制，不是 v27 NaN 的第一根因。

## 6. 为什么 high LR 是核心放大器

V27 使用：

```text
actor_lr=1.5e-6
ppo_epochs=1
kl=0.03
max_grad_norm=1.0
```

相比 V23：

```text
actor_lr=1e-6
ppo_epochs=1
kl=0.03
max_grad_norm=1.0
```

唯一核心优化强度差异就是 LR 增大 50%。

在 reward 平滑时，`1.5e-6` 可能只是更快；但在 step 45-49 这种 format/zero-reward 波动时，它会把尖峰 advantage 转化成过大的 policy update。

这解释了：

```text
step 45-49 梯度尖峰
→ step 70+ entropy > 0.8
→ step 92 first non-finite
```

也解释为什么 V28 不应该继续 `1.5e-6`。

## 7. 为什么 emergency brake 没救住 V27

V27 触发了：

```text
NONFINITE EMERGENCY BRAKE
```

但日志随后出现：

```text
Cooldown ended. LR restored to 1.50e-06
```

这说明 scheduler/cooldown 覆盖了所谓“永久降 LR”。结果是：

1. 第一次 non-finite 后没有真正进入低 LR 区；
2. 后续仍以高 LR 反复进入 non-finite；
3. NaN skip 累积到 88；
4. 44.4% optimizer step 被浪费。

这不是 reward 层面的根因，而是训练保护机制的实现漏洞。

V28 已恢复为 cooldown-only 主策略：

```text
skip bad update
temporary cooldown LR factor = 0.1
GRAD_SPIKE_BRAKE_MAX=999
GRAD_NONFINITE_BRAKE_MAX=999
fail-fast stops repeated entropy/format/non-finite failures
```

## 8. 因果链总结

完整链条如下：

```text
1. V27 用 lr=1.5e-6，更新强度高于 V23。
2. process prompt 截断，使中间轮次停止/answer 指令不完整。
3. strict JSON + format rejection 正确执行，但把格式波动样本置零。
4. step 45-49 出现 format_fail 14%-27%、zero_reward 27%、low_var 36%。
5. BoK/DrGRPO advantage 在这种离散 reward 分布上变尖。
6. 高 LR 把尖峰梯度转成过大的 policy update，出现 grad spike #3-#5。
7. entropy 随后从 0.5 区间升到 0.8、0.9、1.0。
8. 高 entropy / 高 KL / 不稳定 logits 下，FSDP grad_norm 返回 non-finite。
9. emergency brake 被 scheduler 恢复 LR，导致 non-finite skip burst。
10. 训练有效步数大幅下降，validation 持续低于 baseline。
```

## 9. 对 V28 的直接启示

V28 应针对链条中的每个环节前置修复：

| V27 问题 | V28 对策 |
|---|---|
| `lr=1.5e-6` 放大尖峰 | `ACTOR_LR=1e-6` |
| process prompt 截断 | fixed complete process prompt |
| 格式波动变 zero-reward | strict JSON 保留，但 prompt 降低格式波动 |
| zero-reward/low-var 触发 spike | 保留 GradSpikeProtect skip + cooldown；`GRAD_SPIKE_BRAKE_MAX=999` 默认不触发 emergency LR drop |
| non-finite 后处理策略 | V28 默认 skip bad update + temporary cooldown；repeated non-finite 用 fail-fast 停训，不对单个 outlier 永久降 LR |
| entropy 早涨无硬停止 | step 40 前监控 entropy，必要时 `KL_COEF=0.05` |
| smart_filter 被误判 | 保留 `smart_filter=0.955`，必要时调 `BOK_EASY_SCALE` 而非关闭 |

## 10. 最终判断

V27 NaN 的第一性原理解释是：

> 高 LR 在格式/零奖励分布波动时放大了 group-relative policy gradient，先造成梯度尖峰和 entropy drift，再在高熵/高 KL 状态下触发 non-finite；而 emergency brake 不持久导致 NaN skip burst 无法停止。

因此 V28 的正确方向不是简单关闭某个算法组件，而是：

```text
降低更新强度
修复 process prompt
保留 smart_filter 的 point 学习优势
V28 默认使用 temporary cooldown + fail-fast，不再依赖永久降 LR
用早期 entropy / format / low_var 监控做 fail-fast
```
