# BoK-GRPO Trajectory-Level 与 Step-Level Point-to-Count Credit Assignment 兼顾方案（2026-04-27）

## 0. 结论先行

结论：**可以兼顾，但不能只靠当前 `PROCESS_REWARD_ENABLE=1`。**

当前系统已经有两部分能力：

1. Reward worker 可以把每步 point reward 放在对应 `</point>` token 附近。
2. 当前 BoK-GRPO 会把所有 token reward 求和成整条 trajectory score，再做 pass@K → pass@1 的 trajectory-level 优化。

因此，在当前 `ADV_ESTIMATOR=bok_grpo` 下，即使打开 `PROCESS_REWARD_ENABLE=1`，BoK-GRPO 仍会执行：

```text
scores = token_level_rewards.sum(dim=-1)
```

这会把 per-step reward 重新聚合为 trajectory-level scalar。最终 advantage 仍然是整条 trajectory 级别，不能真正做到“第 k 步 point 错，就主要惩罚第 k 步生成 token”。

但可以新增一个 hybrid advantage estimator：

```text
ADV_ESTIMATOR=bok_grpo_step
PROCESS_REWARD_ENABLE=1
```

核心思想：

```text
final_advantage = trajectory_BoK_advantage + λ * gated_step_advantage
```

其中：

- `trajectory_BoK_advantage` 继续由最终 trajectory score / answer reward 主导，保证评测目标不偏离 final answer pass@1。
- `step_advantage` 只从 point step reward 中计算，用于 point-to-count 的局部 credit assignment。
- `gated_step_advantage` 用 answer score / trajectory quality 做门控，避免模型为了局部 point reward 牺牲最终 answer。
- `λ` 是小权重，例如 `0.1-0.3`，只作为辅助 shaping，不取代 BoK 主目标。

推荐实现名：

```text
bok_grpo_step
```

推荐主线先不直接替换 BoK-GRPO，而是新建 ablation 脚本测试。

---

## 1. 为什么当前 BoK-GRPO 是 trajectory-level 优化

当前 BoK-GRPO 的第一性原理目标是：

> 在同一个 prompt 的 K 条 rollout 中，把最高 reward trajectory 的概率质量转移到 pass@1。

也就是：

```text
pass@K 能做对，但 pass@1 不稳定 → 训练让模型更倾向于采样高 reward 路径
```

当前实现中，BoK-GRPO 在 advantage 入口执行：

```text
scores = token_level_rewards.sum(dim=-1)
```

这说明无论 reward 是只放在最后 token，还是分散在多个 token，都会先被聚合为一个 scalar：

```text
score_i = sum_t reward_{i,t}
```

然后对同一 prompt 的 K 条 trajectory 做 group-level 路由：

- all-correct filter / smart filter
- easy DrGRPO path
- BoK softmax path
- all-wrong cap

最后得到每条 trajectory 的 advantage，并广播到整段 response。

### 优点

- 直接优化最终 answer / trajectory outcome。
- 非常符合当前评测：`val/reward_score = final answer binary`。
- 能把 pass@K 能力蒸馏到 pass@1。
- 与 `BOK_SMART_FILTER_THRESHOLD=0.955` 配合，可以继续释放 answer-correct 但 point-low group。

### 缺点

- 如果 trajectory 最终 answer 错，不容易知道是哪一步 point 导致错误。
- 如果最终 answer 对，但中间 point 有偏差，除非 smart filter 释放，否则可能丢失局部 point 梯度。
- 所有 token 共享同一个 trajectory advantage，credit assignment 粗糙。

---

## 2. 为什么单独 step-level GSPO 不够

当前已有 `grpo_step` / GSPO 路线，设计目标是对每个 reward-bearing token position 独立做 advantage normalization。

它的机制是：

1. reward worker 把每个 point step reward 放到对应 `</point>` token。
2. `compute_grpo_step_level_advantage()` 找到所有 reward positions。
3. 对每个 reward position，在同一 prompt 的 K 条 rollout 间做 step-level normalization。
4. 把该 step 的 advantage broadcast 到该段 token。

这确实能提供 step-level credit assignment。

但单独 GSPO 有问题：

1. 它更关注局部 point reward，可能削弱 final answer pass@1。
2. counting 任务存在误差累计：某一步 point 好，不代表最终 count answer 对。
3. 如果 step reward 与 final answer reward 冲突，单独 step-level 可能鼓励局部最优。
4. 用户当前总目标是 answer accuracy，尤其 sparse 场景要接近/超过 95%，所以不能让 step objective 主导。

因此，单独 GSPO 不适合作为主线替代 BoK-GRPO。

---

## 3. 为什么当前 process reward 默认关闭是合理的

用户判断基本正确：当前 process reward 默认关闭，是因为在 BoK-GRPO 主线下，所有 reward 最终仍会被计算成整条 trajectory 的总分。

更精确地说：

- Reward function 支持 `_step_rewards`。
- Reward worker 支持把 `_step_rewards` 放到每个 `</point>` token。
- 但 `compute_bok_grpo_advantage()` 会把 token reward 求和为 trajectory score。

所以：

```text
PROCESS_REWARD_ENABLE=1 + ADV_ESTIMATOR=bok_grpo
```

不是严格意义上的 step-level credit assignment。

它只改变 reward 在 token tensor 中的位置，但 BoK advantage 还是 trajectory-level。

真正 step-level credit assignment 至少需要：

```text
PROCESS_REWARD_ENABLE=1
ADV_ESTIMATOR=grpo_step
```

但这又会放弃 BoK-GRPO 的 trajectory-level pass@K→pass@1 优势。

这就是需要 hybrid estimator 的根本原因。

---

## 4. 第一性原理：为什么要做 hybrid

StepCount interleaved point-to-count 的真实因果结构是：

```text
point_1 → marked image_1 → point_2 → marked image_2 → ... → final answer
```

最终 answer 是评测目标，但每一步 point 是状态转移动作。

如果只做 trajectory-level：

```text
final answer wrong → 整条轨迹负反馈
```

问题是模型不知道是哪一步错。

如果只做 step-level：

```text
point step hit mask → 局部正反馈
```

问题是局部 point 对不等于最终 count 对。

所以正确目标应是层级式：

```text
全局：最终 answer / trajectory 选择必须正确
局部：在不破坏全局目标的前提下，让每一步 point 更可归因
```

数学形式：

```text
A_total(i, t) = A_traj(i) + λ * G(i) * A_step(i, t)
```

其中：

- `A_traj(i)`：第 i 条 trajectory 的 BoK trajectory-level advantage。
- `A_step(i,t)`：第 i 条 trajectory 在第 t 个 point step 附近的 step-level advantage。
- `G(i)`：trajectory quality gate，避免 wrong-answer 轨迹因为局部 point 好而被过度奖励。
- `λ`：step auxiliary weight，建议 `0.1-0.3`。

---

## 5. 推荐算法：BoK-Step GRPO / BOK-GSPO

### 5.1 总体流程

新增 estimator：

```text
ADV_ESTIMATOR=bok_grpo_step
```

训练时设置：

```text
PROCESS_REWARD_ENABLE=1
```

流程：

1. Reward function 计算整条 trajectory 的 `overall`。
2. Reward function 同时输出每个 point step 的 `_step_rewards`。
3. Reward worker 将 point rewards 放在 `</point>` token，将 answer/format remainder 放在最后 token。
4. Hybrid estimator 复制当前 token-level reward：
   - `trajectory_rewards = token_level_rewards`
   - `point_step_rewards = token_level_rewards` 但把每条 response 最后 token 的 answer/format remainder 清零
5. 对 `trajectory_rewards` 调用原始 `compute_bok_grpo_advantage()`，得到 `A_traj`。
6. 对 `point_step_rewards` 调用 step-level advantage，得到 `A_step`。
7. 用 answer / trajectory gate 缩放 `A_step`。
8. 组合：

```text
A_total = A_traj + λ * G * A_step
```

9. 对最终 advantage clip，防止 step auxiliary 放大梯度。

### 5.2 为什么要去掉最后 token reward 再算 step advantage

reward worker 会把 answer/format remainder 放在最后 token。

如果 step-level advantage 直接用完整 `token_level_rewards`，那么 final answer reward 会同时进入：

- trajectory BoK advantage
- step-level advantage 的最后 reward position

这会造成 final answer 双重计算。

因此 hybrid estimator 中的 `A_step` 应只基于 point step rewards：

```text
point_step_rewards = token_level_rewards.clone()
point_step_rewards[i, last_valid_token_i] = 0
```

这样：

- final answer 仍由 BoK trajectory-level 主导。
- point-to-count credit assignment 只影响中间 point segments。

### 5.3 Gate 设计

建议默认 gate：

```text
G(i) = clamp(answer_score_i, min_gate, 1.0)
```

其中：

```text
min_gate = 0.2
```

原因：

- 如果只给 answer-correct trajectory step credit，则 answer-wrong 但部分 point 正确的轨迹完全没有 dense learning，浪费 point signal。
- 如果不给 gate，则 answer-wrong 但 point 高的轨迹可能被过度鼓励，偏离 final answer。
- 用 `answer_score` 做 soft gate 可以折中：final answer 越可信，step credit 越强；answer wrong 时仍保留少量 point 学习。

可选 gate：

| Gate | 公式 | 特点 |
|---|---|---|
| none | `G=1` | step 信号强，但可能冲突 final answer |
| answer_soft | `G=max(answer_score, min_gate)` | 推荐默认 |
| answer_hard | `G=1(answer_score>=0.5)` | 更保守，但 all-wrong group 没有 point 学习 |
| traj_positive | `G=1(A_traj>0)` | 只增强 BoK 认为好的轨迹 |

推荐默认：

```text
BOK_STEP_GATE=answer_soft
BOK_STEP_MIN_GATE=0.2
```

### 5.4 Step weight 设计

建议：

```text
BOK_STEP_WEIGHT=0.2
```

原因：

- `λ=0` 退化为原始 BoK-GRPO。
- `λ=1` 容易让 step objective 过强。
- `0.1-0.3` 是合理 auxiliary range。
- 当前 point reward 权重是 `0.3`，所以 step auxiliary 不应超过 final trajectory 主项。

推荐 ablation：

```text
λ = 0.1, 0.2, 0.3
```

主线先用 `0.2`。

---

## 6. 关键风险与解决方案

### 风险 1：step reward 双重计入

解决：`A_step` 只用 point step rewards，清零每条 response 最后 token reward。

### 风险 2：局部 point objective 压过 final answer

解决：

- `BOK_STEP_WEIGHT` 默认 `0.2`。
- `BOK_STEP_GATE=answer_soft`。
- final answer 仍在 trajectory BoK advantage 中主导。

### 风险 3：step-level 方差过高

解决：

- step advantage clip，例如：`BOK_STEP_CLIP=2.5`。
- final combined clip，例如：`BOK_HYBRID_CLIP=4.0`。
- 如果 entropy 上升，降低 `BOK_STEP_WEIGHT` 或改 hard gate。

### 风险 4：不同 trajectory step 数不同

当前 step-level advantage 通过 token reward positions 寻找 step rewards。若某些 trajectory 提前 answer 或格式失败，则 reward positions 不完全对齐。

解决：

- 缺失 step reward 的 trajectory 在该 position reward 为 0，会自然得到低 step advantage。
- turns exceeded / missing answer 由 overall reward 置零，trajectory BoK 主项惩罚。

### 风险 5：all-correct group 与 smart filter 重叠

smart filter 已经释放 answer-correct but point-low group。

Hybrid step advantage 会进一步提供局部 point credit。

解决：

- 保留 `BOK_SMART_FILTER_THRESHOLD=0.955`。
- 监控 `ac_released`。
- 若 `ac_released` 过高且 entropy 上升，可降低 `BOK_STEP_WEIGHT` 或设置 `BOK_EASY_SCALE=0.7`。

---

## 7. 实现方案

### 7.1 新增 AdvantageEstimator

在 `verl/trainer/ray_trainer.py` 中新增：

```python
BOK_GRPO_STEP = "bok_grpo_step"
```

### 7.2 新增 core algorithm

在 `verl/trainer/core_algos.py` 中新增函数：

```python
def compute_bok_grpo_step_advantage(
    token_level_rewards,
    response_mask,
    index,
    bok_tau,
    bok_clip,
    bok_uniform_mix,
    bok_tau_init,
    bok_tau_final,
    global_step,
    total_steps,
    answer_scores=None,
):
    ...
```

核心伪代码：

```python
traj_adv, _ = compute_bok_grpo_advantage(
    token_level_rewards,
    response_mask,
    index,
    bok_tau=bok_tau,
    bok_clip=bok_clip,
    bok_uniform_mix=bok_uniform_mix,
    bok_tau_init=bok_tau_init,
    bok_tau_final=bok_tau_final,
    global_step=global_step,
    total_steps=total_steps,
    answer_scores=answer_scores,
)

point_rewards = token_level_rewards.clone()
last_pos = response_mask.sum(dim=-1).long() - 1
point_rewards[torch.arange(bsz), last_pos] = 0.0

step_adv, _ = compute_grpo_step_level_advantage(
    point_rewards,
    response_mask,
    index,
)

gate = build_gate(answer_scores, traj_adv, mode=BOK_STEP_GATE)
step_adv = step_adv * gate.unsqueeze(-1)
step_adv = clamp(step_adv, -BOK_STEP_CLIP, BOK_STEP_CLIP)

combined = traj_adv + BOK_STEP_WEIGHT * step_adv
combined = clamp(combined, -BOK_HYBRID_CLIP, BOK_HYBRID_CLIP)
return combined, combined
```

### 7.3 ray_trainer 分支

新增：

```python
elif adv_estimator == AdvantageEstimator.BOK_GRPO_STEP:
    advantages, returns = core_algos.compute_bok_grpo_step_advantage(...)
```

复用 BoK 参数，并新增 env：

```text
BOK_STEP_WEIGHT=0.2
BOK_STEP_GATE=answer_soft
BOK_STEP_MIN_GATE=0.2
BOK_STEP_CLIP=2.5
BOK_HYBRID_CLIP=4.0
BOK_STEP_EXCLUDE_FINAL=1
```

### 7.4 脚本配置

新建脚本而不是覆盖主线：

```text
examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_bok_step_hybrid.sh
```

关键配置：

```text
ADV_ESTIMATOR=bok_grpo_step
PROCESS_REWARD_ENABLE=1
BOK_STEP_WEIGHT=0.2
BOK_STEP_GATE=answer_soft
BOK_STEP_MIN_GATE=0.2
BOK_STEP_CLIP=2.5
BOK_HYBRID_CLIP=4.0
TRAJ_RETURN_POINT_STEP_SCORES=1
TRAJ_POINT_STRICT_JSON=1
TRAJ_FORMAT_REJECTION=1
ANSWER_WEIGHT=0.6
POINT_WEIGHT=0.3
TRAJECTORY_FORMAT_WEIGHT=0.1
```

---

## 8. 推荐实验顺序

### 实验 A：BoK-GRPO 主线 baseline

目的：确认当前恢复配置后的稳定基线。

```text
ADV_ESTIMATOR=bok_grpo
PROCESS_REWARD_ENABLE=0
```

### 实验 B：Hybrid λ=0.1

目的：低风险验证 step auxiliary 是否改善 point，不明显损害 answer。

```text
ADV_ESTIMATOR=bok_grpo_step
PROCESS_REWARD_ENABLE=1
BOK_STEP_WEIGHT=0.1
```

### 实验 C：Hybrid λ=0.2

目的：推荐主 hybrid 强度。

```text
BOK_STEP_WEIGHT=0.2
```

### 实验 D：Hybrid λ=0.3

目的：测试 step credit 上限。

```text
BOK_STEP_WEIGHT=0.3
```

### 实验 E：纯 GSPO 对照

目的：证明纯 step-level 是否牺牲 final answer。

```text
ADV_ESTIMATOR=grpo_step
PROCESS_REWARD_ENABLE=1
```

---

## 9. 监控指标

必须同时监控：

| 指标 | 目标 |
|---|---|
| `val/answer_reward` | 必须高于 BoK baseline，否则 hybrid 失败 |
| `val/point_reward` | 应提升，但不能以 answer 下降为代价 |
| `val/format_fail_reward` | 保持 `<0.05` |
| `val/consistency_violation_reward` | 不应持续上升 |
| `reward/point` | 训练 point 应提升 |
| `reward/answer` | 不应明显下降 |
| `actor/entropy_loss` | 不应进入 `>0.9` 区间 |
| `ac_released` | 建议 3%-8% rollout 区间 |
| `adv_std` | step hybrid 后不能暴涨 |
| non-finite / spike | 不应超过原 BoK baseline |

判定标准：

- 如果 point 提升但 answer 不提升：step weight 过强或 gate 太宽。
- 如果 answer 提升且 point 提升：hybrid 成功。
- 如果 format fail 上升：step reward 正在破坏格式，应降低 λ 或加强 gate。
- 如果 entropy 上升：step auxiliary 引入高方差，应降低 λ。

---

## 10. 与总目标的关系

用户总目标是：

> VLM interleaved point to count，通过动态显式注意力转移，使 sparse counting 达到 95%，dense counting 大幅提升。

Hybrid BoK-Step GRPO 正好对应这个目标：

- BoK-GRPO 负责“最终 answer pass@1”这个评测指标。
- Step-level auxiliary 负责“每一步 point 的局部 credit assignment”。
- Smart filter 负责“answer-correct 但 point-low 的继续学习”。
- Strict JSON + format rejection 负责“格式可靠性”。

这比单纯调 reward 权重更符合问题因果结构。

---

## 11. 当前是否应立即实现？

建议：**可以实现，但应作为新 estimator 和新脚本，不覆盖当前 BoK-GRPO 主线。**

原因：

1. 当前主线还需要先恢复 V12-quality baseline。
2. Hybrid 是更复杂算法，可能带来方差风险。
3. 新 estimator 可以完全不影响原始 BoK-GRPO、GRPO_STEP、DrGRPO。
4. 通过 env 控制 `BOK_STEP_WEIGHT=0` 可以退化回原始 BoK 行为。

如果确认实现，推荐下一步改动文件：

```text
verl/trainer/core_algos.py
verl/trainer/ray_trainer.py
examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_bok_step_hybrid.sh
```

并新增测试/验证：

```text
bash -n examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_bok_step_hybrid.sh
python3 -m py_compile verl/trainer/core_algos.py verl/trainer/ray_trainer.py
```

---

## 12. 最终结论

可以兼顾 trajectory-level BoK-GRPO 与 step-level point-to-count credit assignment，但必须用 hybrid advantage estimator，而不是单纯打开 process reward。

最终推荐算法：

```text
BoK-Step GRPO / BOK-GSPO
A_total = A_BoK_trajectory + λ * G(answer/trajectory) * A_step_point
```

默认建议：

```text
ADV_ESTIMATOR=bok_grpo_step
PROCESS_REWARD_ENABLE=1
BOK_STEP_WEIGHT=0.2
BOK_STEP_GATE=answer_soft
BOK_STEP_MIN_GATE=0.2
BOK_STEP_EXCLUDE_FINAL=1
```

但在用户确认前，不建议直接替换主线训练；应先作为独立 ablation 实现。
