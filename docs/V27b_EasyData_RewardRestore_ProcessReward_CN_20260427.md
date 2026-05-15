# V27b Easy-Data Reward 恢复与 Process Reward Credit Assignment 结论（2026-04-27）

## 0. 本次完成内容

本次基于 reward 代码、训练调用路径和 BoK-GRPO advantage 代码复核后，完成两项恢复/确认：

1. 已将 V27b easy-data 脚本恢复为：

```text
TRAJ_POINT_STRICT_JSON=1
ANSWER_WEIGHT=0.6
POINT_WEIGHT=0.3
TRAJECTORY_FORMAT_WEIGHT=0.1
```

2. 确认 reward 函数本身暂不需要修改；当前主要问题是脚本配置与训练目标不完全一致。

目标脚本：

```text
examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_easy_data.sh
```

已执行语法验证：

```text
bash -n examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_easy_data.sh
```

结果通过。

---

## 1. 为什么恢复 reward 到 0.6 / 0.3 / 0.1

V27b 原配置为：

```text
ANSWER_WEIGHT=0.5
POINT_WEIGHT=0.3
TRAJECTORY_FORMAT_WEIGHT=0.2
```

恢复为：

```text
ANSWER_WEIGHT=0.6
POINT_WEIGHT=0.3
TRAJECTORY_FORMAT_WEIGHT=0.1
```

原因不是简单复刻 V12，而是 reward margin 的第一性原理。

### 1.1 对正确答案合法格式轨迹，两套权重等价

设 point score 为 `P`，format score 为 `1`。

V12 / restored reward：

```text
overall = 0.6 * 1 + 0.3 * P + 0.1 * 1 = 0.7 + 0.3P
```

V27b old reward：

```text
overall = 0.5 * 1 + 0.3 * P + 0.2 * 1 = 0.7 + 0.3P
```

因此，对 answer-correct 且 format-valid 的 trajectory，二者完全等价，不会损失 point-quality 梯度。

### 1.2 对错误答案合法格式轨迹，V27b old reward 更宽容

错误答案在 `TRAJ_SOFT_ANSWER_DECAY=1` 时会得到 `A`，其中 `0 <= A <= 0.4`。

Restored reward：

```text
overall = 0.6A + 0.3P + 0.1
```

V27b old reward：

```text
overall = 0.5A + 0.3P + 0.2
```

差值：

```text
old_v27b - restored = 0.1 - 0.1A >= 0.06
```

也就是说，旧 V27b reward 会系统性提高错误答案轨迹的 score，降低 answer-correct 与 answer-wrong 之间的 BoK ranking margin。

这与评测目标不一致，因为 validation / benchmark 最终只看 final answer pass@1。

### 1.3 format 安全不靠 format weight，而靠 format rejection

当前 reward 中真正防 format fail 的机制是：

```text
TRAJ_FORMAT_REJECTION=1
```

当 `_trajectory_format_reward()` 判定格式错误时，`format_score=0`，随后 training overall 被置零。

因此，format 保护应主要依赖 rejection，而不是把 format weight 从 `0.1` 提到 `0.2`。

恢复 `0.6/0.3/0.1` 的效果是：

- 保留正确答案轨迹中的 point 梯度；
- 增强 answer margin；
- 让 format safety 由 `TRAJ_FORMAT_REJECTION=1` 单独负责，语义更清楚。

---

## 2. 为什么 `TRAJ_POINT_STRICT_JSON=1` 需要开启

当前 `_trajectory_format_reward()` 已经严格检查每个 `<point>` 内容必须是合法 JSON 且包含 `point_2d`。

如果 `TRAJ_FORMAT_REJECTION=1`，format-broken trajectory 的 overall 会被置零。

因此，理论上仅靠 format rejection 已能防止大多数 malformed JSON 轨迹通过 overall reward hacking 拿分。

但仍建议开启：

```text
TRAJ_POINT_STRICT_JSON=1
```

原因：

1. 关闭 strict JSON 后，`_parse_pred_point()` 会在 JSON 失败时 fallback 到 regex。
2. 这会让 point diagnostic、point count、consistency 统计与 format reward 的严格语义不一致。
3. smart filter 分析依赖 point / overall / format 的日志可信度，regex fallback 会让日志解释变复杂。
4. 关闭 strict JSON 不会提升效率；JSON 失败后跑 regex fallback 反而可能更慢。
5. 真正的效率优化方向是合并 format validation 与 point parsing，共享一次 strict JSON parse，而不是放松 parser。

结论：

```text
TRAJ_FORMAT_REJECTION=1
TRAJ_POINT_STRICT_JSON=1
```

这是主线最安全组合。

---

## 3. Process reward 默认关闭是否因为无法实现逐步 credit assignment？

结论：**在当前 BoK-GRPO 主线下，是的：仅打开 process reward 并不能真正实现逐步 point-to-count credit assignment。**

更准确地说，当前代码有两层：

### 3.1 Reward worker 可以把 step reward 放到每个 `</point>` token

当：

```text
PROCESS_REWARD_ENABLE=1
```

reward 函数会生成：

```text
score["_step_rewards"] = [point_weight * step_score_1, ...]
```

reward worker 会把这些 step rewards 放到每个 `</point>` token 位置，并把剩余 answer / format reward 放到最后 token。

所以，**reward placement 层面已经支持 per-turn reward placement**。

### 3.2 但 BoK-GRPO advantage 会重新聚合成整条 trajectory score

当前 `compute_bok_grpo_advantage()` 的核心是：

```text
scores = token_level_rewards.sum(dim=-1)
```

也就是说，即使 token 上有每步 reward，BoK-GRPO 也会先把整条 response 的 token reward 求和，得到一个 trajectory-level scalar score。

之后优势值仍是整条 trajectory 级别的 advantage，再乘到整段 response mask。

因此：

```text
PROCESS_REWARD_ENABLE=1 + ADV_ESTIMATOR=bok_grpo
```

并不等价于真正逐步 credit assignment。

它只是改变 reward 在 token 上的放置位置，但 advantage 计算仍然回到 trajectory 总分。

---

## 4. 真正逐步 credit assignment 需要什么？

真正 step-level credit assignment 至少需要：

```text
PROCESS_REWARD_ENABLE=1
ADV_ESTIMATOR=grpo_step
```

也就是 GSPO / step-level GRPO 路线。

其含义：

- 每个 point step 的 reward 在对应 step 上形成局部 advantage；
- 不只是把整条 trajectory 总分分散到 token 上；
- 才能让第 k 个 point 的错误主要影响第 k 步附近 token，而不是整条 response 共享同一个 trajectory advantage。

但这会改变主算法，不再是当前 BoK-GRPO 主线。

---

## 5. 为什么主线暂时仍建议关闭 process reward

主线默认：

```text
PROCESS_REWARD_ENABLE=0
ADV_ESTIMATOR=bok_grpo
```

是合理的，原因：

1. 当前核心目标是先稳定恢复 / 超过 V12 sparse answer accuracy。
2. BoK-GRPO 的主要目标是 pass@K → pass@1，把高 reward trajectory 的概率集中到单次输出。
3. 在 BoK-GRPO 下，process reward placement 不会真正变成 step-level advantage。
4. 强行打开 process reward 可能改变 reward token 分布，但不改变 trajectory-level BoK credit，本质收益有限。
5. 若切换到 GSPO，则会引入更高方差，并可能削弱 final answer objective，需要单独 ablation。

因此推荐主线保持：

```text
PROCESS_REWARD_ENABLE=0
ADV_ESTIMATOR=bok_grpo
```

后续单独设计 ablation：

```text
PROCESS_REWARD_ENABLE=1
ADV_ESTIMATOR=grpo_step
```

---

## 6. 最终主线配置建议

当前已经恢复的部分：

```text
TRAJ_FORMAT_REJECTION=1
TRAJ_POINT_STRICT_JSON=1
ANSWER_WEIGHT=0.6
POINT_WEIGHT=0.3
TRAJECTORY_FORMAT_WEIGHT=0.1
PROCESS_REWARD_ENABLE=0
```

仍建议后续再考虑是否同步优化 optimizer：

```text
ACTOR_LR=1.5e-6
ppo_epochs=1
max_grad_norm=0.5
kl_coef=0.03
BOK_SMART_FILTER_THRESHOLD=0.955
```

本次没有修改 optimizer，只按用户要求恢复 strict JSON 与 reward。

---

## 7. 一句话结论

- Reward 代码本身暂不需要改。
- V27b 脚本 reward 权重已恢复为更适合 answer-margin 的 `0.6/0.3/0.1`。
- `TRAJ_POINT_STRICT_JSON=1` 需要开启，用来保证 point parser、format reward、diagnostic 和 smart-filter 统计语义一致。
- `process reward` 默认关闭是合理的：在当前 BoK-GRPO 中 token-level step rewards 会被 `sum(dim=-1)` 聚合回 trajectory score，不能真正实现逐步 point-to-count credit assignment；真正 step-level credit 需要 GSPO / `ADV_ESTIMATOR=grpo_step` 单独实验。
