# V36 训练复盘与后续 RL 优化 Plan

- 文档日期：2026-07-21
- 适用仓库：`/mnt/shared-storage-user/zhangchenhao/work/EasyR1-hy-0703`
- 训练任务：StepCount 多轮视觉计数 RL
- 本文状态：V36 实验复盘完成；V37 核心、原子四 cell A/B launcher、可重算 benchmark evidence 与 promotion gate 已实现，formal 数据、完整 trajectory replay、canary 和 formal A/B 待执行

## 1. 文档目标

本文回答四个问题：

1. V36 实际取得了什么结果，哪些结论已经被完整 eval 和真实 rollout 验证？
2. 为什么继续沿用当前配置训练，不能稳定提高 `pixmo-test` 与 `stepcount-500`？
3. 下一轮训练应如何修改 KL、BoK-GRPO、step reward、数据和 frontier sampling？
4. 在什么条件下可以从机制 pilot 晋级到正式训练？

本文第 3–12 节保留 V36 复盘和当时的因果设计。2026-07-21 集成状态以第 13 节和 `V37_STRICT_WINNER_STEP_RL_PLAN.md` 为准：核心机制与 fail-closed launcher 已落地，但尚未生成 formal frontier 证据、运行 canary 或产生新分数，不能把“代码完成”写成“性能提升”。

## 2. 一页结论

### 2.1 V36 最终结果

V36 使用 BF16、greedy decoding、逐样本 `min(task_cap, GT+3)`、显式闭合 `<answer>`，并禁止累计 point 数作为答案 fallback。

| checkpoint | pixmo-test | stepcount-500 | bias | countqa |
|---|---:|---:|---:|---:|
| step60 | 426/529 = 80.53% | **76/500 = 15.20%** | 280/1992 = 14.06% | **398/491 = 81.06%** |
| step77 | **433/529 = 81.85%** | 70/500 = 14.00% | **314/1992 = 15.76%** | 390/491 = 79.43% |

两个 checkpoint 形成能力 trade-off，没有单一 checkpoint 同时继承两者最好得分。

### 2.2 下一阶段目标

同一个 checkpoint 同时达到：

- `pixmo-test raw >= 83%`：至少 `440/529`；相对当前最好还差 7 个 case。
- `stepcount-500 raw >= 18%`：至少 `90/500`；相对当前最好还差 14 个 case。

SFT 的 `pass@16` raw 上限为：

- pixmo-test：`491/529 = 92.82%`。
- stepcount-500：`215/500 = 43.00%`。

因此目标在能力上限内，是合理的 stretch goal，但不能由当前 V36 配置自然保证。

### 2.3 总判断

- V36 证明：BoK 采样能够从 SFT 的随机成功轨迹中学习部分 dense counting 能力。
- V36 也证明：当前 KL 量纲、winner 排序、trajectory broadcast 和数据分布会造成后期漂移。
- 当前状态：**launcher/contract dependency-light 回归已通过；训练环境 E2E、canary 与 formal training 仍为 NO-GO**。
- KL 解耦、correctness-first routing、native action-level reward 和 frontier/preflight/gate 工具已实现；正式训练前仍必须生成可审计 frontier 数据与 machine-checkable coverage/filtering 证据，并先通过 canary。

## 3. V36 已验证事实

### 3.1 Eval 协议

当前结果属于 `strict_oracle_gt_plus_v3`：

- BF16；
- greedy decoding；
- 自适应最大轮数 `GT+3`；
- 必须显式输出可解析的 `<answer>...</answer>`；
- 不允许用累计 point 数替代答案；
- 到达最大轮数但没有 answer 的轨迹判错。

注意：使用 GT 决定 rollout 上限属于 oracle-efficiency 协议，不等同于 GT-blind fixed-turn 官方协议。后续目标和基线必须始终使用同一协议。

完整评测报告：

`/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/eval/v36_candidate_selection/v36_checkpoint_selection.md`

### 3.2 V36 后期不是单调变好

- step60 在 dense counting 和 CountQA 上更好。
- step77 在 pixmo-test 和 bias 上更好。
- step68、75、76、77 的 actor update 被 `GradSpikeProtect` 跳过。
- step77 的模型权重实际等价于 step74 更新后状态，而不是完成 77 次有效更新。
- step60 到 step77 期间 KL、entropy、BoK group variance 和 all-wrong 比例整体上升。

因此，后续不能使用“step 越大越好”的选模方式，必须按冻结的 held-out 指标 early stop。

## 4. 真实 Rollout 审计

### 4.1 75 行周期性训练 debug 摘要

从 resumed V36 日志中的周期性 reward detail 提取到 75 行摘要，覆盖 74 个唯一 sample。它们包含 answer、point count、count_number、cap 等诊断字段，但不包含每条 response 的完整 token、action span 和逐步 ledger，因此不能称为 75 条完整 trajectory，也不能用于精确 advantage replay。

| 指标 | 数量 |
|---|---:|
| 显式答案正确 | 20/75 |
| 正确答案但存在 `count_number` 缺陷 | 7/20 |
| 正确答案但 point 数不等于 GT | 1/20 |
| 正确答案但 shaped `answer_reward < 0.5` | 3/20 |
| capped 或没有正常 answer | 17/75 |

这些日志摘要足以支持“不能把 strict trajectory 作为 answer reward hard gate”的诊断，但表中比例不是完整 rollout 总体的无偏估计。否则会错误丢弃摘要中已经观察到的一部分正确答案。

正确原则是：

- 答案正确可以获得 answer reward；
- point、编号、重复、格式和停止缺陷分别扣分；
- strict trajectory 用于正确 winner 内部排序，而不是决定“答案是否正确”。

### 4.2 真实 point 质量

截至 step77，训练日志累计评估约 116 万个 point：

| point 类型 | 比例 | 含义 |
|---|---:|---|
| unique mask hit | 21.21% | 命中尚未使用的目标实例 |
| duplicate | 9.00% | 重复点击已使用实例 |
| miss | 69.80% | 未命中可用 GT mask |

process reward 存在真实区分信号，但 miss 中可能包含 mask false negative，因此当前阶段不应统一给予强负奖励。

### 4.3 Broadcast 的具体问题

真实 case 表明：模型可能已经完成 11 个高质量 point，但 GT=12 时提前回答 11。当前 trajectory-level broadcast 会处罚此前全部 point，尽管真正错误集中在最后的 continue/stop/answer 决策。

另有以下缺口：

- capped trajectory 在 reward 端不计算 point step detail；
- process reward 只标记 `</point>`，没有 answer/stop action；
- 超过 GT 的 extra point 不进入逐 mask step 计算；
- 通过字符串搜索 `</point>` 映射 token，结构异常时可能错位；
- step auxiliary re-center 会把 offset 扩散到 answer tail。

## 5. KL 根因与修复

### 5.1 当前机制

当前配置：

```bash
USE_KL_LOSS=false
KL_TYPE=adaptive
KL_COEF=0.08
KL_TARGET=0.15
KL_HORIZON=10000
```

当前 KL 是 reward-side adaptive KL：

```text
token_level_reward = task_score - beta * token_KL
BoK score = sum(token_level_reward)
```

由于多轮 trajectory 很长，KL 被按 token 累加后直接进入 BoK winner 排序。

### 5.2 真实量纲

| step | task score mean | BoK selector mean | token-mean KL | beta |
|---|---:|---:|---:|---:|
| 43 | 0.205 | -28.237 | 0.351 | 0.076 |
| 77 | 0.198 | -110.611 | 0.818 | 0.121 |

task reward 位于 `[0,1]`，而 selector score 已达到几十到上百的负值。BoK 实际主要按长度和 reference proximity 排序，不再以计数正确性为主。

### 5.3 目标机制

改为独立的 adaptive token-mean actor KL：

```text
BoK selector = task/outcome/process reward only
actor loss = policy_loss + beta * token_mean_KL(new_policy, reference_policy)
```

必须同时实现：

1. driver 使用 old/ref logprob 计算监控 KL，并更新 adaptive controller；
2. 每个 batch 把更新前的 `beta` 传给 actor；
3. actor 使用 new/ref logprob 计算可微的 token-mean KL loss；
4. checkpoint 保存和恢复 KL controller 的 `beta/target/horizon` 状态；
5. 分开记录 `selector_task_score`、`actor_kl_loss`、`kl_coef_before/after`。

不能只设置 `USE_KL_LOSS=true`。当前代码在该模式下不会更新 adaptive controller，并使用固定 `worker.actor.kl_coef`。

实现后的 V37 实际映射为 `algorithm.adaptive_actor_kl=true`、`algorithm.use_kl_loss=false`、`algorithm.kl_penalty=low_var_kl`。当前核心明确拒绝 adaptive actor KL 与旧 `use_kl_loss` 同时为 true；reward-side KL 不参与 selector。初始 beta/target/horizon 仍为 0.08/0.15/10000。

初始 canary 可保持 `beta=0.08`、`target=0.15`、`horizon=10000`，避免同时修改机制和初值。是否调整 beta 必须由 canary 的 actor KL/PG loss 比值决定。

## 6. Correctness-First BoK-GRPO

### 6.1 三个独立通道

每条 trajectory 输出三个互不混淆的信号：

1. `answer_exact`：显式可解析答案是否等于 GT；用于保留部分 answer reward。
2. `raw_success`：V37 strict winner 中要求 `answer_exact`、正常结束、未超过自适应 cap、无 duplicate 且完整通过 strict format；用于主 outcome advantage。
3. `trajectory_quality`：point count、count_number、mask unique hit、duplicate、format 和 answer-point consistency。

### 6.2 Mixed group

主 outcome advantage：

```text
A_outcome_i = raw_success_i - mean_group(raw_success)
```

性质：

- 正确轨迹一定为正；
- 错误轨迹一定为负；
- 不再让 KL、长度或 point shaping 翻转正确性排序。

`trajectory_quality` 只在正确轨迹集合内中心化并作为小幅 residual，不能改变正确/错误 advantage 的符号。

### 6.3 All-Wrong 与 All-Correct

当前 `BOK_ALLWRONG_NEG_ONLY=1` 把 advantage 限制到 `[-cap,0]`，但仍会间接增加零 advantage 的 best-wrong 相对概率。

目标规则：

- all-wrong：所有 terminal outcome advantage 为 0；只训练可验证的 process action；
- all-correct：terminal outcome advantage 为 0；只优化 process/quality；
- mixed：使用 correctness-first outcome，并叠加受限 process residual。

正式 V37 应设置：

```bash
BOK_WINNER_BOOST=0
```

当前 `BOK_WINNER_BOOST=1.5` 在正确轨迹 advantage 已为负时会把它放大得更负，不能保留。

## 7. 完整 Action-Level Reward

### 7.1 Point action

第一阶段建议：

| action 结果 | process value |
|---|---:|
| 新实例 unique hit | +1 |
| duplicate | -1 |
| miss | 0 |

miss 暂时保持 0，直到 mask 语义抽检证明 false-negative 足够低。

### 7.2 Answer/Stop action

必须把每一轮建模为明确事件：

- `point`：继续计数；
- `answer`：停止并给出答案；
- `cap`：达到最大轮数仍未正常回答。

answer action 至少记录：

- 是否 exact；
- 是否与累计 point/count_number 一致；
- 是否在 GT 附近正确停止；
- 是否在 cap 前正常闭合。

这允许在同一 turn 比较“继续 point”与“现在 answer”，而不是把终局错误平均广播到此前所有正确 point。

### 7.3 实现约束

- capped trajectory 仍必须输出已有 action 的 process values；
- extra point 必须显式标记 overshoot；
- reward worker 使用 rollout 事件 token offset，不再二次搜索拼接字符串；
- process advantage 只作用于对应 point/answer span；
- answer tail 不接受无关的 point auxiliary offset。

## 8. 数据方案

### 8.1 现有 focused10k 的限制

当前分布：

| bucket | 样本数 | 比例 |
|---|---:|---:|
| 2-10 | 0 | 0% |
| 11-20 | 4000 | 40% |
| 21-30 | 4500 | 45% |
| 31-40 | 1000 | 10% |
| 41-50 | 500 | 5% |

mask `ok_ratio` 在高难桶明显偏低：

- 11-20：53.19%；
- 21-30：2.49%；
- 31-40：5.77%；
- 41-50：6.82%。

因此 `maskcomplete` 只表示文件和映射存在，不表示 mask 语义统一高质量。

### 8.2 目标驱动的 candidate pool

真实 benchmark：

- pixmo-test 529 条全部位于 2-10；
- stepcount-500 在 11-20/21-30/31-40/41-50 分别为 130/130/120/120；
- V36 dense 的主要缺口位于 21-40；
- 11-20 已相对较强，需要维护而不是继续占据最大比例。

建议 frontier candidate pool 初始配比：

```text
2-10 / 11-20 / 21-30 / 31-40 / 41-50
40%  / 10%   / 20%   / 20%   / 10%
```

该比例是首轮实验配比，不是永远固定的最优值。后续根据独立 held-out 的 bucket deficit 和 frontier yield 更新。

### 8.3 数据红线

- train/held-out/pixmo/stepcount/countqa/bias 的 ID、图像 SHA256 和 perceptual hash 均不得重叠；
- benchmark 的 `pass@16` trajectory 和标签不得进入训练；
- metadata 必须逐样本 100% 匹配；
- basename fallback 必须唯一且可回溯到图像 hash；
- 每个 bucket 至少人工抽检 200 条 mask/point 语义质量；
- selection manifest 固定记录源数据、模型 hash、seed、scorer hash 和筛选原因。

## 9. Frontier Sampling 与 Strict Winner

### 9.1 两个概念必须分开

- `frontier prompt set`：供在线 BoK-GRPO 重新采样的 prompt 数据。
- `strict winner trajectory`：离线生成的高质量 response，只能供单独 RFT/SFT 阶段使用。

当前 `RLHFDataset` 只消费 prompt、answer 和 image，不会自动学习离线 winner response。因此不能把“保存 strict winner JSON”误认为 BoK RL 已经训练了这些 response。

### 9.2 Mining 流程

对独立训练 pool 中每个 prompt：

1. 使用 clean SFT checkpoint-476；
2. 使用与训练一致的 `temperature=0.7`、`top_p=1.0`；
3. 至少生成 `M=32` candidates；
4. 使用两个独立 seed；
5. 分别记录 raw success、strict quality、unique hit、duplicate、miss、cap 和长度；
6. 用另一 seed 验证 frontier 难度，减少 winner's curse。

建议保留 raw winner 数约 `3-20/32` 的 prompt。其在线 N=16 rollout 大概率产生同时包含正确与错误轨迹的 mixed group。

训练池建议：

- 70%-80% outcome-frontier prompts；
- 20%-30% process-hard prompts：没有 exact winner，但 candidates 之间存在可靠 point progress 差异；
- process-hard prompts 的 terminal outcome 固定为 0，只训练 process spans。

模型更新后 frontier 会移动，建议每 10-20 个有效 optimizer update 重新 mining 或更新 online buffer。

## 10. Policy Loss 与 GSPO-Token

本地 `POLICY_LOSS_IS_LEVEL=sequence_token` 的 stop-gradient 结构接近 GSPO-token：使用 length-normalized sequence ratio 的数值，同时保留 per-token gradient，适合多轮 process advantage。

但当前仍使用：

```bash
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
```

这是 DAPO/GRPO 量级，不是 GSPO 论文使用的 sequence clip 量级。V36 的 `pg_clipfrac_higher/lower` 基本为 0，说明 clipping 没有形成实际保护。

正式训练前必须做独立 canary：

- 保持其他变量不变；
- 比较当前 clip 与更小的 sequence clip；
- 观察 clip fraction、effective updates、grad norm、KL、entropy 和 held-out raw；
- 不能直接照搬论文数值，也不能把当前配置称为已验证 GSPO。

## 11. 分阶段执行计划

### P0：实现与单元测试

实现状态：下列核心项与针对性单测已经落地；launcher 契约测试另覆盖两臂展开、formal manifest、数据模式、CP 和 final gate 分发。这里保留原验收清单，不能据此推断训练效果。

必须完成：

- adaptive actor-side token-mean KL；
- KL controller checkpoint state；
- correctness-first outcome routing；
- all-wrong terminal zero；
- `BOK_WINNER_BOOST=0`；
- point/answer/cap event offsets；
- capped 与 extra-point process values；
- 每 response audit 字段落盘。

必须新增测试：

- 正确答案无论 point quality 如何，`answer_exact=1`；
- strict 缺陷不能翻转 `answer_correct` 或清零 partial answer reward，但 duplicate/format 缺陷必须令 strict-winner `raw_success=0`；
- mixed group 中 correct advantage 恒正；
- all-wrong terminal advantages 全零；
- KL 不改变 BoK ranking；
- actor KL 可反向传播且 controller 可续训恢复；
- capped trajectory 保留已有 process values；
- answer event 与对应 token span 对齐。

### P1：离线 Replay 与兼容性证据

当前没有保存可逐 token 重建的 75 条训练 trajectory，因此完整 action-level advantage replay 仍待补采。已完成的替代验证为：

- 在 3512 条完整 V36 eval trajectory 上比较 26 个 legacy reward 字段，V37 开关关闭时差异为 0；
- 对 500 组随机 BoK 输入比较新旧结果，437 组 bit-exact，全部 500 组都在一个 float32 ULP 内，没有 advantage 符号、排序或实质数值回归；
- 上述证据只能证明 feature-off/legacy 数学兼容，不能替代 native action span 的真实训练 replay。

后续必须保存完整 response、token IDs、action ledger 和 reward extra info，再检查：

- 20 条 exact trajectory 不再被 shaped reward 误判；
- GT=12、point=11、answer=11 等 case 只处罚 terminal decision，不抹掉全部正确 prefix；
- all-wrong 组不再选择 best-wrong；
- selector score 回到 task reward 的可解释量纲；
- process reward 没有写入 answer tail。

### P2：2-4 Step Canary

目的仅是检查机制，不判断最终性能。

硬门：

- `selector_task_score` 不含 KL；
- correct advantage sign error = 0；
- all-wrong positive/relative terminal reinforcement = 0；
- process event mapping coverage = 100%；
- `progress_degraded=0`；
- nonfinite gradient = 0；
- 无连续 skipped update；
- checkpoint 恢复后 beta 连续。

### P3：12-Step 双 Seed A/B

两臂除 process auxiliary 外完全一致：

- Arm A：correctness-first trajectory outcome；
- Arm B：Arm A + action-level process reward。

建议固定：

| 参数 | 初始值 |
|---|---:|
| base model | checkpoint-476 |
| rollout N | 16 |
| temperature | 0.7 |
| top_p | 1.0 |
| actor LR | 2e-7 |
| KL beta/target/horizon | 0.08 / 0.15 / 10000 |
| adaptive max turns | GT+3 |
| winner boost | 0 |
| process lambda | 从 0.05-0.10 canary 起步 |

晋级条件：

- 两个 seed 方向一致；
- frontier outcome mixed groups >=70%；
- outcome-frontier 的 raw all-wrong <=25%；
- process-hard pool 单独统计，不混入上述 all-wrong 门槛；
- Arm B raw 不低于 Arm A；
- Arm B unique hit 提高，duplicate/cap 不升；
- 五个 bucket held-out macro 不退化。

### P4：30-50 个有效 Update

- 只统计实际执行 `optimizer.step()` 的 effective update；
- 每 5 个有效 update 保存 checkpoint；
- 每 5 个有效 update 跑独立 held-out；
- 每 10-20 个有效 update 更新 frontier；
- 按预先冻结的 dual-benchmark selector early stop；
- 出现连续 grad spike、KL 不可恢复或 held-out 连续两次下降时停止。

## 12. 最终 Eval 与晋级标准

最终只对 held-out 选出的少量 checkpoint 跑完整 benchmark，禁止按 test 反复调参。

正式协议：

- BF16；
- greedy；
- GT+3 adaptive cap；
- explicit answer only；
- no point-count fallback；
- 同一份固定代码和 dataset hash。

最终硬目标：

| suite | 晋级线 |
|---|---:|
| pixmo-test | >=440/529 |
| stepcount-500 | >=90/500 |

附加安全门：

- 必须由同一个 checkpoint 同时达到；
- 至少两个训练 seed 的方向一致；
- strict trajectory rate 不显著下降；
- count_number continuity、duplicate、cap rate 不恶化；
- CountQA 不出现明显能力坍塌；
- 报告 paired case flips，而不只报告总分。

## 13. 当前状态与下一步

当前已完成：

- V36 完整 eval 与 checkpoint 对比；
- step40-77 日志趋势分析；
- 75 行 training reward debug 摘要审计（74 个唯一 sample；非完整 trajectory replay）；
- point mask 统计；
- KL、BoK、process reward 和 frontier 方案审计；
- adaptive actor-side response-token mean KL、controller 更新与 checkpoint state；
- correctness-first `raw_success` routing、homogeneous terminal zero、`BOK_WINNER_BOOST=0`；
- strict structural winner gate：duplicate/format 缺陷不会污染 `raw_success`，同时保留独立 `answer_correct` 与 partial answer reward；
- rollout 原生 point/answer/cap action ledger 和 progress step credit；
- adaptive actor LR scheduler 只随实际执行的 optimizer update 推进，全跳过或非法 counter fail closed；
- 两 seed M32 frontier builder、formal preflight、双 seed A/B final gate；
- `run_v37_strict_ab.sh` 原子 preregistration、两个 seed 的四 cell 严格串行调度与 evidence index；
- formal external image/runtime/execution-environment snapshot，所有下游训练使用显式环境而非任意 ambient shell；
- schema v2 benchmark evidence：从 canonical dataset 与逐样本 transcript 重算，冻结 BF16/greedy/GT+3/explicit-answer 协议，并强制 eval model 为 checkpoint 自身的 `actor/huggingface`；
- V37 `debug|canary|formal` / `frontier_rl|strict_winner_rft` fail-closed launcher；
- baseline/progress 契约测试、formal manifest hash/quota 测试、CP>1 和 RFT 数据模式拒绝测试。

当前尚未完成：

- formal frontier 数据及 selection/coverage/filtered manifest 的现场产物；
- 含完整 token/action ledger 的 V36 真实 trajectory 离线新旧 advantage replay；
- 2-4 step canary 和双 seed 12-step A/B。

另外，adaptive actor KL 当前对 Ulysses sequence parallel `>1` fail closed；`V37_CP_SIZE=1` 是唯一验证配置。正式五桶配额为 40/10/20/20/10，不再使用早期文档中的 25/25/20/20/10。

因此，在 formal 数据证据与 P2 canary 验收前，不应启动正式长周期训练。`pixmo >=440/529` 与 `stepcount-500 >=90/500` 只属于 final promotion gate，并且必须由同一 checkpoint 同时达到；它们不是 canary 前置条件。当前没有新训练或 eval，不能宣称已达到 V37 性能目标。
