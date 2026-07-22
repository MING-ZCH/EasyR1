# V37 得分目标与 8xH200 16 小时执行计划

- 冻结日期：2026-07-23
- 训练起点：`StepCount-7B-SFT-1M-resume-from-1963/checkpoint-476`
- 目标：同一个 V37 checkpoint 同时提升 sparse、dense 和 extreme counting，不再按 benchmark 分别挑 checkpoint。
- 当前状态：代码机制仍在 canary 前，性能结论为 `NO-GO`。

## 1. 统一评测协议

所有正式比较必须使用同一协议，否则只能报告历史点估计，不能声明超过 SFT 或进入 top3：

- BF16、greedy decoding；
- 每个 sample 使用 `min(task_cap, GT_answer + 3)`；
- 首个完整 `</point>` 结束当前 point turn，首个完整 `</answer>` 结束 sample；
- 只接受唯一、闭合、纯整数 `<answer>`；
- 禁止累计 point 数作为 answer fallback；
- 使用完整 canonical sample 集和固定顺序，不能删除失败、超时或无答案样本；
- V37、SFT-1M 与候选 checkpoint 使用相同 evaluator、prompt、图像 bytes 和分母；
- 同一个 checkpoint 必须同时满足全部目标，禁止跨 checkpoint 拼接最好成绩。

该协议依赖 GT 来限制 rollout 长度，属于 oracle-efficiency eval，不等同于 GT-blind deployment。两种结果必须分开报告。

## 2. 冻结得分目标

### 2.1 V37 core promotion

这是从 V36 进入正式 production 长跑的最低性能门槛，不等于 top3：

| suite | V36 最好结果 | core promotion | 至少增加 |
|---|---:|---:|---:|
| pixmo-test | 433/529 = 81.85% | 440/529 = 83.18% | +7 cases |
| stepcount-500 | 76/500 = 15.20% | 90/500 = 18.00% | +14 cases |

两个门槛必须由同一 checkpoint 同时达到，并通过 paired sample-level 检验、训练完整性和 provenance gate。

### 2.2 V37 target：远超过 SFT-1M 且 Dense/Extreme 达到 top3 点估计

| suite | SFT-1M bf16 历史点估计 | V37 target | 目标含义 |
|---|---:|---:|---|
| pixmo-test | 约 78.45%--79.58% | >=445/529 = 84.12% | 相对冻结 SFT 约 +5pp 以上 |
| stepcount-500 | 约 13.39%--14.00% | >=95/500 = 19.00% | 相对冻结 SFT 约 +5pp 以上 |
| StepCount-Dense | 137/1229 约 11.15% | >=204/1229 = 16.60% | 严格超过当前第三名 16.52% |
| StepCount-Extreme | 7/1471 约 0.48% | >=35/1471 = 2.38% | 严格超过当前第三名 2.31%，约为 SFT 的 5 倍 |

Dense/Extreme leaderboard 混合了不同模型、prompt 和推理协议，因此 `204/1229`、`35/1471` 在完成同协议外部复评前只能称为 **top3-equivalent point-estimate threshold**，不能写成严格同协议排名。

### 2.3 “远超过 SFT-1M”的统计条件

正式表述还必须满足：

- 现场重跑 canonical SFT-1M，而不是从多个历史 SFT 数字中选择较低者；
- 保存逐题 correct vector，报告 discordant pairs、exact McNemar 和 paired/cluster bootstrap 95% CI；
- 四个 suite 使用 Holm correction；
- PixMo、StepCount-500、Dense：绝对提升至少 5pp，且 paired 95% CI 下界大于 +2pp；
- Extreme：绝对提升至少 1pp、至少 3x SFT EM，且 paired 95% CI 下界大于 0；
- 只达到 aggregate 点估计但未满足上述条件时，只能写“高于历史点估计”。

## 3. 第一性原理差距分析

### 3.1 V36 已学到什么

- Dense EM 从历史 RL 的约 5% 恢复到 step60 的 9.36%，但仍低于 SFT-1M 的 11.15%。
- Extreme EM 仍只有 0.34%，但 MAE 从 SFT 的 229.859 降到 39.909，Within-10% 从 5.10% 升到 48.54%。
- 这说明 V36 学会了更合理的数量尺度、停止和少重复，不等于学会 exact enumeration。
- 约 116 万个训练 point 中 unique mask hit 21.21%、duplicate 9.00%、miss 69.80%。把所有 mask miss 都定义为 answer loser 会让 mask false negative 主导 winner 路由。

### 3.2 V37 必须修复的机制

1. `outcome_success` 只回答“最终答案与轨迹结构是否可作为正确 winner”；mask miss/duplicate 进入 `trusted_trajectory` 和 process credit，不抹掉正确 answer reward。
2. baseline/progress 只改变 native action-level credit：baseline=`bok_grpo`，progress=`bok_grpo_step`；两臂共享数据、seed、outcome、KL、LR 和 scheduler。
3. adaptive actor KL 使用 valid response token 的全局均值，只进入 actor loss，不进入 BoK selector。
4. GT+3、首 action tag stop、strict integer answer 和 count_number 连续性必须保持 fail-closed。
5. OOM、nonfinite 或 skipped optimizer update 都使当前 cell 无效，不能把 attempted step 当 effective update。

### 3.3 Extreme 的不可绕过支持集问题

现有 V37 训练数据只覆盖 2--50，而 Extreme 是 51--500。仅在 2--50 上继续 RL，可能改善停止与尺度迁移，但没有足够监督让模型稳定完成 51--500 exact counting。因此：

- 本轮 16 小时只验证 V37 credit assignment 和系统稳定性，不承诺 Extreme top3；
- 下一阶段必须从 benchmark 之外生成独立 51--100 数据和 mask，先覆盖 Extreme 中最大的 51--100 bucket（800/1471）；
- 初始 long-count curriculum 建议 51--70 / 71--85 / 86--100 = 40% / 35% / 25%，并混入 2--50 replay 防止遗忘；
- 101--500 需要另行验证 context length、turn scheduler、KV cache 和 context parallel，未验证前不能直接扩大到 500 turns。

只要在 51--100 的 800 个样本上新增约 35 个 exact case，就可能跨过 Extreme top3 点估计；这比一开始强行训练 500 turns 更可验证、更节省算力。

### 3.4 真实 broad 错误结构对数据方案的约束

2026-07-23 的 3512 题 item-aligned 分析显示，V36 step60 相比本地 SFT raw 的净变化为 `v36-only=185`、`SFT-only=117`，但它同时产生了 `1517/3512` 个无显式整数答案；step77 为 `1524/3512`。这不是单纯的视觉计数误差，而是视觉覆盖、停止和答案闭合共同造成的系统误差。该比较的 SFT 与 V36 推理协议并未证明完全一致，因此只作为训练设计证据，不能作为 model-only 提升声明。

production 数据不能只继续增加 11--30 dense 样本。预注册的能力/来源 sampling prior 为：broad natural 20%、dense 11--30 20%、dense 31--50 15%、独立 extreme 51+ 15%、bias/repeated pattern 12%、clutter/aerial/retail hard source 8%、format/explicit-answer 5%、clean replay 5%。与之正交的错误角色 prior 为：late-stop/overcount 25%、early-stop/undercount 20%、no-answer/cap 25%、answer-trace inconsistency 10%、duplicate/coverage 15%、clean anchor 5%。两组比例都必须各自合计 100%，并通过消融验证，不能把相关性直接当成最优配方。

本轮 16 小时 diagnostic 仍使用 focused10k，只验证机制和系统稳定性；它不验证上述 production mixture，也不能据此声明达到最终得分目标。

## 4. 16 小时窗口的实验设计

历史 8xH200 实测平均每个 optimizer update 约 2670.8 秒（44.5 分钟）。完整 `2 seeds x 2 arms x 12 updates` 约需 35.6 小时，尚未包含四次启动、validation、checkpoint 和 merge，因此不能在 16 小时内诚实完成。

本窗口执行四个 clean-start diagnostic cell，使用 AB/BA 顺序抵消部分时间漂移：

1. seed11 baseline，3 effective updates；
2. seed11 progress，3 effective updates；
3. seed22 progress，3 effective updates；
4. seed22 baseline，3 effective updates。

总计 12 updates，纯训练估计 8.9 小时；其余预算用于四次加载、独立 heldout validation、checkpoint 和 Ray teardown。所有 cell 串行独占 8 张 H200。

固定安全配置：

```text
micro_update=4
micro_exp=8
vllm_num_gpu_blocks=20480
gpu_memory_utilization=0.50
max_num_batched_tokens=49152
context_parallel=1
rollout_n=16
```

该窗口不做超参网格、不做 CP>1、不并行 merge、不用 benchmark 作训练 val。结果只能回答：

- outcome winner 与 native action ledger 是否在真实 rollout 中对齐；
- progress 是否产生非零且方向正确的 action advantage；
- 两个 seed 的 reward/KL/grad/unique-hit/duplicate 趋势是否一致；
- 保守资源配置是否零 OOM、零 nonfinite、零 skipped update。

它不能回答模型是否已经达到最终四套 benchmark 目标，因此所有产物永久标记 `non-promotable diagnostic`。

## 5. Diagnostic GO/NO-GO

进入正式 frontier A/B 的必要条件：

- 四个 cell 全部完成预注册 effective updates；
- OOM、validation fallback、nonfinite、GradSpike skip 均为 0；
- baseline/progress 的唯一差异是 native action-level credit；
- 两个 seed 中 progress 的 exact-answer/五桶 macro 均不低于 baseline；
- progress 的 unique valid hit 提高，duplicate、format fail、cap/early-stop 不恶化；
- selector KL contribution=0，adaptive beta 有限、可恢复且变化方向正确；
- checkpoint、manifest、训练日志和实际 step 一致。

任一条件失败即 `NO-GO` 或 `INCONCLUSIVE`。不得自动挑 winner，不得从部分完成 cell 续出一个“formal”结果。

## 6. 后续 production plan

1. 用本轮 diagnostic 的真实 winner yield、每步 wall time、P95/P99 response length 和显存峰值校准正式预算。
2. 生成两个 seed、M=32 的 frontier audit；只保留未参与 benchmark 的代表性 heldout。
3. 完成 `2 seeds x 2 arms x 12 effective updates` formal mechanism A/B。
4. 方向一致后，从 clean SFT 起点按预注册 30--50 effective updates 训练 production candidate，不把 12-step canary 当最终模型。
5. 增加独立 51--100 curriculum 后再追求 Extreme top3；保留 2--50 replay 和 KL retention gate。
6. 最终只评一次冻结的 PixMo、StepCount-500、Dense、Extreme，并与现场重跑 SFT 做 paired 比较。

## 7. 结论

V37 的合理顺序是：先用 16 小时证明 reward routing、action-level credit 和 8xH200 系统稳定，再投入正式 frontier 与 production；Dense top3 可以由 2--50 能力直接追求，Extreme top3 必须补 51--100 训练支持。任何绕过支持集、统计口径或完整分母得到的“top3”，都不是本计划接受的结果。
