# 2025-2026 GRPO / Agentic RL 中 Trajectory-Level 与 Step-Level Reward 设计调研、多 Agent Debate 与 StepCount 方案结论

日期：2026-04-27

## 0. 执行摘要

本轮基于联网检索、多个子 Agent 交叉分析和 reviewer-style debate，针对以下问题做了系统调研：

> 2025-2026 年 GRPO / RLVR / agentic RL / VLM active perception 相关论文与 GitHub 项目中，面对 long-horizon tasks 和 agentic-rl 场景时，是否已有兼顾 trajectory-level final outcome reward 与 step-level / process reward 的具体设计与实现？这些设计对 StepCount 的 BoK-GRPO + step-level point reward hybrid 有什么启发？

核心结论：

1. **已有大量“全局 outcome + 局部 process/action signal”的相关设计，但没有与 StepCount 完全同构的 visual point-to-count 方案。**
2. 相关工作给出的共同规律是：
   - final answer / task success 必须保持主导；
   - step/process reward 很有价值，但容易 reward hacking；
   - long-horizon agentic RL 必须做 turn/action/trace 级诊断；
   - step reward 最好 gated、masked、clipped，并且单独监控 advantage scale。
3. 对 StepCount 最直接的外部支撑来自：
   - **DeepEyes / DeepEyesV2**：VLM interleaved active perception / tool use；
   - **HopChain**：numeric final-answer VLM RLVR 与 instance-grounded multi-hop structure；
   - **ToolRL**：GRPO/PPO 下 decomposed tool/action reward；
   - **RAGEN / Agent Lightning**：long-horizon agent RL 的 trajectory / turn-level credit 与 diagnostics；
   - **PRIME / PURE / ImplicitPRM**：outcome reward 与 process reward 的结合，以及 process reward hacking 风险；
   - **ReTool**：interleaved tool execution 中 outcome-driven reward 的强 cautionary evidence；
   - **Agentic-R**：local utility 与 global answer correctness 的 local-global coupling；
   - **verl / slime**：实现层基础设施，不是方法论 novelty 依据。
4. 多 Agent debate 后的推荐不是“纯 step-level GSPO”，也不是“把 process reward 简单加到 BoK 总分”，而是：

```text
Soft-Gated BoK-Step GRPO
A_total(i,t) = A_BoK_traj(i) + λ * G_soft(i) * M_point(i,t) * A_point_step(i,t)
```

其中：

- `A_BoK_traj`：保留当前 BoK-GRPO trajectory-level pass@K → pass@1 主目标；
- `A_point_step`：只基于 point step reward 的局部 advantage；
- `M_point`：只作用于 point-step token span，排除 final answer token；
- `G_soft`：使用 answer/trajectory quality 的 soft gate，避免 hard answer gate 的冷启动问题；
- `λ`：建议先 `0.1`，主候选 `0.2`，不建议一开始超过 `0.3`。

---

## 1. 调研范围与方法

### 1.1 重点问题

本次调研关注的不是泛泛的 RLHF，而是以下交集：

- GRPO / RLVR / PPO / DAPO / Dr.GRPO / GSPO 相关 reasoning RL；
- long-horizon agentic RL；
- multi-turn tool use / search / code execution / active perception；
- 同时存在 final trajectory outcome 与 step/process/action reward 的设计；
- 有公开 GitHub / framework / reward implementation 的项目优先。

### 1.2 多 Agent 分工

本轮使用了三类角色：

1. **Literature Scout Agent**：联网搜索 2025-2026 相关论文和项目，输出结构化文献图谱。
2. **Reviewer Agent**：以顶会 reviewer 视角质疑 hybrid 方案，指出缺失证据、overclaim、ablation 和 reward hacking 风险。
3. **StepCount Specialist Agent**：结合 StepCount 因果结构，把外部文献映射为具体算法建议和低算力优先级。

### 1.3 检索和验证来源

重点核验过的公开来源包括：

- DeepEyes: https://arxiv.org/abs/2505.14362, https://github.com/Visual-Agent/DeepEyes
- DeepEyesV2: https://arxiv.org/abs/2511.05271, https://github.com/Visual-Agent/DeepEyesV2
- HopChain: https://arxiv.org/abs/2603.17024
- ToolRL: https://arxiv.org/abs/2504.13958, https://github.com/qiancheng0/ToolRL
- RAGEN / RAGEN-2: https://arxiv.org/abs/2504.20073, https://github.com/mll-lab-nu/RAGEN, https://arxiv.org/abs/2604.06268
- Agent Lightning: https://arxiv.org/abs/2508.03680, https://github.com/microsoft/agent-lightning
- PRIME: https://arxiv.org/abs/2502.01456, https://github.com/PRIME-RL/PRIME
- PURE: https://arxiv.org/abs/2504.15275, https://github.com/CJReinforce/PURE
- Agentic-R: https://arxiv.org/abs/2601.11888, https://github.com/8421BCD/Agentic-R
- ReTool: https://openreview.net/forum?id=tRk1nofSmz, https://github.com/ReTool-RL/ReTool
- verl: https://github.com/verl-project/verl, https://verl.readthedocs.io/en/latest/algo/grpo.html, https://verl.readthedocs.io/en/latest/start/agentic_rl.html
- slime: https://github.com/THUDM/slime

说明：用户提到的 “SMILE” 在本次检索中没有验证到与 VLM / agentic GRPO 直接相关的同名核心项目；较可能指的是 **THUDM slime**，本文按 slime 作为 infrastructure 讨论。

---

## 2. 相关工作总览表

| 工作 / 项目 | 年份 | 领域 | 优化器 / 框架 | Reward 粒度 | 是否显式结合 final + step/process | 公开实现 | 对 StepCount 的启发 |
|---|---:|---|---|---|---|---|---|
| DeepEyes | 2025/2026 | VLM active perception | 基于 verl 的 RL，论文/项目强调 end-to-end RL | outcome reward / tailored reward strategy；README 强调无直接 intermediate supervision | 弱到中：主要是 outcome-driven active perception，不是 point-level reward | 有 | 最接近“think with images / active visual action”的 VLM 参考；说明 interleaved visual action 能提升 grounding |
| DeepEyesV2 | 2025/2026 | Agentic multimodal model | cold-start SFT + RL，使用 DeepEyes RL codebase | final accuracy / format / tool-use patterns | 弱：RL 主要 refine tool invocation，直接复杂 tool reward 需谨慎 | 有 | 说明 direct RL 难以自发学稳定 tool protocol，cold-start 和 reward safety 重要 |
| HopChain | 2026 | Multi-hop VLM RLVR | SAPO / RLVR | final numeric answer reward；结构性 multi-hop grounding | 否：结构化中间过程，但 reward 仍是 final numeric | 论文为主 | 是 StepCount 最强 counterpoint：不一定需要 step reward；必须证明 point reward 有额外价值 |
| ToolRL | 2025 | Tool learning | GRPO / PPO | tool name / args / values / correctness / format 等 decomposed rewards | 是：细粒度工具调用 reward 与任务 reward 结合 | 有 | 支持 decomposed action reward，但也要求 scale、granularity、dynamic schedule |
| RAGEN / RAGEN-2 | 2025/2026 | Multi-turn agent RL | StarPO，支持 PPO / GRPO | trajectory-level 和 turn-wise | 框架层支持；GRPO 下可把 normalized reward 分配到 full trajectory | 有 | 强调 long-horizon diagnostics、reward variance filtering、collapse detection |
| Agent Lightning | 2025 | 通用 agent RL framework | LightningRL / hierarchical credit | trace / transition / arbitrary reward | 是：把 agent trace 分解成 transitions 做 credit assignment | 有 | 支持 StepCount 把 point 视为状态转移动作，而非普通 token |
| PRIME | 2025 | math/code process RL | PPO / RLOO-style advantage | outcome verifier + implicit process reward | 是：outcome return 与 process return 组合 | 有 | 直接支持 outcome + process 混合 advantage 的思想 |
| PURE | 2025 | PRM + VR credit assignment | OpenRLHF / verl 分支 | process reward / verifiable reward | 是：PURE-PRM、PURE-VR、PURE-PRM+VR | 有 | 关键 warning：sum-form process reward 可能诱发 reward hacking，credit assignment 形式很重要 |
| ReTool | 2026 ICLR | Interleaved code tool use | PPO | final outcome reward；作者反馈 intermediate syntax/execution rewards 可能有害 | 主方法偏 outcome-driven；shaped step reward 被视为 caution | 有部分 | 强 caution：中间 reward 可能让模型产出短而安全但无解题能力的 tool calls |
| Agentic-R | 2026 | Agentic search retriever | Search-R1/PPO + retriever training | local passage relevance + global answer correctness | 对 retriever 是显式 local-global 结合；agent policy 本身偏 final EM | 有 | 类比 StepCount：point 的局部价值应由最终 count 贡献校准 |
| verl | 2024-2026 | RL infrastructure | GRPO/PPO/DAPO/GSPO/DrGRPO 等 | function reward / model reward / multi-turn tool reward | 基础设施支持，不提供唯一方法 | 有 | 当前 EasyR1-latest 的底层思想来源；支持 multi-turn/tool/token API |
| slime | 2025-2026 | RL scaling infrastructure | Megatron + SGLang | custom data generation / verifier output | 基础设施支持 | 有 | 可作为 scaling reference；不是 StepCount 方法 novelty 来源 |

---

## 3. 分方向深入分析

## 3.1 VLM active perception：DeepEyes / DeepEyesV2

### 3.1.1 已验证信息

DeepEyes 的 arXiv 摘要和 GitHub README 强调：

- VLM 不应只在文本 reasoning 中处理图像，而应“think with images”；
- 模型通过 active perception 学会在推理中使用视觉信息；
- README 明确强调其能力通过 end-to-end RL 学得，且不依赖 pre-collected reasoning data 的 cold-start SFT；
- README 还提到没有直接 intermediate supervision，但 grounding IoU 和 tool-calling efficiency 在 RL 中上升。

DeepEyesV2 进一步扩展到 code execution / image-text search / reasoning 统一 agentic multimodal model。其 arXiv 摘要和 GitHub README 强调：

- direct RL alone 难以诱导 robust tool-use behavior；
- 采用 cold-start SFT + RL 两阶段；
- tool invocation 在 RL 后变得更复杂、更上下文自适应；
- 训练需要高资源和外部 judge / sandbox / search 服务。

### 3.1.2 对 StepCount 的意义

DeepEyes 系列对 StepCount 的支撑非常强，但主要支撑的是 **范式**，不是具体 reward 公式：

```text
VLM 可以通过 interleaved visual actions 改变后续观察，从而提升视觉 reasoning。
```

StepCount 的 point-to-count 与 DeepEyes 的 crop/zoom/search 类似，都是显式动态视觉注意力转移。但 StepCount 更特殊：

- point action 不只是观察工具调用，也是 object instance grounding；
- 每一步 point 会被红点标注回图像，改变下一步视觉输入；
- 最后 count answer 与 point 序列有强因果关系。

### 3.1.3 对 hybrid 的启发

DeepEyes 提醒我们：

1. 不一定必须强行给每个中间 action 高 reward；outcome-driven RL 也可能诱导中间能力出现。
2. 但 StepCount 有比 DeepEyes 更强的 step supervision：mask point reward 是可验证的，因此可以作为辅助 credit。
3. DeepEyesV2 的 cold-start 和 tool protocol 经验说明：如果格式和 action protocol 不稳，RL 会先学坏格式而不是学视觉过程。

因此 StepCount 的正确设计应是：

```text
trajectory-level final answer 主导 + bounded step-level point auxiliary
```

而不是：

```text
step reward 与 answer reward 等权竞争
```

---

## 3.2 Numeric VLM RLVR：HopChain

### 3.2.1 已验证信息

HopChain 关注 multi-hop VLM reasoning data synthesis。其摘要中明确提到：

- 当前 VLM 在 fine-grained vision-language reasoning 上仍有不足；
- long CoT 会暴露 perception、reasoning、knowledge、hallucination 等错误，并会跨中间步骤累积；
- HopChain 合成 logically dependent chain of instance-grounded hops；
- final answer 是 specific, unambiguous number，适合 verifiable rewards；
- 训练后在 24 个 benchmark 中 20 个提升。

### 3.2.2 对 StepCount 的意义

HopChain 是 StepCount 的强相关工作，也是强 counterpoint。

支持点：

- StepCount 同样是 final numeric answer；
- 同样需要 instance-grounded intermediate reasoning；
- 同样关注中间错误累计。

反对点：

- HopChain 通过数据结构引入 multi-hop grounding，但 reward 仍偏 final numeric；
- 如果 HopChain-style outcome-only RLVR 已经足够，那么 StepCount 的 step reward 必须证明额外价值。

### 3.2.3 对 hybrid 的启发

HopChain 说明：

```text
结构化中间过程 + final numeric reward 已经是一条强 baseline。
```

因此 StepCount 论文/实验必须包含：

1. outcome-only BoK-GRPO；
2. BoK-GRPO + process reward 但仍 sum 到 trajectory；
3. 真正 step-level hybrid；
4. point-answer correlation 和 error accumulation 分析。

否则 reviewer 会问：为什么 final-only 不够？

---

## 3.3 Tool / action reward decomposition：ToolRL

### 3.3.1 已验证信息

ToolRL 的 arXiv 摘要和 GitHub README 显示：

- 目标是 tool selection and application；
- 研究 reward design 的类型、尺度、粒度和 temporal dynamics；
- 使用 GRPO 训练 LLM tool use；
- GitHub 明确提供多种 reward variants：length reward、dynamic scale、fine-grained reward、intermediate reward、coarse reward 等；
- 支持 GRPO 和 PPO training scripts。

### 3.3.2 对 StepCount 的意义

ToolRL 说明 decomposed reward 是合理方向。StepCount point action 可以类比 tool call：

| ToolRL | StepCount |
|---|---|
| tool name 正确 | `<point>` 格式和 point action 类型正确 |
| parameter key/value 正确 | `point_2d` JSON 和坐标准确 |
| final task correctness | final count answer 正确 |
| tool-call intermediate reward | point mask hit reward |

### 3.3.3 风险

ToolRL 也提醒：

- reward scale 会决定训练行为；
- coarse / fine-grained / intermediate reward 不是越多越好；
- 如果局部 reward 太强，模型可能只优化工具参数而不是最终任务。

StepCount 对应风险：

- duplicate points；
- point spam；
- 坐标格式过拟合；
- points 看似命中 mask，但最终 count 错；
- answer 对但 point 错，造成“伪 grounding”。

---

## 3.4 Long-horizon agent RL diagnostics：RAGEN / RAGEN-2

### 3.4.1 已验证信息

RAGEN GitHub README 和 arXiv 摘要显示：

- RAGEN 是 multi-turn LLM agent RL framework；
- StarPO 把 agent-environment interaction 表述为 MDP；
- rollout stage 生成多条 trajectory；
- update stage 支持 PPO 和 GRPO；
- GRPO 分支中 normalized reward assigned to full trajectory；
- RAGEN-2 引入 SNR-Adaptive Filtering，用 reward variance 过滤高信号 prompts；
- RAGEN-2 关注 template collapse，使用 conditional entropy / mutual information 诊断。

### 3.4.2 对 StepCount 的意义

StepCount 与 RAGEN 的共同点：

```text
都是 long-horizon trajectory，局部 action 会改变后续 state，final reward 可能稀疏。
```

RAGEN 对 StepCount 的最大启发不是某个公式，而是诊断体系：

- 不要只看 `val/answer_reward`；
- 要看 point sequence 的局部质量；
- 要看 reward variance；
- 要看是否出现 template collapse 或 reasoning collapse；
- 要按 group / prompt / step 做统计。

### 3.4.3 对 hybrid 的启发

当前 StepCount 的 BoK-GRPO 正是 trajectory-level：它类似 RAGEN 中 GRPO 把 reward assigned to full trajectory 的做法。

因此若要真正 step-level credit，需要显式新增：

```text
point-step advantage path
```

而不是只把 `_step_rewards` 放入 token reward tensor 后再 sum。

---

## 3.5 通用 agent trace credit：Agent Lightning

### 3.5.1 已验证信息

Agent Lightning arXiv 摘要和 GitHub README 显示：

- 目标是训练任意 AI agents；
- agent execution 与 training decoupled；
- 将 agent execution 表述为 MDP；
- LightningRL 包含 credit assignment module；
- 可把 arbitrary agent trajectories 分解为 training transitions；
- 支持 text-to-SQL、RAG、math tool-use 等场景。

### 3.5.2 对 StepCount 的意义

StepCount 现在是在 EasyR1 trajectory 模式中实现，而不是完整 agent framework。但概念上完全符合 Agent Lightning 的 trace-to-transition 思想：

```text
point_k 不是普通文本，而是一次 transition action。
```

因此 StepCount hybrid 的理论解释可以写成：

```text
trajectory-level answer reward handles global outcome selection;
point-level auxiliary advantage handles transition-local visual grounding correction.
```

---

## 3.6 Outcome + process reward：PRIME / PURE / ImplicitPRM

### 3.6.1 PRIME 已验证信息

PRIME arXiv 和 README 显示：

- dense process rewards 可缓解 sparse outcome reward 的 credit assignment 问题；
- 难点是 process labels 昂贵，PRM 容易 reward hacking；
- PRIME 使用 implicit PRM，从 outcome labels 在线更新 process reward；
- policy update 中 combination of outcome rewards and process rewards；
- 使用 RLOO-style advantage。

### 3.6.2 PURE 已验证信息

PURE arXiv 和 README 显示：

- PRM reward 在 RL 中会出现 reward hacking；
- 问题部分来自 canonical summation-form credit assignment；
- 提出 min-form credit assignment；
- 支持 PURE-PRM、PURE-VR、PURE-PRM+VR；
- NeurIPS 2025 accepted。

### 3.6.3 对 StepCount 的意义

PRIME/PURE 是 StepCount hybrid 的方法论最强支持之一：

```text
outcome reward + process reward 可以组合，但 process reward 不能无脑求和。
```

StepCount 的 `_step_rewards` 如果只是被 BoK sum 到 trajectory score，本质仍是 sum-form shaping，不能解决局部 credit assignment。

PURE 还提醒：如果把每一步 point reward 简单累计，模型可能学会“刷局部奖励”：

- 多点 spam；
- 重复命中同一个大目标；
- 靠坐标先验而非真实视觉；
- 把 point 数量当 answer shortcut。

因此 StepCount 不应只增加 point weight，而要在 advantage 层做：

```text
step-only path + gate + mask + clip + anti-hacking metrics
```

---

## 3.7 Interleaved tool execution caution：ReTool

### 3.7.1 已验证信息

ReTool OpenReview 页面和 GitHub README 显示：

- ReTool 训练 LLM 在 reasoning 中动态调用 code interpreter；
- 使用 cold-start code-augmented reasoning data；
- RL 阶段主要基于 final task outcome reward；
- reviewer / author discussion 中明确提到 intermediate syntax/executability rewards 可能鼓励短而安全的 code，而不是解决完整问题；
- PPO 是其默认优化器，作者表示 PPO 和 GRPO 并非核心差异。

### 3.7.2 对 StepCount 的意义

ReTool 是 StepCount 的重要反例警告：

```text
中间 action reward 不一定帮助 long-horizon task，有时会压制探索和最终任务表现。
```

对应到 StepCount：

- 如果 point reward 过强，模型可能更重视“给出看起来正确的点”而非 final count；
- 如果格式/坐标 reward 过强，模型可能产生短、安全、格式漂亮但视觉错误的轨迹；
- 如果 hard gate 只奖励 answer-correct 轨迹，near-correct 轨迹的 point learning 会被抑制。

因此 ReTool 支持我们选择：

```text
soft-gated, low-weight, point-only auxiliary advantage
```

而不是 direct step reward dominance。

---

## 3.8 Local-global utility：Agentic-R

### 3.8.1 已验证信息

Agentic-R arXiv 和 GitHub README 显示：

- 目标是为 agentic search 训练 retriever；
- 与普通 RAG 只用 local passage relevance 不同，Agentic-R 同时使用 local query-passage relevance 和 global answer correctness 衡量 passage utility；
- 搜索 agent 使用 Search-R1/PPO 训练；
- retriever 训练通过 agent trajectory 生成 query / candidates / local utility / global utility。

### 3.8.2 对 StepCount 的意义

Agentic-R 与 StepCount 的核心类比：

| Agentic-R | StepCount |
|---|---|
| passage 是 local retrieval action 的结果 | point 是 local visual grounding action |
| local relevance 不等于 final answer correctness | point hit mask 不等于 final count correctness |
| global answer correctness 校准 passage utility | final count correctness 校准 point utility |

因此 StepCount 的 point reward 不应被视为绝对正确，而应被 final trajectory quality soft-gate。

---

## 3.9 Infrastructure：verl / slime

### 3.9.1 verl 已验证信息

verl GRPO 文档显示：

- GRPO 是 critic-less；
- 对每个 prompt 采样多个 completions；
- reward 在 group 内归一化；
- verl 提醒 long-CoT 中 sample-level loss aggregation 可能不稳；
- 支持 DrGRPO、DAPO、GSPO、RLOO、REINFORCE++ 等 recipe；
- agentic RL 文档强调 multi-turn/tool calling、server-based async rollout、token-based API，以避免 tool call text/token drift 影响 advantage 计算。

### 3.9.2 slime 已验证信息

slime 是 THUDM 的 RL scaling framework：

- 连接 Megatron training 与 SGLang rollout；
- 支持 flexible data generation 和 verifier output；
- 有多个 agentic / omni-modal / verifiable environment 项目 built upon slime；
- 属于 scaling infrastructure。

### 3.9.3 对 StepCount 的意义

verl / slime 应在论文或文档中作为 implementation context，而不是 scientific novelty 证据。

对当前 EasyR1-latest：

- 当前已具备 trajectory 模式；
- 当前已有 `bok_grpo` 和 `grpo_step` 两条路径；
- 当前缺少的是中间层 hybrid estimator；
- 实现时应遵循 verl agentic RL 的 token/mask 严谨性，避免 retokenization / reward position drift。

---

## 4. 多 Agent Debate 结论

## 4.1 Literature Scout Agent 的主张

Scout 认为：

1. StepCount 相关工作可以分成四类：
   - VLM active perception：DeepEyes / DeepEyesV2；
   - agentic tool/search RL：ToolRL / ReTool / Agentic-R；
   - long-horizon agent framework：RAGEN / Agent Lightning / verl / slime；
   - outcome-process reward：PRIME / PURE / ImplicitPRM。
2. 这些工作共同支持 StepCount 使用 trajectory reward + process reward 的方向。
3. 但没有一个工作直接实现“VLM interleaved point-to-count + point mask step reward + final answer BoK”。
4. 推荐 StepCount 以 outcome-gated process reward 和 local-global coupling 作为理论线索。

## 4.2 Reviewer Agent 的反驳

Reviewer 认为当前 hybrid 仍有高风险：

1. “point 变好 → count 变好”是核心假设，不是已证明事实。
2. answer-correct hard gate 会 suppress near-correct but answer-wrong trajectories。
3. `A_BoK_traj` 与 `A_point_step` 可能产生梯度冲突。
4. step advantage 的 scale、normalization、token mask、final answer leakage 没有讲清就实现，会被认为是 reward engineering。
5. 如果没有 outcome-only、ungated、hard-gated、soft-gated、mask/λ ablation，论文很容易被认为 incremental。

## 4.3 StepCount Specialist Agent 的综合

StepCount Specialist 认为：

1. StepCount 的因果图确实不同于普通 VQA：每一步 point 是 state transition action。
2. 所以最终设计应是 hierarchical hybrid，而不是 pure trajectory 或 pure step。
3. 推荐默认：

```text
A_total(i,t) = A_BoK_traj(i) + λ * G_soft(i) * M_point(i,t) * A_point_step(i,t)
```

4. 默认 gate 不应是 hard exact answer，而应是 soft trajectory quality / soft count closeness。
5. 实现前或同时必须加入 diagnostics：point-answer correlation、duplicate/spam、advantage scale、gate distribution、mask metadata alignment。

## 4.4 Debate 后最终裁决

最终裁决：

```text
支持实现 bok_grpo_step，但不支持无诊断地直接替换主线。
```

优先级：

1. 保留 outcome-only BoK-GRPO 作为安全 baseline；
2. 新增 `bok_grpo_step` estimator；
3. 新增独立 ablation 脚本；
4. 使用 `PROCESS_REWARD_ENABLE=1`；
5. 默认 `BOK_STEP_WEIGHT=0.1` 安全验证，`0.2` 主候选；
6. 默认 `BOK_STEP_GATE=soft_count_closeness` 或 `answer_soft`；
7. `BOK_STEP_EXCLUDE_FINAL=1`；
8. 加强日志诊断。

---

## 5. 对 StepCount 当前 BoK-GRPO 的直接启发

### 5.1 为什么不能只打开 `PROCESS_REWARD_ENABLE=1`

当前 BoK-GRPO 的关键问题是：

```text
scores = token_level_rewards.sum(dim=-1)
```

这意味着：

```text
point step reward -> token tensor -> sum -> trajectory scalar -> BoK advantage
```

所以 `PROCESS_REWARD_ENABLE=1 + ADV_ESTIMATOR=bok_grpo` 只改变 reward 位置，不改变 credit assignment 粒度。

这对应文献中的 warning：

- PURE 反对 naive sum-form process reward；
- RAGEN 说明 GRPO 会把 normalized reward assign 到 full trajectory；
- ReTool 警告 intermediate reward 可能 harmful；
- Agent Lightning 强调 trace-to-transition credit assignment 必须显式建模。

### 5.2 为什么不能纯 `grpo_step`

纯 step-level 会让 point reward 过度主导。

对 StepCount 来说，评测只看 final answer pass@1；point 是手段，不是最终目标。

因此 pure GSPO 的风险是：

- point reward 上升但 answer 不升；
- 模型 point spam；
- final answer 被局部 reward 干扰；
- 训练不再服务 pass@K → pass@1。

这与 ToolRL / ReTool / PURE 的 warning 一致。

### 5.3 为什么 soft-gated hybrid 更合理

正确目标是：

```text
global outcome selection + local action correction
```

其中：

- BoK-GRPO 做 global selection；
- StepGRPO 做 local correction；
- Soft gate 做 local-global consistency calibration；
- Mask/clip 防止泄漏和高方差。

推荐公式：

```text
A_total(i,t)=clip(
    A_BoK_traj(i)
    + λ * G_soft(i) * M_point(i,t) * clip(A_point_step(i,t), -c_step, c_step),
    -c_total,
    c_total
)
```

推荐默认值：

```text
ADV_ESTIMATOR=bok_grpo_step
PROCESS_REWARD_ENABLE=1
BOK_STEP_WEIGHT=0.1  # first safety run
BOK_STEP_WEIGHT=0.2  # main candidate
BOK_STEP_GATE=soft_count_closeness 或 answer_soft
BOK_STEP_MIN_GATE=0.15 或 0.20
BOK_STEP_EXCLUDE_FINAL=1
BOK_STEP_CLIP=2.0~2.5
BOK_HYBRID_CLIP=4.0
```

---

## 6. 推荐 gate 设计

### 6.1 不推荐 hard answer gate 作为默认

Hard gate：

```text
G = 1[answer_correct]
```

优点：

- 最保守；
- 不会增强 final-wrong 轨迹的 point reward。

缺点：

- 冷启动时 gate 激活太少；
- near-correct but wrong-answer 轨迹完全得不到 point 学习；
- counting 中错一个物体就 answer wrong，但前面多个 point 可能正确，硬门控浪费这些学习信号。

### 6.2 推荐 soft count closeness gate

若能 parse final answer，可以使用：

```text
q_i = exp(-abs(pred_count_i - gt_count) / max(1, sqrt(gt_count)))
G_i = g_min + (1 - g_min) * stop_gradient(q_i)
```

建议：

```text
g_min = 0.15 或 0.20
```

优点：

- answer 越接近 GT，point auxiliary 越强；
- answer 完全错也保留少量 point 学习；
- dense counting 中允许 near-miss 轨迹提供 point correction。

### 6.3 简化版 answer_soft gate

如果初期不想引入 continuous count closeness，可先用 reward function 已经给出的 `answer_scores`：

```text
G_i = max(answer_score_i, g_min)
```

这更容易实现，但对 off-by-one 和 count-distance 的区分不如 soft count closeness。

---

## 7. 推荐 ablation 矩阵

| 优先级 | 实验 | Estimator | Process reward | Gate | λ | 要回答的问题 |
|---:|---|---|---|---|---:|---|
| 0 | 离线 rollout 诊断 | none | none | none | 0 | point reward 是否真的预测 answer correctness |
| 1 | outcome-only BoK baseline | `bok_grpo` | off | none | 0 | 当前安全基线是什么 |
| 2 | BoK summed process control | `bok_grpo` | on | none | 0 | 只改变 reward placement 是否有影响 |
| 3 | soft hybrid safety | `bok_grpo_step` | on | soft | 0.1 | 弱 point auxiliary 是否无害 |
| 4 | soft hybrid main | `bok_grpo_step` | on | soft | 0.2 | 主候选是否提升 answer/point |
| 5 | ungated hybrid | `bok_grpo_step` | on | none | 0.2 | gate 是否必要 |
| 6 | hard-gated hybrid | `bok_grpo_step` | on | hard answer | 0.2 | hard gate 是否抑制 near-correct 学习 |
| 7 | strong hybrid | `bok_grpo_step` | on | soft | 0.3 | step auxiliary 的上限在哪里 |

低算力优先只跑前三个 GPU-consuming 实验：

```text
1. outcome-only BoK baseline
2. soft-gated hybrid λ=0.1
3. soft-gated hybrid λ=0.2
```

---

## 8. 必须新增或至少监控的诊断

| 诊断 | 目的 | 红旗 |
|---|---|---|
| point-answer correlation | 证明 point reward 与 final count 有关 | 相关性接近 0 或负相关 |
| answer-correct bad-point rate | 检查是否存在伪 grounding | 高说明 answer gate 会强化坏点 |
| answer-wrong good-point rate | 检查 hard gate 是否浪费信号 | 高说明不能用 hard gate |
| first bad point step | 分析错误累计 | first bad step 与 final error 无关 |
| duplicate point rate | 检测 point spam | hybrid 后上升 |
| point-count vs answer-count consistency | 检测点数和 answer 矛盾 | consistency violation 上升 |
| reward scale stats | 防止 point reward 主导 | point std 大于 answer/traj std |
| `A_point_std / A_BoK_std` | 直接看 auxiliary 强度 | 长期 > 0.5 |
| gate mean/p10/p50/p90 | 监控 gate 是否饱和 | 总在 min 或总接近 1 |
| format fail rate | 防止格式被破坏 | 持续上升 |
| entropy / KL / grad spike | 稳定性 | entropy spike 或 non-finite |

---

## 9. 推荐 novelty framing

不要把 novelty 写成：

```text
我们首次提出 outcome + process reward。
```

这是不成立的，PRIME/PURE/ToolRL/Agent Lightning 等都已有相似宏观思想。

也不要写成：

```text
我们首次提出 active perception VLM。
```

DeepEyes 系列已经非常接近。

更安全、更强的 framing 是：

```text
StepCount 是一个 counting-specific interleaved point-to-count RL paradigm：每一步 point 是可验证的 visual grounding action，也是改变下一步输入状态的显式 visual attention shift。我们在 trajectory-mode RL 中把 final count answer 作为主 outcome objective，同时用 soft-gated point-level auxiliary advantage 做局部 credit assignment，并系统诊断 point error 如何累积成 count error。
```

一句话版本：

```text
Outcome-aligned process reinforcement learning for visual counting with verifiable point-level grounding actions.
```

---

## 10. 对当前实现的具体结论

### 10.1 是否应该实现 `bok_grpo_step`

结论：应该实现，但必须作为独立 estimator 和独立 ablation 脚本，不覆盖当前 BoK 主线。

原因：

- 外部文献支持 hybrid 的必要性；
- reviewer debate 指出不能无诊断替换主线；
- 新 estimator 可通过 `BOK_STEP_WEIGHT=0` 退化为 BoK，风险可控；
- 独立脚本可保证当前 V27b restored baseline 不被污染。

### 10.2 最推荐实现版本

```text
Soft-Gated BoK-Step GRPO
```

代码命名：

```text
ADV_ESTIMATOR=bok_grpo_step
```

核心实现：

1. `A_BoK_traj` 调用现有 BoK-GRPO；
2. `A_point_step` 调用 step-level GRPO，但只用 point reward；
3. 清零每条 response final valid token 上的 answer/format reward，避免 final answer 被 step path 双重计算；
4. 用 soft gate 缩放 step auxiliary；
5. step component 和 total advantage 都 clip；
6. 新增日志记录 gate 和 advantage scale。

### 10.3 不推荐默认项

不推荐：

```text
BOK_STEP_GATE=hard_answer
BOK_STEP_WEIGHT>=0.3
pure grpo_step as main
PROCESS_REWARD_ENABLE=1 + bok_grpo 并声称是 step-level credit
```

---

## 11. 最终行动建议

如果继续实现，我建议下一步按以下顺序：

1. 新增 `bok_grpo_step` estimator；
2. 新增 `BOK_STEP_*` env 参数；
3. 新增独立脚本 `qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_bok_step_hybrid.sh`；
4. 默认：

```text
PROCESS_REWARD_ENABLE=1
BOK_STEP_WEIGHT=0.1
BOK_STEP_GATE=answer_soft 或 soft_count_closeness
BOK_STEP_MIN_GATE=0.2
BOK_STEP_EXCLUDE_FINAL=1
BOK_STEP_CLIP=2.5
BOK_HYBRID_CLIP=4.0
```

5. 跑 `py_compile` 和 `bash -n`；
6. 首个训练只作为 safety run，不直接宣称主线替换；
7. 并行准备离线诊断脚本，输出 point-answer correlation、duplicate rate、advantage scale。

---

## 12. 本轮最终判断

结合 2025-2026 相关论文与 GitHub 项目，StepCount 当前想兼顾 BoK-GRPO trajectory-level final answer 优化与 step-level point-to-count credit assignment 是合理且有文献支撑的。

但文献同时强烈警告：process/step reward 不是越强越好，naive summation 或 ungated auxiliary 很容易 reward hacking。

因此最稳妥、最符合第一性原理的实现路线是：

```text
trajectory-primary + soft-gated point-auxiliary + strict masking + advantage scale diagnostics
```

即：

```text
A_total = A_BoK_trajectory + λ * G_soft * M_point * A_step_point
```

这也是当前 StepCount 从“能训练”走向“可解释、可诊断、可发表”的关键下一步。
