# V26-C 第一性原理证明（v2 证据修正版）

**日期**: 2026-04-24（v2 修正）
**脚本**: `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v26c_sparse_verified.sh`
**基底**: `v26b.sh`（已参数化 RUN_NAME_PREFIX/SAVE_FREQ/VAL_FREQ）

## v2 相对 v1 的修正

| 参数 | v1（错误） | v2（修正） | 修正依据 |
|------|----------|----------|---------|
| BOK_CLIP | 3.0（引用 V12>V8 +1.89pp） | **4.0** | `docs/FirstPrinciples_NaN_and_DataStrategy_Analysis.md:160` 明确："V12 vs V8 differs in 5 hyperparameters simultaneously. The 1.89pp gap CANNOT be attributed to data alone." → CLIP=3.0 的 A/B 证据**不成立**。V23_SC（唯一 NaN=0 且 peak=0.7769）用的是 CLIP=4.0，应跟随该干净锚点。 |
| 数据 | easy-only | **mixed easy_plus_hard** | 核对 v25 原始日志：lr=1.5e-6 + mixed 前 120 步 val_ans 0.74-0.77、format 0.96-0.98 **完全稳定**；step 135 崩塌是 format 0.9792→0.0284 的**单次事件**，不是 mixed 数据的因果失败。之前"mixed 必导致 NaN"的推论过度归因。 |
| v25 归因 | "lr=2e-6 导致崩塌" | **lr 实际是 1.5e-6** | 原始日志显示 `"lr": 1.5e-06`，我的 v1 表格把 v25 记成 2e-6 是错的。v25 是"lr=1.5e-6 + mixed 能跑到 peak 0.7826" 的**成功证据**，不是失败样本。 |

---

## 0. 修正后的证据基线（已核对原始日志）

| Run | lr | ppo | kl | max_grad | BOK_CLIP | data | peak val_ans | format末 | NaN | 结局 |
|-----|----|----|------|----------|----------|------|--------------|----------|-----|------|
| v12 | 1e-6 | 2 | 0.04 | 0.5 | 3.0 | easy | 0.7731 @s45 | 0.97 | 273 | 漂移 |
| v22 | 1e-6 | 1 | 0.03 | 0.5 | 3.0 | easy | 0.7769 | 0.97 | 271 | 漂移 |
| **v23_SC** | **1e-6** | **1** | **0.03** | **1.0** | **4.0** | **easy** | **0.7769** | **0.97** | **0** | **收敛（唯一 NaN=0 锚点）** |
| v23+hard | 1e-6 | 1 | 0.03 | 1.0 | 4.0 | +hard | — | — | 4 | step45 entropy 1.447 崩 |
| v24 | 1e-6 | 1 | 0.03 | 1.0 | 3.0 | mixed | 0.7713 | 0.98 | 34 | 漂移 |
| **v25** | **1.5e-6** | **1** | **0.03** | **1.0** | **3.0** | **mixed** | **0.7826 @s150** | **0.0624** | **138** | **前 120 步稳定；step 135 单次事件后 format 失灵但 answer 继续爬** |
| v12fix | 2e-6 | 1 | 0.04 | 1.0 | 3.0 | mixed | 0.7845 | 0.0000 | 241 | format 崩 |

关键观察：
- v25 是 v26c-v2 的"最近实证邻居"：相同 lr、mixed、唯一差异是 CLIP 3.0 vs 4.0 和 SAVE/VAL_FREQ 15→10 和 GRAD_SPIKE_ABSOLUTE_CAP 5.0→2.5
- v25 前 120 步的稳定性是 v26c-v2 能超 v12 的直接证据
- v25 step 135 崩塌事件 = 必须加 spike cap=2.5 的直接理由

---

## 1. 第一性原理逐参数（修正版）

### 1.1 数据 = `mixed easy_plus_hard`
- 之前说"mixed 必 NaN" → 过度归因。v25 前 120 步 mixed 运行完好。
- 之前说 "v23+hard 崩塌" → 实际 v23+hard 用的是**单独 hard_only 路径**（很可能 oversample_factor>1），与这里标准 mixed 拼接数据不同。
- 结论：在 lr=1.5e-6 已有实证的前提下，mixed 数据未被证实有害，且能让模型接触难样本提升泛化。采用。

### 1.2 `actor_lr=1.5e-6`
- v25 原始日志 `"lr": 1.5e-06` 证明这个 lr **完全可以稳定训练 120+ 步**并达到 peak 0.7826。
- 比 v23_SC 的 1e-6 高 50%，有效学习步长够大能突破 v23_SC 的 0.7769 天花板。
- 结论：**基于实证，不是插值猜测**。v2 保留。

### 1.3 `BOK_CLIP=4.0`（v2 修正）
- 没有干净的 3.0 vs 4.0 A/B 证据。
- 唯一的 NaN=0 锚点 V23_SC 用 4.0 → 跟随。
- 结论：修正为 4.0。

### 1.4 `ppo_epochs=1` / `kl=0.03` / `max_grad_norm=1.0`
- v22/v23_SC/v24/v25 全部采用此组合，稳定性证据一致。
- 保留。

### 1.5 `GRAD_SPIKE_ABSOLUTE_CAP=2.5 / THRESHOLD=3.0`
- v25 step 135 事件是**唯一阻挡 v26c-v2 达到全部硬指标的已知风险**。
- cap=2.5 是针对这类单次 spike 的**唯一已知干预手段**。
- 注意：这是一个**尚未被生产级验证过**的干预（v25 用的是默认 cap=5.0，没拦住）。可能仍不足以完全防住，但是目前证据链下最好的防线。

### 1.6 `SAVE_FREQ=VAL_FREQ=10`
- v25 peak 出现在 step 150，v23_SC 在 step 45-60；10 步粒度都能捕获。

---

## 2. 中间会话期"优化提议"落实对照（修正版）

| 提议 | v2 落实 | 理由 |
|------|--------|------|
| ppo_epochs=1 | ✅ | 所有 peak-touching run 共享 |
| max_grad_norm=1.0 | ✅ | v23_SC 证明 0 NaN |
| kl_coef=0.03 | ✅ | 所有成功 run 公约数 |
| **BOK_CLIP=4.0** | ✅（修正） | V23_SC 唯一干净 NaN=0 锚点 |
| lr=1.5e-6 | ✅ | v25 实证可达 peak 0.7826 |
| **mixed 数据** | ✅（修正） | v25 证实前 120 步稳定，过度归因已撤回 |
| GRAD_SPIKE_ABSOLUTE_CAP=2.5 | ✅ | v25 step 135 事件的唯一已知干预 |
| SAVE_FREQ=10 | ✅ | 捕获 peak |
| lr=2e-6 | ❌ | v12fix 验证崩塌（kl=0.04 增正则也救不回）|
| BOK_WINNER_BOOST | ❌ | v25 代码 bug，从未生效 |
| BOK_ALLWRONG_NEG_ONLY=1 | ❌ | v25 代码 bug，从未生效 |
| BOK_SMART_FILTER_THRESHOLD>0 | ❌ | 未 A/B，与 sign-GD 交互未知 |
| TRAJ_ANSWER_GATE_MODE=soft | ❌ | 增加 reward 耦合噪声 |
| ROLLOUT_N=32 | ❌（推迟）| 显存风险 |
| temperature 提高 | ❌（推迟）| 抬 entropy 破坏稳定性 |

---

## 3. 是否能达到硬指标？（诚实评估）

| 硬指标 | v26c-v2 预期 | 置信度 |
|--------|-------------|-------|
| peak val_ans ≥ 0.7731（超 v12） | ✅ 大概率 | **高**（v25 已实证到 0.7826）|
| format_reward ≥ 0.95 全程 | ⚠️ 取决于 cap 能否接住 step-135-type 事件 | **中**（cap=2.5 从未在生产环境验证过）|
| NaN 步数 < 30 | ⚠️ 取决于 cap | **中** |

**诚实边界**：
- 如果 cap=2.5 成功接住 spike → v26c-v2 很可能达到所有三项硬指标，peak 在 0.78-0.79。
- 如果 cap=2.5 仍然接不住 → 需要回退到"**安全底**"：V23_SC 复刻（easy-only + lr=1e-6 + CLIP=4.0），已知 NaN=0 但 peak 封顶 0.7769。
- 真正突破 0.80+ 需**算法层改动**（rollout_n=32、answer-gate soft、point matching 升级、SFT 起点升级）。

---

## 4. 启动 & 消融

```bash
cd /mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest

# 主推：v26c-v2
bash examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v26c_sparse_verified.sh

# 安全底（若主推 cap 失效可回退）
STEPCOUNT_TRAIN_DATA=/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_0_10 \
  ACTOR_LR=1e-6 BOK_CLIP=4.0 RUN_NAME_PREFIX=v26c_v23SC_repro \
  bash examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v26c_sparse_verified.sh
```

## 5. 成功判定

- peak(val_ans) ≥ 0.7731 ← 必须
- NaN_steps < 30 ← 必须
- format_reward 末值 ≥ 0.95 ← 必须
- entropy_loss ∈ [0.3, 0.9] 全程 ← 监测
- 若任一硬指标未达 + cap=2.5 接住 spike 失败 → 切 v26c_v23SC_repro
