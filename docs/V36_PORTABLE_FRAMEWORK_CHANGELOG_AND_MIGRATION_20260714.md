# V36 StepCount RL 框架改造、训练状态与集群迁移指南

更新时间：2026-07-14 HKT

## 1. 一页结论

本分支把远程历史分支 `hy-0703` 上的 StepCount RL 能力整理为一套可迁移的
V36 dense counting 训练链，当前生产入口为：

```text
examples/rl_launch/run_v36_8gpu_stable_focused10k.sh
  -> examples/rl_launch/run_v36_8gpu_stable.sh
  -> examples/rl_launch/run_v36_8gpu_fast_probe.sh
  -> examples/rl_launch/mn_trainer_dense_v36.sh
  -> examples/v36_dense_11_50_full_from_1m_ckpt476.sh
  -> examples/v32_sparse_0_10_stable_drfix.sh
  -> python3 -m verl.trainer.main
```

版本身份：

| 名称 | 值 |
|---|---|
| 原远程仓库 | `https://github.com/MING-ZCH/EasyR1.git` |
| 历史基础分支 | `hy-0703` |
| 历史基础 commit | `5e149138ab28f442489566d6dddd6b7c4d6331db` |
| 本次新分支 | `v36-dense-rl-portable-20260714` |
| 发布历史形式 | clean snapshot；文档保留逻辑 base，不继承含旧 credential 的 Git parent |
| 当前训练算法 | `bok_grpo`，trajectory-level reward 为主 |
| 当前稳定硬件 | 单节点 `8x H200 140GB` |
| 当前生产 SP 设置 | `V36_ENABLE_ULYSSES_SP=0` |

当前训练环境快照：

| 组件 | 版本 |
|---|---|
| PyTorch | `2.6.0` |
| transformers | `4.52.3` |
| vLLM | `0.8.2` |
| verl | `0.3.1.dev0` |
| Ray | `2.55.1` |
| FlashAttention | `2.7.3` |

当前稳定 run 已证明：8 卡配置可以连续完成 16 个训练 step，无 OOM、无
non-finite gradient、无 NCCL error。Ulysses SP2 仍属于实验开关，未被当前 run
使用，不应在其他集群直接作为生产默认值。

## 2. 当前训练状态

当前 run：

| 项目 | 值 |
|---|---|
| W&B project | `easy_r1` |
| W&B run id | `7rqnh6lh` |
| 数据集 | `StepCountQA-RL-Traj_11_30_Focused10k_quality_maskcomplete_v36_20260713` |
| 总 step | `80` |
| 已完整输出指标 | `Step 16` |
| 当前阶段 | `Step 17 rollout` |
| 首次 checkpoint/val | `Step 20` |
| 预计完成时间 | 2026-07-16 02:20-05:30 HKT |

关键配置：

```text
rollout_batch/global_batch=128/128
rollout_n=16
micro_update/micro_experience=4/8
gpu_memory_utilization=0.50
num_gpu_blocks=20480
max_num_batched_tokens=49152
temperature=0.7
actor_lr=1e-6
KL_TYPE=adaptive
KL_COEF=0.06
KL_TARGET=0.10
KL_HORIZON=50000
adaptive_max_turns=true
adaptive_margin=2
global_max_turns=53
```

训练窗口趋势：

| 指标 | Step 1-5 | Step 6-10 | Step 11-15 | 解读 |
|---|---:|---:|---:|---|
| 平均耗时 | 42.85 min | 44.27 min | 44.47 min | 近期稳定，rollout 约占 79% |
| throughput | 247.0 | 245.7 | 240.5 | 小幅下降 |
| overall | 0.1982 | 0.2012 | 0.2086 | 有轻微上升，尚需 val 确认 |
| answer | 0.1006 | 0.1086 | 0.1116 | 缓慢改善 |
| point | 0.3030 | 0.2990 | 0.3096 | 基本持平 |
| coverage 命中率 | 19.4% | 18.7% | 20.3% | 尚未形成稳定上升趋势 |
| point/GT 数量比 | 62.6% | 61.3% | 63.9% | 仍存在少点/早停 |
| format_fail | 31.4% | 33.2% | 30.0% | 略有恢复，仍偏高 |
| turns_exceeded | 30.8% | 33.2% | 30.0% | 与 format failure 高度重合 |
| count number score | 0.949 | 0.953 | 0.955 | 稳定且略升 |
| KL | 0.011 | 0.142 | 0.223 | 当前首要算法风险 |

最新单点 `Step 16` 为：`overall=0.212`、`answer=0.111`、`point=0.325`、
`count_number=0.963`、`format_fail=turns_exceeded=0.289`、coverage
`4.603/23.995=19.2%`、KL `0.235`、`grad_norm=4.264`。该点结构指标改善，但
coverage 没有同步改善，必须等待 Step 20 双 val，不能只凭 train batch 判断泛化增益。

资源状态：

- `max_memory_allocated_gb=107.688`
- `max_memory_reserved_gb=131.552/140.06`
- `nonfinite_grad_count=0`
- prompt/response `clip_ratio=0`
- 最大 prompt 已到 `11875/12000`，不能继续下调 prompt 上限
- 当前缺少 FlashInfer，vLLM 使用 PyTorch sampler fallback；这是效率问题，不是正确性问题

ETA 使用日志自身的 `time_per_step` 最近五步中位数计算。monitor 已修复原先
2 小时批量采样会把多个 step 记为同一时间、进而得到 `median_step=0.0` 的问题。

## 3. 相对 `hy-0703` 的主要改造

### 3.1 Interleaved rollout 正确停止

涉及文件：

- `verl/workers/rollout/vllm_rollout_spmd.py`
- `verl/workers/rollout/config.py`
- `verl/trainer/ray_trainer.py`

改造内容：

- 第一个 `</point>` 结束当前 point turn，并进入下一轮。
- 第一个 `</answer>` 结束整个 sample rollout。
- point turn 同时监听 `</point>` 和 `</answer>`，避免模型提前回答时被当作普通 point 输出。
- `include_stop_str_in_output=True` 保留闭合标签，保证 reward/parser 能看到完整结构。
- `INTERLEAVED_POINT_TURN_USE_ANSWER_BUDGET=true` 时，token 数只是上限，不会预分配为固定输出长度；正常 point 会在 `</point>` 立即停止。
- 对重复 count number、空输出、未闭合 point/answer 等状态增加诊断。

原因：多轮模型可能在任意 turn 提前输出 answer。若每轮只给 256/512 token，提前
answer 可能被截断；若不在闭合 tag 立即停止，又会产生额外文本、重复点和无效显存开销。

### 3.2 每个 sample 的 adaptive max turns

有效规则：

```text
effective_max_turn = min(INTERLEAVED_MAX_TURNS, GT_answer + 1 + margin)
```

当前 `margin=2`，`+1` 是 answer turn。训练和 val 都从 batch 的 `ground_truth`
解析 GT，并把 per-sample cap 传入 vLLM interleaved loop。无法解析 GT 时回退全局 cap。
V36 master 与 wrapper 的全局默认值已统一为 53，覆盖 `GT=50 + answer turn + margin=2`。

当前训练 `effective_max_turns` 平均约 26-27，而全局 cap 为 53，说明该机制已显著
缩短可生成的长尾轨迹。它既节省 rollout，也减少模型学习无意义超长 trajectory 的机会。

### 3.3 Point/answer token budget 分离

配置：

```text
INTERLEAVED_PER_TURN_MAX_TOKENS=512
INTERLEAVED_ANSWER_TURN_MAX_TOKENS=8192
V31_MAX_RESPONSE_LENGTH=16384
V31_MAX_MODEL_LEN=32768
```

answer budget 是动态生成上限。正常输出遇到停止 tag 即结束，不会因为上限为 8192
就固定占用 8192 token。全 trajectory 仍受 response/model length 约束。

### 3.4 双 benchmark val 独立上报

涉及文件：

- `verl/trainer/data_loader.py`
- `verl/trainer/ray_trainer.py`
- `tools/prepare_stepcount500_val100.py`

`data.val_files` 支持：

```text
pixmo-test::<path>,stepcount-500-v36-val-100::<path>
```

两个 suite 使用独立 dataloader 和 metric prefix，不能混成一个平均分：

```text
val/pixmo-test/...
val/stepcount-500-v36-val-100/...
```

`stepcount-500-v36-val-100` 是按 answer 11-50 分层抽样的 100 条代表集，用于高计数
answer-level 压力测试。其原始记录没有可匹配 mask sequence path，因此不能把该 suite
的 per-step duplicate/hit 指标与 pixmo 的 mask-derived 指标等价解读。

### 3.5 Reward 与 trajectory 完整性

涉及文件：

- `examples/reward_function/StepCount_mask_reward.py`

当前 V36 默认不是 hard strict gate：answer 正确仍可获得 answer 部分奖励，同时由 point
coverage、duplicate、extra point、format、count number、stop timing 等项扣分。这比
“point 结构任一错误就清零 answer reward”更适合 dense 11-50 的训练早期。

主要机制：

- mask metadata 多前缀/sequence id 匹配与 cache
- dense continuous answer reward
- soft answer gate
- missing coverage penalty
- extra/duplicate point penalty
- count number required，但跳号默认不 hard reject
- point-step 指标用于诊断；当前 `BOK_STEP_WEIGHT=0.0`，训练仍以 trajectory reward 为主

### 3.6 BoK-GRPO 与稳定性保护

当前使用：

- `ADV_ESTIMATOR=bok_grpo`
- `ROLLOUT_N=16`
- all-wrong negative-only/cap
- smart filter 和 DrGRPO fallback
- sequence-token policy loss 聚合
- reward-side adaptive KL
- gradient spike skip-only guard

当前没有启用真正的 per-step policy credit。`point_step_n_*` 主要用于展示、debug 和
后续 ablation，不应被误解为当前每个 point 都获得独立 advantage。

### 3.7 H20/H200 环境隔离

涉及文件：

- `verl/workers/rollout/vllm_rollout_spmd.py`
- `examples/v32_sparse_0_10_stable_drfix.sh`
- `examples/rl_launch/mn_trainer_dense_v36.sh`

H20 的 cuBLAS/TF32 SIGFPE workaround 只在 `STEPCOUNT_HARDWARE_PROFILE=h20` 或显式
开关下启用。H200 默认不再继承 H20 专用 cuBLAS 设置，避免无关 warning 和算法路径改变。

### 3.8 8/16 GPU、checkpoint 与 resume

8 卡稳定默认：

```text
micro_update=4
micro_experience=8
blocks=20480
gpu_mem=0.50
batched_tokens=49152
```

`TRAIN_SAVE_LIMIT=3` 会滚动保留最近三个 `global_step_*`。8 卡 FSDP checkpoint 可以在
8 卡下修改 micro batch、vLLM blocks、gpu mem util 后续训；不应直接把 16 卡 FSDP
shard checkpoint 当作 8 卡 checkpoint 加载。跨 world size 应先合并为 HF 权重再新开 run。

新 run 会在 save 目录写入 `v36_run_manifest.tsv`，记录原始 dataset、model、mask、GPU
数量、`ROLLOUT_N` 和算法标识。resume 优先读取 manifest；旧 checkpoint 只对明确包含
`11_30_quality_focused10k` 或 `11_50_maskcomplete` 的 run 名使用兼容 fallback。无法判定时
直接失败并要求显式设置 `STEPCOUNT_TRAIN_DATA`，不再静默切到 full dataset。

8 卡 resume 还会直接校验 `actor/model_world_size_8_rank_0..7.pt` 以及对应 optimizer/
extra-state shards，不能靠显式 dataset 覆盖绕过 world-size 检查。测试假 checkpoint 只有在
`V32_DRY_RUN=1 V36_ALLOW_INCOMPLETE_DRY_RUN_CHECKPOINT=1` 时允许跳过 shard 完整性。

### 3.9 数据集聚焦与可复现构建

新增工具：

- `tools/prepare_v36_focused10k_dataset.py`
- `tools/prepare_v36_focused10k_quality_dataset.py`
- `tools/prepare_stepcount500_val100.py`

focused10k 把训练重点放在 11-30，同时保留 31-50 长计数样本防止能力遗忘。quality 版本
按 mask metadata 的 quality tier 优先替换低质量记录，保持 answer quota 不变。选择过程
使用稳定 hash，可复现。

这些工具现在优先读取 `STEPCOUNT_DATA_ROOT`/`LOCAL_ROOT`，不再依赖唯一开发机目录。

### 3.10 Monitor 与 W&B

新增：

- `tools/monitor_v36_training.py`
- `tools/run_v36_hourly_watchdog.sh`

支持周期解析 step、错误、val、趋势和 ETA，并同步 offline W&B。文件名保留了历史
`hourly` 名称，但实际周期由 `V36_WATCH_INTERVAL_SECONDS` 控制；当前设置为 7200 秒。

本次修复：

- ETA 优先使用训练日志的 `time_per_step`，不再依赖 monitor 看到 event 的时间。
- Ray worker 在另一个 namespace/host 不可见时，只要日志仍新鲜，不再误报训练死亡。
- `logs/` 已加入 `.gitignore`，避免把 W&B 和训练日志推到 Git。

### 3.11 历史凭证脱敏

当前 snapshot 中历史 `examples/archive/` 脚本的硬编码 `WANDB_API_KEY` 已全部改为读取运行环境：

```bash
export WANDB_API_KEY=${WANDB_API_KEY:-}
```

这不会改变 active V36 训练行为。新分支使用无 parent 的 clean snapshot commit 发布，
避免该分支自身继承旧 credential 历史。由于旧值已存在远程仓库的其他历史分支，仍必须
在 W&B 控制台轮换或撤销对应旧 key；新分支无法清除其他远程 refs 已暴露的 secret。

## 4. Context parallel / Ulysses 的真实状态

本地框架保留了可插拔配置和 actor/ref/critic 侧的 sequence parallel 实验路径：

```text
V36_ENABLE_ULYSSES_SP=1
V36_ULYSSES_SEQUENCE_PARALLEL_SIZE=2
```

但当前结论是 **Qwen2.5-VL SP2 尚不可用于生产训练**：

- 当前生产 run 使用 `SP=1`，也就是关闭 Ulysses。
- 官方 EasyR1 README 仍把 “VLM 与 Ulysses 不兼容” 列为 known bug。
- 官方 verl 支持 Ulysses，但“框架支持 Ulysses”不自动等于本地 Qwen2.5-VL 多模态路径已验证。
- 当前 mRoPE `position_ids` 为 `(3, 1, sequence)`，而通用 Ulysses helper 的既有约束按
  `(1, sequence)` 设计；即使放宽该 shape，SP rank 的 image token 与完整 vision features
  仍需重新对齐。单纯通过 dry-run 不能证明 logprob 正确。
- 本地旧逻辑在 image token/features mismatch 时可能返回全零 logprob。新分支已改为默认
  fail-fast，`EASYR1_ALLOW_ZERO_MM_LOGPROB=0`，防止静默污染 policy ratio。
- SP 主要降低 actor/ref 长序列训练内存；当前总耗时约 79% 在 vLLM rollout，因此即使
  SP 正确，整体 wall-clock 收益也不会线性等于 micro batch 增益。

官方参考：

- EasyR1: https://github.com/hiyouga/EasyR1
- verl: https://github.com/volcengine/verl

开发验证顺序：

1. 从同一 checkpoint、同一小 batch 做 SP1/SP2 forward-logprob 数值对齐。
2. 检查 image token、position id、padding-free unpad/pad 和 loss mask 对齐。
3. 先用 `SP2 + micro_update=4` 跑 1-2 step，验证正确性和显存。
4. 再测试 `SP2 + micro_update=8`，不能直接长跑。
5. 任一 reward/logprob/grad 分布明显漂移，立即回退 SP1。

实验入口默认需要双重确认，避免把未验证 SP2 误当成节省显存的生产开关：

```bash
V36_ACK_EXPERIMENTAL_VLM_SP=1 \
V36_ALLOW_UNVALIDATED_QWEN25_VL_SP=1 \
  bash examples/rl_launch/run_v36_8gpu_focused10k_ulysses_sp2_probe.sh
```

## 5. 路径名称与迁移映射

### 5.1 当前开发机默认值

| 变量 | 当前默认路径 |
|---|---|
| `LOCAL_ROOT` | `/mnt/shared-storage-user/zhangchenhao` |
| `EASYR1_REPO_ROOT` | `/mnt/shared-storage-user/zhangchenhao/work/EasyR1-hy-0703` |
| `STEPCOUNT_WORK_ROOT` | `${LOCAL_ROOT}/work/StepcountModel` |
| `STEPCOUNT_DATA_ROOT` | `${STEPCOUNT_WORK_ROOT}/dataset` |
| `STEPCOUNT_MODEL_ROOT` | `${STEPCOUNT_WORK_ROOT}/model` |
| `EASYR1_LOG_ROOT` | `${EASYR1_REPO_ROOT}/logs` |
| `EASYR1_CHECKPOINT_ROOT` | `/mnt/shared-storage-user/puyuan/zhangchenhao/EasyR1-latest/model` |

`EASYR1_CHECKPOINT_ROOT` 是本次新增的统一名称，替代多个脚本中重复写死的 Puyuan
共享盘路径。迁移时覆盖这一个变量即可。

### 5.2 历史源路径

仓库 archive/legacy 脚本仍保留以下历史路径用于实验追溯：

```text
/data/workspace/hyleochang/EasyR1-latest
/apdcephfs_hldy2/share_305110755/hunyuan/chenhaoz
```

它们不在 V36 active launch chain。`scripts/preflight_v36_portable.sh` 会扫描生产链，发现
上述路径即失败。不要用全仓库 grep 的结果误判 active V36，因为 archive 有意保留历史值。

### 5.3 不进入 Git 的大文件

新分支只同步框架、脚本、配置和小型审计报告，不包含：

- model checkpoint
- parquet dataset
- mask PNG/metadata 大目录
- FSDP `global_step_*`
- `logs/`、W&B offline run、Ray session

## 6. 新集群部署

### 6.1 Clone

```bash
git clone --single-branch --branch v36-dense-rl-portable-20260714 \
  https://github.com/MING-ZCH/EasyR1.git
cd EasyR1
```

### 6.2 设置路径

```bash
export LOCAL_ROOT=/new/shared/user
export STEPCOUNT_WORK_ROOT=/new/shared/user/work/StepcountModel
export STEPCOUNT_DATA_ROOT=/new/shared/user/work/StepcountModel/dataset
export STEPCOUNT_MODEL_ROOT=/new/shared/user/work/StepcountModel/model
export EASYR1_CHECKPOINT_ROOT=/new/checkpoint/volume/StepCount-v36
export EASYR1_LOG_ROOT=$PWD/logs
```

`STEPCOUNT_H200_SOCKET_NCCL=auto` 是默认值：已有 `NCCL_SOCKET_IFNAME` 时尊重现值；
当前 host 确实存在 `brainpf0` 时沿用它；其他集群则交给 NCCL/Gloo 自动探测。因此不再
无条件写死开发机网卡。只有目标集群确认需要指定 socket 模式时才显式设置：

```bash
export STEPCOUNT_H200_SOCKET_NCCL=1
export STEPCOUNT_NCCL_IFNAME=<target-cluster-interface>
```

`scripts/preflight_v36_portable.sh` 会验证显式指定的网卡在当前 host 存在。

如目录名不同，可继续显式覆盖：

```bash
export STEPCOUNT_1M_RESUME_CKPT476_MODEL_PATH=/path/to/checkpoint-476
export STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA=/path/to/focused10k
export STEPCOUNT_DENSE_11_50_MASKS_METADATA=/path/to/masks_metadata.json
export STEPCOUNT_DENSE_11_50_MASKS_DIR=/path/to/masks
export STEPCOUNT_STEPCOUNT500_V36_VAL100_DATA=/path/to/val100
```

### 6.3 Preflight

```bash
bash scripts/preflight_v36_portable.sh
```

### 6.4 稳定启动

```bash
WANDB_MODE=offline bash examples/rl_launch/run_v36_8gpu_stable_focused10k.sh
```

### 6.5 8 卡续训

```bash
bash examples/rl_launch/resume_v36_8gpu_from_ckpt.sh \
  /path/to/run/global_step_20 stable
```

### 6.6 W&B sync

```bash
WANDB_MODE=online python3 -m wandb sync \
  --include-offline --no-mark-synced --no-sync-tensorboard \
  --project easy_r1 /path/to/offline-run-*
```

## 7. 下一阶段效率优化

按风险从低到高排列。

### P0：当前 run 保持不变

当前 `reserved=131.552/140.06GB`，不应热改 `micro_update=8`。继续到 Step 20，先拿到
checkpoint 和两个 val suite 的真实变化。

### P1：安装匹配版本的 FlashInfer 后做 1-step A/B

当前 vLLM 明确回退 PyTorch sampler。FlashInfer 可能降低 sampling 开销，但需要与当前
CUDA/PyTorch/vLLM ABI 严格匹配，并在离线集群预先准备 wheel。它不改变 reward 算法，
是优先级较高的纯效率实验。

### P1：仅提高 vLLM cache 的短 probe

从 Step 20 checkpoint 分叉 1-2 step：

```text
micro_update=4
blocks=24576
gpu_mem=0.55
batched_tokens=49152
```

只在 generation throughput 明显提升且 actor update 不逼近 OOM 时保留。不要同时改变
micro、blocks、gpu mem，否则无法归因。

若当前配置再次 OOM，优先从最近的同一 `8-GPU global_step_*` 使用以下保守资源参数续训，
训练目标和 batch 统计口径保持不变：

```text
micro_update=2
micro_experience=8
blocks=16384
gpu_mem=0.45
```

只有在 old/ref logprob forward 本身 OOM 时才把 `micro_experience` 从 8 降到 4。

### P2：Ulysses SP2

先 `SP2 + micro4` 做正确性 probe，再考虑 `SP2 + micro8`。上游 EasyR1 对 VLM 仍有
known bug 标记，因此这项优化不能仅凭 dry-run 判定完成。

### 不推荐作为效率优化

- actor/optimizer CPU offload：可降低显存，但会增加 PCIe/CPU 传输，通常让当前 run 更慢。
- 关闭 gradient checkpointing：当前显存余量不足。
- 降低 max prompt：真实 prompt 已达到 11875/12000。
- 直接把 `ROLLOUT_N=16` 改为 8：会改变 BoK group 分布和模型性能，不是等价加速。

`ROLLOUT_N=12` 可作为从 Step 20 checkpoint 新开的独立效率 Arm，预期 rollout 样本数
减少 25%，但端到端收益通常只有约 15%-25%，且会改变 BoK winner/advantage 分布。不能在
当前 run 中途修改，也不能与 KL/reward 改动放在同一个 Arm。

框架不再在单个 rollout 内自动把 `n=16` 降为 8。trainer、FSDP worker、gradient
accumulation 和 BoK UID group 都按启动时的 `ROLLOUT_N` 初始化，step 内动态改变会破坏
batch/advantage 口径。OOM 后应从 checkpoint 用显式较小的 `ROLLOUT_N` 新开实验。

## 8. 下一阶段算法与模型性能优化

### 8.1 Step 20 决策阈值

使用 Step 16-20 五步均值和双 val，不看单个 batch：

| 状态 | 建议阈值 |
|---|---|
| 健康 | overall >= 0.21，answer >= 0.12，coverage >= 22%，point/GT >= 65% |
| coverage 警告 | coverage < 18% 或 point/GT < 58% |
| format 警告 | format_fail 或 turns_exceeded > 0.33 |
| KL 警告 | 五步均值 > 0.25 |
| KL 高风险 | > 0.35 或持续上升且 kl_coef 不响应 |
| gradient 高风险 | 任意 nonfinite，或 EMA > 4/单步 > 6 |
| memory 高风险 | allocated > 125GB 或 reserved > 138GB |

### 8.2 KL controller

当前 KL 最近五步均值约 0.223，高于 target 0.10，而 `KL_HORIZON=50000` 对 80-step
短 run 响应偏慢。更重要的是，当前 `USE_KL_LOSS=false` 会先执行：

```text
token_level_rewards = token_level_scores - kl_coef * token_kl
```

随后 BoK 对 `token_level_rewards.sum(-1)` 排名。Step 14-15 的 BoK score 均值约
`-15`，范围约 `[-34, -5]`，而 task overall reward 约 0.2。这说明累计 token KL 和
trajectory length 可能主导 BoK 排序，使模型偏向短/低 KL 轨迹，而不是更高的真实
counting quality。

当前 run 不热改。Step 20 后应先从同一 checkpoint 做独立 Arm：

```text
USE_KL_LOSS=true
KL_TYPE=fixed
KL_COEF=0.01-0.03
```

这样 BoK 使用 task score 排名，KL 在 actor loss 侧约束。该 Arm 必须使用新的
experiment/save 目录，不能覆盖当前 baseline checkpoint。若仍保留 reward-side KL，
才考虑单独把 horizon 从 50000 降到 10000-20000。两类改动不能同时做，否则无法归因。

示例命令：

```bash
V31_EXPERIMENT_NAME=v36_focused10k_step20_actor_kl002 \
V31_SAVE_CHECKPOINT_PATH=/new/checkpoint/volume/v36_focused10k_step20_actor_kl002 \
USE_KL_LOSS=true KL_TYPE=fixed KL_COEF=0.02 \
bash examples/rl_launch/resume_v36_8gpu_from_ckpt.sh \
  /path/to/baseline/global_step_20 stable
```

效率 Arm 需要从同一个 Step 20 checkpoint 另开目录，并保持 KL/reward 不变：

```bash
V31_EXPERIMENT_NAME=v36_focused10k_step20_rollout_n12 \
V31_SAVE_CHECKPOINT_PATH=/new/checkpoint/volume/v36_focused10k_step20_rollout_n12 \
ROLLOUT_N=12 \
bash examples/rl_launch/resume_v36_8gpu_from_ckpt.sh \
  /path/to/baseline/global_step_20 stable
```

先各跑 2-5 step 比较 `time_per_step`、BoK valid-group 比例、KL、coverage、format fail 和
双 val；不要把两个 Arm 的结论从单个 train batch 外推到模型性能。

### 8.3 Step-level reward

当前 coverage 约 20%，trajectory reward 对每个 point 的 credit 较弱。若 Step 20 val
确认 answer 略升但 coverage 不升，下一轮可从相同 checkpoint 比较：

- Arm A：继续 `bok_grpo`，作为严格基线。
- Arm B：小权重 step-level signal，保持 answer-soft gate 和 duplicate penalty。

不要直接给大 step weight。先验证 mask 命中、duplicate、count-number 和 answer gate
没有 reward hacking，再逐步从 0.05 附近试验。该变化会改变训练目标，不能在当前 run
中途静默切换。

### 8.4 GSPO

GSPO 使用 sequence-level importance ratio 和 clipping，理论上更贴合 trajectory-level
reward，也可能降低 token-level ratio outlier。参考：https://arxiv.org/abs/2507.18071 。

但 StepCount 是多轮 point trajectory，仍需要细粒度 point credit；直接从 BoK-GRPO 切
GSPO 可能提高稳定性却弱化局部 credit。因此建议把 GSPO 作为同 checkpoint 的独立 ablation，
而不是当前 V36 的无条件替代。

## 9. 验证清单

已执行：

```bash
bash -n examples/local_path_env.sh \
  examples/rl_launch/mn_trainer_dense_v36.sh \
  examples/rl_launch/run_v36_8gpu_fast_probe.sh \
  examples/v36_dense_11_50_full_from_1m_ckpt476.sh \
  scripts/preflight_v36_portable.sh

python3 -m py_compile tools/monitor_v36_training.py \
  tools/prepare_v36_focused10k_dataset.py \
  tools/prepare_v36_focused10k_quality_dataset.py \
  tools/prepare_stepcount500_val100.py

python3 -m unittest discover -s tests -p test_monitor_v36_training.py -v
bash scripts/preflight_v36_portable.sh
```

预期结果：

- shell syntax 通过
- Python compile 通过
- monitor 三个标准库测试通过
- model/data/val/mask/checkpoint 路径全部 `[OK]`
- active V36 launch chain legacy path scan 为 0

## 10. 已知限制

- 当前生产 run 尚未到 Step 20，因此新的双 benchmark val 趋势和 checkpoint 实写仍待验证。
- Ulysses SP2 只有可插拔实现和 probe 入口，不等于 Qwen2.5-VL 生产正确性已确认。
- 当前 Qwen2.5-VL mRoPE/image-feature SP 对齐尚未实现；双重 opt-in 只允许开发 probe，
  不代表该路径能完成一个正确训练 step。
- `rl_envirment.txt` 是历史保留的环境快照文件名，拼写未改以避免已有引用失效。
- archive 脚本保留旧集群路径；迁移只保证本文列出的 active V36 链。
- 数据、mask 和 checkpoint 不随 Git 分支分发，必须在目标集群单独同步并通过 preflight。
