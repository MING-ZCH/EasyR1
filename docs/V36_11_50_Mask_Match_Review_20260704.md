# V36 11-50 Dataset / Mask / Training 参数复核

时间：2026-07-04

## 结论

`StepCountQA-RL-Traj_11_50_Combined` 路径是正确的，dataset schema 和 `images[0].path` 能被 `StepCount_mask_reward.py` 的 sequence/turn 解析逻辑识别。

但当前 full dataset 不是 100% GT mask-backed：

- dataset 总行数：39,230，unique sequence：39,230，answer 范围：11-50。
- answer 总点数：920,888。
- metadata 可匹配 sequence：38,132。
- 完整匹配行：38,097，占 97.11%。
- 不完整行：1,133，占 2.89%。
- 没有任何 metadata 的行：1,098，涉及 GT 点数约 26,720。
- 有 metadata 但少 turn 的行：35，缺失 turn 总数 112。
- metadata 相关的 894,056 个 mask 文件全部存在，且全部通过原始绝对 `mask_path` 命中；没有落到 `STEPCOUNT_MASKS_DIR` basename fallback，也没有 mask 文件缺失。

因此，如果训练目标允许 `TRAJ_NO_SEQUENCE_FALLBACK=stat_mask_sim`，当前脚本可以启动；如果要求每个 sample 都必须使用真实 GT mask reward，则需要先过滤掉 1,133 个不完整样本，或补齐这 1,098 + 35 个 sequence 的 metadata/mask。

## Filtered 训练集

已新建 mask-complete filtered 训练集，原始 `StepCountQA-RL-Traj_11_50_Combined` 未改动：

- 新路径：`/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_11_50_Combined_maskcomplete_38332`
- source rows：39,230
- kept rows：38,332
- excluded rows：898
- kept answer sum：896,691
- excluded answer sum：24,197
- 过滤规则：只保留 metadata turns 对 sequence 恰好等于 `0..answer-1`，且 `metadata_count == answer`、无 duplicate turn、无 extra turn 的样本。

剔除分布：

- `train-00000-of-00004.parquet`：保留 9,765 行，剔除 43 行。
- `train-00001-of-00004.parquet`：保留 9,521 行，剔除 287 行。
- `train-00002-of-00004.parquet`：保留 9,525 行，剔除 283 行。
- `train-00003-of-00004.parquet`：保留 9,521 行，剔除 285 行。

复验结果：

- filtered dataset 总行数：38,332。
- 38,332 行全部完整匹配当前 metadata，`bad=0`。
- answer 范围仍为 11-50。

manifest：

- `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_11_50_Combined_maskcomplete_38332/filter_manifest_maskcomplete_20260705.json`

2026-07-05 新 metadata 更新：

- metadata：`/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/sam_mask/StepCount-RL_masks_output_merged/final_rl_mask_manifest_grouped_progressive12h8_20260704_062627_grouped_progressive_issue_filtered_pointin_quality_upgraded/masks_metadata.easy_r1_compatible.grouped_progressive_issue_filtered.json`
- metadata total entries：970,214。
- full dataset 可完整覆盖行数从 38,097 提升到 38,332。
- 仍不完整行：898，其中 18 行完全没有 metadata，其余为 partial/missing turn。
- 相关 mask 文件存在性：0 missing；850,938 条通过原始绝对 `mask_path` 命中，58,567 条通过 `STEPCOUNT_MASKS_DIR` basename fallback 命中。
- coverage 报告：`docs/v36_new_metadata_coverage_20260705.json`
- filtered dataset 复验报告：`docs/v36_maskcomplete_38332_verify_20260705.json`

## DeepCheck 复核

新增 deepcheck 报告：

- `docs/v36_11_50_mask_presence_deepcheck_20260704.json`
- `docs/v36_11_50_mask_presence_deepcheck_20260704.sequence_prefix.json`

复核结果：

- 1,098 个 `no_metadata_sequence` 不只是在 reward 的 `image_path` sequence index 中不存在；在 metadata 的 `id`、`mask_filename`、`mask_path`、`image_path` prefix 索引中也均为 0 命中。
- 35 个 partial 行缺失的 turn，在 `id/mask/path` prefix 兜底索引中也没有找到，`alt_prefix_missing_turn_hits=0`。
- 38,097 个完整样本严格一一对应：answer sum = 893,065，metadata entries = 893,065，无 extra turn、无 duplicate turn、无重复 mask path。
- 38,097 个完整样本对应的 893,065 条 metadata，在 `id`、`mask_filename`、`mask_path` basename 的 sequence prefix 上全部一致，`seq_prefix_mismatch={}`，`seq_prefix_missing={}`。
- 当前 metadata 文件整体有多余内容：总 954,765 条，其中 893,065 条对应完整样本，991 条对应 35 个 partial 样本，60,709 条不属于当前 `StepCountQA-RL-Traj_11_50_Combined` dataset。

注意：`mask_filename` / `id` 后缀不能直接当作 turn。turn 以 metadata 的 `image_path` 为准，这是 reward 文件实际采用的逻辑；例如 turn0 的 `image_path` 可能是 `seq.jpg`，而 mask 文件名可能是 `seq_5.npz`，其中 `5` 是 mask/object id，不是 trajectory turn。

## 校验方法

按 `/mnt/shared-storage-user/zhangchenhao/work/EasyR1-hy-0703/examples/reward_function/StepCount_mask_reward.py` 的核心逻辑复现：

- dataset image path 取 `images[0].path`。
- `extract_sequence_id()` 去 basename、去扩展名，并把末尾 `_<turn>` 去掉。
- `extract_turn_number()` 取末尾 `_<turn>`，没有则为 0。
- metadata 的 `_k\d+` alias 会归一到 base sequence。
- mask 文件查找顺序与 reward 等价：先查 metadata 的绝对 `mask_path`，再查 `STEPCOUNT_MASKS_DIR` 下 basename fallback。

报告 JSON：

- `docs/v36_11_50_mask_match_report_20260704.json`

最新复验字段：

- `docs/v36_maskcomplete_38332_verify_20260705.json`
- `rows = 38332`
- `complete = 38332`
- `bad_count = 0`
- `answer_sum = 896691`
- `answer_min = 11`
- `answer_max = 50`
- `docs/v36_new_metadata_coverage_20260705.json`
- `row_match.complete_rows = 38332`
- `row_match.incomplete_rows = 898`
- `mask_files.missing_relevant_entries_total = 0`
- `mask_files.resolve_mode_counts = {"original_abs": 850938, "masks_dir_basename": 58567}`

## V36 参数适配

已把 v36 启动脚本对齐到当前 11-50 dense counting 目标：本轮主训练使用 trajectory-level `bok_grpo`，不启用 step-level estimator。

- 模型：`/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/model/StepCount-7B-SFT-1M-resume-from-1963/checkpoint-476`
- 数据：`/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_11_50_Combined_maskcomplete_38332`
- mask metadata：`/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/sam_mask/StepCount-RL_masks_output_merged/final_rl_mask_manifest_grouped_progressive12h8_20260704_062627_grouped_progressive_issue_filtered_pointin_quality_upgraded/masks_metadata.easy_r1_compatible.grouped_progressive_issue_filtered.json`
- mask dir：`/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/sam_mask/StepCount-RL_masks_output_merged/masks`
- val：`pixmo-test::/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/pixmo-test,stepcount-500-v36-val-100::/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/stepcount-500-v36-val-100`
- `ADV_ESTIMATOR=bok_grpo`
- `PROCESS_REWARD_ENABLE=0`
- `BOK_STEP_WEIGHT=0.0`
- `INTERLEAVED_ADAPTIVE_MAX_TURNS=true`
- `INTERLEAVED_ADAPTIVE_MAX_TURNS_MARGIN=2`
- `INTERLEAVED_MAX_TURNS=53`
- `effective_max_turn = min(53, GT_answer + 1 + 2)`
- `INTERLEAVED_POINT_TURN_USE_ANSWER_BUDGET=true`
- `INTERLEAVED_PER_TURN_MAX_TOKENS=512`
- `INTERLEAVED_ANSWER_TURN_MAX_TOKENS=8192`
- `ROLLOUT_N=16`
- `ROLLOUT_TEMPERATURE=0.7`
- `DISABLE_KL=false`
- `USE_KL_LOSS=false`
- `KL_TYPE=adaptive`
- `KL_COEF=0.06`
- `KL_TARGET=0.10`
- `KL_HORIZON=50000`
- `KL_PENALTY=low_var_kl`
- `ANSWER_WEIGHT=0.7`
- `POINT_WEIGHT=0.2`
- `TRAJECTORY_FORMAT_WEIGHT=0.1`
- `TRAJ_ANSWER_GATE_MODE=soft`
- `TRAJ_SOFT_GATE_BASE=0.7`
- `TRAJ_DENSE_CONTINUOUS_WRONG_CAP=0.05`
- `TRAJ_DENSE_CONTINUOUS_WITHIN1_CAP=0.12`
- `BOK_ALLWRONG_NEG_ONLY=1`
- `BOK_ALLWRONG_CAP=0.5`
- `BOK_DAPO_FILTER=0`
- `BOK_TAU_INIT=0.7`
- `BOK_TAU_FINAL=0.4`
- `BOK_SMART_FILTER_THRESHOLD=0.955`
- `V31_GPU_MEM_UTIL=0.65`
- `V31_MICRO_BATCH_UPDATE=8`
- `V31_MICRO_BATCH_EXP=8`

保留/适配项：

- `V31_MAX_RESPONSE_LENGTH=16384`、`V31_MAX_PROMPT_LENGTH=12000`、`V31_MAX_MODEL_LEN=32768`、`V31_MAX_NUM_BATCHED_TOKENS=65536`，适配 11-50 long-chain rollout。
- H200 16 卡默认 `EASYR1_VLLM_NUM_GPU_BLOCKS=32768`、`V31_ENFORCE_EAGER=false`、`REWARD_NUM_WORKERS=32`。
- `V31_NNODES=2`、`V31_N_GPUS_PER_NODE=8`、`V31_ROLLOUT_BATCH_SIZE=256`、`V31_GLOBAL_BATCH_SIZE=256`、`V31_VAL_BATCH_SIZE=256`，直接运行 v36 主脚本时也会覆盖到 `trainer.nnodes`、`trainer.n_gpus_per_node`、`data.rollout_batch_size`、`worker.actor.global_batch_size`、`data.val_batch_size`。
- `V31_FILTER_OVERLONG_NUM_PROC=64`，覆盖到 `data.filter_overlong_num_proc`，避免 `RLHFDataset` 在 Ray Runner 中只用 1 个进程做 overlong prompt filtering。可临时用 `V31_FILTER_OVERLONG_NUM_PROC=96/128` 覆盖；不建议默认 256。
- `TRAIN_SAVE_LIMIT=4`、`TRAIN_SAVE_FREQ=75`、`TRAIN_VAL_FREQ=30`，在 150 step 一轮训练中控制 checkpoint 数量，同时保留较密 validation 观测。
- `BOK_TOTAL_STEPS=150` 仅作为 fallback，trainer 会按 dataloader 的真实 steps 覆盖。

## 启动命令

多节点一键启动：

```bash
cd /mnt/shared-storage-user/zhangchenhao/work/EasyR1-hy-0703
bash examples/rl_launch/mn_trainer_dense_v36.sh 2 256
```

直接启动主脚本：

```bash
cd /mnt/shared-storage-user/zhangchenhao/work/EasyR1-hy-0703
bash examples/v36_dense_11_50_full_from_1m_ckpt476.sh
```

## 风险提示

原始 full dataset 仍有 898 个 incomplete rows，因此正式训练使用 filtered dataset `StepCountQA-RL-Traj_11_50_Combined_maskcomplete_38332`。该 filtered dataset 的复验结果是 38332/38332 complete，`bad_count=0`。

`stepcount-500-v36-val-100` 是 answer-level high-count validation，不是 mask-sequence validation；mask trajectory 训练和 reward 依赖训练集的 metadata/masks。
