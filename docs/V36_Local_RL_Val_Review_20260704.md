# V36 本地 RL 框架与 Val Review

日期：2026-07-04

## 结论

1. `v36` active 训练链路已适配本开发机器路径；定向检查未发现 `hyleochang`、`/apdcephfs`、`/data/workspace`、`share_305110755` 残留。旧路径仍存在于 archive/legacy 脚本，不在 `v36` 实际调用链。
2. `v36` 训练参数已对齐 `v35 adaptive RL` 的稳定方向，并针对 11-50 长链路扩展：clean start、turn=51、reward-side adaptive KL、temperature=0.8、GPU mem util=0.50、strict format、soft answer gate、BoK all-wrong neg-only。
3. `stepcount-500` 可作为 `pixmo-test` 之外的第二个 validation benchmark，但应单独 report，不应混成一个总分。已生成代表性 100-sample 子集。
4. adaptive max turn 若只放在 reward 层只能事后判错，不能省 vLLM 生成算力；真正提速需要 validation rollout 读取 per-sample `GT answer + 3` cap。建议先 val-only，不进入训练采样。

## V36 路径与参数

关键文件：

- `examples/local_path_env.sh`
- `examples/v36_dense_11_50_full_from_1m_ckpt476.sh`
- `examples/rl_launch/mn_trainer_dense_v36.sh`
- `examples/v32_sparse_0_10_stable_drfix.sh`
- `examples/reward_function/StepCount_mask_reward.py`
- `examples/config.yaml`
- `verl/trainer/ray_trainer.py`

当前 v36 默认：

- model: `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/model/StepCount-7B-SFT-1M-resume-from-1963/checkpoint-476`
- train data: `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_11_50_Combined`
- val data 默认：`/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/pixmo-test`
- 11-50 mask metadata: `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/sam_mask/StepCount-RL_masks_output_merged/final_rl_mask_manifest_extra_h20_4h_nonoverlap_20260703_181252_componentclean_pointin_quality_upgraded/masks_metadata.easy_r1_compatible.json`
- mask dir: `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/sam_mask/StepCount-RL_masks_output_merged/masks`
- save/log: repo-local `save/` 和 `logs/`

重要修正：

- `v36` 单脚本不再继承泛用 `STEPCOUNT_MASKS_METADATA/DIR`，避免脏 shell 环境误用旧 mask；只允许 `V36_STEPCOUNT_MASKS_METADATA/DIR` 显式覆盖。
- `examples/config.yaml` 默认 save path 改为 `./save/easy_r1_default`。
- `StepCount_mask_reward.py` 默认 mask fallback 改为本机路径。
- `ray_trainer.py` debug log fallback 改为当前 repo 下 `logs/debug`。

参数确认：

- `INTERLEAVED_MAX_TURNS=51`
- `INTERLEAVED_PER_TURN_MAX_TOKENS=512`
- `INTERLEAVED_ANSWER_TURN_MAX_TOKENS=4096`
- `V31_MAX_RESPONSE_LENGTH=16384`
- `V31_MAX_MODEL_LEN=32768`
- `V31_MAX_NUM_BATCHED_TOKENS=49152`
- `DISABLE_KL=false`
- `USE_KL_LOSS=false`
- `KL_TYPE=adaptive`
- `KL_COEF=0.06`
- `KL_TARGET=0.10`
- `KL_HORIZON=50000`
- `ROLLOUT_TEMPERATURE=0.7`
- `V31_GPU_MEM_UTIL=0.60`
- `V31_MICRO_BATCH_UPDATE=4`
- `V31_MICRO_BATCH_EXP=8`
- `TRAJ_ANSWER_GATE_MODE=soft`
- `BOK_ALLWRONG_NEG_ONLY=1`

## Stepcount-500 Val100

源数据：

- `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/stepcount-500/data/train-00000-of-00001.parquet`
- rows: 500
- answer: 11-50
- 11-30 每个 answer 13 条；31-50 每个 answer 12 条
- `images.bytes` 完整；`images.path` 全部为 `null`

已生成子集：

- dataset: `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/stepcount-500-v36-val-100/train.parquet`
- manifest: `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/stepcount-500-v36-val-100/selection_manifest.json`
- script: `tools/prepare_stepcount500_val100.py`
- hash salt: `stepcount-500:v36-val-100`

子集统计：

- rows: 100
- bin: `11-20=26`, `21-30=26`, `31-40=24`, `41-50=24`
- mean: 30.1
- median: 30.0
- population std: 11.536
- duplicate images: 0
- `images.path` non-null: 0

使用建议：

- 该子集适合作为 11-50 answer-level 高计数压力 val。
- 不适合作为 trajectory/mask sequence val，因为没有 image path/sequence id；reward 会走 no-sequence fallback。
- `pixmo-test` 是低计数 benchmark，`stepcount-500-v36-val-100` 是高计数 benchmark，checkpoint selection 应看两个独立 metric。
- 当前 EasyR1 只有一个 `data.val_files` 和一个 `val_dataloader`；同一次 `_validate()` 同时输出两个 benchmark 独立 metric 需要新增 multi-val dataloader 支持。直接合并两个 parquet 会得到混合总分，不建议。

## Adaptive Max Turn

目标规则：

- 正常正确轨迹约为 `GT answer` 个 `<point>` turn + 1 个 `<answer>` turn。
- 若 turn 超过 `GT answer + 3`，该 sample 默认错误。

现状：

- reward 可拿到 `ground_truth` 与 `is_validation=True`。
- rollout 生成阶段当前只读取全局 `interleaved_max_turns`，不知道每个 sample 的 GT。
- 现有 reward 判定是 `turns_exceeded = len(pred_points) >= max_turns`。

结论：

- reward-only 可以事后判错，但 vLLM 已经完成生成，不能提升 eval efficiency。
- 真正提速需要在 validation generation 前把 per-sample `turn_cap = min(global_cap, GT + 3)` 传给 rollout，并在 interleaved loop 内按样本提前关闭 `active_flags`。
- 小 cap 样本的最后允许 turn 要使用 `answer_turn_max_tokens`，否则可能剥夺正常回答空间。
- 训练 rollout 不建议默认启用 GT adaptive cap；它会改变 GRPO/BoK 采样分布和实验可比性。建议先 val-only。

推荐实现面：

- `verl/trainer/ray_trainer.py`：validation `test_gen_batch` 带上 per-sample cap 字段。
- `verl/workers/rollout/config.py`：新增显式开关，例如 `interleaved_adaptive_max_turns_scope=val`、`interleaved_adaptive_max_turns_margin=3`。
- `verl/workers/rollout/vllm_rollout_spmd.py`：interleaved generation loop 支持 per-sample cap。
- `examples/reward_function/StepCount_mask_reward.py`：reward 侧同步同一个 effective cap，避免生成与判分不一致。

## Verification

已执行：

```bash
bash -n examples/local_path_env.sh
bash -n examples/v36_dense_11_50_full_from_1m_ckpt476.sh
bash -n examples/rl_launch/mn_trainer_dense_v36.sh
/root/miniconda3/bin/python -m py_compile tools/prepare_stepcount500_val100.py examples/reward_function/StepCount_mask_reward.py verl/trainer/ray_trainer.py
grep -RInE "hyleochang|/apdcephfs|/data/workspace|share_305110755|WANDB_API_KEY=wandb" examples/local_path_env.sh examples/v36_dense_11_50_full_from_1m_ckpt476.sh examples/rl_launch/mn_trainer_dense_v36.sh examples/v32_sparse_0_10_stable_drfix.sh examples/config.yaml examples/reward_function/StepCount_mask_reward.py verl
```

结果：

- shell syntax: OK
- Python compile: OK
- active v36 链路旧路径残留：0
- `stepcount-500-v36-val-100` 可被 `datasets.load_dataset("parquet")` 通过目录和文件两种方式读取。
