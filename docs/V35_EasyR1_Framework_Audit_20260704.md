# V35 EasyR1 Framework Audit

审计时间: 2026-07-04

审计对象:

- Repo: `https://github.com/MING-ZCH/EasyR1.git`
- Branch: `hy-0703`
- Commit: `5e149138ab28f442489566d6dddd6b7c4d6331db`
- 本地目录: `/mnt/shared-storage-user/zhangchenhao/work/EasyR1-hy-0703`

## 结论

`v35` 训练脚本需要的 EasyR1 主链路已具备，支持启动
`StepCount` 11-20 dense RL、interleaved point-to-count rollout、
BoK-GRPO/BoK-GRPO-Step、dense continuous answer reward、strict point key
format scoring、coverage/stop-none shaping、GSPO-token importance sampling、
adaptive reward-side KL、resume checkpoint、grad spike skip-only guard 和
Ray worker env 透传。

审计中发现一个框架完整性问题: `verl/workers/critic/dp_critic.py` 存在
空 `if self.rank == 0:` 代码块导致 `IndentationError`。该文件在 `v35`
的 `bok_grpo/bok_grpo_step` 路径下通常不会被实例化，但会导致全框架
Python 编译检查失败。已在本地补 `pass` 修复，不改变训练逻辑。

2026-07-04 已完成本地路径适配: 新增 `examples/local_path_env.sh` 作为统一
路径入口，`v35/v34/v32/resume/mn_trainer` 脚本均改为读取本机
`/mnt/shared-storage-user/zhangchenhao` 下的 model、data、mask、reward、
log 和 save 路径。

## V35 脚本链

- `examples/rl_launch/mn_trainer_dense_v35.sh`
  多机 launch wrapper，设置 32-GPU/Ray/WandB/环境变量后执行 `examples/v35_dense_11_20.sh`。
  当前已改为从脚本位置解析 repo root，不再 `cd` 到远端 EasyR1-latest。
- `examples/v35_dense_11_20.sh`
  v35 主参数层: 1M SFT base、EM-align reward、`NEG_ONLY=1`、`KL_COEF=0.06`、
  `POLICY_LOSS_IS_LEVEL=sequence_token`、grad skip-only guard、`BOK_DAPO_FILTER=0`。
  当前默认 base 为
  `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/model/StepCount-7B-SFT-1M-merged`。
- `examples/v35_resume_0703_adaptivekl.sh`
  0703 resume probe: 从 `global_step_20` 或 `global_step_40` resume，
  切到 reward-side adaptive KL，降低 temperature/gpu mem util。
- `examples/v34_dense_curriculum_11_20.sh`
  继承 v34 11-20 dense curriculum、Arm A/B/C 选择和 long-chain rollout 参数。
- `examples/v32_sparse_0_10_stable_drfix.sh`
  最终 trainer 入口，调用 `python3 -m verl.trainer.main` 并传入所有 Hydra override。
  当前默认 reward 为本 repo 下的
  `examples/reward_function/StepCount_mask_reward.py:compute_score`。

## 本地路径适配

统一入口:

- `examples/local_path_env.sh`

当前默认路径:

| 类型 | 路径 |
|---|---|
| repo | `/mnt/shared-storage-user/zhangchenhao/work/EasyR1-hy-0703` |
| 1M model | `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/model/StepCount-7B-SFT-1M-merged` |
| SFT fallback model | `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/model/StepCount-7B-SFT-10k-high/checkpoint-1296` |
| v35 intended train data | `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_11_20_plus_replay` |
| v35 current fallback train data | `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_11_50_Combined` |
| replay data | `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_0_10` |
| val data | `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/pixmo-test` |
| mask metadata | `/mnt/shared-storage-user/zhangchenhao/StepCount-RL_masks_output/masks_metadata.json` |
| mask dir | `/mnt/shared-storage-user/zhangchenhao/StepCount-RL_masks_output/masks` |
| log root | `/mnt/shared-storage-user/zhangchenhao/work/EasyR1-hy-0703/logs` |
| save root | `/mnt/shared-storage-user/zhangchenhao/work/EasyR1-hy-0703/save` |

辅助脚本:

- `scripts/preflight_v35_local_paths.sh`: 只检查本地路径，不启动 Ray/GPU。
- `examples/prepare_v35_dense_11_20_plus_replay_local.sh`: 使用本地
  `11_50_Combined/data` 和 `0_10/data` 生成 v35 目标训练集。

## 特性支持矩阵

| v35 需求 | 代码支持位置 | 审计结论 |
|---|---|---|
| `bok_grpo` / `bok_grpo_step` | `verl/trainer/ray_trainer.py`, `verl/trainer/core_algos.py` | 支持 |
| DAPO filter off/on | `BOK_DAPO_FILTER` in `core_algos.py` | 支持，v35 默认 off |
| all-wrong neg-only / cap | `BOK_ALLWRONG_NEG_ONLY`, `BOK_ALLWRONG_CAP` in `core_algos.py` | 支持 |
| GSPO-token | `POLICY_LOSS_IS_LEVEL=sequence_token` in `compute_policy_loss` | 支持 |
| adaptive KL | `AdaptiveKLController`, `apply_kl_penalty` | 支持，需 `USE_KL_LOSS=false` |
| resume checkpoint | `trainer.load_checkpoint_path` | 支持 |
| optional `MAX_STEPS` | `trainer.max_steps` | 支持 |
| interleaved point-to-count rollout | `RolloutConfig`, `vllm_rollout_spmd.py` | 支持 |
| auto-protect rollout `n` | `auto_protect_min_n` in rollout config/impl | 支持 |
| dense continuous answer reward | `TRAJ_DENSE_CONTINUOUS_*` in `StepCount_mask_reward.py` | 支持 |
| strict canonical `point_2d` format scoring | `TRAJ_FORMAT_STRICT_KEY`, `TRAJ_FORMAT_TYPO_CREDIT` | 支持 |
| parser typo normalization | `TRAJ_POINT_KEY_NORMALIZE` | 支持 |
| coverage penalty | `TRAJ_COVERAGE_PENALTY_*` | 支持 |
| stop-none bonus / early-stop penalty | `TRAJ_STOP_NONE_*` | 支持 |
| per-step progress/stoptiming values | `_point_step_value`, `BOK_STEP_SIGNAL` | 支持 |
| grad spike skip-only guard | `GRAD_SPIKE_*`, `GRAD_NONFINITE_*` in `dp_actor.py` | 支持 |
| Ray worker env 透传 | `verl/trainer/main.py` whitelist | 支持 |

## 已执行验证

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
bash -n examples/v35_dense_11_20.sh examples/v35_resume_0703_adaptivekl.sh \
  examples/rl_launch/mn_trainer_dense_v35.sh examples/v34_dense_curriculum_11_20.sh \
  examples/v32_sparse_0_10_stable_drfix.sh
python -m compileall -q verl examples/reward_function scripts tools
bash scripts/preflight_v35_local_paths.sh
```

结果:

- shell 语法检查通过。
- 修复 `dp_critic.py` 后 Python 编译检查通过。
- Hydra override 静态字段检查通过；`worker.reward.reward_function_kwargs.*`
  是 dict 动态参数，不需要 dataclass 固定字段。
- 本地路径 preflight 通过；`model/reward/mask/val/fallback train/save/log` 均可见。

## 运行前置条件

当前本地路径已适配，但未启动 Ray/GPU dry-run，因为当前环境没有完整训练依赖
和 GPU/Ray 运行上下文。

仍需注意:

- v35 最理想训练集
  `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/StepCountQA-RL-Traj_11_20_plus_replay`
  当前不存在。脚本会 fallback 到 `StepCountQA-RL-Traj_11_50_Combined` 以便路径预检通过；
  若要严格按 v35 设计训练，应先执行:
  `bash examples/prepare_v35_dense_11_20_plus_replay_local.sh`。
- 当前 Python 环境缺少 `pyarrow`，因此本地无法直接运行上述数据准备脚本；
  在训练环境安装 `requirements.txt` 后即可运行。
- 0703 resume 所需 checkpoint 默认位于
  `/mnt/shared-storage-user/zhangchenhao/work/EasyR1-hy-0703/save/StepCount-7B_v35_dense_11_20_A_emalign_20260701_0432/global_step_20`
  或 `global_step_40`。如源 checkpoint 在其他位置，用 `V35_SOURCE_RUN_ROOT`
  或 `V32_LOAD_CHECKPOINT_PATH` 覆盖。

当前 Python 环境也未安装 EasyR1 运行依赖，例如 `codetiming`。`requirements.txt`
中包含相关依赖，训练 pod 需先安装或使用已配置镜像。

## 风险与建议

1. 本地已修复 `dp_critic.py` 语法错误，但远端 `hy-0703` 原始 commit 仍存在该问题。
   建议把该小修推回分支，避免后续 CI/全量编译/critic 路径失败。
2. `v35_dense_11_20.sh` 注释引用的部分 v34 review 文档在当前 repo 中未找到。
   不影响训练启动，但影响追溯性。
3. `mn_trainer_dense_v35.sh` 已移除硬编码 `WANDB_API_KEY` 默认值，改为读取环境变量；
   本地默认 `WANDB_MODE=offline`。
4. 训练前建议先执行路径预检:
   `bash scripts/preflight_v35_local_paths.sh`。
5. 真实训练 smoke 仍需在有 GPU/Ray 和依赖的环境执行:
   `V34_DRY_RUN=1 V34_ARM=A bash examples/v35_dense_11_20.sh`。
