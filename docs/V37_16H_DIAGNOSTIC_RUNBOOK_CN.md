# V37 16h 非晋级诊断 Runbook

本流程只做四个 3-step 机制诊断，不产生 winner，不跑 benchmark，不 merge，不上传在线 W&B，也不能用于晋级。训练严格串行：`seed11 baseline → seed11 progress → seed22 progress → seed22 baseline`。每个 cell 都从干净的 SFT `checkpoint-476` 开始，使用 focused10k debug fallback。

## 1. 构建独立 heldout

默认输出路径由 `examples/local_path_env.sh` 的 `STEPCOUNT_V37_DIAGNOSTIC_VAL_DATA` 提供。输出必须不存在；builder 拒绝覆盖。下面的 `--benchmark` 要覆盖本机全部 benchmark，不能少传：

```bash
source examples/local_path_env.sh
python3 tools/build_v37_diagnostic_heldout.py \
  --source-0-10 "$STEPCOUNT_REPLAY_DATA" \
  --source-11-50 "$STEPCOUNT_DENSE_11_50_MASKCOMPLETE_DATA" \
  --focused-selection-manifest "$STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA/selection_manifest.json" \
  --benchmark "$STEPCOUNT_PIXMO_CANONICAL_JSON" \
  --benchmark "$STEPCOUNT_STEPCOUNT500_CANONICAL_JSON" \
  --benchmark "$STEPCOUNT_COUNTQA_DATA" \
  --benchmark "$STEPCOUNT_BIAS_DATA" \
  --benchmark "$STEPCOUNT_DENSE_CANONICAL_JSON" \
  --benchmark "$STEPCOUNT_EXTREME_CANONICAL_JSON" \
  --output-dir "$STEPCOUNT_V37_DIAGNOSTIC_VAL_DATA"
```

builder 使用固定 `batch_size=8` 流式扫描 parquet，不会物化 0--10 数据中约 1.07GB 的单个 row-group；内存中只保留五个有界候选堆。它按 `2-10/11-20/21-30/31-40/41-50 = 200/50/100/100/50` 选 500 条，排除 focused selection manifest 中的 `(source_file, local_idx)`，并按可解析文件或 parquet 内嵌 image bytes 的 SHA256 排除 PixMo、StepCount-500、CountQA、bias、Dense、Extreme 六套 canonical benchmark。最终目录通过同一文件系统内的 rename 原子发布，已有输出一律失败。

若既没有内嵌 image bytes 也无法解析图像路径、任一桶不足、benchmark 没有可核验图像、规范化后的源 schema 不一致或输出已存在，builder 会 fail closed。

## 2. 启动前检查

必须独占 8 张 H200：每卡 `memory.total >= 139000 MiB`、`memory.used <= 5120 MiB`、MIG 为 Disabled，且没有 compute process。launcher 会自动执行这些检查；不满足时不要绕过。

确认使用 GNU coreutils `timeout`，并选择一个全新的输出目录：

```bash
timeout --version | head -n1
export V37_DIAGNOSTIC_ROOT=/absolute/new/path/v37-diagnostic-$(date +%Y%m%d-%H%M%S)
source examples/local_path_env.sh
bash examples/rl_launch/run_v37_16h_diagnostic.sh
```

离开机器前使用后台一键命令：

```bash
cd /mnt/shared-storage-user/zhangchenhao/work/EasyR1-hy-0703
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=/mnt/shared-storage-user/puyuan/zhangchenhao/EasyR1-latest/model/v37_diag_${RUN_TS}
LOG=$PWD/logs/rl/v37_diag_${RUN_TS}.log
mkdir -p "$(dirname "$LOG")"
setsid nohup env -u BASH_ENV -u ENV \
  V37_DIAGNOSTIC_ROOT="$RUN_ROOT" \
  WANDB_MODE=offline \
  bash examples/rl_launch/run_v37_16h_diagnostic.sh \
  >"$LOG" 2>&1 < /dev/null &
echo $! | tee "${LOG}.pid"
printf 'PID=%s\nLOG=%s\nROOT=%s\n' "$!" "$LOG" "$RUN_ROOT"
```

heldout 缺失时 launcher 会停止并打印完整 builder 命令。launcher 还会复核 manifest、parquet SHA256、固定五桶分布以及六套 benchmark hash universe。不要把 `STEPCOUNT_V37_DIAGNOSTIC_VAL_DATA` 指向任何 benchmark，也不要复用 `STEPCOUNT_VAL_DATA`。

## 3. 固定资源和时间预算

- 资源：micro update 4、micro experience 8、vLLM blocks 20480、GPU memory utilization 0.50、max batched tokens 49152、CP 1。
- update micro 8 被明确拒绝。pilot 的可插拔入口是 `V37_MICRO_BATCH_UPDATE`、`V37_MICRO_BATCH_EXP`、`V37_VLLM_NUM_GPU_BLOCKS`、`V37_GPU_MEM_UTIL`、`V37_MAX_NUM_BATCHED_TOKENS`；旧 `V31_*` 与它们冲突时直接失败。
- formal 始终锁定上述保守值；`V37_ALLOW_INDEPENDENT_VAL_OUTSIDE_TRAIN_RANGE=1` 只允许 debug，canary/formal 拒绝。
- 全局 deadline 使用 monotonic clock，固定 55,800 秒（15.5 小时）。baseline cell 的 P90/timeout 为 12,600 秒，progress 为 13,500 秒。开始下一个 cell 前，若余额小于该 cell 的完整 P90，立即停止。

## 4. 停止规则和清理

任一 cell 出现以下情况都会停止剩余序列：OOM、nonfinite/NaN/Inf、Python Traceback、任何 `GradSpike ... skip`、GNU timeout/非零退出、缺少 `global_step_3`，或 `ray stop --force` 后 900 秒内 GPU 仍未回到空闲阈值。

每个 cell 后以及 EXIT/INT/TERM trap 都执行 `ray stop --force`。清理逻辑不会删除任何 checkpoint；需要人工清理时先保存 `status.json`、日志和路径记录。

## 5. 产物解释

诊断目录包含：

- `status.json`：原子更新的总状态和四个 cell 状态；
- `command_manifest.json`：固定顺序、环境、资源与命令；
- `logs/*.log`：每个 cell 结束后原子发布；
- `中文总结.md`：最终状态和 checkpoint 表；
- 四个 cell 目录及其 `global_step_3`。

`completed` 只表示四个诊断 cell 均满足机械完整性要求，不表示 progress 优于 baseline。不要据此选 winner、运行 formal gate、触发 benchmark、merge checkpoint 或恢复在线 W&B。
