# V27 训练计划（含 2GPU H200 对比版）

## 两版脚本对比

| 参数 | v27_sparse_guarded | v27b_2gpu_h200 | 变化原因 |
|------|-------------------|----------------|---------|
| actor_lr | **1.5e-6** | **2e-6** | 探索更高LR上界 |
| n_gpus_per_node | 4 | 2 | H200机器配置 |
| micro_batch_update | 8 | 8 | 保守值（trajectory多轮高显存） |
| gpu_memory_util | 0.4 | 0.65 | H200 140GB更充裕 |
| GRAD_SPIKE_THRESHOLD | 3.0x | 3.0x | 同 |
| TRAJ_POINT_STRICT_JSON | 1 | 1 | 核心修复 |
| data | easy+hard | easy+hard | 同 |
| BOK_CLIP | 4.0 | 4.0 | 同 |
| BOK_SMART_FILTER | 0.955 | 0.955 | 同 |

## 启动命令

```bash
# v27 (4GPU A100, lr=1.5e-6)
cd /mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest
bash examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27_sparse_guarded.sh

# v27b (2GPU H200, lr=2e-6, 对比探索)
bash examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v27b_2gpu_h200.sh

# 如需切换 easy-only 数据：
STEPCOUNT_TRAIN_DATA=.../StepCountQA-RL-Traj_0_10 bash v27_sparse_guarded.sh
```

## 关键修复（两版共有）

1. **TRAJ_POINT_STRICT_JSON=1**（核心）: 关闭 `_parse_pred_point` regex兜底路径，阻断 reward hacking
2. **GRAD_SPIKE_THRESHOLD=3.0x**: v22-v24 实证 NaN=0 最稳定（vs v26 3.5x）
3. **无 GRAD_SPIKE_ABSOLUTE_CAP**: v26 的 abs_cap=2.5 有害（加速崩溃）

## 预期结果

| 版本 | 预期 val_peak | 预期 format | 对比基线 |
|------|-------------|------------|---------|
| v27 | ≥ 0.7826 | ≥ 0.97 全程 | v25: 0.7826但崩 |
| v27b | ≥ 0.7826 (可能更高) | ≥ 0.97 全程 | v27探索2e-6上界 |

## 历史评估基准（pixmo-test 529样本）

- SFT base (step 0): 0.7561
- v23 峰值: 0.7769
- v25 峰值: 0.7826（崩于 step135）
- v12fix 峰值: 0.7845（崩于 step200）
- **目标**: ≥ 0.7826 且 format 全程稳定

## Kill v26 命令

```bash
# 查找当前 v26 进程
ps aux | grep training_interleaved_traj_v26
# 杀掉
kill -9 <PID>
# 或直接杀 ray
ray stop
```
