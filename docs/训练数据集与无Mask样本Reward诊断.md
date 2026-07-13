
---

## 8. 代码修改记录

### 修改1: 新增 `stat_mask_sim` fallback 模式

**文件**: `examples/reward_function/StepCount_mask_reward.py` L1112-1123

**原理**: 根据V23+hard训练日志中有mask样本的统计特征 (hit_rate=0.736, miss_rate=0.237, miss_decay_avg≈0.15)，构造一个统计模拟公式，使无mask样本的point reward分布与有mask样本**统计一致**。

**公式**:
```
eval_steps = min(pred_count, target_count)
per_step = (HIT_RATE + (1 - HIT_RATE) * MISS_DECAY_AVG) / target_count
score = eval_steps * per_step - max(0, pred_count - target_count) / target_count
```

**效果对比** (pred==GT时):
| 方案 | point_score | 与mask reward的误差 |
|------|-------------|-------------------|
| count_iou | 1.000 | +0.228 (膨胀) |
| calibrated_iou | 0.736 | -0.036 |
| **stat_mask_sim** | **0.776** | **+0.004** |

**使用方式**:
```bash
export TRAJ_NO_SEQUENCE_FALLBACK=stat_mask_sim
# 可选调参:
export TRAJ_SIM_HIT_RATE=0.736       # 从训练日志统计得到
export TRAJ_SIM_MISS_DECAY_AVG=0.15  # 估计的miss distance decay平均得分
```

### 架构说明: PATH A vs PATH B vs eval_answer_only_on_no_mask

| 场景 | is_eval | 路径 | 说明 |
|------|---------|------|------|
| 训练 rollout | False | PATH B | 加权 `0.6*answer + 0.3*point + 0.1*format` |
| 训练时 val | True | PATH A | 纯answer-only二值(不受point影响) |
| 无mask样本(期望) | - | PATH B + answer-only | `eval_answer_only_on_no_mask` 应跳过point权重 |

**当前BUG**: `eval_answer_only_on_no_mask` 因 `expected_steps == 0` 条件永不触发。
**推荐**: 使用 `stat_mask_sim` 代替修复此BUG，因为它保留了point reward的训练信号而不产生膨胀。

---

## 9. Interleaved Rollout 图片路径调查结论

### 9.1 `with_images=1024/1024` 矛盾解决

**不存在矛盾。** 完整机制：

```
ray_trainer.py: batch.pop("multi_modal_data") → gen_batch
  → fsdp_workers.py: deepcopy(multi_modal_data) → cached_multi_modal_data [保存原始dict]
    → process_image() → PIL Image (供vLLM生成)
      → vllm_rollout_spmd.py: .copy() + draw → 修改后的PIL (仅用于下一轮生成)
    → fsdp_workers.py: 用 cached_multi_modal_data 覆写output [恢复原始dict]
  → ray_trainer.py: batch.union(gen_batch_output) → multi_modal_data重新注入
    → reward拿到原始 {bytes, path} dict → with_images=1024/1024 ✅
```

**关键**: `fsdp_workers.py` 中的 **deepcopy + restore** 机制将rollout内部状态与reward输入完全隔离。

### 9.2 PIL.Image.copy() filename丢失 — 无实际影响

| 阶段 | 格式 | path保留? |
|------|------|----------|
| Dataset加载 | `{bytes, path}` dict | ✅ |
| fsdp_workers deepcopy | `{bytes, path}` dict深拷贝 | ✅ |
| vLLM rollout process_image | PIL Image | ❌ (无.filename) |
| rollout .copy() + draw | PIL Image | ❌ (.filename丢失) |
| **fsdp_workers恢复** | **原始 {bytes, path} dict** | **✅ 不受rollout影响** |
| **Reward函数接收** | **{bytes, path} dict** | **✅ path完好** |

Rollout中的PIL Image和reward收到的图片是**完全独立的两条路径**。`.copy()`丢失`.filename`**不传播到reward**。

### 9.3 结论

**严重程度: 无 (Non-issue)**

- Reward永远不接触rollout中被修改的PIL Image
- 图片路径通过HF dict的`path`字段保留，与PIL `.filename`属性无关
- `_infer_image_path_from_images()`中的`.filename` fallback在正常情况下永远不触发
- 已确认V12/V23/V23+hard中的`mask_no_sequence`不是由此引起，而是由189个eval样本缺失mask metadata导致

---

## 10. V12 mask_no_sequence=7060 完整调查结论

### 10.1 问题描述

V12训练日志中出现两组看似矛盾的counter：
- `[mask_reward][debug]`: calls=7060, **mask_no_sequence=7060**, mask_used=0 (100% fallback)
- `[traj_mask_reward][debug]`: calls=1150, traj_hit_unused=960, **traj_no_sequence=0** (0% fallback)

表面看，一个说全部找不到sequence，另一个说全部找到了sequence。

### 10.2 根因分析

**两组counter来自不同的reward worker进程，处理不同的数据：**

| | 训练进程 (pid=1622835) | 验证进程 (pid=1622914) |
|---|---|---|
| 数据源 | StepCountQA-RL-Traj_0_10 (11,455样本) | **pixmo-test** (529样本) |
| batch_size | 1024 | 64 |
| [mask_reward] | **无触发**（从不走fallback） | calls=7060, mask_no_sequence=7060 |
| [traj_mask_reward] | calls=1,788,860, traj_no_sequence=0, hit=82.6% | calls=1150, traj_no_sequence=0, hit=83.5% |

### 10.3 pixmo-test数据在mask metadata中的覆盖率

```
pixmo-test: 529 total
  - 在mask metadata中找到sequence: 11 (2.1%)
  - 在mask metadata中找不到sequence: 518 (97.9%)
```

pixmo-test是外部评测数据集，绝大部分图片不在训练用的mask metadata中。

### 10.4 数学验证

V12进行了约13.6次验证评估（87个training steps中约每6步评估一次），每次处理全部529个pixmo-test样本：

- `mask_no_sequence` = 13.6 × 518 = **7045 ≈ 7060** ✅
- `traj_calls` = 13.6 × 11 × 7.7(avg steps) = **1152 ≈ 1150** ✅

### 10.5 关键结论

1. **V12训练reward完全正常**：训练进程处理11,455个样本，178万步评估全部成功匹配mask sequence（traj_no_sequence=0），hit_unused_rate=82.6%
2. **mask_no_sequence=7060仅来自验证进程**：pixmo-test是外部数据集，97.9%样本不在mask metadata中，这是**预期行为**
3. **V12训练质量无问题**：mask-based point reward在训练中100%生效，不存在之前担心的"全部走count_iou fallback"的问题
4. **验证进程的518个无mask样本走count_iou fallback**：这就是之前诊断的count_iou膨胀问题（+0.228），但由于验证指标`is_eval=True`走PATH A（纯answer_score），这不影响验证得分

### 10.6 对之前分析的影响

- 之前担忧的"V12训练数据mask匹配失败"问题 → **不存在**，训练完全正常
- stat_mask_sim优化 → 主要解决后续版本（如V23 mixed）中含无mask训练样本（189/576个）的reward精度问题
- V12的81.47% pixmo-test得分 → 是在完全正常的mask reward训练下获得的，可作为可靠baseline

