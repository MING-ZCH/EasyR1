# OOM 问题修复总结：利用率只有一半但报 OOM

## 问题原因分析

### 为什么利用率只有 50% 但会 OOM？

1. **内存碎片化** ⭐⭐⭐⭐⭐ (最主要原因)
   - PyTorch 的内存分配器会产生碎片
   - 虽然总使用率只有 50%，但内存被分割成很多小块
   - 无法分配连续的大块内存（如大张量）
   - **表现**：`memory_allocated()` 显示 50%，但分配新张量时失败

2. **Reserved vs Allocated 内存差异**
   - `memory_allocated()`：实际使用的内存（可能只有 50%）
   - `memory_reserved()`：PyTorch 缓存保留的内存（可能接近 100%）
   - **表现**：显示使用率低，但实际可用内存很少

3. **峰值内存使用**
   - 某些操作（forward/backward）会瞬间需要大量内存
   - 虽然平均使用率是 50%，但峰值可能超过 100%
   - **表现**：大部分时间正常，偶尔 OOM

4. **批次大小不一致**
   - 某些批次可能特别大（特别是多模态数据）
   - 导致瞬间内存需求激增
   - **表现**：大部分批次正常，偶尔遇到大批次就 OOM

## 修复方案

### 1. 更智能的内存检查
- **之前**：只检查 `memory_allocated()`，阈值 99%
- **现在**：同时检查 `memory_allocated()` 和 `memory_reserved()`，使用较大的值，阈值降低到 85%

```python
# 检查 reserved memory（可能更准确反映实际使用）
reserved_mem = torch.cuda.memory_reserved()
reserved_ratio = reserved_mem / total_mem if total_mem > 0 else 0.0
effective_ratio = max(usage_ratio, reserved_ratio)
```

### 2. 更激进的缓存清理
- **之前**：只调用 `torch.cuda.empty_cache()`
- **现在**：
  - 先 `torch.cuda.synchronize()` 同步所有 CUDA 操作
  - 再 `torch.cuda.empty_cache()` 清理缓存
  - 如果使用率仍然高，再次清理
  - 在 backward 后也进行清理

### 3. 更详细的 OOM 日志
- **之前**：只打印错误信息
- **现在**：打印详细的内存统计信息
  - allocated memory
  - reserved memory
  - free memory
  - 帮助诊断问题

### 4. 更早的预防措施
- **之前**：阈值 99%，几乎满了才清理
- **现在**：阈值 85%，更早触发清理，避免 OOM

## 关键改进点

### 改进 1：双重内存检查
```python
# 同时检查 allocated 和 reserved
usage_ratio = used_mem / total_mem
reserved_ratio = reserved_mem / total_mem
effective_ratio = max(usage_ratio, reserved_ratio)  # 使用较大的值
```

### 改进 2：同步后清理
```python
torch.cuda.synchronize()  # 确保所有操作完成
torch.cuda.empty_cache()  # 清理缓存
```

### 改进 3：backward 后清理
```python
loss.backward()
# 清理缓存，如果使用率仍然高，再次清理
torch.cuda.empty_cache()
if usage_ratio > 0.8:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
```

### 改进 4：OOM 时的详细日志
```python
print(f"Memory: allocated={allocated/1024**3:.2f}GB, "
      f"reserved={reserved/1024**3:.2f}GB, "
      f"free={free_mem/1024**3:.2f}GB/{total_mem/1024**3:.2f}GB")
```

## 使用建议

### 如果仍然遇到 OOM：

1. **进一步降低阈值**
   - 将 85% 改为 80% 或更低
   - 在 `dp_actor.py` 中修改 `effective_ratio >= 0.85`

2. **减小 batch size**
   - 在 `config.yaml` 中减小 `micro_batch_size_per_device_for_update`

3. **启用更多 offload**
   - 在 `config.yaml` 中设置：
     ```yaml
     worker:
       actor:
         offload:
           offload_params: true
           offload_optimizer: true
     ```

4. **限制最大内存使用**
   - 在训练脚本开始时添加：
     ```python
     torch.cuda.set_per_process_memory_fraction(0.8)  # 限制最大使用 80%
     ```

5. **检查是否有内存泄漏**
   - 监控 `max_memory_allocated` 是否持续增长
   - 如果持续增长，可能有内存泄漏

## 监控建议

在训练过程中，关注以下指标：
- `perf/max_memory_allocated_gb`：峰值分配内存
- `perf/max_memory_reserved_gb`：峰值保留内存
- 如果两者差异很大，说明有内存碎片化问题

## 预期效果

修复后应该能够：
- ✅ 更早检测到内存压力
- ✅ 更有效地清理缓存
- ✅ 减少 OOM 错误
- ✅ 提供更详细的诊断信息










