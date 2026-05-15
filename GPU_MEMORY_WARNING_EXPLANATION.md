# GPU 内存警告解释

## 警告信息

```
Warning: GPU memory high before backward (allocated=39.0%, reserved=86.0%). Cleaning...
Warning: GPU memory high before backward (allocated=49.0%, reserved=96.0%). Cleaning...
Warning: GPU memory high before backward (allocated=50.2%, reserved=97.2%). Cleaning...
Warning: GPU memory high before backward (allocated=48.6%, reserved=95.6%). Cleaning...
```

## 这是什么意思？

### 1. **Allocated vs Reserved 内存的区别**

- **Allocated (已分配内存)**：实际被张量占用的内存
  - 例如：39.0% = 实际使用的 GPU 内存
  - 这是 `torch.cuda.memory_allocated()` 返回的值

- **Reserved (保留内存)**：PyTorch 内存分配器保留的内存缓存
  - 例如：86.0% = PyTorch 缓存保留的内存
  - 这是 `torch.cuda.memory_reserved()` 返回的值
  - **关键**：即使没有使用，PyTorch 也会保留这些内存以便快速分配

### 2. **为什么 Reserved 比 Allocated 高这么多？**

这是**正常现象**，原因包括：

1. **内存分配器策略**
   - PyTorch 使用内存池（memory pool）来管理 GPU 内存
   - 为了避免频繁分配/释放，会保留一些内存块
   - 即使释放了张量，内存也不会立即还给系统

2. **内存碎片化**
   - 多次分配/释放后，内存被分割成小块
   - 虽然总使用率不高，但无法分配连续的大块内存
   - PyTorch 会保留这些碎片化的内存

3. **FSDP 的特殊情况**
   - FSDP 会在不同 rank 之间同步模型参数
   - 需要临时分配大量内存用于通信
   - 这些临时内存会被保留在缓存中

### 3. **为什么在 Backward 之前检查？**

Backward pass 是**内存密集型**操作：
- 需要存储中间激活值（activation）
- 需要计算梯度
- FSDP 需要同步梯度（ALLGATHER 操作）
- 如果内存不足，会导致 OOM 或性能下降

## 代码逻辑

在 `verl/workers/actor/dp_actor.py` 中：

```python
# 检查内存（在 backward 之前）
free_mem, total_mem = torch.cuda.mem_get_info()
used_mem = total_mem - free_mem
usage_ratio = used_mem / total_mem  # allocated 比例

# 检查 reserved memory（更准确反映实际可用内存）
reserved_mem = torch.cuda.memory_reserved()
reserved_ratio = reserved_mem / total_mem  # reserved 比例

# 使用较大的值（更保守）
effective_ratio = max(usage_ratio, reserved_ratio)

# 如果超过 85%，触发清理
if effective_ratio >= 0.85:
    print(f"Warning: GPU memory high before backward "
          f"(allocated={usage_ratio*100:.1f}%, "
          f"reserved={reserved_ratio*100:.1f}%). Cleaning...")
    
    # 清理缓存
    torch.cuda.synchronize()  # 同步所有 CUDA 操作
    torch.cuda.empty_cache()  # 清理未使用的缓存
    time.sleep(0.5)  # 等待清理完成
```

## 这个警告是否严重？

### 不严重（正常情况）✅

如果：
- Reserved 在 85%-95% 之间
- Allocated 在 40%-50% 之间
- 警告后能继续训练
- 没有 OOM 错误

**说明**：这是正常的内存管理机制，代码会自动清理缓存。

### 需要关注 ⚠️

如果：
- Reserved 持续 > 95%
- Allocated 持续 > 60%
- 频繁出现警告
- 出现 OOM 错误

**说明**：GPU 内存压力较大，可能需要优化。

## 解决方案

### 如果警告频繁出现：

1. **减少 batch size**
   ```yaml
   worker:
     actor:
       micro_batch_size_per_device_for_update: 2  # 从 4 减到 2
   ```

2. **启用 CPU offload**
   ```yaml
   worker:
     actor:
       offload:
         offload_params: true
         offload_optimizer: true
   ```

3. **减少 vLLM 内存使用**
   ```yaml
   worker:
     rollout:
       gpu_memory_utilization: 0.6  # 从 0.75 减到 0.6
   ```

4. **降低内存检查阈值**（如果误报太多）
   - 在 `dp_actor.py` 中修改 `effective_ratio >= 0.85` 为 `0.90`

### 如果出现 OOM：

1. **进一步减少 batch size**
2. **启用更多 offload**
3. **检查是否有内存泄漏**（监控 `max_memory_allocated` 是否持续增长）

## 当前情况分析

从你的日志看：
- Allocated: 39%-50% ✅ 正常
- Reserved: 86%-97% ⚠️ 较高但可接受
- 警告后继续训练 ✅ 清理机制工作正常

**结论**：这是**正常的预防性内存管理**，代码在 backward 之前主动清理缓存，避免 OOM。如果训练能正常进行，可以忽略这些警告。

## 总结

- ✅ **这是正常的**：代码在 backward 前检查并清理内存
- ✅ **预防性措施**：避免在 backward 时 OOM
- ⚠️ **如果频繁出现**：可能需要优化配置
- ❌ **如果出现 OOM**：需要减少 batch size 或启用 offload

这些警告是**保护机制**，不是错误！




