# OOM 问题分析：利用率只有一半但报 OOM

## 问题现象
- GPU 利用率显示只有 ~50%
- 但训练时出现 OOM (Out of Memory) 错误

## 可能原因分析

### 1. **内存碎片化** ⭐⭐⭐⭐⭐ (最可能)
**问题**：PyTorch 的内存分配器会产生碎片。虽然总使用率只有 50%，但内存被分割成很多小块，无法分配连续的大块内存。

**表现**：
- `torch.cuda.memory_allocated()` 显示使用率不高
- 但尝试分配新张量时失败
- `torch.cuda.max_memory_reserved()` 可能接近 100%

**解决方案**：
```python
# 定期清理缓存，减少碎片
torch.cuda.empty_cache()
# 或者更激进的方式
torch.cuda.reset_peak_memory_stats()
```

### 2. **峰值内存使用** ⭐⭐⭐⭐
**问题**：虽然平均使用率是 50%，但在某些操作（forward pass、backward pass）时会瞬间达到峰值。

**表现**：
- 正常时使用率 ~50%
- forward/backward 时瞬间飙升到 100%+
- 导致 OOM

**解决方案**：
- 减小 batch size
- 使用 gradient checkpointing
- 更频繁地清理缓存

### 3. **PyTorch 缓存未释放** ⭐⭐⭐
**问题**：PyTorch 会缓存已释放的内存，不立即归还给系统。

**表现**：
- `memory_allocated()` 低
- `memory_reserved()` 高
- 实际可用内存少

**解决方案**：
```python
# 在关键位置清理缓存
torch.cuda.empty_cache()
# 或者设置内存分配器
torch.cuda.set_per_process_memory_fraction(0.8)  # 限制最大使用 80%
```

### 4. **批次大小不一致** ⭐⭐⭐
**问题**：某些批次可能特别大（特别是多模态数据），导致瞬间 OOM。

**表现**：
- 大部分批次正常
- 偶尔遇到大批次就 OOM

**解决方案**：
- 动态调整 batch size
- 跳过过大的批次
- 限制 max_prompt_length

### 5. **多进程/分布式问题** ⭐⭐
**问题**：在分布式训练中，每个进程看到的内存使用可能不同。

**表现**：
- 主进程显示 50%
- 但某个 worker 进程可能已经 100%

## 诊断方法

### 1. 检查内存统计
```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print(f"Max Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
```

### 2. 检查内存碎片
```python
# 查看内存分配情况
torch.cuda.memory_summary()
```

### 3. 监控峰值内存
在 forward 和 backward 前后记录内存使用。

## 推荐的修复方案

### 方案 1：更激进的内存清理（推荐）
在关键位置添加更频繁的内存清理。

### 方案 2：限制最大内存使用
设置 PyTorch 的最大内存使用比例。

### 方案 3：动态批次大小
根据可用内存动态调整 batch size。

### 方案 4：使用内存池
使用 PyTorch 的内存池来减少碎片。










