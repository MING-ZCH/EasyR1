# 修复 CPU Offload 与梯度累积冲突错误

## 错误信息
```
ValueError: actor cannot use FSDP's CPU offload when gradient accumulation is enabled.
```

## 问题原因

FSDP 的 CPU offload (`enable_cpu_offload: true`) 与梯度累积（gradient accumulation）不兼容。

**梯度累积发生条件**：
- 当 `global_batch_size_per_device != micro_batch_size_per_device_for_update` 时
- 例如：`global_batch_size_per_device = 4`, `micro_batch_size_per_device_for_update = 2`
- 这意味着需要 2 步梯度累积

## 解决方案

### 方案 1：禁用 CPU Offload（如果 GPU 内存足够）⭐ 推荐

在配置文件中修改：

```yaml
worker:
  actor:
    fsdp:
      enable_cpu_offload: false  # 改为 false
    offload:
      offload_params: true        # 使用 parameter offload 代替
      offload_optimizer: true
```

**优点**：
- 性能更好（CPU-GPU 传输减少）
- 仍然可以通过 parameter offload 节省内存

**缺点**：
- 需要更多 GPU 内存

### 方案 2：禁用梯度累积

确保 `micro_batch_size_per_device_for_update` 等于 `global_batch_size_per_device`：

```yaml
worker:
  actor:
    global_batch_size: 32
    micro_batch_size_per_device_for_update: 4  # 必须等于 global_batch_size / num_gpus
```

**计算方式**：
- 如果有 8 个 GPU：`global_batch_size_per_device = 32 / 8 = 4`
- 所以 `micro_batch_size_per_device_for_update` 必须等于 4

**优点**：
- 可以继续使用 CPU offload

**缺点**：
- 可能增加内存使用（因为不能使用更小的 micro batch）

### 方案 3：使用 Parameter Offload 代替（推荐）⭐⭐⭐

Parameter offload 与梯度累积兼容，是更好的选择：

```yaml
worker:
  actor:
    fsdp:
      enable_cpu_offload: false  # 禁用 FSDP CPU offload
    offload:
      offload_params: true        # 使用 parameter offload
      offload_optimizer: true
```

**优点**：
- 与梯度累积兼容
- 仍然可以节省内存
- 性能通常比 FSDP CPU offload 更好

## 当前配置分析

根据你的配置：
- `global_batch_size: 32`
- `micro_batch_size_per_device_for_update: 4`
- `n_gpus_per_node: 8`
- `enable_cpu_offload: true`

**计算**：
- `global_batch_size_per_device = 32 / 8 = 4`
- `micro_batch_size_per_device_for_update = 4`
- 所以 `4 == 4`，理论上不应该有梯度累积

**但错误仍然发生，可能原因**：
1. 实际 GPU 数量不是 8
2. 或者有其他配置覆盖了这些值

## 推荐修复

**最简单的方法**：禁用 FSDP CPU offload，使用 parameter offload

```yaml
worker:
  actor:
    fsdp:
      enable_cpu_offload: false  # 改为 false
    offload:
      offload_params: true       # 保持 true
      offload_optimizer: true   # 保持 true
```

这样既节省内存，又与梯度累积兼容。










