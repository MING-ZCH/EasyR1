# 修复 Image Features 和 Tokens 不匹配问题

## 问题分析

错误信息：`ValueError: Image features and image tokens do not match: tokens: 18557, features 20167`

**根本原因**：
- 当 `max_prompt_length` 设置过小时，prompt 会被截断
- 截断发生在 image tokens 位置时，会导致 image tokens 数量减少
- 但 `image_grid_thw`（image features）没有被相应调整
- 从而造成 tokens 和 features 数量不匹配

## 解决方案

### 方案1：增加 `max_prompt_length`（推荐）

这是最简单直接的解决方案，可以避免 prompt 被截断。

#### 修改配置文件（config.yaml）

```yaml
data:
  max_prompt_length: 8192  # 从 4096 增加到 8192 或更大
  # 或者根据你的数据情况设置：
  # max_prompt_length: 16384  # 如果数据中有很长的 prompt
```

#### 修改启动脚本（.sh 文件）

如果使用命令行参数启动，可以添加：

```bash
--data.max_prompt_length=8192 \
```

#### 注意事项

1. **内存消耗**：增加 `max_prompt_length` 会增加内存使用
   - 需要确保有足够的 GPU/CPU 内存
   - 可能需要相应调整 `worker.rollout.gpu_memory_utilization`

2. **性能影响**：更长的 prompt 会：
   - 增加计算时间
   - 可能需要调整 batch size

3. **建议值**：
   - 如果错误显示 tokens: 18557，建议设置 `max_prompt_length >= 20000`
   - 可以设置为 `max_prompt_length: 25000` 或更大，留一些余量

### 方案2：减少 `max_pixels`（备选）

如果无法增加 `max_prompt_length`（内存限制），可以尝试减少 `max_pixels`：

```yaml
data:
  max_pixels: 10485760  # 从 12845056 减少
```

这会减少每个图像的 token 数量，从而减少总的 prompt 长度。

### 方案3：使用数据校验和跳过机制（已实现）

如果某些数据确实无法匹配，可以使用已实现的校验和跳过机制：
- 自动检测不匹配的数据
- 跳过有问题的 batch，继续训练
- 记录统计信息

## 推荐配置

根据错误信息 `tokens: 18557, features 20167`，建议：

```yaml
data:
  max_prompt_length: 25000  # 留足够余量
  max_response_length: 4096
  max_pixels: 12845056
  filter_overlong_prompts: true  # 过滤过长的 prompt
```

或者如果内存充足：

```yaml
data:
  max_prompt_length: 32768  # 更大的值
  max_response_length: 4096
```

## 验证

修改配置后，重新运行训练，应该不会再出现该错误。

如果仍然出现，可以：
1. 进一步增加 `max_prompt_length`
2. 检查是否有异常长的数据样本
3. 使用 `filter_overlong_prompts: true` 过滤掉过长的 prompt












