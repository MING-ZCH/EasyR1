# V17/V18 Corrected First-Principles Analysis (修正版)

## 一、Skip Update 深入分析 — 模型不是NaN污染

### 1.1 代码确认

```python
# verl/workers/actor/dp_actor.py L314-323
def _optimizer_step(self) -> torch.Tensor:
    grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
    if not torch.isfinite(grad_norm):
        print("Gradient norm is not finite. Skip update.")
        self.actor_optimizer.zero_grad()  # 清零梯度
        return grad_norm                   # ← 不调 optimizer.step(), 权重不更新
    # ... 正常走下面的 optimizer.step()
```

**Skip update = 真正跳过**:
1. `zero_grad()` 清零所有梯度缓冲区
2. **不调用** `optimizer.step()` → 权重不变, Adam state 不更新
3. 仅返回 NaN grad_norm 给日志

### 1.2 V17 Step 90 模型的真实状态

V17 Grad Norm 完整轨迹:
```
Step 75: 1.824 (✓正常)
Step 76: 4.932 (✓有限值, 更新成功)
Step 77-80: NaN (skip, 权重冻结在step 76状态)
Step 81: 2.009 (✓短暂恢复, 更新成功)
Step 82-90: NaN (skip, 权重冻结在step 81状态)
```

**结论: V17 step 90 checkpoint ≈ step 81 的模型权重, 不是NaN损坏的模型。**

### 1.3 PPO Epochs=2 的微妙交互

代码中 `_optimizer_step()` 在 ppo_epochs 循环内被调用:
```python
for _ in range(self.config.ppo_epochs):  # 2次
    for mini_batch in mini_batches:
        grad_norm = self._optimizer_step()  # 每次独立判断
```

这意味着对于"NaN步":
- PPO Epoch 1 **可能成功** → 权重被更新一次
- PPO Epoch 2 NaN → 跳过
- 日志记录的 grad_norm 是两个 epoch 的聚合 (avg包含NaN = NaN)

**所以 step 90 的模型并非简单"冻结在step 76"**, 而可能包含 step 77-81 的 PPO Epoch 1 的部分更新。
但核心事实不变: **模型权重不会被NaN值写入**, 所有 NaN 步的更新都被安全跳过。

### 1.4 真正的问题: 不是NaN污染, 是过拟合

V17 验证集轨迹:
```
Val call 50: 0.8594 (PEAK)
Val call 60: 0.8125 (下降中)
Val call 63: 0.7647 (终点)
```

**Val peak 在 step ~50, 之后持续下降**。这不是 NaN 导致的退化, 而是**训练数据用完一遍后(1 epoch = 90 steps)的经典过拟合/reward hacking**。

Step 90 的评测结果差(StepCount-Bench 12.2% vs V12-step132 15.45%), 根本原因是:
1. 评测的是 step 90 模型 (已过拟合/过训练)
2. Val peak 在 step 50 但该 checkpoint 未被保存 (save_freq=30)
3. 如果评测 step 50 的模型, 结果可能超越 V12

---

## 二、BF16 精度问题 — 论文研究结论

### 2.1 关键论文发现

**arXiv:2510.26788** "Defeating the Training-Inference Mismatch via FP16" (Sea AI Lab):

| 维度 | BF16 | FP16 |
|:---:|:---:|:---:|
| 指数位 | 8 bits (大动态范围) | 5 bits |
| **尾数位** | **7 bits (低精度)** | **10 bits (8倍精度)** |
| GRPO 稳定性 | 训练崩溃 (73-84%) | 稳定收敛 (~99%) |
| Log-prob mismatch | ~24× | ~1× |

**核心发现**: 即使 training (FSDP) 和 inference (vLLM) 都用 BF16, 它们使用不同 CUDA kernel 实现, 
产生不同的舍入误差。在 autoregressive 生成中, 这些误差 **逐token累积**, 导致:
$$\frac{\pi_\theta(y|x)}{\mu_\theta(y|x)} \neq 1.0$$
其中 $\pi_\theta$ = training forward, $\mu_\theta$ = vLLM rollout, **两者参数完全相同但计算路径不同**。

### 2.2 你的 EasyR1 精度配置

```
Rollout (vLLM):    dtype = bf16
Actor FSDP:        param_dtype = bf16, reduce_dtype = fp32, buffer_dtype = fp32
Reference Model:   torch_dtype = bfloat16
```

**问题**: Rollout (vLLM bf16) 和 Training Forward (FSDP bf16) 使用不同 CUDA kernel:
- vLLM: PagedAttention kernel, 高度优化的 autoregressive inference kernel
- FSDP + FlashAttention2: 不同的 matmul kernel 和 attention kernel
- 两者即使输入相同, 产生的 log_probs 有微小差异
- 这些差异在 importance ratio $\exp(\log\pi - \log\mu)$ 中被放大
- 随训练进行, policy 偏离 → ratio 偏离加剧 → 梯度不稳定

### 2.3 BF16 是否是 V17 NaN 的原因?

**部分是, 但不是直接原因。** 因果链:

```
BF16 training-inference mismatch
    → importance ratio 有系统性bias
    → PPO epoch 2 (更stale的ratio) 放大 bias
    → 累积 ~76 steps 后梯度不稳定
    → Step 76 grad_norm spike 到 4.932
    → Step 77 NaN (不可逆)
```

V17 NaN 的直接诱因是 **ppo_epochs=2 + 75步累积偏漂移**, 而 BF16 mismatch 是底层加速因子。

### 2.4 是否需要切换到 FP16?

**优先级分析:**

| 方案 | 收益 | 实施难度 | 推荐优先级 |
|:---:|:---:|:---:|:---:|
| ppo_epochs 2→1 | 消除最大NaN源 | 改一个参数 | **★★★★★** |
| save_freq 30→10 | 捕捉val peak | 改一个参数 | **★★★★★** |
| easy data代替mixed | 降低hard-group不稳定 | 改一个参数 | **★★★★☆** |
| **FP16** | 消除training-inference mismatch | 需验证Flash-Attn兼容性 | **★★★☆☆** |
| max_grad_norm 1.0→0.5 | 更强裁剪 | 改一个参数 | **★★★☆☆** |

**FP16 不是当前最紧急的**。论文中 BF16 崩溃发生在 73-84% accuracy (对应我们的 0.73-0.84 answer_mean), 
与 V17 的 NaN 范围 (val 0.67-0.86) 吻合。**但 ppo_epochs=1 + easy data 可能已经足够避免 NaN**, 
因为 V9/V11 (ppo_epochs=1, easy data, BF16) 训练完成 0 NaN。

**建议**: 先尝试 ppo_epochs=1 + easy data + save_freq=10。如果仍出现 NaN, 再考虑 FP16。

---

## 三、修正后的评测对比表

### StepCount-Bench-500 (count range 11-50, 密集对象)

| Version | Config | Acc | Clean Steps | NaN Onset |
|:---:|:---:|:---:|:---:|:---:|
| SFT-3537 | baseline | 9.60% | — | — |
| V7 (GRPO) | easy, 1ep | 10.60% | Full | 0 NaN |
| V11 (BoK) | easy, 1ep | 12.00% | Full | 0 NaN |
| V12-final | easy, 2ep | 13.00% | 84 | NaN@85 |
| **V12-step132** | easy, 2ep (best ckpt) | **15.45%** | — | checkpoint pre-NaN |
| V14-mixed | mixed, 1ep | 11.40% | Full | 0 NaN |
| V16 | easy, 2ep | 13.20% | Full | 0 NaN |
| **V17** | mixed, 1ep, Cap=1.0 | **12.20%** | 76 | NaN@77 |

### CountBench-491 (count range 0-10) & PixMo-529

| Version | CountBench | PixMo (w/h) |
|:---:|:---:|:---:|
| SFT-10k-1296 | 78.82% | — |
| V12-step132 | 79.43% | 82.80% |
| V15 | 79.02% | 81.66% |
| V16 | 77.19% | 81.29% |
| **V17** | **79.43%** | **81.29%** |

**V17 在 sparse 场景 (CountBench/PixMo) 与 V12 持平, 在 dense 场景 (StepCount-Bench) 落后 3pp。**

---

## 四、根因排序 (修正版)

| 排名 | 因素 | 对评测分数的影响 | 证据 |
|:---:|:---|:---:|:---|
| 1 | **Step 90 过拟合 (val peak在step 50)** | -3~5pp | Val: 0.8594→0.7647, 但只保存了step 90 |
| 2 | **Mixed data不适配** | -1~2pp | V14-mixed 11.4% < V11-easy 12.0% |
| 3 | **NaN导致训练中断@step 77** | -0~1pp | 如果无NaN, step 77-90可能继续学习 |
| 4 | **BF16 training-inference mismatch** | 系统性ceiling | 限制所有版本的最终精度上限 |

**最关键的修正: NaN不是评测退化的主因 (权重被安全保护), 真正的问题是 val peak 未被保存。**

---

## 五、最终建议

### Tier 1: V18 紧急修改
- **save_freq 90→10** (否则唯一保存的也是最后可能已过拟合的checkpoint)

### Tier 2: V19 配置
```bash
PPO_EPOCHS=1          # 消除 stale ratio NaN 源头    
DATA=easy             # 最稳定的训练数据
SAVE_FREQ=10          # 捕捉 val peak
TOTAL_EPOCHS=2        # 358步, 弥补ppo_epochs减半
MAX_GRAD_NORM=0.5     # 更严梯度裁剪
GRAD_SPIKE_PROTECT=1  # 保留保护, threshold可降至3x
```

### Tier 3: 精度升级 (FP16)
- 修改 `verl/workers/actor/config.py`: `mp_param_dtype: str = "fp16"` 
- 修改 `verl/workers/rollout/config.py`: `dtype: str = "fp16"`  
- 需要验证: FlashAttention2 + Qwen2.5-VL 在 FP16 下的兼容性和内存占用
- 预期效果: 消除 training-inference mismatch, 可能允许更长训练和更高 ppo_epochs

