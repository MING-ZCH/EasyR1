# 全日志对比分析 & DrGRPO vs BoK-GRPO 算法选择

## 一、14个训练日志 Trajectory Mask Reward 统计

### 核心结论：所有训练日志 `traj_no_sequence=0`，训练mask reward从未失败

| 实验 | TrajEval步数 | Hit率 | Entropy(末) | 备注 |
|------|-------------|-------|------------|------|
| v6_bok_grpo (主训练) | 201,790 | **83.3%** | 0.513 | low_var=2048/2048 |
| v6_bok_grpo (续训练) | 1,490 | **82.7%** | 0.785 | low_var=1024/1024 |
| v5_bok_grpo | 1,460 | 81.7% | 3.871 | entropy偏高 |
| 10k_bok_grpo | 1,139,480 | 79.5% | **13.143** | entropy爆炸! |
| bok_grpo_early | 3,130 | 81.5% | 0.890 | |
| bok_grpo_long | 1,239,240 | **82.5%** | 0.763 | |
| drgrpo_0_early (有KL) | 345,150 | 79.7% | **0.014** | entropy崩塌! |
| drgrpo_0_DisableKL | 986,910 | 76.3% | 1.728 | entropy健康 |
| drgrpo_1_DisableKL | 442,200 | 79.3% | 1.310 | entropy健康 |
| 10k_gspo | 215,900 | **72.6%** | 0.356 | hit率最低 |
| gspo | 20,060 | 81.5% | 0.487 | |
| 10k_standard_grpo_1 | 1,440 | 80.4% | 0.467 | |
| 10k_standard_grpo_2 | 1,560 | 81.3% | 0.488 | |
| 10k_process_reward | 1,420 | 80.3% | 0.478 | |

### 关键发现

1. **`mask_no_sequence`只出现在验证进程**(pixmo-test数据)，所有训练进程`traj_no_sequence=0`
2. **Hit率范围: 72.6%-83.3%**，v6 BoK-GRPO最高(83.3%)
3. **Entropy异常**:
   - DrGRPO有KL → 0.014 (崩塌，不可用)
   - 10k_bok_grpo → 13.143 (爆炸，可能config问题)
   - v5 → 3.871 (偏高)

---

## 二、DrGRPO vs BoK-GRPO 算法分析

### 2.1 当前真实行为（low_var=100%时）

**关键发现：当`low_var=100%`时，BoK-GRPO的zscore fallback与DrGRPO在advantage计算上完全等价！**

| 维度 | DrGRPO | BoK-GRPO (zscore fallback) |
|------|--------|---------------------------|
| advantage计算 | `(score - batch_mean) / (batch_std + eps)` | `(score - batch_mean) / (batch_std + eps)` |
| KL惩罚 | 标准KL / 禁用 | `low_var_kl`: `kl.exp() - kl - 1` |
| Entropy趋势 | 无KL: 上升至1.3-1.7; 有KL: 崩塌至0.014 | 稳定在0.5-0.8 |
| BoK机制 | N/A | **从未激活** (需group_std > threshold) |

### 2.2 行为差异本质

两种算法在advantage计算上**完全相同**（都走batch-level zscore），差异仅在**KL正则化**：

- **BoK + low_var_kl**: Schulman's `exp(kl) - kl - 1`，温和惩罚远离ref policy → entropy稳定(0.5-0.8)
- **DrGRPO 无KL**: 无任何正则化 → entropy自由上升(1.3-1.7)，更多探索但有漂移风险
- **DrGRPO 有KL**: 标准KL惩罚过强(在低方差情况下) → entropy崩塌至0.014

### 2.3 结论与建议

**无需切换算法**。当前v6的BoK-GRPO（zscore fallback + low_var_kl）已经是合理配置：

1. **Advantage计算等价**: 因为low_var=100%，zscore fallback = DrGRPO
2. **KL正则化优于DrGRPO**: low_var_kl比标准KL温和，比无KL更稳定
3. **真正的瓶颈不在算法层**: 是VLM坐标级确定性导致16个rollout完全相同

### 2.4 根本问题与优化方向

**Root Cause**: temperature=1.2不足以让VLM在坐标精度级别产生多样性
→ 16个rollout生成几乎相同的point坐标 → 相同的mask reward → group_std≈0 → low_var=100%

**优化方向（按优先级）**:
1. **提升采样多样性**: temperature>1.5, top_p调低(如0.9→0.8), top_k限制
2. **坐标级噪声**: 在point坐标生成后添加小幅随机扰动
3. **Per-rollout图像增强**: 对同一prompt的不同rollout做轻微图像扰动(色调/亮度)
4. **Prompt变体**: 对同一题目的16个rollout使用略微不同的prompt措辞

---

## 三、修改记录

### eval_multi_task.sh 修改 (本次)
- **Per-task默认参数**: 每个数据集独立配置 dot_radius/max_rounds/history
  - pixmo-test & countqa: dot=20, max_turn=11, history=0
  - stepcount-500: dot=10, max_turn=11, history=0  
  - bias: dot=20, max_turn=32, history=0
- **默认no-think**: `DISABLE_THINKING=True`
- **History命名**: 开启history时输出名追加 `_with_history_{N}` 后缀
- **CLI覆盖**: 命令行参数覆盖task默认值 (`__unset__`标记未指定)
- **移除emoji**: 改用`[INFO]`/`[OK]`/`[FAIL]`/`[ERROR]`标签


---

## 重大发现：`SamplingParams.seed=1` 是 low_var=100% 的根因

### 发现时间
2025-03-07

### 问题描述
v6 BoK-GRPO 训练中 `low_var=100%`（1024/1024 prompts 的 16 个 rollout 全部 reward 相同），导致：
- BoK 永远不激活（需要 best-of-k 对比，但所有 k 相同）
- DrGRPO zscore fallback 接管，但组内零方差无法提供有效的 per-sample advantage
- 实际变成了 "全局 baseline 回退" 而非 RL 多样性探索

用户关键矛盾：`pass@32=95% >> pass@1=79.2%`（评测），但训练时 `low_var=100%`。如果模型有能力产生多样输出（pass@32 高），为什么训练中16个rollout完全一样?

### 根因分析

#### 证据链（完整传播路径）

1. **源头**: `verl/workers/rollout/config.py:30`
   ```python
   @dataclass
   class RolloutConfig:
       seed: int = 1   # ← 默认 seed=1
   ```

2. **传播到 vLLM Engine 构造**: `verl/workers/rollout/vllm_rollout_spmd.py:452`
   ```python
   self._engine_init_args = dict(
       ...
       seed=config.seed,   # ← seed=1 传到 vLLM engine（用于初始化，这是正确的）
       ...
   )
   ```

3. **BUG: 泄漏到 SamplingParams**: `vllm_rollout_spmd.py:478-481`
   ```python
   default_sampling_params = SamplingParams()
   for key in config.to_dict().keys():
       if hasattr(default_sampling_params, key):
           sampling_kwargs[key] = getattr(config, key)
   ```
   - `RolloutConfig` 有 `seed=1` 字段
   - vLLM 的 `SamplingParams` 也有 `seed` 属性
   - **循环匹配 → seed=1 被复制到 SamplingParams**

4. **训练日志确认** (v6 bok_grpo_continued.log):
   - Config 输出: `"seed": 1` (在 RolloutConfig section, line 175)
   - 4个GPU Worker: `Sampling params: {..., 'seed': 1, ...}` (lines 511, 516, 736, 744)

5. **Interleaved 模式放大**: `vllm_rollout_spmd.py:682-694,794`
   ```python
   # 每个 prompt 被复制 16 次（= desired_n）
   repeat_times = max(desired_n, 1)
   for _ in range(repeat_times):
       active_prompt_ids.append(list(item["prompt_token_ids"]))
   
   # 然后以 n=1 逐步生成
   with self.update_sampling_params(n=1):  # 只改 n，seed 仍保持 =1
       ...
   ```
   **结果: 16份相同prompt × n=1 × seed=1 → 16次完全相同的输出**

#### vLLM 的 seed 行为
- `SamplingParams.seed = None` (默认): 每次请求使用随机种子 → **非确定性输出**
- `SamplingParams.seed = 1` (被泄漏): 每次请求使用固定种子 → **确定性输出**
- 相同 prompt + 相同 seed → **完全相同的 token 序列**，无论 temperature 多高

#### 为什么 pass@32=95% 但 low_var=100%
- **评测**: eval 脚本没有设置 seed（默认 None），使用 `do_sample=False` 或 temperature 采样 → 非确定性 → pass@k 随 k 增大而提高
- **训练**: RolloutConfig.seed=1 泄漏到 SamplingParams → 16个 rollout 全部确定性地产生相同输出 → low_var=100%

### 修复方案

修复位置: `verl/workers/rollout/vllm_rollout_spmd.py:478`

```python
_SAMPLING_PARAMS_SKIP_KEYS = {"seed"}
for key in config.to_dict().keys():
    if key in _SAMPLING_PARAMS_SKIP_KEYS:
        continue
    if hasattr(default_sampling_params, key):
        sampling_kwargs[key] = getattr(config, key)
```

**状态**: 已应用 ✅ (在代码文件中)。下次训练运行将生效。

### 预期影响
修复后：
1. **SamplingParams.seed = 不会被设置** → 默认 None → vLLM 每次生成不同输出
2. **16 个 rollout 将产生多样性** → low_var 大幅降低
3. **BoK-GRPO 正常激活**: best-of-k 可以选出最优轨迹
4. **真正的 RL 探索**: 组内有方差 → advantage 有区分度 → 梯度有方向
5. **Temperature=1.2 开始生效**: 不再被 seed=1 覆盖，采样真正有随机性

### 附注
- Engine seed (`LLM(seed=config.seed)`, line 452) 保持不变 → 模型加载和初始化仍可复现
- DataConfig.seed=42 (data shuffling) 不受影响
- 仅排除了 SamplingParams 的 seed，不影响其他 sampling 参数 (temperature, top_p, etc.)

---

## seed修复后参数调整建议 (2025-03-07)

### 当前v6核心参数回顾

| 参数 | v6值 | 说明 |
|------|------|------|
| `adv_estimator` | `bok_grpo` | Best-of-K GRPO |
| `bok_tau` cosine | 0.8 → 0.3 / 178步 | softmax温度递减 |
| `BOK_LOW_VAR_THRESHOLD` | 1e-5 | low_var门槛 |
| `BOK_FALLBACK_MODE` | `zscore` | 低方差时batch z-norm |
| `temperature` | 1.2 | 采样温度 |
| `n` | 16 | rollout数量 |
| `kl_penalty` | `low_var_kl` | Schulman's KL: exp(kl)-kl-1 |
| `kl_coef` | 0.05 | KL系数 |
| `lr` | 1e-6 | 学习率 |
| `global_batch_size` | 64 | 训练批次 |
| reward weights | 0.6/0.3/0.1 | answer/point/format |
| `history_mode` | 0 | 无文本历史 |
| `max_turns` | 11 | 最大轮数 |

### seed修复后训练动态变化

- 修复前: seed=1 → 16 rollout完全相同 → low_var=100% → BoK全部退化为zscore fallback
- 修复后: seed=None → 16 rollout多样化 → BoK正常工作 → per-group梯度信号

### 参数建议

- 保持: bok_grpo, tau 0.8→0.3, temperature=1.2, n=16, lr=1e-6, reward weights
- 监控: low_var比例(预期<30%), entropy(0.5-2.0), KL(<5.0)
- 可调: kl_coef如KL>5→0.08-0.10; tau如不稳定→0.5→0.2

---

## history_mode=0上下文分析 + train/eval对比

### history_mode=0核心逻辑

模型每轮看不到历史信息(上一轮output和process prompt)，只看到:
1. base_prompt (dataset prompt + first_turn_prompt，含JSON格式+红点说明)
2. process_prompt (追加为新user turn)
3. 带红点的更新图片 (唯一的视觉历史)

代码路径:
- vllm_rollout_spmd.py:752: base_prompt_ids保存初始prompt
- vllm_rollout_spmd.py:937: active_prompt_ids重置到base_prompt_ids (丢弃所有生成)
- vllm_rollout_spmd.py:941: 追加process_prompt

### 训练vs评测结构差异

| 维度 | 训练 | 评测 |
|------|------|------|
| 系统prompt | 默认"You are Qwen..." | 自定义(含JSON格式) |
| JSON格式位置 | user turn (first_turn_prompt) | system prompt |
| 红点引导 | first_turn_prompt + process_prompt | system prompt无, process说"observation image" |
| 图片位置 | dataset的user turn | 当前process的user turn |
| user消息数 | 3 (dataset+first_turn+process) | 1 (process+image) |

影响: 信息内容基本一致但消息结构不同，可能导致eval分布偏移。

---

## process prompt修改记录

### 变更说明
- 日期: 2025-03-07
- 文件: examples/format_prompt/StepCount_interleaved_process_prompt.txt
- 备份: StepCount_interleaved_process_prompt.txt.bak_v6

### 修改内容
```diff
- outputting <point> and </point> as before
+ outputting <point>{"point_2d": [x, y], "label": "object", "count_number": "n"}</point>
```

### 原因
1. 虽然history_mode=0下base_prompt含first_turn_prompt(有JSON格式)，但"as before"是隐式引用
2. RL训练中模型逐渐偏离SFT分布，显式格式可减少format reward=0的边缘情况
3. 增加约15 tokens/轮，成本极低
4. JSON格式与first_turn_prompt一致("object", "n" placeholder)
