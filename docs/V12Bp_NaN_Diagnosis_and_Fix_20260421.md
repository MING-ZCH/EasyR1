# V12-B' NaN 诊断与修复（2026-04-21）

## 结论概览
- 首次 NaN：step 91
- 形态：间歇性持续 NaN（step 94/95 曾短暂恢复，随后再次频繁 NaN）
- 本质：不是一次性模型彻底损坏，而是高激进参数在部分 batch 上反复触发数值非有限

## 证据（来自 training_interleaved_traj_v12Bp_bok_grpo_bok_grpo_20260420_014458.log）
1. 首次触发窗口
   - `[BoK-GRPO] ... step=91/213 ... adv_range=[-2.500, 4.000]`
   - 紧接着：`Gradient norm is not finite. Skip update.`
   - 同步 actor 指标：`grad_norm: nan`, `entropy_loss: 1.11`, `kl_loss: 0.148`, `lr: 2.0e-06`
2. 触发前已显著升温
   - step 83~90: grad_norm 从 2.763、4.584、5.838、6.407 波动
   - 最高有限值：step 89, grad_norm=6.407
3. 触发后非永久坏死
   - step 94/95 恢复为有限 grad_norm（2.405, 2.389）
   - 说明是“批次诱发的数值不稳定”，不是权重完全炸毁
4. 但后续仍频繁复发
   - step 91~108 共 16 个 NaN step

## 根因分析（按贡献排序）
1. 学习率 + sign-GD 组合过激进
   - `LR=2e-6 + max_grad_norm=0.5 + ppo=1` 在该 mixed 数据分布下仍偏激进
2. BoK优势上限过高
   - `BOK_CLIP=4.0`，优势上限较高，增加某些 batch 的更新风险
3. 原有保护只“跳过本步”，缺少“非有限后的自愈降速”
   - 旧逻辑：non-finite 仅 zero_grad + return
   - 导致下一步仍在相同 LR 上反复触发

## 已实施修复

### 1) 代码级修复（核心）
文件：`verl/workers/actor/dp_actor.py`
- 新增 non-finite 统计与历史：`_nonfinite_count`, `_nonfinite_history`
- 当 `grad_norm` 非有限时：
  - 记录并打印详细事件（含 opt_step、count、lr）
  - 触发临时降 LR + cooldown（复用 spike 机制）
  - 在窗口内非有限事件过多时触发 emergency brake（永久减半 base LR）
- 新增可观测指标：`actor/nonfinite_grad_count`

### 2) 启动脚本级修复（稳态版本）
新脚本：`examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v12_Bp_fix.sh`
- `ACTOR_LR: 2e-6 -> 1.5e-6`
- `BOK_CLIP: 4.0 -> 3.0`
- `kl_coef: 0.02 -> 0.03`
- `BOK_ADV_NORMALIZE: 0 -> 1`
- `GRAD_SPIKE_ABSOLUTE_CAP: 4.0 -> 3.0`
- 新增 non-finite 自愈参数：
  - `GRAD_NONFINITE_COOLDOWN=8`
  - `GRAD_NONFINITE_LR_FACTOR=0.3`
  - `GRAD_NONFINITE_BRAKE_WINDOW=20`
  - `GRAD_NONFINITE_BRAKE_MAX=2`

## 运行建议
- 当前 v12Bp run 已结束（进程不在）
- 建议使用 `v12_Bp_fix.sh` 重新启动
- 重点观测：
  - `actor/nonfinite_grad_count` 是否停止增长
  - `grad_norm` 是否稳定回到 <3 区间
  - `format_fail` 是否维持 <5%


## V2 稳定版脚本（后续优化）
- 脚本：`examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v12_Bp_fix_v2.sh`
- 进一步收敛稳定性：
  - `LR=1.2e-6`
  - `GRAD_SPIKE_ABSOLUTE_CAP=2.5`
  - `GRAD_SPIKE_THRESHOLD=3.0`
  - `GRAD_NONFINITE_COOLDOWN=10`, `GRAD_NONFINITE_LR_FACTOR=0.2`
  - `data.max_prompt_length=7000`, `data.max_response_length=2800`
  - `INTERLEAVED_PER_TURN_MAX_TOKENS=1600`, `INTERLEAVED_ANSWER_TURN_MAX_TOKENS=1400`
