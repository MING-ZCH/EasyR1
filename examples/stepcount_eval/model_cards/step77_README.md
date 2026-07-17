---
language:
- zh
- en
license: other
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- qwen2.5-vl
- stepcount
- counting
---

# StepCount-7B v36 focused10k step77

这是 V36 的 secondary checkpoint。训练日志表明 step68、75、76、77 的更新被 GradSpikeProtect 完整跳过，因此该 actor 应标注为 **post-step74 weights**；这来自训练控制流与日志证据，不是 step74/step77 的逐 tensor checksum 等价证明。它在 bias 分布上更强，但不是全面优于 step60。

## 严格评测结果

| 任务 | 正确数/样本数 | Accuracy |
|---|---:|---:|
| pixmo-test | 433/529 | 81.85% |
| stepcount-500 | 70/500 | 14.00% |
| countqa | 390/491 | 79.43% |
| bias | 314/1992 | 15.76% |

协议为 `strict_oracle_gt_plus_v3`：BF16、greedy、每个样本 `min(task cap, GT+3)`、必须显式闭合 `<answer>`、禁止 point-count fallback、遇首个 `</point>` 进入下一轮、遇首个 `</answer>` 结束样本、history=0、默认不保留中间图片。

## 定位与局限

- step77 偏向 bias repeated-pattern 鲁棒性，并在 pixmo-test 上略高；step60 在 dense 与 countqa 上更稳。
- bias 的优势集中于特定模板和分辨率结构，不能外推为普遍 OOD 鲁棒性。
- 模型依赖 oracle GT 来设置最大轮数，因此上述分数不是未知 GT 部署场景的端到端吞吐指标。
- stepcount-500 的绝对准确率仍低，长轨迹停止边界与编号连续性并未完全解决。
- “post-step74 weights” 是基于跳步日志的描述，不应表述成已经做过 checkpoint tensor checksum 证明。

评测与上传前请使用随发布 bundle 提供的 preflight、分析器和文件 size 校验。
