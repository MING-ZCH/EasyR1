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

# StepCount-7B v36 focused10k step60

这是 V36 的主 checkpoint。预设选模目标给 dense counting 与 broad 三套任务各 50% 权重；在该口径下，step60 的 composite 为 36.87%，略高于 step77 的 36.51%。

## 严格评测结果

| 任务 | 正确数/样本数 | Accuracy |
|---|---:|---:|
| pixmo-test | 426/529 | 80.53% |
| stepcount-500 | 76/500 | 15.20% |
| countqa | 398/491 | 81.06% |
| bias | 280/1992 | 14.06% |

协议为 `strict_oracle_gt_plus_v3`：BF16、greedy、每个样本 `min(task cap, GT+3)`、必须显式闭合 `<answer>`、禁止 point-count fallback、遇首个 `</point>` 进入下一轮、遇首个 `</answer>` 结束样本、history=0、默认不保留中间图片。

## 定位与局限

- step60 是当前预设 50/50 composite 下的主 checkpoint，在 stepcount-500 与 countqa 上优于 step77。
- 模型依赖 oracle GT 来设置最大轮数，因此上述分数不是未知 GT 部署场景的端到端吞吐指标。
- stepcount-500 的绝对准确率仍低，长轨迹停止边界与编号连续性并未完全解决。
- 四套测试集存在分布差异；小于约 1--2pp 的差值不应直接解释为稳定总体能力差异。
- bias 含重复语义模板和多分辨率变体，不能代表所有 OOD counting 场景。

评测与上传前请使用随发布 bundle 提供的 preflight、分析器和文件 size 校验。
