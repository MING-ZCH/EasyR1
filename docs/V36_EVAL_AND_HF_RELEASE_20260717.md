# V36 Eval 与 Hugging Face 发布记录（2026-07-17）

## 目的与改动边界

本次在 `examples/stepcount_eval/` 增加独立、可移植的 V36 StepCount eval/release bundle，并补充本说明文档；未修改任何训练代码。Git/HF 发布状态在完成远端校验后以本文末尾 revision/receipt 为准。

## 代码来源

核心语义来自 StepcountModel 的以下权威材料，移植时按 basename 记录，避免把源集群用户目录固化到产物：

- `dataset/eval_fix_pixels_one_count_per_time.py`
- `dataset/eval_multi_task.sh`
- `dataset/run_v36_candidate_eval.sh`
- `dataset/analyze_v36_candidate_eval.py`
- `dataset/preflight_eval_bundle.py`
- `dataset/test_eval_adaptive_turns.py`
- `eval/v36_candidate_selection/step60_vs_step77_deep_analysis.md`
- `eval/v36_candidate_selection/v36_checkpoint_selection.md`

`eval_stepcount.py` 保留了已完成全量评测验证的核心生成路径：Qwen2.5-VL processor/model、逐轮标点、首个完整 closing tag 双重停止、严格 answer、GT+3、无进度轮继续、LPT worker cost balancing、原子结果汇总和严格 resume fingerprint。移植测试移除了源测试对 pass16 脚本的无关依赖。

launcher 不再运行时修改 Python 源码，也不拼接本机默认路径；全部运行参数通过固定 CLI 进入 evaluator。

## 固定评测协议

唯一正式协议是 `strict_oracle_gt_plus_v3`：

| 项目 | 固定值 |
|---|---|
| dtype | BF16 |
| decoding | greedy，`do_sample=False`、`num_beams=1`、单返回序列；覆盖 checkpoint 中可能改变 mode 的配置 |
| sample turn cap | `min(task cap, GT+3)` |
| final answer | 必须有显式闭合 `<answer>整数</answer>` |
| fallback | 禁止 point-count fallback |
| point stop | 第一个 `</point>` 后立即下一轮 |
| answer stop | 第一个 `</answer>` 后结束 sample |
| history | 0 |
| intermediate images | 默认不保留；样本结束清理 |

task cap 与顺序：

| 顺序 | suite | 样本数 | GT 范围 | cap |
|---:|---|---:|---:|---:|
| 1 | pixmo-test | 529 | 2--10 | 13 |
| 2 | stepcount-500 | 500 | 11--50 | 53 |
| 3 | countqa | 491 | 2--10 | 13 |
| 4 | bias | 1992 | 2--51 | 54 |

模型顺序固定为 step77 后 step60。每个 suite 使用全部可见 GPU，完成并汇总后才进入下一 suite；默认期望 8 GPU，每卡 worker 可选 1/2/3。

## 可移植路径接口

模型：

- `V36_STEP77_MODEL` / `--step77-model`
- `V36_STEP60_MODEL` / `--step60-model`

数据：

- `V36_PIXMO_JSON`
- `V36_STEPCOUNT500_JSON`
- `V36_COUNTQA_JSON`
- `V36_BIAS_JSON`
- CLI 统一使用可重复的 `--dataset SUITE=JSON`

图片根目录：全局 `V36_IMAGE_ROOT`，或四个 `V36_*_IMAGE_ROOT`；CLI 使用 `--image-root SUITE=DIR`。相对路径只相对显式 image root 或 JSON 所在目录解析，并按 realpath 拒绝通过 `..` 或 symlink 逃逸该根目录。

绝对路径迁移使用 `--image-remap FROM=TO` 或 `V36_IMAGE_REMAPS`。FROM/TO 都必须为绝对路径；实现使用路径组件边界匹配和最长前缀优先，拒绝 `/` FROM 与重复冲突，不做 basename fallback。正式 preflight 遍历全部样本并验证 remap 后图片存在。

输出根使用 `V36_EVAL_OUTPUT_ROOT` / `--output-root`；GPU 与 worker 使用 `V36_EVAL_NUM_GPUS`、`V36_EVAL_WORKERS_PER_GPU` 或对应 CLI。CLI 高于 env。

## 结果验证与选模

分析器重新读取四个数据 JSON，验证：

- 权威样本数、ID 集合与顺序、dataset SHA256、sample ID SHA256；
- 每 row 的 GT、`min(cap, GT+3)`、BF16、greedy、history=0；
- explicit-answer 配置与实际 model response、no fallback、首 closing tag 后无尾随输出、终止原因；
- model label/identity、重新计算的当前 evaluator code hash、row/manifest suite config fingerprint；
- 同一 checkpoint 四套 model identity 与代码 hash 一致；
- paired flips、exact McNemar、dense/broad 50/50 composite。

权威完整评测分数：

| checkpoint | pixmo-test | stepcount-500 | countqa | bias | 50/50 composite |
|---|---:|---:|---:|---:|---:|
| step77 | 433/529 (81.85%) | 70/500 (14.00%) | 390/491 (79.43%) | 314/1992 (15.76%) | 36.51% |
| step60 | 426/529 (80.53%) | 76/500 (15.20%) | 398/491 (81.06%) | 280/1992 (14.06%) | 36.87% |

因此 model card 将 step60 标为主 checkpoint；step77 标为 secondary、偏 bias。step77 的 step68/75/76/77 更新被保护逻辑跳过，因此描述为 post-step74 actor/weights，但明确说明这不是逐 tensor checksum 结论。

## HF 仓库与安全策略

- `SI-Lab/StepCount-7B-v36-focused10k-step60`
- `SI-Lab/StepCount-7B-v36-focused10k-step77`

`hf_release.py` 没有 token CLI 参数，只读取 `HF_TOKEN` 环境变量。dry-run 不读取 token、不访问目录/网络、不落盘。正式上传前检查：

1. `config.json`、`preprocessor_config.json`、`model.safetensors.index.json`；
2. index 的 shard 路径安全、文件齐全、每个 tensor 的 key-to-shard 位置精确一致，且不存在额外未索引 safetensors；
3. 所有 tensor dtype 恰为 BF16；
4. 在独立 staging 中生成完整文件清单、size 与 SHA256；只 hardlink 不改写的 `.safetensors`，JSON/README 独立复制，不改写 source checkpoint；
5. 安装已审阅的中文 model card，并清除 config 中的内部 `_name_or_path`；
6. 使用 `HfApi.upload_large_folder`；默认禁用 Xet 且单 worker 串行 LFS，以限制发布端 CPU RAM，内存充足时可插拔启用 `--enable-xet`；
7. 上传后重新列远端文件，逐文件核对 size，并核验大文件 Git LFS SHA256。

仓库默认 private，只有显式 `--public` 才请求 public。token 不进入命令行、model card、manifest 或日志。

## 验证命令

```bash
bash -n examples/stepcount_eval/run_v36_eval.sh
python3 -m compileall -q examples/stepcount_eval
python3 -m unittest discover -s examples/stepcount_eval/tests -v
python3 examples/stepcount_eval/run_v36_eval.py --dry-run --workers-per-gpu 3
python3 examples/stepcount_eval/hf_release.py \
  --step 60 --source-dir /REQUIRED/model/step60 \
  --staging-dir /REQUIRED/release/step60 --dry-run
python3 examples/stepcount_eval/hf_release.py \
  --step 77 --source-dir /REQUIRED/model/step77 \
  --staging-dir /REQUIRED/release/step77 --dry-run
```

目标集群正式运行前，再按 `examples/stepcount_eval/README.md` 设置真实模型、JSON、图片映射和输出路径。权威结果使用 `transformers==4.57.3`；tokenizer regex/chat template 变化会改变协议。eval preflight 兼容标准 HF cache snapshot 的 immutable blob shard symlink，发布源/staging 仍严格拒绝 symlink。当前开发环境验证静态检查、30 项无 GPU unit tests、离线 dry-run，以及既有 7,024 条正式结果的独立重算；preflight 会逐张 GPU 检查 BF16，但新目标集群的 flash-attn/CUDA ABI 仍需实际加载确认。

## 发布完成与清理

2026-07-17 对两个 private HF repo 做了固定 revision 二次校验：每个 repo 的 19 个 release 文件 size 全匹配，5 个 LFS 对象 SHA256 全匹配，其余 14 个文件按固定 revision 下载后 SHA256 全匹配。HF 自动维护的 `.gitattributes` 是唯一不属于 release manifest 的远端文件。

| checkpoint | 角色 | HF repo | 固定 revision | release bytes | source full manifest SHA256 |
|---|---|---|---|---:|---|
| step60 | primary | [`SI-Lab/StepCount-7B-v36-focused10k-step60`](https://huggingface.co/SI-Lab/StepCount-7B-v36-focused10k-step60) | `ed9c4ef693ad3378cd5876f42d2133a26af5f7de` | 16,600,362,092 | `496696f2050697043606abfb259cab8891d987f61be3540ed79350fbefb452fa` |
| step77 | secondary | [`SI-Lab/StepCount-7B-v36-focused10k-step77`](https://huggingface.co/SI-Lab/StepCount-7B-v36-focused10k-step77) | `b93c9e67e9ee2b32e135c9bca800c8de691b8fce` | 16,600,362,129 | `0c6aa26986d6bfeba2892a0e96e9d333481cf6a552305a34ad5bfa004113d3de` |

上传完成后再次重算两个 source checkpoint 的全部 16 个文件，full manifest 与发布前一致。未选中的 `global_step_70` 已删除，释放约 92.7 GiB；本地只保留 `global_step_60` 与 `global_step_77`。eval/release bundle 与本文位于 EasyR1 `hy-0703` 分支。
