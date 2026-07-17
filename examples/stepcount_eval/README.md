# V36 StepCount 可移植评测与发布 bundle

本目录用于在另一训练集群复现 V36 `step77 -> step60` 的四套严格评测，并在人工确认后发布到 Hugging Face。它不依赖原评测仓库的 pass16 文件，也不包含模型、数据、图片或 token。

正式协议只有一个：`strict_oracle_gt_plus_v3`。

- BF16、greedy（`do_sample=False`、`num_beams=1`、单返回序列）；不提供 sampling/fp16 的正式入口，并覆盖 checkpoint 中可能改变 decoding mode 的 generation 配置。
- 每个样本 `effective_max_turn=min(task cap, GT+3)`。
- 只接受显式闭合且内容为整数的 `<answer>...</answer>`；禁止累计 point-count fallback。
- 解码遇第一个 `</point>` 立即截断并进入下一轮；遇第一个 `</answer>` 结束样本。
- `history=0`，默认不保留中间图片。
- task cap：pixmo-test=13、stepcount-500=53、countqa=13、bias=54。

## 目录

- `eval_stepcount.py`：从已完成四套评测的核心 evaluator 移植，保留逐轮画点、首标签停止、GT+3、LPT worker 调度、断点指纹和原子汇总语义。
- `run_v36_eval.py` / `run_v36_eval.sh`：固定模型和 suite 顺序的串行 launcher。
- `preflight.py`：检查数据集 ID/数量/GT 范围、重映射后图片、模型 index/shard/BF16、逐 GPU BF16 支持和并发显存保守估计。
- `analyze_results.py`：验证结果证据并输出 JSON/Markdown paired checkpoint 报告。
- `hf_release.py`：本地发布校验、model card 安装、低内存 LFS/可选 Xet 上传和远端 size 对账。
- `model_cards/`：step60/step77 中文 model card。
- `tests/`：不加载模型/GPU 的单元测试。

## 1. 安装

建议复用训练集群已经验证的 EasyR1 环境，再只补齐本目录的直接依赖：

```bash
python3 -m pip install -r examples/stepcount_eval/requirements-eval.txt
```

`flash-attn` 需要与集群 CUDA/PyTorch ABI 匹配；不要在已有训练环境中盲目升级 torch。当前 evaluator 使用 Qwen2.5-VL 和 `flash_attention_2`。

## 2. 先运行无 GPU 检查

```bash
bash -n examples/stepcount_eval/run_v36_eval.sh
python3 -m compileall -q examples/stepcount_eval
python3 -m unittest discover -s examples/stepcount_eval/tests -v
python3 examples/stepcount_eval/run_v36_eval.py --dry-run --workers-per-gpu 3
python3 examples/stepcount_eval/hf_release.py \
  --step 60 \
  --source-dir /REQUIRED/model/step60 \
  --staging-dir /REQUIRED/release/step60 \
  --dry-run
```

dry-run 不访问给定路径，不加载 torch/model/GPU，不读取 `HF_TOKEN`，也不访问网络或写发布目录。

权威分数使用 `transformers==4.57.3` 复现。不要在对比实验中静默启用 tokenizer regex 修复或更换 chat template；这会改变 tokenization/生成协议，而不是单纯的性能优化。

## 3. 配置另一集群路径

CLI 优先级高于环境变量。下面的 `/cluster/...` 只是示例占位符：

```bash
export V36_STEP77_MODEL=/cluster/models/v36-step77/huggingface
export V36_STEP60_MODEL=/cluster/models/v36-step60/huggingface

export V36_PIXMO_JSON=/cluster/eval/eval_dataset.json
export V36_STEPCOUNT500_JSON=/cluster/eval/eval_stepcount_bench_500.json
export V36_COUNTQA_JSON=/cluster/eval/eval_dataset_countbench.json
export V36_BIAS_JSON=/cluster/eval/eval_dataset_bias.json

export V36_EVAL_OUTPUT_ROOT=/cluster/outputs/v36-strict
export V36_EVAL_NUM_GPUS=8
export V36_EVAL_WORKERS_PER_GPU=1
```

每卡 worker 只允许 `1/2/3`。先从 1 开始；preflight 会用“模型权重 + 每 worker 24 GiB”做保守估计，但实际峰值仍受图片尺寸、driver、allocator 和 flash-attn 版本影响。

### 图片路径规则

相对 `image_path` 只按以下确定性规则解析：

1. 设置对应 suite 的 image root 时，解析为 `image_root/image_path`；
2. 否则解析为 `dataset JSON 所在目录/image_path`。

可用 `V36_IMAGE_ROOT` 设置四套共用根，或分别设置：

```bash
export V36_PIXMO_IMAGE_ROOT=/cluster/images/pixmo
export V36_STEPCOUNT500_IMAGE_ROOT=/cluster/images/stepcount
export V36_COUNTQA_IMAGE_ROOT=/cluster/images/countqa
export V36_BIAS_IMAGE_ROOT=/cluster/images/bias
```

JSON 中已有绝对路径时，使用边界感知 prefix remap。多个映射在环境变量中以 `;;` 分隔，最长前缀优先：

```bash
export V36_IMAGE_REMAPS='/old/datasets/pixmo=/cluster/images/pixmo;;/old/datasets/countqa=/cluster/images/countqa'
```

也可重复传 `--image-remap FROM=TO`。FROM/TO 都必须是绝对路径。映射不会用 basename 搜索或猜路径；未匹配绝对路径保持原样，随后由 preflight 明确报缺失。禁止把 `/` 作为 FROM，重复冲突映射会失败；相对 `image_path` 使用 `..` 或 symlink 逃逸配置根目录也会被拒绝。

## 4. 一键正式评测

```bash
bash examples/stepcount_eval/run_v36_eval.sh
```

执行顺序不可配置：

1. preflight；
2. step77：pixmo-test → stepcount-500 → countqa → bias；
3. step60：pixmo-test → stepcount-500 → countqa → bias；
4. 严格验证并生成对比报告。

每个 suite 都由 evaluator 使用全部可见 GPU；只有该 suite 全部 worker 成功、结果完整并原子落盘后，launcher 才进入下一项。已有 `results.json` 或可恢复的 worker part 默认 fail-fast，不会静默删除；确需续跑时显式加 `--resume`，且已有 row 的配置指纹必须完全一致。工作图片目录必须是 output root 的真实直属子目录且不能是 symlink。

CLI 示例：

```bash
bash examples/stepcount_eval/run_v36_eval.sh \
  --step77-model /cluster/models/step77 \
  --step60-model /cluster/models/step60 \
  --dataset pixmo-test=/cluster/eval/pixmo.json \
  --dataset stepcount-500=/cluster/eval/dense500.json \
  --dataset countqa=/cluster/eval/countqa.json \
  --dataset bias=/cluster/eval/bias.json \
  --output-root /cluster/outputs/v36-strict \
  --num-gpus 8 --workers-per-gpu 2
```

输出布局为：

```text
OUTPUT_ROOT/
├── step77/{pixmo-test,stepcount-500,countqa,bias}/
│   ├── results.json
│   ├── run_manifest.json
│   ├── results.txt
│   └── eval_time.log
├── step60/{pixmo-test,stepcount-500,countqa,bias}/...
├── v36_checkpoint_comparison.json
└── v36_checkpoint_comparison.md
```

`run_manifest.json` 和每条 row 共同记录 dataset SHA256/ID、sample ID hash、BF16、greedy、GT+3、explicit answer、no fallback、模型 identity、evaluator code hash 与配置指纹。分析器会重新读取数据集、重新计算当前 evaluator code hash，并逐项验证 row/manifest 指纹、首 closing tag 截断、终止原因和实际 answer 文本，不能只凭文件名或自洽元数据认结果。

## 5. 单独重跑分析

```bash
python3 examples/stepcount_eval/analyze_results.py \
  --results-root "$V36_EVAL_OUTPUT_ROOT" \
  --dataset pixmo-test="$V36_PIXMO_JSON" \
  --dataset stepcount-500="$V36_STEPCOUNT500_JSON" \
  --dataset countqa="$V36_COUNTQA_JSON" \
  --dataset bias="$V36_BIAS_JSON"
```

分析器要求同一 checkpoint 的四套 `model_identity` 与 evaluator code hash 一致，并做样本级 paired flips / exact McNemar 统计。它不硬编码任何 `model_run`。

## 6. Hugging Face 发布

默认仓库：

- step60：`SI-Lab/StepCount-7B-v36-focused10k-step60`
- step77：`SI-Lab/StepCount-7B-v36-focused10k-step77`

先看计划：

```bash
python3 examples/stepcount_eval/hf_release.py \
  --step 60 \
  --source-dir /cluster/models/step60 \
  --staging-dir /cluster/release/step60 \
  --dry-run
```

`source-dir` 是只读 merged checkpoint，`staging-dir` 必须不存在或为空且不能与 source 互相包含。发布源与 staging 拒绝 symlink；eval preflight 则允许标准 HF cache snapshot 中指向 immutable blob 的 shard symlink，避免混淆两类信任边界。工具只在 staging 写入经过审阅的 model card、规范化 config 与 manifest；只有不可变的 `.safetensors` 大 shard 会在同文件系统优先 hardlink，JSON/README 始终独立复制，source metadata 不会被改写。

正式上传前，工具验证 safetensors index、每个 tensor 的精确 key-to-shard 位置、无额外未索引 safetensors、全部 tensor 为 BF16，并生成含 SHA256/size 的 `release_manifest.json`。随后调用 `huggingface_hub.HfApi.upload_large_folder`，最后逐文件比较远端 size，并核验所有大文件的 Git LFS SHA256。仓库默认 private；只有显式 `--public` 才请求 public visibility。

默认使用 `HF_HUB_DISABLE_XET=1`、`--num-workers 1` 的低内存 LFS 模式，避免多个 multi-GB shard 同时 chunk 导致控制节点 RAM OOM。内存充足且已验证 Xet 的集群可显式传 `--enable-xet --num-workers 2`；该选项只改变传输实现，不改变模型文件或 eval 分数。

token 只允许来自进程环境，没有 `--token` 参数，也不会写入 manifest/model card：

```bash
read -rsp 'HF token: ' HF_TOKEN; echo
export HF_TOKEN
python3 examples/stepcount_eval/hf_release.py \
  --step 60 \
  --source-dir /cluster/models/step60 \
  --staging-dir /cluster/release/step60
unset HF_TOKEN
```

自动化测试只走离线 dry-run，不读取凭据或访问远端；实际发布必须以 `upload_receipt.json` 和固定 HF revision 为准。

## 已知限制

- 正式分数使用 oracle GT 决定 turn cap，不等价于未知 GT 的真实部署策略。
- evaluator 保留权威实现的临时逐轮图片生成语义；`KEEP_EVAL_IMAGES=False` 只在样本结束后清理，因此磁盘需容纳并发中的临时图。
- model identity 是配置、index、shard 大小/mtime 与头尾内容形成的本地运行指纹；HF 发布阶段另用完整文件 SHA256 与远端 size 校验。
- preflight 的 worker 显存估计是保守近似，不能替代目标集群上的小批量实测。
