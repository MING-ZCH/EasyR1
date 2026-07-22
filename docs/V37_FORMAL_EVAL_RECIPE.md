# V37 Formal Eval Recipe 与 Evidence 契约

- 日期：2026-07-22
- producer：`tools/v37_eval_producer.py`
- postprocess：`tools/v37_postprocess.py`
- gate：`tools/v37_gate.py`
- 当前状态：契约实现已落盘，但训练环境 E2E、production evaluator/recipe 和真实 8×H200 eval 尚未闭环，因此仍为 **NO-GO**。

## 1. 为什么需要 recipe

formal eval 不能在训练完成后临时更换脚本、参数或模型。`ab_plan.json` 在训练前封印 producer、recipe、held-out sample universe 和 implementation SHA256；训练后 postprocess 为每个 cell 生成唯一 nonce，并现场执行同一 recipe。已有 eval JSON 不能直接用于 formal promotion。

postprocess 会从稳定 inode 读取 preregistered producer/recipe，校验 SHA256 后写入 cell 私有 execution snapshot；producer 对每个 evaluator 也执行相同的 stable-read/hash/snapshot 流程。即使原 producer/recipe/evaluator 路径在 snapshot 后被替换，本次命令仍执行已验证 bytes；原路径或 checkpoint 后续漂移继续 fail closed。该机制不对抗拥有全部 run/staging/解释器写权限的同 UID 主体，也不冻结 `PATH/PYTHONPATH/LD_LIBRARY_PATH` 指向内容。

绑定链如下：

```text
sample universe + A/B preregistration
  -> newly generated nonce unique within this four-cell gate input
  -> frozen producer/recipe/program/rendered argv
  -> producer receipt
  -> paired/benchmark output SHA256
  -> gate recomputation
```

## 2. 先生成 Sample Universe

```bash
python3 tools/v37_paired_universe.py \
  --data /abs/path/heldout.parquet \
  --output /abs/path/v37-paired-sample-universe.json
```

输入必须有唯一、非空字符串 `sample_id`/`prompt_id` 和严格 integer `answer`，answer 范围为 2..50，并覆盖五个 bucket。输出冻结完整样本全集、GT、bucket、source-row hash、dataset snapshot 和 ordered-row hash。formal eval 不允许自行改 GT/bucket、抽取子集或补样本。

## 3. Recipe Schema

下面只是 schema 模板，不是可直接运行的 production recipe。`path` 和 SHA256 必须替换为训练集群上真实、可执行且内容冻结的 evaluator。

```json
{
  "schema_version": 1,
  "recipe_type": "v37_formal_eval_recipe",
  "environment": {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "WANDB_MODE": "offline",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONHASHSEED": "{seed}"
  },
  "steps": {
    "paired_eval": {
      "arms": ["baseline", "progress"],
      "program": {"path": "/abs/path/paired_eval", "sha256": "<64-lowercase-hex>"},
      "argv": [
        "--manifest", "{manifest}", "--checkpoint", "{checkpoint}",
        "--model", "{eval_model}", "--nonce", "{invocation_nonce}",
        "--output", "{output}"
      ],
      "output": "paired_eval.json"
    },
    "pixmo_benchmark": {
      "arms": ["progress"],
      "program": {"path": "/abs/path/pixmo_eval", "sha256": "<64-lowercase-hex>"},
      "argv": [
        "--manifest", "{manifest}", "--checkpoint", "{checkpoint}",
        "--model", "{eval_model}", "--nonce", "{invocation_nonce}",
        "--output", "{output}"
      ],
      "output": "pixmo_benchmark_evidence.json"
    },
    "stepcount_benchmark": {
      "arms": ["progress"],
      "program": {"path": "/abs/path/stepcount_eval", "sha256": "<64-lowercase-hex>"},
      "argv": [
        "--manifest", "{manifest}", "--checkpoint", "{checkpoint}",
        "--model", "{eval_model}", "--nonce", "{invocation_nonce}",
        "--output", "{output}"
      ],
      "output": "stepcount_benchmark_evidence.json"
    }
  }
}
```

每个 step 的 argv 必须各出现一次 `{manifest}`、`{checkpoint}`、`{eval_model}`、`{output}`、`{invocation_nonce}`。禁止 shell expansion；producer 使用 argv list 和精确 recipe environment。允许的可选 environment key 仅为代码中的 `OPTIONAL_ENVIRONMENT_KEYS`，敏感 key 和未知 key fail closed。

## 4. Paired Evidence v2

formal paired artifact 顶层必须包含：

- `schema_version=2`、`artifact_type=v37_paired_eval_evidence`；
- producer nonce、sample universe SHA256；
- validation data SHA256；
- `evidence_run_id`、execution environment SHA256；
- final checkpoint ID/path/tree SHA256；
- 完整 `samples` 和 sample-set SHA256。

每个 sample 只能提供 evaluator 原始事实和一个可校验 cache。下面是与 `GT=2` universe row 对应的结构示例；`source_row_sha256` 仍须替换成该真实 universe row 的 hash，因此这段 JSON 不是可直接提交的 artifact：

```json
{
  "sample_id": "...",
  "prompt_id": "...",
  "source_row_sha256": "<universe row hash>",
  "predicted_answer": 2,
  "transcript": [
    "<think>inspect</think><point>{\"point_2d\":[512,384],\"label\":\"object\",\"count_number\":1}</point>",
    "<think>inspect</think><point>{\"point_2d\":[640,384],\"label\":\"object\",\"count_number\":2}</point>",
    "<think>done</think><answer>2</answer>"
  ],
  "point_events": [
    {
      "point_index": 1,
      "point_payload_sha256": "becc6bea8c0359117ae1ff2dde4fb2bba5e5f97159db79c4c50482594a21fbb1",
      "matched_target_id": "mask-instance-1"
    },
    {
      "point_index": 2,
      "point_payload_sha256": "575a49db9b4e2fbdfd5d5bb726cf612f0979e72af2f973b9e9588e5b2a97ccf4",
      "matched_target_id": "mask-instance-2"
    }
  ],
  "termination_reason": "answer",
  "num_rounds": 3,
  "configured_max_turns": 53,
  "effective_max_turns": 5,
  "metrics": {
    "answer_exact": 1.0,
    "format_compliance": 1.0,
    "unique_valid_hit": 1.0,
    "duplicate": 0.0,
    "cap_turn_exceeded": 0.0,
    "early_stop": 0.0
  }
}
```

GT 和 bucket 不允许出现在 sample row，它们来自 universe。gate 从 transcript 重取 strict integer answer、tag 顺序、rounds 和 termination；从 `<point>` JSON 原文解析 `point_2d`/`count_number`，核对 payload hash，再根据 evaluator 输出的 `matched_target_id` ledger 重算 unique/duplicate。cache 任一值不一致即失败。

这里的信任边界必须明确：gate 会验证 transcript、point payload、ledger 与聚合 cache 的内部一致性，但不会重新加载 mask 做几何命中；`matched_target_id` 的真实性依赖 preregistered evaluator/mask matcher 代码和人工审查。producer receipt 绑定 program/argv/output bytes，但不能独立证明程序真的加载指定 checkpoint、使用 8 张 GPU、执行模型推理或没有构造 transcript；它不是 sandbox、外部签名或第三方 attestation。

`kl_recoverability` 与 `progress_degraded` 不属于 sample-level paired metrics：前者由 checkpoint/controller evidence hard gate，后者必须为零，否则整 run 直接失败。

## 5. Benchmark Evidence

两个 progress cell 还必须产出 `v37_benchmark_evidence.py` schema v2 descriptor：

- BF16 greedy，beam/return=1；
- adaptive max turn 为 `min(task_cap, GT+3)`；
- strict explicit integer answer，禁止 point-count fallback；
- pixmo-test 固定 529 samples，stepcount-500 固定 500 samples；
- dataset、ordered IDs、images、checkpoint、HF model、evaluator/validator/prompt/requirements 全部绑定。

baseline cell 不运行 benchmark，避免无必要的 GPU 开销；paired held-out 同时运行 baseline/progress，用于提供有限的 paired mechanism-effect 证据，但两个 seed 和固定执行顺序本身不构成稳定因果或泛化结论。

## 6. 启动与审查

```bash
export V37_AB_PAIRED_UNIVERSE=/abs/path/v37-paired-sample-universe.json
export V37_AB_EVAL_RECIPE=/abs/path/v37-formal-eval-recipe.json

V37_AB_ROOT=/new/abs/path/v37-formal-ab \
V37_AB_SEEDS=11,22 \
V37_AB_POSTPROCESS=1 \
bash examples/rl_launch/run_v37_strict_ab.sh
```

正式 A/B launcher 在 preregistration 时自动调用 `verify_recipe()`；schema、program executable/hash、8-GPU offline environment 或 argv 不合法会在首个 training child 前失败。启动前还必须执行 `bash -n` 和主 plan“验证记录”中的 dependency-light tests。最终仍需人工核对 evaluator 是否真的以 BF16、greedy、GT+3、首个完整 action stop 运行，以及 mask matcher 的 `matched_target_id` 是否来自真实 mask 命中而非答案或累计点数推断。

本地 SHA/receipt 能发现漂移、误拼接和旧产物复用，但不能对抗拥有全部文件写权限且能同时重写 plan/artifacts 的主体。recipe environment 只冻结变量字符串，不冻结 `PATH/PYTHONPATH/LD_LIBRARY_PATH` 指向的文件、Python/native package bytes、CUDA driver/container、cache 内容或网络/文件系统权限；`HF_*_OFFLINE=1` 也不是 sandbox。相同输入的 deterministic gate replay 是允许的；gate 的 `GO` 是 artifact-contract GO，不是外部性能晋级 attestation。“只能 promotion 一次”或跨机器 freshness 由外部登记负责。需要外部来源认证时，应把 `ab_plan.json`/hash 存到独立只读位置或增加签名/远程 attestation。

所有 manifest、checkpoint、recipe、receipt 和 staging 绑定均使用绝对路径，因此 evidence 默认 location-bound。复制或改写路径后不能继续通过原 gate；`STEPCOUNT_IMAGE_PATH_REMAP_JSON` 只处理输入图片根目录，不迁移这些 provenance。若完成后的 formal run 无法在新集群保持相同绝对挂载路径，必须重新 preregister 并重跑 formal training/eval，而不是只重新 produce evidence；执行 gate 的目标主机还必须具有相同 live Git identity 与 implementation bytes。Git 身份命令会清除 ambient `GIT_*` repository/index/object/diff override，A/B child 同样不继承这些变量。`V37_AB_ROOT` 和 evidence root 必须位于 Git worktree 外。sample-universe 文件与 producer step-output promotion 依赖同一 filesystem 支持 POSIX hard link；receipt 使用 `O_EXCL`，plan/postprocess staging 使用同 filesystem atomic replace。不支持对应语义的 FUSE/对象存储挂载会 fail closed。
