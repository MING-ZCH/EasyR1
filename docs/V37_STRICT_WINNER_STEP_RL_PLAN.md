# V37 Frontier RL / Strict Winner RFT 实施与实验计划

- 日期：2026-07-22
- launcher：`examples/v37_strict_winner_step_rl_pilot.sh`
- 当前结论：训练机制、四 cell A/B preregistration、可重算 benchmark evidence 与 fail-closed gate 的实现及 dependency-light 回归已完成；尚未完成训练 conda/CUDA E2E、V37 canary/formal 或新 eval，因此整体与性能晋级均为 **NO-GO**。

## 0. 分步实施记录

| 标记 | 修改 | 可插拔/回滚边界 | 当前验证 |
|---|---|---|---|
| V37-01 | correctness-first `answer_correct/raw_success` 与 homogeneous terminal zero | 仅 `BOK_CORRECTNESS_FIRST/BOK_ALLWRONG_TERMINAL_ZERO` | 对抗输入与随机 BoK parity |
| V37-02 | token-native `native_action_ledger_v2` 与局部 point credit | 两臂 `ACTION_EVENT_LEDGER_ENABLE=1`；仅 progress `ACTION_EVENT_REWARD_ENABLE=1` | ledger mutation、span/tail 隔离 probe |
| V37-03 | response-token global-mean adaptive actor KL | `algorithm.adaptive_actor_kl=true` | 数学/符号/controller 静态与 probe |
| V37-04 | adaptive state 的 manifest-first checkpoint 发布与恢复 | adaptive 模式专用；legacy save 路径保留 | incomplete save、idempotent publish 测试 |
| V37-05 | 严格 integer answer winner 路由 | V37 wrapper 强制 1；V36 默认 0 | ambiguous/valid integer 对抗 probe |
| V37-06 | torch fallback sign fail-closed | V37=`error`；V36=`legacy` | resolver 合约与 launcher 测试 |
| V37-07 | 双 seed M32 frontier、真实 loader preflight | 独立 builder/preflight 工具 | synthetic parquet 的真实 builder→launcher→RLHFDataset preflight E2E 已通过；生产数据待运行 |
| V37-08 | formal artifact/config/environment/checkpoint gate | 仅 promotion gate，不进训练热路径 | gate 正/负向专项 tests |
| V37-09 | action metadata 一次 CPU snapshot，移除逐 event CUDA scalar sync | 仅 native action 分支 | 数学等价/no-tail probe；CUDA profiler 待 canary |
| V37-10 | A/B launcher、文档和 fail-closed contract | `V37_CONTRACT_ONLY/PREFLIGHT_ONLY/GATE_ONLY` | launcher contract tests 与 dependency-light 真实 preflight E2E；训练 conda/CUDA 待复核 |
| V37-11 | strict structural winner gate：duplicate 或不完整格式不能进入 `raw_success` | `V37_RAW_SUCCESS_STRICT_WINNER=1`；V36 默认 0 | duplicate/typo/flag-off 对抗 probe |
| V37-12 | LR scheduler 与实际 optimizer update 对齐 | 仅 adaptive actor KL 分支；legacy 保持逐 RPC 推进 | scheduler contract tests |
| V37-13 | 外部图片、runtime package 与显式 execution environment snapshot | 仅 formal promotion evidence | preflight mutation/symlink/环境污染负向测试 |
| V37-14 | 原子 A/B plan、两 seed 四 cell 串行 launcher 与 evidence index | `run_v37_strict_ab.sh` 可独立 plan/contract-only | A/B plan、串行顺序与 artifact contract tests |
| V37-15 | schema v2 benchmark evidence：逐 sample 重算并绑定 dataset/image/code/model | 仅 final gate；旧 schema v1 明确拒绝 | descriptor/gate 集成与篡改负向 tests |
| V37-16 | canonical tree hash v2、路径组件/internal symlink 与精确 publication entry 校验 | 仅 provenance；不进入训练数值路径 | 有效/悬空 symlink、额外目录负向测试 |
| V37-17 | trainer-owned schema v2 training evidence、绝对 global step continuation 与 rank-agreed nonfinite counter | `V37_TRAINING_EVIDENCE_REQUIRED=1`；V36 不强制 | optimizer 发布前门禁、continuation identity/tamper tests |
| V37-18 | formal 终点 checkpoint 在隐藏 staging 内 merge BF16 HF 权重后再写 manifest/publish | `V37_FINALIZE_HF_CHECKPOINT=1`；仅 formal 终点 | rank/schema/safetensors-header contract tests；8×H200 实跑待验证 |
| V37-19 | 训练前 paired sample universe：冻结 held-out 全集、GT、bucket、source-row/data hash | formal 必填 `V37_AB_PAIRED_UNIVERSE` | 缺样本、换 GT/source、artifact 漂移均 NO-GO |
| V37-20 | preregistered eval recipe、invocation challenge/receipt、完整 rendered argv 与输出 hash | formal postprocess 强制现场 producer；禁止直接投喂旧 artifact，计算 freshness 仍需外部 attestation | producer/recipe/evaluator verified execution snapshot 与原路径替换对抗测试 |
| V37-21 | formal paired evidence 从 transcript 和 point JSON 重算，禁止 eval 自报 GT/分数 | canary 保留轻量诊断格式；formal 强制 schema v2 | metric cache、subset、count/tag/receipt 篡改测试 |
| V37-22 | `runtime_mask_coverage` 只证明 mask route 可用，少点/多点/坏 payload 留在模型质量指标 | 仅 V37 strict mask evidence；V36 默认语义不变 | point count/payload 对抗测试；formal coverage 不再要求模型输出完美 |
| V37-23 | Git 身份采集移除 ambient `GIT_*` repository/diff override | formal launcher/gate/continuation 与 A/B child sanitation | 伪造 `GIT_DIR/GIT_WORK_TREE` 的真实工作树回归通过 |
| V37-24 | V36/V37 Ray semantic env 隔离，并让 strict point parser 成为两臂 remote 必填 | `ACTION_EVENT_REWARD_ENABLE` 仅显式 V37 run 可转发；V36 继续 feature-off | stray action/V37 marker 与丢失 parser 负向测试 |
| V37-25 | checkpoint-476 固定 content identity 与可恢复 validation OOM 累计 telemetry | formal plan/manifest/effective env/gate 强制；debug/canary 仍不可晋级 | 19-file model hash、OOM monotonic/reseal/NO-GO 测试 |

表中“验证”只代表机制/合约证据；性能收益必须由 paired canary/formal A/B 证明。

## 1. 已实现边界

本轮以 V37 wrapper 和显式 feature flag 为入口，增量修改了 core、reward、rollout、checkpoint、frontier/preflight/gate 与测试；未把 V37 训练目标默认开启到 V36 launcher。以下实现已落盘，但真实训练闭环仍以 canary/formal 结果为准：

- 显式 `V37_RUN_CLASS=debug|canary|formal`；
- 显式 `V37_DATA_MODE=frontier_rl|strict_winner_rft`；
- baseline/progress 两臂、correctness-first、native action-event、adaptive actor KL；
- `build_v37_frontier_dataset.py` selection manifest、`preflight_v37_training.py` 和 `v37_gate.py`；
- run manifest 的数据、配置、代码、CP 和 promotability 审计字段；
- 不依赖训练资产的 `V37_CONTRACT_ONLY=1` launcher 契约测试。
- V37-only `TRAJ_STRICT_ANSWER_INTEGER_PARSE=1`，含糊 answer 文本不能进入 winner；
- V37-only `V37_RAW_SUCCESS_STRICT_WINNER=1`，duplicate 或 format 不完整的轨迹不能进入 strict winner，但仍保留答案正确性和部分 answer reward；
- `native_action_ledger_v2`、reward 侧 parser contract fail-closed；
- `TORCH_LOGPROB_FALLBACK_MODE=error`，V37 缺失 FlashAttention CE 时拒绝训练，V36 默认仍为 `legacy`；
- formal gate 对初始模型、训练/验证数据、metadata/mask、真实 loader preflight、完整 audited environment、实现文件和最终 checkpoint/eval 做实体交叉验证。canonical checkpoint-476 的 `content_snapshot` 固定为 `f9e6b1e8031bdbc509d34249745cdcf75af85c320d6b88918444b1abb4f580a3`（19 files，16,600,416,818 bytes）；formal plan 发布前、single wrapper 与 final gate 会在 driver 侧重算实际 snapshot，remote worker 只校验转发的 `V37_EXPECTED_INITIAL_MODEL_SHA256` 等于同一冻结常量，不能据此声称每个 worker 都重读了模型树。
- `run_v37_strict_ab.sh` 在训练前原子发布一个 `ab_plan.json`，按 seed-major 顺序串行运行 `seed11-baseline`、`seed11-progress`、`seed22-baseline`、`seed22-progress`；每个 manifest 必须反向绑定同一个 plan/hash/cell。
- formal downstream 使用 allowlist-filtered and sealed environment：wrapper 把允许的环境字符串写入 `v37_execution_environment.json` 后用该快照启动 V36；`BASH_ENV`、`ENV` 和 exported shell function 会失败。`PATH`、`PYTHONPATH`、`LD_LIBRARY_PATH` 等仍是字符串绑定，不等于其指向内容或容器镜像已被 attestation。
- `v37_benchmark_evidence.py` 不信任 `correct/total` 汇总，而是从 canonical dataset、source image hashes、完整 transcript 和 frozen protocol 重算每个 sample；final gate 还强制 eval model 为对应 checkpoint 的 `actor/huggingface`。

`frontier_rl` 是 prompt-only 在线 RL 数据。`strict_winner_rft` 是带 winner response 的 SFT/RFT 数据，二者不能互换。当前仓库没有经本 launcher 审计的独立 RFT 训练入口，所以 `V37_DATA_MODE=strict_winner_rft` 会立即失败；不会把 winner-only 数据传给在线 BoK-GRPO。

paired gate 只比较真正的 sample-level 指标：五桶 answer macro、`answer_exact`、`format_compliance`、`unique_valid_hit`、`duplicate`、`cap_turn_exceeded`、`early_stop`。`kl_recoverability` 是 run/checkpoint 状态，`progress_degraded` 是 evidence 完整性状态，二者继续作为 trainer-owned hard gate，不能由 evaluator 逐样本自报后参与 paired 比较。

## 2. 开关表

| 开关 | debug | canary | formal |
|---|---|---|---|
| `V37_RUN_CLASS` | 必填；永不可晋级 | 必填；2–4 effective updates | 必填；双 seed A/B 每臂 12 updates |
| formal 入口 | 不适用 | 不适用 | 必须由 `run_v37_strict_ab.sh` 生成并传入 `V37_AB_*` plan/cell 绑定 |
| `V37_DATA_MODE` | `frontier_rl` | `frontier_rl` | `frontier_rl` |
| `V37_ARM` | `baseline` / `progress` | 同左 | 同左 |
| `V37_FRONTIER_MANIFEST` | 使用 frontier 时可提供 | 必须，且 `run_class=canary` | 必须，且 `run_class=formal` |
| focused10k fallback | 仅显式 `V37_ALLOW_FOCUSED10K_PILOT=1` | 禁止 | 禁止 |
| benchmark 作训练 val | 仅显式 debug，run 永不可晋级 | 禁止 | 禁止 |
| `V37_METADATA_COVERAGE_REPORT` | 可选 | 可选，建议预演 formal | 必须且 hash/coverage=1.0 |
| `V37_FILTERED_MANIFEST` | 可选 | 可选 | 必须，与 post-filter 数据 hash/样本/桶一致 |
| `V37_CP_SIZE` | 默认 1；大于 1 失败 | 同左 | 同左 |
| `TRAJ_STRICT_ANSWER_INTEGER_PARSE` | 强制 1 | 强制 1 | 强制 1 |
| `V37_RAW_SUCCESS_STRICT_WINNER` | 强制 1 | 强制 1 | 强制 1 |
| `TORCH_LOGPROB_FALLBACK_MODE` | 强制 `error` | 强制 `error` | 强制 `error` |
| promotability | false | false，需后续 formal/final gate | 仅 candidate；仍须 final gate |

旧变量 `STEPCOUNT_V37_BALANCED_DATA` 仍作为 `V37_FRONTIER_DATA` 的兼容别名，但推荐新名称。V37 专用开关在 V36 入口默认关闭；共享 config/core、数据、Ray 与 checkpoint 路径已有改动，目前只验证了局部 feature-off contract，尚无 V36 端到端数值等价或 bit-for-bit 兼容结论。

## 3. Baseline / Progress A/B

两臂共享 checkpoint-476、frontier 数据、seed、严格 action-close scheduler、EOS/cap 终止语义、reward outcome、correctness-first 和 adaptive actor KL。两臂都固定 `ACTION_EVENT_LEDGER_ENABLE=1`，使用相同 token-bound parser/ledger 做 `cap/abort/answer` 判定并 fail closed；`ACTION_EVENT_REWARD_ENABLE` 只决定是否导出 action values 并进入 step reward，因此预期实验差异仅为 native action-level process credit：

| 机制 | baseline | progress |
|---|---|---|
| `ADV_ESTIMATOR` | `bok_grpo` | `bok_grpo_step` |
| `ACTION_EVENT_LEDGER_ENABLE` | 1（只读） | 1（只读） |
| `ACTION_EVENT_REWARD_ENABLE` | 0 | 1 |
| `PROCESS_REWARD_ENABLE` | 0 | 0 |
| `BOK_STEP_WEIGHT` | 0 | 0.1 |
| `BOK_STEP_GATE` / `BOK_STEP_MIN_GATE` | 不适用 | `answer_soft` / `0.2` |

`PROCESS_REWARD_ENABLE=0` 在两臂都固定，关闭旧的 decode 后搜索 `</point>` 并重定位 token 的 retokenized process 路径。两臂都校验 rollout 直接产生的 `action_event_ledger`，错误 span、缺行或非 sequential reward manager 都会失败；baseline 只记录 ledger diagnostics，不生成 `_action_event_values` 或 `action_step_*` 训练张量。

共同 outcome 语义固定为：

```text
BOK_CORRECTNESS_FIRST=1
BOK_ALLWRONG_TERMINAL_ZERO=1
BOK_WINNER_BOOST=0
```

mixed group 中 `raw_success=1` 的正确 winner 必须为正 advantage，错误轨迹必须为负；all-correct/all-wrong homogeneous group terminal advantage 为零。V37 的 `outcome_success` strict winner 同时要求 exact terminal integer answer、正常闭合、未超自适应 cap、point/count_number 结构完整且无 trajectory hard reject。mask miss 或 duplicate 不再抹掉正确 answer outcome；它们进入独立的 `trusted_trajectory`/process diagnostics 和局部 credit。精确答案但 capped、format/count_number 不完整或 invalid 的轨迹仍保留 `answer_correct=1` 和 shaped answer partial credit，但不能进入 correctness-first winner 集合。`V37_WINNER_MODE=legacy_all_hit` 保留旧行为，仅用于 feature-off 回归；V37 pilot 锁定 `outcome_success`。

V37 的 exact answer 只接受完整 signed integer，例如 `5`、`+5`、`005`；`5 or 6`、`5.0`、`5 objects` 不进入 `answer_correct/raw_success`。这是 winner 路由的严格化，不会把“答案确实正确但 point 较差”的 partial answer reward 清零。未经过 V37 wrapper 时仍使用 V36 的宽松 parser。

rollout 对每个 sample 使用 `min(INTERLEAVED_MAX_TURNS, GT_answer + 1 + 2)`，即 GT+3。native action 模式的请求同时以首个 `</point>` 和 `</answer>` 为 stop sequence；首个 `</answer>` 结束 sample，首个 `</point>` 进入下一轮。为允许小于 GT turn 的 early answer 在同一请求完整生成，point 请求上限使用 answer budget 8192；vLLM 按实际生成动态占用，但 malformed/unclosed action 可能放大调度与 KV 峰值，因此 8×H200 canary 必须实测 OOM、tokens/s、KV 使用和长尾长度，不能仅由静态审计宣称效率正收益。

## 4. Adaptive actor KL 的实际契约

下游实际配置为：

```text
algorithm.adaptive_actor_kl=true
algorithm.use_kl_loss=false
algorithm.kl_type=adaptive
algorithm.kl_penalty=low_var_kl
algorithm.kl_coef=0.08
algorithm.kl_target=0.15
algorithm.kl_horizon=50
KL_HORIZON_UNIT=executed_optimizer_updates
```

launcher 在每个 run 目录生成只属于 V37 的 `v37_config.yaml`，并显式打开 `algorithm.adaptive_actor_kl`；`USE_KL_LOSS=false` 仍通过 V32 命令下传。共享 `examples/config.yaml` 与 core 已修改，V36 兼容状态必须通过冻结 checkpoint/seed/data/environment 的差分回归确认，不能由默认值关闭推导为“行为不变”。

这里不能按字面同时设置 `use_kl_loss=true`：当前 `ray_trainer.py` 明确拒绝 `adaptive_actor_kl && use_kl_loss`，因为前者已经在 actor loss 中加入由 adaptive controller 驱动的全局 response-token mean `low_var_kl`；后者是旧固定 actor KL 分支。adaptive 模式下 selector 只使用 task score，reward-side KL 不参与 winner 选择，beta 和 actor LR scheduler 都仅在至少一个 `optimizer.step()` 实际执行后推进；全跳过的 RPC 不会消耗 warmup/decay step，非法或缺失 update counter 会 fail closed。controller 公式为 `beta *= 1 + clip(KL/target - 1, -0.2, 0.2) * executed_updates / horizon`；`horizon=50` 时每个实际 optimizer update 的 beta 最大变化为 0.4%，12-update formal A/B 的单向累计上界约 4.9%。adaptive controller state、`horizon_unit` 与 effective-update counter 由 checkpoint 保存/恢复。若未来要把 `use_kl_loss=true` 作为同义选择器，必须先修改并验证核心契约，本 wrapper 不绕过校验。

核心目前也对 Ulysses sequence parallel `>1` fail closed。`V37_CP_SIZE` 保留为可插拔接口，但本轮只允许 1。默认是单机 8×H200、micro update/experience 4/8、rollout/global/val batch 128、vLLM blocks 20480、GPU memory utilization 0.50。CP>1 的精确 token 全局均值与多模态路径是下一轮验证项，不宣称已经支持。

## 5. Frontier 数据与 formal 证据

formal 数据必须来自 `tools/build_v37_frontier_dataset.py`。builder 输入两个不同 seed 的 audit；每个 source prompt 在每个 seed 都必须有 M=32 个唯一 candidate。输出 `frontier_rl.parquet` 不能含 response/trajectory 列，并由相邻 `selection_manifest.json` 固化：

- `schema_version=3`、`data_mode=frontier_rl`、`run_class=formal`、`publication=atomic_directory_v1`；
- 两个不同 mining seed、`candidates_per_seed=32`；
- 已知且非零的 source SHA256 和两个 audit/miner SHA256；
- `requested_sample_count == selected_sample_count > 0`；
- 五桶 2–10 / 11–20 / 21–30 / 31–40 / 41–50 配额严格为 40% / 10% / 20% / 20% / 10%，每桶 selection quota 完整；
- `candidate_fields_injected=false`。
- `_COMPLETE.json` 对 parquet、selected IDs 和 selection manifest 的 SHA256 封印完整，publication 目录只能包含这三个 regular files 与 marker；任何额外目录、有效/悬空 symlink 或特殊文件都失败；canary/formal audit 的 checkpoint、generation/sampling config 与 response provenance 由 builder 强制校验。

formal builder 的完整调用模板如下；所有 SHA256 必须由独立冻结步骤预先计算，`--output-dir` 必须不存在：

```bash
python3 tools/build_v37_frontier_dataset.py \
  --source /abs/path/source_rl.parquet \
  --audit /abs/path/mining-seed11.jsonl \
  --audit /abs/path/mining-seed22.jsonl \
  --output-dir /new/abs/path/v37-frontier-formal \
  --run-class formal \
  --sample-count 10000 \
  --selection-seed 37 \
  --candidates-per-seed 32 \
  --expected-source-sha256 '<source-sha256>' \
  --expected-audit-sha256 '<seed11-audit-sha256>' \
  --expected-audit-sha256 '<seed22-audit-sha256>' \
  --mining-checkpoint /abs/path/mining-checkpoint \
  --expected-mining-checkpoint-sha256 '<checkpoint-tree-sha256>' \
  --image-root /abs/path/images
```

formal preflight 还必须机器验证：

- train/held-out/全部 benchmark 的 identity、图像 SHA256 与真实 pHash 不重叠；
- held-out 覆盖五桶，answer/image 可用；
- coverage report 的 dataset/metadata/mask-tree SHA256 与现场一致且 `coverage=1.0`；
- metadata basename 唯一、每个 mask 存在且可解码、每个训练图像有映射；
- filtered manifest 与 RLHFDataset 实际过滤后的 dataset hash、sample count、五桶 count 一致；
- focused10k、benchmark dev val、缺失/unknown hash 或人工 ACK 都不能替代这些证据。

preflight 使用临时目录输出。只有全部检查通过后才原子 claim run 目录；因此 preflight 失败不会预先留下目标 run 目录。并发或已存在目标目录会被拒绝。

formal preflight 与训练现在共同使用 `V31_FILTER_OVERLONG_NUM_PROC`，默认 64，不再以 `num_proc=1` 审计另一套 loader 配置。
`V37_PREFLIGHT_PYTHON` 是被审计并用于启动下游训练 shell 的 control interpreter；production 必须把它指向训练环境中的 Python，不能用另一个仅为通过 preflight 而安装依赖的解释器。

## 6. Manifest 审计

`v37_run_manifest.json` 记录：

- `run_class`、`data_mode`、`arm`、seed、promotability 与不可晋级原因；
- estimator、native action、旧 process off、correctness-first、allwrong-zero、winner boost、adaptive KL 和 CP size；
- frontier selection manifest SHA256、source SHA256、两个 miner/audit SHA256；
- train/val/model snapshot、coverage/filtered/preflight hash；
- dataset、metadata、mask-tree、train/val/benchmark identity hash；
- `config_sha256` 对 selected config 与完整 `audited_environment` 联合封印，A/B 只允许预注册的 arm/run/seed 字段不同；
- `ab_preregistration` 对 `ab_run_id`、绝对 plan path/SHA256 和当前 cell ID 封印；formal gate 重新读取 plan，要求两个不同 integer seed、四个唯一 cell、1×8 H200、12 updates 和严格串行顺序；
- `paired_sample_universe` 在训练前绑定一个 held-out parquet snapshot，冻结有序 `(sample_id,prompt_id)` 全集、GT、五桶、source-row hash 与 bucket counts；formal eval 不允许自行提供或修改 GT/bucket，也不能少评/挑选样本；
- `v37_execution_environment.json` 对传给下游训练的显式环境做无自引用 canonical hash；manifest/log/eval 均绑定同一 execution hash；
- 稳定 `evidence_run_id` 交叉绑定 manifest、训练 log 与 paired eval；
- git commit/status/diff hash，以及 config、reward、single/A-B launcher、frontier builder、preflight、benchmark verifier、gate、core、actor/actor config/FSDP worker、rollout/sharding、action ledger、dataset、checkpoint、trainer、reward manager、training entry、system/process prompt 与 torch functional 代码 hash；Git 与完整 implementation snapshot 进入 `evidence_run_id`，gate/continuation 会现场重算本地 Git 身份；
- formal gate 重新读取并复算初始 model、train/val parquet、metadata、mask tree、preflight report 和实现文件，而不只比较自报 hash；
- 训练成功后实际 `global_step_N` 的 checkpoint ID、路径与 gate 同算法 tree SHA256；step 10 可保留纯 FSDP 诊断/恢复 shard，formal cell 失败后仍须重跑该 preregistered clean-start cell，不能把 continuation 冒充 formal；终点 step 12 必须在隐藏 staging 内生成 `actor/huggingface/*.safetensors`，随后才写 checkpoint manifest 并原子发布。merger 在加载前检查所有 rank key 集，保存前比对 meta-model key/shape，保存后只读 safetensors header/index 校验 key/shape/dtype；formal host-memory preflight 锁定为 `error` 和 `4.0×` shard bytes。这里属于 structural publication validation：不能证明 tensor 数值等价、无 NaN/Inf、HF/vLLM 可加载或 smoke inference 正确，`4.0×` 也只是 heuristic。晋级前仍需真实 8×H200 merge、reload、固定输入对比与 tokenizer/processor/config smoke test。

manifest/evidence 中的 provenance path 是绝对路径并参与绑定，因此 artifact 默认不可直接搬迁后继续晋级。`STEPCOUNT_IMAGE_PATH_REMAP_JSON` 只解决输入图像根目录映射，不会改写 checkpoint、plan、recipe 或 evidence 的 provenance。未训练前可在目标集群重新 preregister/preflight；已经完成的 formal run 若不能保持完全相同的绝对挂载路径，就必须重新 preregister 并重跑 formal training/eval，不能只重建 evidence 或文本替换旧 JSON。执行 final gate 的主机还必须具有与 manifest 相同的 live Git identity 和 implementation bytes。

controlled continuation 仅用于 `debug/canary` 且永久 non-promotable。`tools/v37_continuation.py create-seal` 只为 trainer-owned evidence 已完整结束、checkpoint/frontier/Git 身份现场一致的 source run 原子发布不可覆盖 seal；seal 可 replay，不含外部消费登记或一次性语义，并绑定 source manifest、checkpoint tree 与 world size。当前 run 完成 input snapshot 后，late validator 会重新读取并核对 seal/source/checkpoint/frontier live hash，验证 source 的 `config_sha256`、`evidence_run_id`、现场 Git 身份与 trainer-owned complete training evidence，再比较 dataset、metadata/mask、held-out、coverage/filter manifest、Python/package、实现文件、完整 audited environment 和数值 config。`continuation_config_sha256` 排除且只排除 `pilot_steps`，同时要求 `V37_PILOT_STEPS == V36_MAX_STEPS == BOK_TOTAL_STEPS`、当前 evidence start 等于 source final step；修改 LR、KL、reward、micro batch、GPU/vLLM、NCCL/allocator 配置或代码都会拒绝。seal 的 checkpoint path/tree hash 同时写入 Trainer config 和 Ray environment；Trainer 在任何本地 `torch.load` 或 worker load dispatch 前做一次 trainer-side full-tree rehash，并拒绝 config/environment 不一致或 V37 identity 丢失。返回的 parent lineage 进入新 run 的 `evidence_run_id`。这不是各 rank 紧邻反序列化时分别 rehash：trainer 校验到各 rank 实际反序列化之间仍存在很短窗口，不提供不可变存储，也不对抗同权限写者；FSDP/optimizer/dataloader checkpoint 含可信 `torch.load(weights_only=False)` pickle 边界。formal A/B 始终要求 clean start，真实 interrupted-vs-uninterrupted resume 等价仍待验证。

四 cell 串行运行时，V32 exact-Ray 模式会等待可见 GPU 资源归零，而不是在 `used GPUs != 0` 时立即失败；formal 将外层 `RAY_GPU_WAIT_TIMEOUT_SECONDS` 锁定为 900、单次 `RAY_STATUS_TIMEOUT_SECONDS` 锁定为 10、Ray head 启动锁定为 60 秒，并将 placement-group ready 等待锁定为 900 秒。超时会 fail closed 并清理本次未 ready 的 placement groups。该逻辑仍只是基于 `ray status` 的 best-effort 空闲探测，不是原子 lease，也看不到非 Ray CUDA 进程或实际显存余量；它不能证明 cluster/GCS、detached actor、object store 或外部任务隔离。真实 Ray actor/job census、GPU process census 与四 cell teardown E2E 完成前，不声明“不会复用状态”或“不会 OOM”。

debug 永远 `promotable_candidate=false`；canary 也不能直接成为最终晋级 run。formal 的 true 仅表示输入证据允许进入候选流程，不表示分数已达标。

## 7. 推荐执行顺序

先只检查开关展开：

```bash
V37_RUN_CLASS=debug V37_DATA_MODE=frontier_rl V37_ARM=baseline \
V37_CONTRACT_ONLY=1 bash examples/v37_strict_winner_step_rl_pilot.sh
```

canary 使用由 builder 以 `--run-class canary` 生成的完整 manifest，固定 2–4 updates。canary 的门是机制安全：selector KL contribution=0、correctness sign error=0、native event/mask coverage=100%、无 OOM/nonfinite/skipped update、beta 可恢复。`pixmo raw>=0.83` 和 `stepcount-500 raw>=0.18` 不是 canary 前提，也不应在 canary 前宣称 benchmark gain。

需要从已完成的 debug/canary checkpoint 做受控续训时，先创建不可覆盖但可 replay 的 seal，再将其传给同一 V37 launcher；路径必须是绝对、无 symlink 且输出文件不能预先存在：

```bash
python3 tools/v37_continuation.py create-seal \
  --source-manifest /abs/source-run/v37_run_manifest.json \
  --checkpoint /abs/source-run/global_step_4 \
  --output /abs/source-run/continuation-seal.json

V37_CONTINUATION_MODE=1 \
V37_RUN_PURPOSE=controlled_continuation \
V37_RUN_CLASS=debug \
V37_DATA_MODE=frontier_rl \
V37_ARM=progress \
V37_SEED=11 \
V37_FRONTIER_DATA=/abs/frontier/frontier_rl.parquet \
V37_FRONTIER_MANIFEST=/abs/frontier/selection_manifest.json \
STEPCOUNT_V37_VAL_DATA=v37-heldout::/abs/heldout \
V37_STEPCOUNT_MASKS_METADATA=/abs/masks/masks_metadata.json \
V37_STEPCOUNT_MASKS_DIR=/abs/masks \
V37_RESUME_CHECKPOINT=/abs/source-run/global_step_4 \
V37_RESUME_SEAL=/abs/source-run/continuation-seal.json \
V37_PILOT_STEPS=8 \
bash examples/v37_strict_winner_step_rl_pilot.sh
```

上例中的 `run_class/data_mode/arm/seed`、frontier/validation/mask 路径以及其余非 continuation allowlist 的 audited environment 必须逐项复用 source manifest 的值；示例值只展示字段，不代表任意 source run。launcher 会重算实体 hash，并拒绝只给 checkpoint/seal、但未恢复 source run identity 的 clean-shell 调用。

formal 不再逐臂手工启动。先准备共享输入，再让 A/B launcher 原子预注册并串行完成四个 cell（路径仅示意）：

```bash
cd /abs/path/EasyR1-hy-0703

python3 tools/v37_paired_universe.py \
  --data /path/to/heldout.parquet \
  --output /new/path/v37-paired-sample-universe.json

export V37_FRONTIER_DATA=/path/to/frontier_rl.parquet
export V37_FRONTIER_MANIFEST=/path/to/selection_manifest.json
export STEPCOUNT_V37_VAL_DATA=heldout::/path/to/heldout.parquet
export V37_STEPCOUNT_MASKS_METADATA=/path/to/masks_metadata.json
export V37_STEPCOUNT_MASKS_DIR=/path/to/masks
export V37_METADATA_COVERAGE_REPORT=/path/to/coverage.json
export V37_FILTERED_MANIFEST=/path/to/filtered_manifest.json
export V37_AB_PAIRED_UNIVERSE=/new/path/v37-paired-sample-universe.json
export V37_AB_EVAL_RECIPE=/path/to/frozen-v37-formal-eval-recipe.json

V37_AB_ROOT=/new/path/v37-formal-ab-001 \
V37_AB_SEEDS=11,22 \
V37_AB_POSTPROCESS=1 \
bash examples/rl_launch/run_v37_strict_ab.sh
```

`V37_AB_ROOT` 必须尚不存在、不得含 symlink 组件且必须位于 Git worktree 外；launcher 原子 claim 后生成 `ab_plan.json`，再依次运行四个 cell。production eval recipe 必须预先冻结 evaluator executable path/SHA256、8-GPU environment 与 argv；launcher 在 preregistration 阶段调用 `verify_recipe()`，无效 recipe 会在首个 child 前失败。仓库目前没有可冒充 production 的默认 recipe，具体 schema 和审查项见 `docs/V37_FORMAL_EVAL_RECIPE.md`。缺少 universe/recipe 时 formal 会在训练前失败。

只生成轻量 plan 可用：

```bash
V37_AB_ROOT=/new/path/v37-plan-check V37_AB_PLAN_ONLY=1 \
bash examples/rl_launch/run_v37_strict_ab.sh
```

该命令会 claim `V37_AB_ROOT` 并写 `ab_plan.json`，但不解析训练资产、不启动 GPU。`V37_AB_CONTRACT_ONLY=1` 会调用四次 single-run wrapper，因此 formal 模式仍需提供有效 frontier manifest；它不启动 GPU 训练。不要手工伪造 `V37_AB_*` 来逐臂运行，final gate 会按 plan 的 target directory 和 manifest identity 复算。

真实 formal training 还要求 Git worktree clean 且所有实现已 commit；launcher 在完整 preflight 后、run directory 原子 claim 前检查一次，并在 manifest identity capture 时再次要求 status/diff 都等于空 SHA256；final gate 也拒绝任何非空 formal Git identity。`contract-only`、`preflight-only`、`dry-run` 不发布 promotable checkpoint，因此允许用于 dirty-tree 开发验证。

## 8. GO / NO-GO 与最终 gate

`tools/v37_gate.py` 使用 schema v2：`stage=canary, run_class=canary` 只检查机制/paired held-out 门，不要求最终 benchmark；`stage=final, run_class=formal` 才执行最终 promotion，且 manifest/log/eval 必须全部以 path+SHA256 提供。launcher 的只读分发接口为：

```bash
V37_GATE_ONLY=1 V37_GATE_INPUT=/path/to/final_gate_spec.json \
V37_GATE_JSON_OUT=/path/to/final_gate_result.json \
bash examples/v37_strict_winner_step_rl_pilot.sh
```

gate 要求两个 seed 的 baseline/progress、非 allowlist 配置与输入 hash 一致、12 个实际 optimizer step、零 skipped/nonfinite/OOM、selector KL=0、event/mask coverage=100%、KL 可恢复、零 progress degradation、frontier mixed/allwrong 比例和预注册 paired metrics 通过。任意 `--ack` 只记录请求，不能覆盖硬失败。

formal postprocess 不能直接消费目录中已有的 eval JSON。它为每个 cell 新生成在本次四 cell gate input 内唯一的 256-bit nonce，并调用 preregistered producer；receipt 绑定 invocation request、producer/recipe/program SHA256、完整 rendered argv、最小 offline environment、manifest/checkpoint/HF model 与每个输出 hash。该 nonce 不提供跨 run freshness。paired evidence 只允许引用 frozen sample universe，gate 从原始 transcript、严格 integer `<answer>`、每轮 `<point>` JSON 的 `point_2d/count_number` 和 evaluator 输出的 `matched_target_id` ledger 重算聚合分数；`metrics` 只是必须与重算值一致的 cache，不是可信输入。

sample-universe 文件与 producer step-output promotion 使用同一 filesystem 内的 POSIX hard link (`os.link`)；producer receipt 使用 `O_EXCL` 新建，A/B plan 与 postprocess staging 使用同 filesystem `os.replace`。目标 NFS/本地盘必须分别支持这些原子语义。FUSE/对象存储挂载若不支持会 fail closed，不能改成非原子 copy 后继续声称 formal evidence。

final benchmark 只评测两个 progress checkpoint，并为每个 checkpoint/suite 构建一个 schema v2 descriptor。示例：

```bash
python3 tools/v37_benchmark_evidence.py build \
  --suite pixmo-test \
  --checkpoint-tree /path/to/global_step_12 \
  --eval-model /path/to/global_step_12/actor/huggingface \
  --dataset /abs/path/pixmo-test/eval_dataset.json \
  --external-image-manifest /path/to/pixmo_images_manifest.json \
  --raw-results /path/to/pixmo_raw_results.json \
  --eval-run-manifest /path/to/pixmo_eval_run_manifest.json \
  --evaluator /path/to/eval_fix_pixels_one_count_per_time.py \
  --validator /path/to/frozen_eval_validator.py \
  --prompt examples/format_prompt/StepCount_interleaved_system_prompt.txt \
  --requirements requirements.txt \
  --output /path/to/pixmo_benchmark_evidence.v2.json
```

`stepcount-500` 使用 canonical `eval_stepcount_bench_500.json`。descriptor verifier 强制 BF16、greedy (`do_sample=false`, beam/return=1)、`min(task_cap, GT+3)`、显式 integer answer、首个完整 action tag stop、禁止 point-count fallback，并从 transcript 重算 `predicted_answer/is_correct/termination/rounds`。final gate 额外冻结：

| suite | dataset SHA256 | ordered IDs SHA256 | 晋级线 |
|---|---|---|---:|
| pixmo-test | `9f58c7eabcf0e09e94a9584cd7a4ba2e94541b116daf35e50ff8299991b94da1` | `693bcc82f4a2c6c5e7b548ded47da89b51b2f2cfc2aaeeeb171e04ee6a66a8b0` | 440/529 |
| stepcount-500 | `f839edcf49b73c721395f1765a237ce45a12c15e3f07843d73e48716630fb836` | `cff39f9d4393378815e44f3c754e7c1e8ffa265e9275118111689ec050912e4e` | 90/500 |

旧 V36 result 缺少完整 protocol/run/image/model provenance，仍可作为历史分析数据，但不能被“补字段”升级为 formal evidence。两个 suite 必须使用同一 progress checkpoint 的 `actor/huggingface`，不能跨 checkpoint 拼接。

gate 的边界是给定本地 artifacts 的完整性、一致性和可追溯性：它能发现运行后文件变化、跨 run 误拼接、环境/config 漂移与 checkpoint/eval 不匹配；它不是外部签名服务，也不声称能阻止一个拥有全部 artifacts 写权限的主体同时重写所有文件。receipt 证明“冻结 recipe 指定的程序产生了这些 bytes”，不是第三方 attestation；`matched_target_id` 仍以冻结 evaluator/mask matcher 为可信计算边界，gate 不重新执行几何 mask 命中。相同 evidence 的 deterministic replay 是有意支持的，promotion 的一次性/freshness 必须由外部只读登记或签名服务保证。需要更强来源认证时，应在独立机器保存预注册 spec/hash，人工审查 evaluator/matcher，或增加签名/远程 attestation。

本轮不再保留无法映射到固定 commit/命令的历史 pytest 数字。可复算的 dependency-light 回归命令与结果记录在本文末尾“验证记录”；training conda 环境的 PyTorch/CUDA 数值 probe、8×H200 profiler、真实 distributed checkpoint merge/save/resume 和四 cell 实跑仍必须单独完成。系统 Python 的完整 torch pytest 受 NumPy/torch ABI 冲突时只能标为“未运行”，不能写成通过。

最终 benchmark 的必要条件包括 `pixmo-test >=440/529`（raw ≥0.83）与 `stepcount-500 >=90/500`（raw ≥0.18），而且必须由同一个 checkpoint 同时达到，不能跨 checkpoint 拼接；二者不是充分条件，还必须通过 paired、训练完整性、provenance 和 eval-receipt hard gates。gate 的 `GO` 仅表示本地 artifact contract 通过，不能替代 8×H200 merge/reload parity、production evaluator/mask matcher 人工审查或外部 attestation，因此不能单独等同“性能晋级”。canary 的内嵌 evidence 仅是机制诊断；只有 formal 才强制 path+SHA256 artifact。当前没有运行 canary/formal，也没有任何新 eval，因此现状是：

- launcher/contract：dependency-light 回归通过；训练环境 E2E 仍未验证；
- 数据证据：待生产，NO-GO；
- initial-model identity：代码已冻结 checkpoint-476 content hash，真实集群重算待执行；
- OOM/memory-fallback telemetry：代码累计 validation actor death、OOM/高显存 sample skip，以及 99% 显存阈值触发的 batch→per-sample 预防性降级；gate 要求该累计值为 0，8×H200 实测待执行；
- canary：未运行，NO-GO；
- formal training：NO-GO；
- merge/reload parity 与 production evaluator review/attestation：未完成，NO-GO；
- 性能晋级：无结论，NO-GO。

预期因果收益是让 BoK 排序回到正确性、把过程 credit 限定到真实 action span、让 KL 只作为 actor 稳定项；是否带来分数提升只能由上述 A/B 和 final gate 判断。

### 验证记录

本节只记录 2026-07-23 当前 checkout 上可复算的代码冻结验证：

- `bash -n`：本轮变更的 6 个 shell（V37 single/A-B launcher、V32 entry、resume 与 merger）均通过；
- `python3 -m py_compile`：本轮变更的 51 个 Python 文件均通过；
- `debug + CONTRACT_ONLY`：baseline 展开为 `bok_grpo/action=0/step_weight=0`，progress 展开为 `bok_grpo_step/action=1/step_weight=0.1`，两者均为 `adaptive_actor_kl=true/use_kl_loss=false/CP=1/clean_start`；
- V36 `V32_DRY_RUN=1`、V37 baseline/progress `CONTRACT_ONLY` 与 A/B `PLAN_ONLY` smoke 均通过；
- dependency-light pytest：V37 preflight/A-B/benchmark/continuation/eval/frontier/gate/launcher/universe/path-remap/postprocess/Ray/reward/scheduler/training-evidence/tree-hash 与 merger 共 `337 passed, 26 skipped in 195.55s`；26 个 skip 均为主 Python 缺 `pyarrow` 的显式 skip，`test_real_builder_launcher_preflight_e2e` 通过 `/root/miniconda3/bin/python` 实际执行；
- reviewer 修复后的 training-evidence/gate 定向回归：`67 passed in 91.83s`；
- Git override 与子环境 focused 回归：`4 passed`；mask route/model-quality 解耦及 training-evidence 回归：`40 passed`；
- `/usr/bin/python3`：NumPy `2.2.5` 与系统 torch 的 NumPy-1 ABI 冲突，`import torch` 失败；该环境的 torch 数值测试未运行；
- `/root/miniconda3/bin/python`：torch/NumPy import probe 可用，但该环境缺 `pytest`，且本 shell 无 `nvidia-smi`、`ray` 和可见 H200，不能替代训练集群数值验证；
- 8×H200 canary、OOM/nonfinite、tokens/s、distributed merge/reload、resume parity 与新 benchmark eval：均未运行，保持 **NO-GO**。

## 9. 关闭与回滚

- 不调用 V37 wrapper 时，新增训练目标默认关闭，V36 仍选择宽松 answer parser/logprob `legacy` 分支；共享 core/config/Ray/data/checkpoint 已有增量实现，V36 feature-off 仅局部 contract 通过，端到端 parity 状态为 **UNVERIFIED**。
- 在 V37 中选择 `V37_ARM=baseline` 会关闭 native action-event auxiliary。
- `PROCESS_REWARD_ENABLE=0` 始终关闭旧 retokenized process 路径。
- 若 progress 未过 canary/A/B gate，保留 baseline，不扩大训练；不要通过 resume、人工 ACK、放宽 hash 或改 benchmark val 绕过失败。
- `strict_winner_rft` 必须等待独立、正确命名并经验证的 RFT entrypoint；在此之前保持 fail closed。
