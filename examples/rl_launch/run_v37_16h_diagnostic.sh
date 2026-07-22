#!/usr/bin/env bash
# V37 non-promotable 15.5h diagnostic: fixed AB/BA order, strictly serial.
set -euo pipefail

_diag_error() { echo "[V37-16h诊断][错误] $*" >&2; exit 1; }
if [[ -n "${BASH_ENV:-}" || -n "${ENV:-}" ]]; then
  _diag_error "禁止 BASH_ENV/ENV"
fi
if env | awk -F= '$1 ~ /^BASH_FUNC_/ { found=1 } END { exit !found }'; then
  _diag_error "禁止导出的 BASH_FUNC_*"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_DIR}/examples/local_path_env.sh"

PYTHON_BIN="${V37_DIAGNOSTIC_PYTHON:-python3}"
NVIDIA_SMI_BIN="${V37_DIAGNOSTIC_NVIDIA_SMI:-nvidia-smi}"
RAY_BIN="${V37_DIAGNOSTIC_RAY_BIN:-ray}"
TIMEOUT_BIN="${V37_DIAGNOSTIC_TIMEOUT_BIN:-timeout}"
WRAPPER="${V37_DIAGNOSTIC_SINGLE_RUN_WRAPPER:-${REPO_DIR}/examples/v37_strict_winner_step_rl_pilot.sh}"
for _diag_command in "${PYTHON_BIN}" "${TIMEOUT_BIN}" flock setsid; do
  command -v "${_diag_command}" >/dev/null 2>&1 || _diag_error "缺少命令: ${_diag_command}"
done
"${TIMEOUT_BIN}" --version 2>/dev/null | head -n 1 | grep -q 'GNU coreutils' \
  || _diag_error "每 cell 必须使用 GNU coreutils timeout"
[[ -f "${WRAPPER}" && -x "${WRAPPER}" && ! -L "${WRAPPER}" ]] \
  || _diag_error "单 cell wrapper 必须是可执行非软链接文件: ${WRAPPER}"
exec 9>"${V37_DIAGNOSTIC_GPU_LOCK:-/tmp/easyr1-v37-h200-gpu0-7.lock}"
flock -n 9 || _diag_error "GPU 0-7 已被另一个 V37 diagnostic launcher 占用"

[[ -n "${V37_DIAGNOSTIC_ROOT:-}" ]] || _diag_error "必须设置 V37_DIAGNOSTIC_ROOT"
DIAGNOSTIC_ROOT="$(${PYTHON_BIN} - "${V37_DIAGNOSTIC_ROOT}" <<'PY'
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)"
[[ "${DIAGNOSTIC_ROOT}" =~ ^/[A-Za-z0-9_./-]+$ ]] \
  || _diag_error "V37_DIAGNOSTIC_ROOT 仅允许绝对 ASCII 路径字符 [A-Za-z0-9_./-]"
[[ ! -e "${DIAGNOSTIC_ROOT}" && ! -L "${DIAGNOSTIC_ROOT}" ]] \
  || _diag_error "拒绝覆盖已有诊断目录: ${DIAGNOSTIC_ROOT}"

VAL_DATA="${STEPCOUNT_V37_DIAGNOSTIC_VAL_DATA:-}"
_diag_builder_command() {
  printf '%q ' "${PYTHON_BIN}" "${REPO_DIR}/tools/build_v37_diagnostic_heldout.py" \
    --source-0-10 "${STEPCOUNT_REPLAY_DATA}" \
    --source-11-50 "${STEPCOUNT_DENSE_11_50_MASKCOMPLETE_DATA}" \
    --focused-selection-manifest "${STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA}/selection_manifest.json" \
    --benchmark "${STEPCOUNT_PIXMO_CANONICAL_JSON}" \
    --benchmark "${STEPCOUNT_STEPCOUNT500_CANONICAL_JSON}" \
    --benchmark "${STEPCOUNT_COUNTQA_DATA}" \
    --benchmark "${STEPCOUNT_BIAS_DATA}" \
    --benchmark "${STEPCOUNT_DENSE_CANONICAL_JSON}" \
    --benchmark "${STEPCOUNT_EXTREME_CANONICAL_JSON}" \
    --output-dir "${STEPCOUNT_V37_DIAGNOSTIC_VAL_DATA}"
  echo
}
if [[ -z "${VAL_DATA}" || ! -e "${VAL_DATA}" ]]; then
  echo "[V37-16h诊断][错误] 缺少独立 diagnostic heldout；先运行：" >&2
  _diag_builder_command >&2
  exit 1
fi
[[ -d "${VAL_DATA}" && ! -L "${VAL_DATA}" ]] \
  || _diag_error "diagnostic val 必须是 builder 原子发布的非软链接目录"
VAL_REAL="$(readlink -f -- "${VAL_DATA}")"
VAL_PARQUET="${VAL_REAL}/diagnostic_heldout.parquet"
VAL_MANIFEST="${VAL_REAL}/selection_manifest.json"
FOCUSED_MANIFEST="${STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA}/selection_manifest.json"
SOURCE_0_10="${STEPCOUNT_REPLAY_DATA}"
SOURCE_11_50="${STEPCOUNT_DENSE_11_50_MASKCOMPLETE_DATA}"
[[ -f "${VAL_PARQUET}" && ! -L "${VAL_PARQUET}" ]] \
  || _diag_error "diagnostic heldout 缺少 regular diagnostic_heldout.parquet"
[[ -f "${VAL_MANIFEST}" && ! -L "${VAL_MANIFEST}" ]] \
  || _diag_error "diagnostic heldout 缺少 regular selection_manifest.json"
BENCHMARKS=(
  "${STEPCOUNT_PIXMO_CANONICAL_JSON}" "${STEPCOUNT_STEPCOUNT500_CANONICAL_JSON}"
  "${STEPCOUNT_COUNTQA_DATA}" "${STEPCOUNT_BIAS_DATA}"
  "${STEPCOUNT_DENSE_CANONICAL_JSON}" "${STEPCOUNT_EXTREME_CANONICAL_JSON}"
)
for _diag_benchmark in "${BENCHMARKS[@]}"; do
  if [[ -e "${_diag_benchmark}" ]]; then
    _diag_benchmark_real="$(readlink -f -- "${_diag_benchmark}")"
    [[ "${VAL_REAL}" != "${_diag_benchmark_real}" \
       && "${VAL_REAL}" != "${_diag_benchmark_real}/"* \
       && "${_diag_benchmark_real}" != "${VAL_REAL}/"* ]] \
      || _diag_error "diagnostic val 与 benchmark 路径重叠: ${_diag_benchmark}"
  fi
done
"${TIMEOUT_BIN}" --signal=TERM --kill-after=30 600 "${PYTHON_BIN}" - \
  "${VAL_MANIFEST}" "${VAL_PARQUET}" "${FOCUSED_MANIFEST}" \
  "${SOURCE_0_10}" "${SOURCE_11_50}" "${BENCHMARKS[@]}" <<'PY' \
  || _diag_error "diagnostic heldout manifest 完整性或六套 benchmark 隔离绑定失败"
import hashlib
import json
import os
import re
import sys
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path):
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise SystemExit(f"benchmark input is missing: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise SystemExit(f"benchmark input is empty: {path}")
    entries = [
        {
            "path": item.relative_to(path).as_posix(),
            "size": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in files
    ]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

manifest_path, parquet_path, focused_path = map(Path, sys.argv[1:4])
source_paths = [Path(item).resolve() for item in sys.argv[4:6]]
expected_benchmarks = {str(Path(item).resolve()) for item in sys.argv[6:]}
if len(sys.argv[6:]) != 6 or len(expected_benchmarks) != 6:
    raise SystemExit("launcher requires six distinct canonical benchmarks")
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
expected_quotas = {"2-10": 200, "11-20": 50, "21-30": 100, "31-40": 100, "41-50": 50}
if payload.get("schema_version") != 2 or payload.get("dataset_kind") != "v37_nonbenchmark_diagnostic_heldout":
    raise SystemExit("wrong heldout schema/kind")
if payload.get("promotable") is not False or payload.get("benchmark_overlap_selected") != 0:
    raise SystemExit("heldout promotion/overlap contract is invalid")
if payload.get("selected_rows") != 500 or payload.get("quotas") != expected_quotas:
    raise SystemExit("heldout row/quota contract is invalid")
if payload.get("selected_distribution") != expected_quotas:
    raise SystemExit("heldout distribution is invalid")
batch_size = payload.get("parquet_batch_size")
if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 16:
    raise SystemExit("heldout was not built with bounded parquet batches")
recorded_benchmarks = {str(Path(item).resolve()) for item in payload.get("benchmark_paths", [])}
if recorded_benchmarks != expected_benchmarks:
    raise SystemExit("heldout does not bind exactly the six canonical benchmarks")
hashes = payload.get("benchmark_input_sha256")
if not isinstance(hashes, dict) or set(hashes) != expected_benchmarks:
    raise SystemExit("heldout benchmark hash universe differs")
if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes.values()):
    raise SystemExit("heldout benchmark hash is invalid")
current_hashes = {str(Path(item).resolve()): sha256_path(Path(item).resolve()) for item in sys.argv[6:]}
if hashes != current_hashes:
    raise SystemExit("canonical benchmark content changed after heldout construction")
if payload.get("focused_selection_manifest") != str(focused_path.resolve()):
    raise SystemExit("heldout focused manifest path binding differs")
if payload.get("focused_selection_manifest_sha256") != sha256_file(focused_path.resolve()):
    raise SystemExit("focused selection manifest changed after heldout construction")
if payload.get("source_paths") != [str(path) for path in source_paths]:
    raise SystemExit("heldout source path binding differs")


def source_parquet_files(path):
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    files = sorted((path / "data").glob("*.parquet")) if path.is_dir() else []
    if not files and path.is_dir():
        files = sorted(path.glob("*.parquet"))
    if not files:
        raise SystemExit(f"heldout source parquet is missing: {path}")
    return files


current_source_hashes = {
    str(item.resolve()): sha256_file(item.resolve())
    for source in source_paths
    for item in source_parquet_files(source)
}
if payload.get("source_parquet_sha256") != current_source_hashes:
    raise SystemExit("heldout source parquet changed after heldout construction")
for field in (
    "focused_sequence_keys_sha256", "focused_exact_keys_sha256", "benchmark_exact_keys_sha256",
):
    if re.fullmatch(r"[0-9a-f]{64}", str(payload.get(field, ""))) is None:
        raise SystemExit(f"heldout {field} is invalid")
focused_exact_count = payload.get("focused_exact_key_count")
focused_duplicate_count = payload.get("focused_exact_duplicate_count")
if (
    payload.get("focused_sequence_key_count") != 10_000
    or isinstance(focused_exact_count, bool)
    or not isinstance(focused_exact_count, int)
    or not 0 < focused_exact_count <= 10_000
    or isinstance(focused_duplicate_count, bool)
    or not isinstance(focused_duplicate_count, int)
    or focused_duplicate_count != 10_000 - focused_exact_count
):
    raise SystemExit("heldout focused exclusion key count is invalid")
digest = hashlib.sha256()
with parquet_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
if payload.get("artifact_sha256") != {"diagnostic_heldout.parquet": digest.hexdigest()}:
    raise SystemExit("heldout parquet hash mismatch")
selected = payload.get("selected_indices")
if not isinstance(selected, list) or len(selected) != 500:
    raise SystemExit("heldout selected index count is invalid")
if any(not any(str(key).startswith("sha256:") for key in row.get("image_exact_keys", [])) for row in selected):
    raise SystemExit("heldout row lacks exact image SHA evidence")
PY
for _diag_command in "${NVIDIA_SMI_BIN}" "${RAY_BIN}"; do
  command -v "${_diag_command}" >/dev/null 2>&1 || _diag_error "缺少命令: ${_diag_command}"
done
mkdir -p -- "$(dirname "${DIAGNOSTIC_ROOT}")"
mkdir -- "${DIAGNOSTIC_ROOT}" || _diag_error "无法原子占用诊断目录"
mkdir -- "${DIAGNOSTIC_ROOT}/logs"

readonly GLOBAL_BUDGET_SECONDS=55800  # 15.5h
readonly GLOBAL_FINALIZE_RESERVE_SECONDS=300
readonly BASELINE_P90_SECONDS=12600
readonly PROGRESS_P90_SECONDS=13500
readonly GPU_IDLE_WAIT_SECONDS=900
readonly GPU_IDLE_POLL_SECONDS=10
readonly NVIDIA_SMI_TIMEOUT_SECONDS=20
readonly RAY_STOP_TIMEOUT_SECONDS=30
readonly LOG_DRAIN_TIMEOUT_SECONDS=30
CELL_IDS=(seed11-baseline seed11-progress seed22-progress seed22-baseline)
CELL_SEEDS=(11 11 22 22)
CELL_ARMS=(baseline progress progress baseline)

_diag_monotonic() {
  "${PYTHON_BIN}" - <<'PY'
import time
print(time.monotonic_ns() // 1_000_000_000)
PY
}

START_MONOTONIC="$(_diag_monotonic)"
DEADLINE_MONOTONIC=$((START_MONOTONIC + GLOBAL_BUDGET_SECONDS))
STATUS_PATH="${DIAGNOSTIC_ROOT}/status.json"
COMMAND_MANIFEST="${DIAGNOSTIC_ROOT}/command_manifest.json"
SUMMARY_PATH="${DIAGNOSTIC_ROOT}/中文总结.md"

_diag_status() {
  local action="$1"; shift
  "${PYTHON_BIN}" - "${STATUS_PATH}" "${action}" "${START_MONOTONIC}" "${DEADLINE_MONOTONIC}" "$@" <<'PY'
import json, os, secrets, sys, time
from pathlib import Path

path = Path(sys.argv[1]); action = sys.argv[2]
start, deadline = int(sys.argv[3]), int(sys.argv[4]); args = sys.argv[5:]
if path.exists():
    payload = json.loads(path.read_text(encoding="utf-8"))
else:
    identities = [("seed11-baseline", 11, "baseline", 12600),
                  ("seed11-progress", 11, "progress", 13500),
                  ("seed22-progress", 22, "progress", 13500),
                  ("seed22-baseline", 22, "baseline", 12600)]
    payload = {"schema_version": 1, "kind": "v37_16h_diagnostic",
               "promotable": False, "winner_selected": False,
               "benchmark_used": False, "online_wandb": False,
               "global_budget_seconds": 55800,
               "started_monotonic_seconds": start,
               "deadline_monotonic_seconds": deadline,
               "state": "running", "reason": None,
               "cells": [{"cell_id": cell, "seed": seed, "arm": arm,
                          "p90_budget_seconds": budget, "state": "pending"}
                         for cell, seed, arm, budget in identities]}
if action == "cell":
    cell_id, state, reason, rc, checkpoint, log_path, started, finished = args
    cell = next(item for item in payload["cells"] if item["cell_id"] == cell_id)
    cell.update({"state": state, "reason": reason or None,
                 "exit_code": None if rc == "" else int(rc),
                 "checkpoint": checkpoint or None, "log": log_path or None,
                 "started_monotonic_seconds": None if started == "" else int(started),
                 "finished_monotonic_seconds": None if finished == "" else int(finished)})
elif action == "final":
    payload["state"], payload["reason"] = args[0], (args[1] or None)
    payload["finished_monotonic_seconds"] = int(args[2])
elif action != "init":
    raise SystemExit(f"unknown status action: {action}")
payload["updated_monotonic_seconds"] = time.monotonic_ns() // 1_000_000_000
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(5)}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
os.replace(temporary, path)
PY
}

_diag_summary() {
  "${PYTHON_BIN}" - "${STATUS_PATH}" "${SUMMARY_PATH}" <<'PY'
import json, os, secrets, sys
from pathlib import Path
status_path, output = map(Path, sys.argv[1:])
s = json.loads(status_path.read_text(encoding="utf-8"))
lines = ["# V37 16h 诊断总结", "",
         f"- 状态：{s['state']}", f"- 原因：{s.get('reason') or '无'}",
         "- 性质：不可晋级；未选择 winner；未运行 benchmark/merge；W&B 仅离线。",
         "- 顺序：seed11 baseline → seed11 progress → seed22 progress → seed22 baseline。",
         "", "| cell | 状态 | 退出码 | checkpoint |", "|---|---:|---:|---|"]
for cell in s["cells"]:
    lines.append(f"| {cell['cell_id']} | {cell['state']} | {cell.get('exit_code', '')} | {cell.get('checkpoint') or ''} |")
temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.{secrets.token_hex(5)}")
with temporary.open("x", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n"); handle.flush(); os.fsync(handle.fileno())
os.replace(temporary, output)
PY
}

_diag_status init
export _V37_DIAG_ROOT="${DIAGNOSTIC_ROOT}" _V37_DIAG_WRAPPER="${WRAPPER}" _V37_DIAG_VAL="${VAL_REAL}"
"${PYTHON_BIN}" - "${COMMAND_MANIFEST}" <<'PY'
import json, os, secrets, sys
from pathlib import Path
path = Path(sys.argv[1]); root = Path(os.environ["_V37_DIAG_ROOT"])
cells = [("seed11-baseline", 11, "baseline", 12600),
         ("seed11-progress", 11, "progress", 13500),
         ("seed22-progress", 22, "progress", 13500),
         ("seed22-baseline", 22, "baseline", 12600)]
fixed = {"V37_RUN_CLASS": "debug", "V37_DATA_MODE": "frontier_rl",
         "V37_RUN_PURPOSE": "debug_mechanism", "V37_CONTINUATION_MODE": "0",
         "V37_ALLOW_FOCUSED10K_PILOT": "1", "V37_ALLOW_BENCHMARK_DEV": "0",
         "V37_ALLOW_INDEPENDENT_VAL_OUTSIDE_TRAIN_RANGE": "1",
         "V37_PILOT_STEPS": "3", "V37_STEP_WEIGHT": "0.1",
         "V37_STEP_GATE": "answer_soft", "V37_STEP_MIN_GATE": "0.2",
         "V37_MICRO_BATCH_UPDATE": "4",
         "V37_MICRO_BATCH_EXP": "8", "V37_VLLM_NUM_GPU_BLOCKS": "20480",
         "V37_GPU_MEM_UTIL": "0.50", "V37_MAX_NUM_BATCHED_TOKENS": "49152",
         "V37_CP_SIZE": "1", "V31_NNODES": "1", "V31_N_GPUS_PER_NODE": "8",
         "HOST_NUM": "1", "HOST_GPU_NUM": "8", "INDEX": "0",
         "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "WANDB_MODE": "offline",
         "STEPCOUNT_V37_VAL_DATA": os.environ["_V37_DIAG_VAL"]}
payload = {"schema_version": 1, "kind": "v37_16h_diagnostic_commands",
           "promotable": False, "global_deadline_seconds": 55800,
           "initial_model_contract": "clean_sft_checkpoint-476",
           "postprocessing": {"select_winner": False, "benchmark": False,
                              "merge": False, "online_wandb": False},
           "resource_contract": {"gpus": 8, "model": "H200", "memory_total_mib_min": 139000,
                                 "memory_used_mib_max": 5120, "mig": False, "compute_processes": 0},
           "cells": [{"cell_id": cell, "p90_budget_seconds": budget,
                      "environment": {**fixed, "V37_SEED": str(seed), "V37_ARM": arm,
                                      "V31_SAVE_CHECKPOINT_PATH": str(root / cell)},
                      "argv": [os.environ["_V37_DIAG_WRAPPER"]]}
                     for cell, seed, arm, budget in cells]}
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(5)}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
os.replace(temporary, path)
PY
unset _V37_DIAG_ROOT _V37_DIAG_WRAPPER _V37_DIAG_VAL

_diag_gpu_idle() {
  local gpu_csv process_csv
  gpu_csv="$(mktemp "${TMPDIR:-/tmp}/v37-diag-gpu.XXXXXX")"
  process_csv="$(mktemp "${TMPDIR:-/tmp}/v37-diag-proc.XXXXXX")"
  if ! "${TIMEOUT_BIN}" --signal=KILL "${NVIDIA_SMI_TIMEOUT_SECONDS}" \
      "${NVIDIA_SMI_BIN}" --query-gpu=index,name,memory.total,memory.used,mig.mode.current \
      --format=csv,noheader,nounits >"${gpu_csv}"; then
    rm -f -- "${gpu_csv}" "${process_csv}"; return 1
  fi
  if ! "${TIMEOUT_BIN}" --signal=KILL "${NVIDIA_SMI_TIMEOUT_SECONDS}" \
      "${NVIDIA_SMI_BIN}" --query-compute-apps=pid --format=csv,noheader,nounits >"${process_csv}"; then
    rm -f -- "${gpu_csv}" "${process_csv}"; return 1
  fi
  "${PYTHON_BIN}" - "${gpu_csv}" "${process_csv}" <<'PY'
import csv, sys
from pathlib import Path
rows = list(csv.reader(Path(sys.argv[1]).read_text().splitlines()))
if len(rows) != 8:
    raise SystemExit(f"必须恰好 8 张 GPU，实际 {len(rows)}")
for expected, row in enumerate(rows):
    if len(row) != 5:
        raise SystemExit("nvidia-smi GPU CSV 列数错误")
    index, name, total, used, mig = (value.strip() for value in row)
    if int(index) != expected or "H200" not in name:
        raise SystemExit(f"GPU {expected} 不是预期 H200: {row}")
    if float(total) < 139000 or float(used) > 5120:
        raise SystemExit(f"GPU {expected} 显存不满足 total>=139000MiB/used<=5120MiB: {row}")
    if mig.lower() != "disabled":
        raise SystemExit(f"GPU {expected} MIG 未关闭: {mig}")
processes = [line.strip() for line in Path(sys.argv[2]).read_text().splitlines()
             if line.strip() and "no running processes" not in line.lower()]
if processes:
    raise SystemExit(f"检测到 compute process: {processes}")
PY
  local rc=$?
  rm -f -- "${gpu_csv}" "${process_csv}"
  return "${rc}"
}

_diag_ray_stop_bounded() {
  "${TIMEOUT_BIN}" --signal=TERM --kill-after=5 "${RAY_STOP_TIMEOUT_SECONDS}" \
    "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
}

_diag_ray_stop_and_wait() {
  _diag_ray_stop_bounded
  local now stop_deadline
  now="$(_diag_monotonic)"
  stop_deadline=$((now + GPU_IDLE_WAIT_SECONDS))
  if (( stop_deadline > DEADLINE_MONOTONIC - GLOBAL_FINALIZE_RESERVE_SECONDS )); then
    stop_deadline=$((DEADLINE_MONOTONIC - GLOBAL_FINALIZE_RESERVE_SECONDS))
  fi
  while (( $(_diag_monotonic) <= stop_deadline )); do
    if _diag_gpu_idle >/dev/null 2>&1; then return 0; fi
    sleep "${GPU_IDLE_POLL_SECONDS}"
  done
  return 1
}

_diag_log_has_nonfinite() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import re
import sys

metric = re.compile(
    r"(?:actor/)?non[-_ ]?finite(?:_grad)?_count\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
agreement = re.compile(
    r"(?:actor/)?non[-_ ]?finite_counter_agreement\s*[:=]\s*[-+]?\d+(?:\.\d+)?",
    re.IGNORECASE,
)
scalar = re.compile(r"(?<![A-Za-z0-9_])(?:nan|[-+]?inf(?:inity)?)(?![A-Za-z0-9_])", re.IGNORECASE)
with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
        counts = [float(value) for value in metric.findall(line)]
        if any(value > 0 for value in counts):
            raise SystemExit(0)
        scrubbed = agreement.sub("", metric.sub("", line))
        if re.search(r"non[-_ ]?finite", scrubbed, re.IGNORECASE) or scalar.search(scrubbed):
            raise SystemExit(0)
raise SystemExit(1)
PY
}

ACTIVE_LOG_TMP=""
ACTIVE_LOG_FINAL=""
ACTIVE_FIFO=""
ACTIVE_CHILD_PID=""
ACTIVE_TEE_PID=""
WATCHDOG_PID=""
FINISHED=0
_diag_on_exit() {
  local rc=$?
  trap - EXIT
  if [[ -n "${WATCHDOG_PID:-}" ]]; then
    kill "${WATCHDOG_PID}" 2>/dev/null || true
    wait "${WATCHDOG_PID}" 2>/dev/null || true
  fi
  if [[ -n "${ACTIVE_CHILD_PID:-}" ]]; then
    kill -TERM -- "-${ACTIVE_CHILD_PID}" 2>/dev/null || true
    for _diag_wait in 1 2 3 4 5; do
      kill -0 -- "-${ACTIVE_CHILD_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${ACTIVE_CHILD_PID}" 2>/dev/null || true
    wait "${ACTIVE_CHILD_PID}" 2>/dev/null || true
  fi
  if [[ -n "${ACTIVE_TEE_PID:-}" ]]; then
    kill "${ACTIVE_TEE_PID}" 2>/dev/null || true
    wait "${ACTIVE_TEE_PID}" 2>/dev/null || true
  fi
  [[ -z "${ACTIVE_FIFO:-}" ]] || rm -f -- "${ACTIVE_FIFO}"
  _diag_ray_stop_bounded
  if [[ -n "${ACTIVE_LOG_TMP}" && -f "${ACTIVE_LOG_TMP}" ]]; then
    mv -f -- "${ACTIVE_LOG_TMP}" "${ACTIVE_LOG_FINAL}.interrupted"
  fi
  if [[ "${FINISHED}" == "0" && -f "${STATUS_PATH}" ]]; then
    _diag_status final interrupted "trap_exit_${rc}" "$(_diag_monotonic)" || true
    _diag_summary || true
  fi
  return "${rc}"
}
trap _diag_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

MAIN_PID="${BASHPID}"
WATCHDOG_SECONDS=$((DEADLINE_MONOTONIC - $(_diag_monotonic)))
(( WATCHDOG_SECONDS > 0 )) || _diag_error "preflight 已耗尽全局 15.5h budget"
"${PYTHON_BIN}" - "${WATCHDOG_SECONDS}" "${MAIN_PID}" <<'PY' >/dev/null 2>&1 &
import os
import signal
import sys
import time

time.sleep(int(sys.argv[1]))
try:
    os.kill(int(sys.argv[2]), signal.SIGTERM)
except ProcessLookupError:
    pass
PY
WATCHDOG_PID=$!

_diag_gpu_idle || _diag_error "启动前 8×H200/MIG/显存/compute-process 检查失败"

FINAL_STATE=completed
FINAL_REASON="all_four_cells_completed"
for _diag_index in 0 1 2 3; do
  cell_id="${CELL_IDS[${_diag_index}]}"
  seed="${CELL_SEEDS[${_diag_index}]}"
  arm="${CELL_ARMS[${_diag_index}]}"
  if [[ "${arm}" == baseline ]]; then cell_budget="${BASELINE_P90_SECONDS}"; else cell_budget="${PROGRESS_P90_SECONDS}"; fi
  cell_start="$(_diag_monotonic)"
  remaining=$((DEADLINE_MONOTONIC - cell_start))
  required=$((cell_budget + GLOBAL_FINALIZE_RESERVE_SECONDS))
  if (( remaining < required )); then
    FINAL_STATE=stopped
    FINAL_REASON="remaining_${remaining}s_below_${cell_id}_required_${required}s"
    break
  fi
  target_dir="${DIAGNOSTIC_ROOT}/${cell_id}"
  checkpoint="${target_dir}/global_step_3"
  log_final="${DIAGNOSTIC_ROOT}/logs/${cell_id}.log"
  log_tmp="${DIAGNOSTIC_ROOT}/logs/.${cell_id}.log.tmp.$$"
  ACTIVE_LOG_TMP="${log_tmp}"; ACTIVE_LOG_FINAL="${log_final}"
  ACTIVE_FIFO="${DIAGNOSTIC_ROOT}/logs/.${cell_id}.fifo.$$"
  mkfifo -- "${ACTIVE_FIFO}"
  _diag_status cell "${cell_id}" running "" "" "${checkpoint}" "${log_final}" "${cell_start}" ""
  child=(
    env -i PATH="${PATH}" HOME="${HOME:-/nonexistent}" USER="${USER:-}"
      LOGNAME="${LOGNAME:-}" SHELL="${SHELL:-/bin/bash}" LANG="${LANG:-C.UTF-8}"
      LC_ALL=C.UTF-8 TZ="${TZ:-Asia/Hong_Kong}" TMPDIR="${TMPDIR:-/tmp}"
      CONDA_PREFIX="${CONDA_PREFIX:-}" VIRTUAL_ENV="${VIRTUAL_ENV:-}"
      PYTHONPATH="${PYTHONPATH:-}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
      CUDA_HOME="${CUDA_HOME:-}" HF_HOME="${HF_HOME:-}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-}"
      PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
      V37_RUN_CLASS=debug V37_DATA_MODE=frontier_rl V37_RUN_PURPOSE=debug_mechanism
      V37_CONTINUATION_MODE=0 V37_ALLOW_FOCUSED10K_PILOT=1 V37_ALLOW_BENCHMARK_DEV=0
      V37_ALLOW_INDEPENDENT_VAL_OUTSIDE_TRAIN_RANGE=1 STEPCOUNT_V37_VAL_DATA="${VAL_REAL}"
      V37_PILOT_STEPS=3 V37_ARM="${arm}" V37_SEED="${seed}"
      V37_STEP_WEIGHT=0.1 V37_STEP_GATE=answer_soft V37_STEP_MIN_GATE=0.2
      V37_MICRO_BATCH_UPDATE=4 V37_MICRO_BATCH_EXP=8 V37_VLLM_NUM_GPU_BLOCKS=20480
      V37_GPU_MEM_UTIL=0.50 V37_MAX_NUM_BATCHED_TOKENS=49152 V37_CP_SIZE=1
      V31_NNODES=1 V31_N_GPUS_PER_NODE=8 HOST_NUM=1 HOST_GPU_NUM=8 INDEX=0
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      WANDB_MODE=offline V31_SAVE_CHECKPOINT_PATH="${target_dir}" "${WRAPPER}"
  )
  _diag_gpu_idle || _diag_error "${cell_id} 启动前 GPU lease/空闲状态发生变化"
  echo "[V37-16h诊断] 开始 ${cell_id}，P90 timeout=${cell_budget}s"
  set +e
  tee "${log_tmp}" <"${ACTIVE_FIFO}" &
  ACTIVE_TEE_PID=$!
  setsid "${TIMEOUT_BIN}" --signal=TERM --kill-after=120 "${cell_budget}s" \
    "${child[@]}" >"${ACTIVE_FIFO}" 2>&1 &
  ACTIVE_CHILD_PID=$!
  wait "${ACTIVE_CHILD_PID}"
  cell_rc=$?
  # The wrapper can exit while a descendant still owns the FIFO. Keep the
  # process-group identity until every residual writer is terminated, then
  # stop detached Ray daemons before waiting for tee to drain.
  kill -TERM -- "-${ACTIVE_CHILD_PID}" 2>/dev/null || true
  for _diag_wait in 1 2 3 4 5; do
    kill -0 -- "-${ACTIVE_CHILD_PID}" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-${ACTIVE_CHILD_PID}" 2>/dev/null || true
  _diag_ray_stop_bounded
  ACTIVE_CHILD_PID=""
  for ((_diag_wait = 0; _diag_wait < LOG_DRAIN_TIMEOUT_SECONDS; _diag_wait++)); do
    kill -0 "${ACTIVE_TEE_PID}" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "${ACTIVE_TEE_PID}" 2>/dev/null; then
    kill -TERM "${ACTIVE_TEE_PID}" 2>/dev/null || true
    wait "${ACTIVE_TEE_PID}" 2>/dev/null || true
    tee_rc=124
  else
    wait "${ACTIVE_TEE_PID}"
    tee_rc=$?
  fi
  ACTIVE_TEE_PID=""
  set -e
  rm -f -- "${ACTIVE_FIFO}"
  ACTIVE_FIFO=""
  "${PYTHON_BIN}" - "${log_tmp}" <<'PY'
import os, sys
with open(sys.argv[1], "rb") as handle: os.fsync(handle.fileno())
PY
  mv -- "${log_tmp}" "${log_final}"
  ACTIVE_LOG_TMP=""; ACTIVE_LOG_FINAL=""
  cleanup_ok=1
  _diag_ray_stop_and_wait || cleanup_ok=0
  cell_finish="$(_diag_monotonic)"
  failure_reason=""
  if (( cleanup_ok == 0 )); then
    failure_reason="gpu_not_idle_after_ray_stop"
  elif (( tee_rc != 0 )); then
    failure_reason="log_capture_exit_${tee_rc}"
  elif grep -Eiq 'CUDA([^[:alnum:]]+)?out of memory|OutOfMemoryError|(^|[^[:alnum:]_])OOM([^[:alnum:]_]|$)' "${log_final}"; then
    failure_reason="oom_detected"
  elif _diag_log_has_nonfinite "${log_final}"; then
    failure_reason="nonfinite_detected"
  elif grep -q 'Traceback' "${log_final}"; then
    failure_reason="traceback_detected"
  elif grep -Eiq 'GradSpike.*skip|skip.*GradSpike' "${log_final}"; then
    failure_reason="gradspike_skip_detected"
  elif (( cell_rc != 0 )); then
    failure_reason="cell_exit_${cell_rc}"
  elif [[ ! -d "${checkpoint}" ]]; then
    failure_reason="missing_global_step_3"
  fi
  if [[ -n "${failure_reason}" ]]; then
    _diag_status cell "${cell_id}" failed "${failure_reason}" "${cell_rc}" "${checkpoint}" "${log_final}" "${cell_start}" "${cell_finish}"
    FINAL_STATE=stopped; FINAL_REASON="${cell_id}:${failure_reason}"
    break
  fi
  _diag_status cell "${cell_id}" completed "" "${cell_rc}" "${checkpoint}" "${log_final}" "${cell_start}" "${cell_finish}"
done

final_now="$(_diag_monotonic)"
if (( final_now > DEADLINE_MONOTONIC )); then
  FINAL_STATE=stopped
  FINAL_REASON="global_deadline_exceeded"
fi
_diag_status final "${FINAL_STATE}" "${FINAL_REASON}" "${final_now}"
_diag_summary
FINISHED=1
kill "${WATCHDOG_PID}" 2>/dev/null || true
wait "${WATCHDOG_PID}" 2>/dev/null || true
WATCHDOG_PID=""
echo "[V37-16h诊断] 状态=${FINAL_STATE}；status=${STATUS_PATH}；中文总结=${SUMMARY_PATH}"
[[ "${FINAL_STATE}" == completed ]]
