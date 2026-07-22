#!/usr/bin/env bash
# V37 formal A/B: two seeds, baseline/progress, strictly serial.
set -euo pipefail

_v37_ab_error() {
  echo "[V37-AB][ERROR] $*" >&2
  exit 1
}

if [[ -n "${BASH_ENV:-}" || -n "${ENV:-}" ]]; then
  _v37_ab_error "BASH_ENV/ENV are forbidden"
fi
if env | awk -F= '$1 ~ /^BASH_FUNC_/ { found=1 } END { exit !found }'; then
  _v37_ab_error "exported BASH_FUNC_* entries are forbidden"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_DIR}/examples/local_path_env.sh"
_V37_FORMAL_INITIAL_MODEL_SHA256=f9e6b1e8031bdbc509d34249745cdcf75af85c320d6b88918444b1abb4f580a3
if [[ -n "${V37_EXPECTED_INITIAL_MODEL_SHA256:-}" \
      && "${V37_EXPECTED_INITIAL_MODEL_SHA256}" != "${_V37_FORMAL_INITIAL_MODEL_SHA256}" ]]; then
  _v37_ab_error "V37_EXPECTED_INITIAL_MODEL_SHA256 cannot override checkpoint-476 identity"
fi
export V37_EXPECTED_INITIAL_MODEL_SHA256="${_V37_FORMAL_INITIAL_MODEL_SHA256}"
PYTHON_BIN="${V37_AB_PYTHON:-python3}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
  || _v37_ab_error "Python executable not found: ${PYTHON_BIN}"

[[ -n "${V37_AB_ROOT:-}" ]] || _v37_ab_error "V37_AB_ROOT is required"
for _flag_name in V37_AB_PLAN_ONLY V37_AB_CONTRACT_ONLY V37_AB_POSTPROCESS; do
  _flag_value="${!_flag_name:-0}"
  [[ "${_flag_value}" == "0" || "${_flag_value}" == "1" ]] \
    || _v37_ab_error "${_flag_name} must be exactly 0 or 1"
done
if [[ "${V37_AB_PLAN_ONLY:-0}" == "1" && "${V37_AB_CONTRACT_ONLY:-0}" == "1" ]]; then
  _v37_ab_error "V37_AB_PLAN_ONLY and V37_AB_CONTRACT_ONLY are mutually exclusive"
fi

SEEDS_TEXT="${V37_AB_SEEDS-11,22}"
[[ "${SEEDS_TEXT}" =~ ^[0-9]+,[0-9]+$ ]] \
  || _v37_ab_error "V37_AB_SEEDS must be exactly two comma-separated nonnegative integers"
IFS=, read -r SEED_A SEED_B <<< "${SEEDS_TEXT}"
_SEED_META="$("${PYTHON_BIN}" - "${SEED_A}" "${SEED_B}" <<'PY'
import sys
seeds = [int(sys.argv[1]), int(sys.argv[2])]
if seeds[0] == seeds[1]:
    raise SystemExit("[V37-AB][ERROR] V37_AB_SEEDS values must be distinct")
print(seeds[0])
print(seeds[1])
PY
)" || _v37_ab_error "V37_AB_SEEDS values must be distinct"
mapfile -t _SEED_META_LINES <<< "${_SEED_META}"
SEED_A="${_SEED_META_LINES[0]}"
SEED_B="${_SEED_META_LINES[1]}"

AB_ROOT="$("${PYTHON_BIN}" - "${V37_AB_ROOT}" "${REPO_DIR}" <<'PY'
import sys
import os
from pathlib import Path

requested = Path(os.path.abspath(os.path.expanduser(sys.argv[1])))
current = Path(requested.anchor)
for component in requested.parts[1:]:
    current /= component
    if current.is_symlink():
        raise SystemExit(f"V37_AB_ROOT contains a symlink component: {current}")
    if not current.exists():
        break
root = requested.resolve(strict=False)
repo = Path(sys.argv[2]).resolve()
if root == repo or repo in root.parents:
    raise SystemExit("V37_AB_ROOT must be outside the Git worktree")
print(root)
PY
)" || _v37_ab_error "V37_AB_ROOT must resolve outside the Git worktree"
[[ ! -e "${AB_ROOT}" && ! -L "${AB_ROOT}" ]] \
  || _v37_ab_error "V37_AB_ROOT must not already exist or be a symlink: ${AB_ROOT}"
if ! mkdir -- "${AB_ROOT}" 2>/dev/null; then
  _v37_ab_error "failed to atomically claim V37_AB_ROOT: ${AB_ROOT}"
fi

PLAN_PATH="${AB_ROOT}/ab_plan.json"
INDEX_PATH="${AB_ROOT}/ab_evidence_index.json"
_INDEX_PUBLISHED=0
_v37_ab_cleanup_unpublished_index() {
  local status=$?
  if [[ "${_INDEX_PUBLISHED}" == "0" ]]; then
    rm -f -- "${INDEX_PATH}" 2>/dev/null || true
  fi
  return "${status}"
}
trap _v37_ab_cleanup_unpublished_index EXIT

PLAN_META="$("${PYTHON_BIN}" - "${AB_ROOT}" "${PLAN_PATH}" "${SEED_A}" "${SEED_B}" \
  "${REPO_DIR}" "${V37_AB_PLAN_ONLY:-0}" "${V37_AB_CONTRACT_ONLY:-0}" <<'PY'
import hashlib
import importlib.util
import json
import os
import secrets
import sys
from pathlib import Path

root = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
seeds = [int(sys.argv[3]), int(sys.argv[4])]
repo = Path(sys.argv[5]).resolve()
lightweight = (
    sys.argv[6] == "1" or sys.argv[7] == "1"
    or bool(os.environ.get("V37_AB_SINGLE_RUN_WRAPPER"))
)
sys.path.insert(0, str(repo))
from tools import v37_gate as gate
path_remap_spec = importlib.util.spec_from_file_location(
    "v37_ab_path_remap", repo / "verl/utils/path_remap.py",
)
if path_remap_spec is None or path_remap_spec.loader is None:
    raise SystemExit("[V37-AB][ERROR] cannot load path remap implementation")
path_remap_module = importlib.util.module_from_spec(path_remap_spec)
path_remap_spec.loader.exec_module(path_remap_module)


def path_value(*names):
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


def file_binding(path_text):
    if not path_text:
        return None
    path = Path(os.path.abspath(os.path.expanduser(path_text)))
    if not path.is_file() or path.is_symlink():
        return None
    return {"path": str(path), "sha256": gate.sha256(path)}


def content_binding(path_text, *, parquet_only=False):
    if not path_text:
        return None
    try:
        return gate.content_snapshot(Path(path_text), parquet_only=parquet_only)
    except (OSError, ValueError, gate.GateError):
        return None


def validation_bindings(spec):
    if not spec:
        return None
    result = []
    for item in spec.split(","):
        suite, separator, path_text = item.partition("::")
        if not separator:
            path_text, suite = suite, "default"
        snapshot = content_binding(path_text, parquet_only=True)
        if snapshot is None:
            return None
        result.append({"suite": suite, "dataset": snapshot})
    return result


model_path = path_value("MODEL_PATH", "STEPCOUNT_1M_RESUME_CKPT476_MODEL_PATH")
expected_model_sha256 = path_value("V37_EXPECTED_INITIAL_MODEL_SHA256")
train_path = path_value("V37_FRONTIER_DATA", "STEPCOUNT_V37_BALANCED_DATA")
validation_spec = path_value("STEPCOUNT_V37_VAL_DATA")
metadata_path = path_value("V37_STEPCOUNT_MASKS_METADATA")
masks_path = path_value("V37_STEPCOUNT_MASKS_DIR")
frontier_manifest_path = path_value("V37_FRONTIER_MANIFEST")
coverage_path = path_value("V37_METADATA_COVERAGE_REPORT")
filtered_path = path_value("V37_FILTERED_MANIFEST")
eval_producer_path = path_value("V37_AB_POSTPROCESS_PRODUCER") or str(
    repo / "tools/v37_eval_producer.py"
)
eval_recipe_path = path_value("V37_AB_EVAL_RECIPE", "V37_AB_POSTPROCESS_RECIPE")
paired_universe_path = path_value("V37_AB_PAIRED_UNIVERSE")
eval_producer_binding = file_binding(eval_producer_path)
eval_recipe_binding = file_binding(eval_recipe_path)
paired_universe_binding = file_binding(paired_universe_path)
if eval_recipe_binding is not None:
    try:
        gate.eval_producer.verify_recipe(eval_recipe_binding["path"])
    except (OSError, gate.eval_producer.EvalProducerError) as exc:
        raise SystemExit(f"[V37-AB][ERROR] invalid formal eval recipe: {exc}")

if lightweight:
    input_snapshots = None
    implementation = None
    missing = ["lightweight_plan_does_not_freeze_training_inputs"]
else:
    model_snapshot = content_binding(model_path)
    if (
        not isinstance(expected_model_sha256, str)
        or len(expected_model_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_model_sha256)
    ):
        raise SystemExit("[V37-AB][ERROR] V37_EXPECTED_INITIAL_MODEL_SHA256 must be a lowercase SHA256")
    if model_snapshot is not None and model_snapshot.get("sha256") != expected_model_sha256:
        raise SystemExit(
            "[V37-AB][ERROR] initial model differs from the frozen checkpoint-476 identity"
        )
    if expected_model_sha256 != gate.FORMAL_INITIAL_MODEL_SHA256:
        raise SystemExit("[V37-AB][ERROR] initial model authority constant mismatch")
    training_snapshot = content_binding(train_path, parquet_only=True)
    validation_snapshots = validation_bindings(validation_spec)
    metadata = file_binding(metadata_path)
    frontier_manifest = file_binding(frontier_manifest_path)
    coverage = file_binding(coverage_path)
    filtered = file_binding(filtered_path)
    mask_tree = None
    if masks_path:
        try:
            mask_tree = {
                "path": str(Path(masks_path).resolve()),
                "sha256": gate.tree_sha256(Path(masks_path)),
            }
        except (OSError, ValueError, gate.GateError):
            pass
    input_snapshots = {
        "model": model_snapshot,
        "training_data": training_snapshot,
        "validation_data": validation_snapshots,
        "mask_metadata": metadata,
        "mask_tree": mask_tree,
        "frontier_manifest": frontier_manifest,
        "coverage_report": coverage,
        "filtered_manifest": filtered,
    }
    missing = sorted(name for name, value in input_snapshots.items() if value is None)
    if eval_producer_binding is None:
        missing.append("eval_producer")
    if eval_recipe_binding is None:
        missing.append("eval_recipe")
    if paired_universe_binding is None:
        missing.append("paired_sample_universe")
    missing = sorted(missing)
    implementation = {
        name: gate.sha256(repo / relative)
        for name, relative in gate.IMPLEMENTATION_PATHS.items()
    }

try:
    remap = json.dumps(
        dict(path_remap_module.parse_image_path_remap(os.environ.get("STEPCOUNT_IMAGE_PATH_REMAP_JSON"))),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) if os.environ.get("STEPCOUNT_IMAGE_PATH_REMAP_JSON") else None
except ValueError as exc:
    raise SystemExit(f"[V37-AB][ERROR] invalid STEPCOUNT_IMAGE_PATH_REMAP_JSON: {exc}")

preregistered = {
    "schema_version": 1,
    "complete": not missing,
    "missing": missing,
    "input_snapshots": input_snapshots,
    "implementation_sha256": implementation,
    "image_path_remap": remap,
    "eval_contract": {
        "paired_metrics": gate.FROZEN_PAIRED_METRICS,
        "benchmark_dataset_sha256": gate.FORMAL_BENCHMARK_DATASET_SHA256,
        "benchmark_ordered_ids_sha256": gate.FORMAL_BENCHMARK_ORDERED_IDS_SHA256,
        "benchmark_thresholds": {
            "pixmo_correct_min": gate.PIXMO_CORRECT_MIN,
            "pixmo_total": gate.PIXMO_TOTAL,
            "stepcount_correct_min": gate.STEPCOUNT_CORRECT_MIN,
            "stepcount_total": gate.STEPCOUNT_TOTAL,
        },
        "pipeline": {
            "producer": eval_producer_binding,
            "recipe": eval_recipe_binding,
        },
        "paired_sample_universe": paired_universe_binding,
    },
}
if missing and not lightweight and not os.environ.get("V37_AB_SINGLE_RUN_WRAPPER"):
    raise SystemExit(
        "[V37-AB][ERROR] formal preregistration inputs are missing or unreadable: "
        + ",".join(missing)
    )
run_id = f"v37-ab-{secrets.token_hex(16)}"
cells = []
for seed in seeds:
    for arm in ("baseline", "progress"):
        cell_id = f"seed{seed}-{arm}"
        cells.append({
            "cell_id": cell_id,
            "seed": seed,
            "arm": arm,
            "target_dir": str((root / cell_id).absolute()),
        })
plan = {
    "schema_version": 2,
    "ab_run_id": run_id,
    "plan_path": str(plan_path.absolute()),
    "seeds": seeds,
    "execution": {
        "run_class": "formal",
        "data_mode": "frontier_rl",
        "optimizer_steps": 12,
        "nnodes": 1,
        "gpus_per_node": 8,
        "world_size": 8,
        "scheduling": "sequential",
    },
    "preregistered": preregistered,
    "cells": cells,
}
encoded = (json.dumps(
    plan, ensure_ascii=False, indent=2, allow_nan=False,
) + "\n").encode("utf-8")
temporary = plan_path.with_name(f".{plan_path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
try:
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, plan_path)
finally:
    temporary.unlink(missing_ok=True)
print(run_id)
print(hashlib.sha256(encoded).hexdigest())
PY
)"
mapfile -t _PLAN_META_LINES <<< "${PLAN_META}"
[[ "${#_PLAN_META_LINES[@]}" == "2" ]] || _v37_ab_error "failed to create plan metadata"
AB_RUN_ID="${_PLAN_META_LINES[0]}"
PLAN_SHA256="${_PLAN_META_LINES[1]}"

CELL_IDS=("seed${SEED_A}-baseline" "seed${SEED_A}-progress" "seed${SEED_B}-baseline" "seed${SEED_B}-progress")
CELL_SEEDS=("${SEED_A}" "${SEED_A}" "${SEED_B}" "${SEED_B}")
CELL_ARMS=(baseline progress baseline progress)
CELL_DIRS=(
  "${AB_ROOT}/${CELL_IDS[0]}"
  "${AB_ROOT}/${CELL_IDS[1]}"
  "${AB_ROOT}/${CELL_IDS[2]}"
  "${AB_ROOT}/${CELL_IDS[3]}"
)

_v37_ab_validate_plan() {
  "${PYTHON_BIN}" - "${PLAN_PATH}" "${PLAN_SHA256}" "${AB_ROOT}" "${AB_RUN_ID}" "${SEED_A}" "${SEED_B}" "${REPO_DIR}" <<'PY'
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path

plan_path, expected_hash, root, run_id = sys.argv[1:5]
seeds = [int(sys.argv[5]), int(sys.argv[6])]
repo = Path(sys.argv[7]).resolve()
sys.path.insert(0, str(repo))
from tools import v37_gate as gate

def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

try:
    metadata = os.lstat(plan_path)
except OSError as exc:
    raise SystemExit(f"[V37-AB][ERROR] cannot inspect ab_plan.json: {exc}")
if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("[V37-AB][ERROR] ab_plan.json must be a non-symlink regular file")
try:
    raw = Path(plan_path).read_bytes()
except OSError as exc:
    raise SystemExit(f"[V37-AB][ERROR] cannot read ab_plan.json: {exc}")
if hashlib.sha256(raw).hexdigest() != expected_hash:
    raise SystemExit("[V37-AB][ERROR] ab_plan.json hash changed")
try:
    plan = json.loads(
        raw,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {value}")),
    )
except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
    raise SystemExit(f"[V37-AB][ERROR] invalid strict ab_plan.json: {exc}")
if not isinstance(plan, dict) or set(plan) != {
    "schema_version", "ab_run_id", "plan_path", "seeds", "execution", "preregistered", "cells",
}:
    raise SystemExit("[V37-AB][ERROR] ab_plan.json exact schema mismatch")
if type(plan["schema_version"]) is not int or plan["schema_version"] != 2:
    raise SystemExit("[V37-AB][ERROR] ab_plan.json schema_version mismatch")
if plan["ab_run_id"] != run_id or not re.fullmatch(r"v37-ab-[0-9a-f]{32}", run_id):
    raise SystemExit("[V37-AB][ERROR] ab_run_id mismatch")
if plan["plan_path"] != str(Path(plan_path).absolute()) or not Path(plan["plan_path"]).is_absolute():
    raise SystemExit("[V37-AB][ERROR] plan_path must be absolute")
if plan["seeds"] != seeds or any(type(seed) is not int or seed < 0 for seed in plan["seeds"]):
    raise SystemExit("[V37-AB][ERROR] seed preregistration mismatch")
expected_execution = {
    "run_class": "formal", "data_mode": "frontier_rl", "optimizer_steps": 12,
    "nnodes": 1, "gpus_per_node": 8, "world_size": 8, "scheduling": "sequential",
}
if plan["execution"] != expected_execution:
    raise SystemExit("[V37-AB][ERROR] execution contract mismatch")
preregistered = plan["preregistered"]
expected_preregistered_keys = {
    "schema_version", "complete", "missing", "input_snapshots", "implementation_sha256",
    "image_path_remap", "eval_contract",
}
if not isinstance(preregistered, dict) or set(preregistered) != expected_preregistered_keys:
    raise SystemExit("[V37-AB][ERROR] preregistration exact schema mismatch")
if preregistered["schema_version"] != 1 or type(preregistered["complete"]) is not bool:
    raise SystemExit("[V37-AB][ERROR] preregistration version/completeness mismatch")
if not isinstance(preregistered["missing"], list) or any(
    not isinstance(item, str) or not item for item in preregistered["missing"]
):
    raise SystemExit("[V37-AB][ERROR] preregistration missing list is invalid")
expected_eval_contract = {
    "paired_metrics": gate.FROZEN_PAIRED_METRICS,
    "benchmark_dataset_sha256": gate.FORMAL_BENCHMARK_DATASET_SHA256,
    "benchmark_ordered_ids_sha256": gate.FORMAL_BENCHMARK_ORDERED_IDS_SHA256,
    "benchmark_thresholds": {
        "pixmo_correct_min": gate.PIXMO_CORRECT_MIN,
        "pixmo_total": gate.PIXMO_TOTAL,
        "stepcount_correct_min": gate.STEPCOUNT_CORRECT_MIN,
        "stepcount_total": gate.STEPCOUNT_TOTAL,
    },
}
eval_contract = preregistered["eval_contract"]
if not isinstance(eval_contract, dict) or set(eval_contract) != set(expected_eval_contract) | {"pipeline", "paired_sample_universe"}:
    raise SystemExit("[V37-AB][ERROR] preregistered eval contract schema changed")
if {key: eval_contract[key] for key in expected_eval_contract} != expected_eval_contract:
    raise SystemExit("[V37-AB][ERROR] preregistered eval contract changed")
pipeline = eval_contract["pipeline"]
if not isinstance(pipeline, dict) or set(pipeline) != {"producer", "recipe"}:
    raise SystemExit("[V37-AB][ERROR] preregistered eval pipeline schema changed")
universe = eval_contract["paired_sample_universe"]
if preregistered["complete"]:
    if not isinstance(universe, dict) or set(universe) != {"path", "sha256"}:
        raise SystemExit("[V37-AB][ERROR] preregistered paired sample universe schema changed")
elif universe is not None and (not isinstance(universe, dict) or set(universe) != {"path", "sha256"}):
    raise SystemExit("[V37-AB][ERROR] incomplete preregistration has an invalid sample universe")
path_remap_spec = importlib.util.spec_from_file_location(
    "v37_ab_validate_path_remap", repo / "verl/utils/path_remap.py",
)
if path_remap_spec is None or path_remap_spec.loader is None:
    raise SystemExit("[V37-AB][ERROR] cannot reload path remap implementation")
path_remap_module = importlib.util.module_from_spec(path_remap_spec)
path_remap_spec.loader.exec_module(path_remap_module)
current_remap = (
    json.dumps(
        dict(path_remap_module.parse_image_path_remap(os.environ.get("STEPCOUNT_IMAGE_PATH_REMAP_JSON"))),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    if os.environ.get("STEPCOUNT_IMAGE_PATH_REMAP_JSON") else None
)
if current_remap != preregistered["image_path_remap"]:
    raise SystemExit("[V37-AB][ERROR] image path remap changed after preregistration")
if preregistered["complete"]:
    if preregistered["missing"]:
        raise SystemExit("[V37-AB][ERROR] complete preregistration has missing inputs")
    try:
        gate.validate_ab_plan(plan_path, expected_hash, run_id, expected_seeds=seeds)
    except (OSError, ValueError, gate.GateError) as exc:
        raise SystemExit(f"[V37-AB][ERROR] shared formal plan validation failed: {exc}")
    snapshots = preregistered["input_snapshots"]
    expected_snapshot_keys = {
        "model", "training_data", "validation_data", "mask_metadata", "mask_tree",
        "frontier_manifest", "coverage_report", "filtered_manifest",
    }
    if not isinstance(snapshots, dict) or set(snapshots) != expected_snapshot_keys:
        raise SystemExit("[V37-AB][ERROR] preregistered input snapshot schema mismatch")
    try:
        if gate.content_snapshot(Path(snapshots["model"]["root"])) != snapshots["model"]:
            raise ValueError("model")
        if gate.content_snapshot(Path(snapshots["training_data"]["root"]), parquet_only=True) != snapshots["training_data"]:
            raise ValueError("training_data")
        for item in snapshots["validation_data"]:
            if gate.content_snapshot(Path(item["dataset"]["root"]), parquet_only=True) != item["dataset"]:
                raise ValueError("validation_data")
        for name in ("mask_metadata", "frontier_manifest", "coverage_report", "filtered_manifest"):
            binding = snapshots[name]
            if gate.sha256(Path(binding["path"])) != binding["sha256"]:
                raise ValueError(name)
        if gate.tree_sha256(Path(snapshots["mask_tree"]["path"])) != snapshots["mask_tree"]["sha256"]:
            raise ValueError("mask_tree")
    except (KeyError, TypeError, OSError, ValueError, gate.GateError) as exc:
        raise SystemExit(f"[V37-AB][ERROR] preregistered input changed or is invalid: {exc}")
    implementation = preregistered["implementation_sha256"]
    if not isinstance(implementation, dict) or set(implementation) != set(gate.IMPLEMENTATION_PATHS):
        raise SystemExit("[V37-AB][ERROR] preregistered implementation schema mismatch")
    for name, relative in gate.IMPLEMENTATION_PATHS.items():
        if gate.sha256(repo / relative) != implementation[name]:
            raise SystemExit(f"[V37-AB][ERROR] preregistered implementation changed: {name}")
else:
    if not preregistered["missing"] or preregistered["input_snapshots"] is not None or preregistered["implementation_sha256"] is not None:
        raise SystemExit("[V37-AB][ERROR] incomplete preregistration is not a lightweight plan")
expected_cells = []
for seed in seeds:
    for arm in ("baseline", "progress"):
        cell_id = f"seed{seed}-{arm}"
        expected_cells.append({
            "cell_id": cell_id, "seed": seed, "arm": arm,
            "target_dir": str((Path(root) / cell_id).absolute()),
        })
cells = plan["cells"]
if not isinstance(cells, list) or len(cells) != 4:
    raise SystemExit("[V37-AB][ERROR] plan must contain exactly four cells")
if any(not isinstance(cell, dict) or set(cell) != {"cell_id", "seed", "arm", "target_dir"} for cell in cells):
    raise SystemExit("[V37-AB][ERROR] cell exact schema mismatch")
identities = [(cell["seed"], cell["arm"]) for cell in cells]
cell_ids = [cell["cell_id"] for cell in cells]
target_dirs = [cell["target_dir"] for cell in cells]
if len(set(identities)) != 4 or len(set(cell_ids)) != 4 or len(set(target_dirs)) != 4:
    raise SystemExit("[V37-AB][ERROR] duplicate cell identity or target directory")
if cells != expected_cells:
    raise SystemExit("[V37-AB][ERROR] missing, extra, or out-of-order A/B cell")
if any(not Path(path).is_absolute() for path in target_dirs):
    raise SystemExit("[V37-AB][ERROR] every target_dir must be absolute")
PY
}

_v37_ab_validate_plan
if [[ "${V37_AB_PLAN_ONLY:-0}" == "1" ]]; then
  echo "[V37-AB] plan=${PLAN_PATH} sha256=${PLAN_SHA256}"
  exit 0
fi

WRAPPER="${V37_AB_SINGLE_RUN_WRAPPER:-${REPO_DIR}/examples/v37_strict_winner_step_rl_pilot.sh}"
WRAPPER="$("${PYTHON_BIN}" - "${WRAPPER}" <<'PY'
import os
import sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)"
[[ -f "${WRAPPER}" && ! -L "${WRAPPER}" && -x "${WRAPPER}" ]] \
  || _v37_ab_error "single-run wrapper must be an executable non-symlink file: ${WRAPPER}"

_ENV_UNSETS=(
  -u V37_AB_ROOT -u V37_AB_SEEDS -u V37_AB_PLAN_ONLY -u V37_AB_CONTRACT_ONLY
  -u V37_AB_SINGLE_RUN_WRAPPER -u V37_AB_PYTHON
  -u V37_AB_POSTPROCESS -u V37_AB_POSTPROCESS_PRODUCER
  -u V37_AB_POSTPROCESS_RECIPE -u V37_AB_EVAL_RECIPE -u V37_AB_PAIRED_UNIVERSE -u V37_AB_POSTPROCESS_OUTPUT
  -u V37_AB_RUN_ID -u V37_AB_PLAN_PATH -u V37_AB_PLAN_SHA256 -u V37_AB_CELL_ID
  -u WANDB_API_KEY -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN
  -u V37_GATE_ONLY -u V37_GATE_INPUT -u V37_GATE_JSON_OUT
  -u V37_CONTRACT_ONLY -u V37_PREFLIGHT_ONLY -u V37_DRY_RUN -u V37_VALIDATE_DOWNSTREAM
  -u V37_CONTINUATION_MODE -u V37_RESUME_MODE -u V37_RESUME_CHECKPOINT -u V37_RESUME_SEAL
  -u V37_EXPECTED_RESUME_CHECKPOINT_PATH -u V37_EXPECTED_RESUME_CHECKPOINT_SHA256
  -u _V37_RESUME_CHECKPOINT -u _V37_RESUME_CHECKPOINT_SHA256 -u _V37_RESUME_SEAL_SHA256
  -u V36_LOAD_CHECKPOINT_PATH -u V32_LOAD_CHECKPOINT_PATH -u V36_ALLOW_V32_LOAD_CHECKPOINT
  -u V32_DRY_RUN -u V37_ALLOW_FOCUSED10K_PILOT -u V37_ALLOW_BENCHMARK_DEV
  -u TRAINER_VAL_ONLY -u V37_RUN_MANIFEST
  -u V37_EFFECTIVE_ENVIRONMENT_PATH
  -u V37_TRAINING_EVIDENCE_PATH -u V37_TRAINING_EVIDENCE_REQUIRED
  -u V37_RUN_CLASS -u V37_DATA_MODE -u V37_RUN_PURPOSE -u V37_PILOT_STEPS
  -u V37_ARM -u V37_SEED -u V31_SAVE_CHECKPOINT_PATH
  -u V31_NNODES -u V31_N_GPUS_PER_NODE -u HOST_NUM -u HOST_GPU_NUM -u INDEX
  -u RAY_ADDRESS -u RAY_ADDRESS_CANDIDATES
  -u V31_EXPERIMENT_NAME -u V37_RUN_TIMESTAMP
  -u GIT_DIR -u GIT_WORK_TREE -u GIT_COMMON_DIR -u GIT_INDEX_FILE
  -u GIT_OBJECT_DIRECTORY -u GIT_ALTERNATE_OBJECT_DIRECTORIES
  -u GIT_CEILING_DIRECTORIES -u GIT_DISCOVERY_ACROSS_FILESYSTEM
  -u GIT_EXTERNAL_DIFF -u GIT_DIFF_OPTS
)

for _cell_index in 0 1 2 3; do
  _cell_id="${CELL_IDS[${_cell_index}]}"
  _seed="${CELL_SEEDS[${_cell_index}]}"
  _arm="${CELL_ARMS[${_cell_index}]}"
  _target_dir="${CELL_DIRS[${_cell_index}]}"
  _contract_only=0
  [[ "${V37_AB_CONTRACT_ONLY:-0}" == "0" ]] || _contract_only=1
  _v37_ab_validate_plan
  [[ ! -e "${_target_dir}" && ! -L "${_target_dir}" ]] \
    || _v37_ab_error "cell target directory must not already exist or be a symlink: ${_target_dir}"
  _child_command=(
    env -u BASH_ENV -u ENV "${_ENV_UNSETS[@]}"
    V37_RUN_CLASS=formal
    V37_DATA_MODE=frontier_rl
    V37_RUN_PURPOSE=formal_ab
    V37_CONTINUATION_MODE=0
    V37_CONTRACT_ONLY="${_contract_only}"
    V37_PILOT_STEPS=12
    V31_NNODES=1
    V31_N_GPUS_PER_NODE=8
    HOST_NUM=1
    HOST_GPU_NUM=8
    INDEX=0
    V37_AB_RUN_ID="${AB_RUN_ID}"
    V37_AB_PLAN_PATH="${PLAN_PATH}"
    V37_AB_PLAN_SHA256="${PLAN_SHA256}"
    V37_AB_CELL_ID="${_cell_id}"
    V37_ARM="${_arm}"
    V37_SEED="${_seed}"
    V31_SAVE_CHECKPOINT_PATH="${_target_dir}"
    "${WRAPPER}"
  )
  echo "[V37-AB] cell=${_cell_id} arm=${_arm} seed=${_seed}"
  if ! "${_child_command[@]}"; then
    _v37_ab_error "child failed: ${_cell_id}"
  fi
done

_v37_ab_validate_plan
if [[ "${V37_AB_CONTRACT_ONLY:-0}" == "1" ]]; then
  echo "[V37-AB] contract-only completed; no manifests or index required"
  exit 0
fi

"${PYTHON_BIN}" - "${PLAN_PATH}" "${PLAN_SHA256}" "${INDEX_PATH}" \
  "${AB_RUN_ID}" "${SEED_A}" "${SEED_B}" "${CELL_DIRS[@]}" <<'PY'
import hashlib
import json
import os
import secrets
import stat
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
plan_hash = sys.argv[2]
index_path = Path(sys.argv[3])
run_id = sys.argv[4]
seeds = [int(sys.argv[5]), int(sys.argv[6])]
target_dirs = [Path(value) for value in sys.argv[7:]]

def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def strict_load(path):
    try:
        return json.loads(
            path.read_bytes(), object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {value}")),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"[V37-AB][ERROR] invalid strict JSON {path}: {exc}")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if sha256(plan_path) != plan_hash:
    raise SystemExit("[V37-AB][ERROR] ab_plan.json was tampered after launch")

expected = [(seed, arm) for seed in seeds for arm in ("baseline", "progress")]
manifest_records = []
evidence_ids = set()
for target_dir, (seed, arm) in zip(target_dirs, expected, strict=True):
    manifest_path = target_dir / "v37_run_manifest.json"
    try:
        if stat.S_ISLNK(os.lstat(target_dir).st_mode) or not target_dir.is_dir():
            raise SystemExit(f"[V37-AB][ERROR] target directory is missing or a symlink: {target_dir}")
        metadata = os.lstat(manifest_path)
    except FileNotFoundError:
        raise SystemExit(f"[V37-AB][ERROR] missing run manifest: {manifest_path}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"[V37-AB][ERROR] run manifest must be a non-symlink regular file: {manifest_path}")
    manifest = strict_load(manifest_path)
    if not isinstance(manifest, dict):
        raise SystemExit(f"[V37-AB][ERROR] run manifest must be an object: {manifest_path}")
    if type(manifest.get("manifest_version")) is not int or manifest["manifest_version"] != 3:
        raise SystemExit(f"[V37-AB][ERROR] manifest_version must be 3: {manifest_path}")
    if manifest.get("run_class") != "formal" or manifest.get("data_mode") != "frontier_rl":
        raise SystemExit(f"[V37-AB][ERROR] manifest is not formal/frontier_rl: {manifest_path}")
    if manifest.get("arm") != arm or type(manifest.get("seed")) is not int or manifest["seed"] != seed:
        raise SystemExit(f"[V37-AB][ERROR] manifest arm/seed identity mismatch: {manifest_path}")
    config = manifest.get("config")
    if not isinstance(config, dict) or type(config.get("world_size")) is not int or config["world_size"] != 8:
        raise SystemExit(f"[V37-AB][ERROR] manifest config.world_size must be 8: {manifest_path}")
    cell_id = f"seed{seed}-{arm}"
    audited = manifest.get("audited_environment")
    expected_linkage = {
        "V37_AB_RUN_ID": run_id,
        "V37_AB_PLAN_PATH": str(plan_path.absolute()),
        "V37_AB_PLAN_SHA256": plan_hash,
        "V37_AB_CELL_ID": cell_id,
        "V37_ARM": arm,
        "V37_SEED": str(seed),
        "V31_SAVE_CHECKPOINT_PATH": str(target_dir.absolute()),
        "V37_RUN_CLASS": "formal",
        "V37_DATA_MODE": "frontier_rl",
        "V37_PILOT_STEPS": "12",
        "V31_NNODES": "1",
        "V31_N_GPUS_PER_NODE": "8",
        "V37_CONTINUATION_MODE": "0",
    }
    if not isinstance(audited, dict) or any(audited.get(key) != value for key, value in expected_linkage.items()):
        raise SystemExit(f"[V37-AB][ERROR] audited_environment A/B linkage mismatch: {manifest_path}")
    preregistration = manifest.get("ab_preregistration")
    expected_preregistration = {
        "schema_version": 1,
        "ab_run_id": run_id,
        "plan_path": str(plan_path.absolute()),
        "plan_sha256": plan_hash,
        "cell_id": cell_id,
    }
    if preregistration != expected_preregistration:
        raise SystemExit(f"[V37-AB][ERROR] manifest A/B preregistration mismatch: {manifest_path}")
    evidence_id = manifest.get("evidence_run_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise SystemExit(f"[V37-AB][ERROR] manifest evidence_run_id is missing: {manifest_path}")
    if evidence_id in evidence_ids:
        raise SystemExit(f"[V37-AB][ERROR] duplicate manifest evidence_run_id: {evidence_id}")
    evidence_ids.add(evidence_id)
    manifest_records.append({"path": str(manifest_path.absolute()), "sha256": sha256(manifest_path)})

index = {
    "plan": {"path": str(plan_path.absolute()), "sha256": plan_hash},
    "manifests": manifest_records,
}
encoded = (json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
temporary = index_path.with_name(f".{index_path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
try:
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, index_path)
finally:
    temporary.unlink(missing_ok=True)
PY

_INDEX_PUBLISHED=1
echo "[V37-AB] evidence_index=${INDEX_PATH}"

if [[ "${V37_AB_POSTPROCESS:-0}" == "1" ]]; then
  _postprocess_producer="${V37_AB_POSTPROCESS_PRODUCER:-${REPO_DIR}/tools/v37_eval_producer.py}"
  _postprocess_recipe="${V37_AB_EVAL_RECIPE:-${V37_AB_POSTPROCESS_RECIPE:-}}"
  [[ -n "${_postprocess_recipe}" ]] \
    || _v37_ab_error "V37_AB_POSTPROCESS=1 requires V37_AB_EVAL_RECIPE"
  _postprocess_output="${V37_AB_POSTPROCESS_OUTPUT:-${AB_ROOT}/v37_final_evidence}"
  _postprocess_args=(
    "${REPO_DIR}/tools/v37_postprocess.py"
    --ab-index "${INDEX_PATH}"
    --output-dir "${_postprocess_output}"
    --producer "${_postprocess_producer}"
    --recipe "${_postprocess_recipe}"
  )
  "${PYTHON_BIN}" "${_postprocess_args[@]}"
  echo "[V37-AB] final_evidence=${_postprocess_output}"
fi
