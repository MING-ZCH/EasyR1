#!/bin/bash
# V37 frontier-RL debug/canary/formal wrapper around the shared V36 training entrypoint.
set -euo pipefail

if [[ -n "${BASH_ENV:-}" || -n "${ENV:-}" ]]; then
  echo "[V37-strict-winner][ERROR] BASH_ENV/ENV are forbidden; launch through the sanitized V37 entrypoint" >&2
  exit 1
fi
if env | awk -F= '$1 ~ /^BASH_FUNC_/ { found=1 } END { exit(found ? 0 : 1) }'; then
  echo "[V37-strict-winner][ERROR] exported shell functions are forbidden for V37" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
_V37_GIT_BIN="$(command -v git)" || {
  echo "[V37-strict-winner][ERROR] git executable is required" >&2
  exit 1
}

# Repository identity must come from REPO_DIR, never from ambient GIT_* path,
# object-store, index, or diff overrides inherited from an outer shell.
_v37_git() {
  env -i PATH="${PATH}" HOME="${HOME:-/nonexistent}" LC_ALL=C \
    "${_V37_GIT_BIN}" -C "${REPO_DIR}" "$@"
}

_v37_error() {
  echo "[V37-strict-winner][ERROR] $*" >&2
  exit 1
}

_v37_is_true() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

_v37_path_is_within() {
  local candidate="$1"
  local root="$2"
  [[ "${candidate}" == "${root}" || "${candidate}" == "${root}/"* ]]
}

_v37_paths_overlap() {
  _v37_path_is_within "$1" "$2" || _v37_path_is_within "$2" "$1"
}

_v37_absolute_path() {
  readlink -m -- "$1"
}

_v37_require_file() {
  local label="$1"
  local path="$2"
  [[ -f "${path}" ]] || _v37_error "${label} must name an existing file: ${path}"
}

_v37_sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

_v37_sha256_tree() {
  "${V37_PREFLIGHT_PYTHON:-python3}" - "$1" <<'PY'
import hashlib
import os
import stat
import sys
from pathlib import Path

target = Path(os.path.abspath(os.path.expanduser(sys.argv[1])))
current = Path(target.anchor)
for part in target.parts[1:]:
    current /= part
    try:
        metadata = os.lstat(current)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise SystemExit(f"cannot hash missing input: {current}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise SystemExit(f"cannot hash symlink input: {current}")
if target.is_dir():
    for root, directories, filenames in os.walk(target, followlinks=False):
        for name in directories + filenames:
            candidate = Path(root) / name
            if candidate.is_symlink():
                raise SystemExit(f"cannot hash tree containing symlink: {candidate}")
files = [target] if target.is_file() else sorted(item for item in target.rglob("*") if item.is_file())
if not files:
    raise SystemExit(f"cannot hash missing/empty input: {target}")
def file_hash(item):
    digest = hashlib.sha256()
    with item.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
if target.is_file():
    print(file_hash(target))
else:
    import json
    entries = [
        {"path": item.relative_to(target).as_posix(), "size": item.stat().st_size, "sha256": file_hash(item)}
        for item in files
    ]
    encoded = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    print(hashlib.sha256(encoded).hexdigest())
PY
}

_V37_PROMOTABLE=1
_V37_NONPROMOTABLE_REASONS=()

_v37_mark_nonpromotable() {
  local reason="$1"
  local existing
  _V37_PROMOTABLE=0
  for existing in "${_V37_NONPROMOTABLE_REASONS[@]}"; do
    [[ "${existing}" == "${reason}" ]] && return 0
  done
  _V37_NONPROMOTABLE_REASONS+=("${reason}")
}

# The final gate is a separate, post-training operation over all four A/B runs.
if _v37_is_true "${V37_GATE_ONLY:-0}"; then
  [[ -n "${V37_GATE_INPUT:-}" ]] \
    || _v37_error "V37_GATE_ONLY=1 requires V37_GATE_INPUT"
  _v37_require_file "V37_GATE_INPUT" "${V37_GATE_INPUT}"
  _V37_GATE_PYTHON="$(readlink -f -- "$(command -v python3)")"
  [[ -x "${_V37_GATE_PYTHON}" && -f "${_V37_GATE_PYTHON}" ]] \
    || _v37_error "a real python3 executable is required for V37_GATE_ONLY"
  _v37_gate_args=("${REPO_DIR}/tools/v37_gate.py" "${V37_GATE_INPUT}")
  [[ -z "${V37_GATE_JSON_OUT:-}" ]] || _v37_gate_args+=(--json-out "${V37_GATE_JSON_OUT}")
  "${_V37_GATE_PYTHON}" "${_v37_gate_args[@]}"
  if [[ -n "${V37_GATE_JSON_OUT:-}" ]]; then
    [[ -s "${V37_GATE_JSON_OUT}" ]] \
      || _v37_error "gate returned without publishing V37_GATE_JSON_OUT"
    "${_V37_GATE_PYTHON}" - "${V37_GATE_JSON_OUT}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("decision") not in {"GO", "NO-GO"}:
    raise SystemExit("invalid V37 gate result decision")
PY
  fi
  exit 0
fi

source "${SCRIPT_DIR}/local_path_env.sh"
_V37_FORMAL_INITIAL_MODEL_SHA256=f9e6b1e8031bdbc509d34249745cdcf75af85c320d6b88918444b1abb4f580a3
if [[ -n "${V37_EXPECTED_INITIAL_MODEL_SHA256:-}" \
      && "${V37_EXPECTED_INITIAL_MODEL_SHA256}" != "${_V37_FORMAL_INITIAL_MODEL_SHA256}" ]]; then
  _v37_error "V37_EXPECTED_INITIAL_MODEL_SHA256 cannot override checkpoint-476 identity"
fi
export V37_EXPECTED_INITIAL_MODEL_SHA256="${_V37_FORMAL_INITIAL_MODEL_SHA256}"

if [[ -n "${STEPCOUNT_IMAGE_PATH_REMAP_JSON:-}" ]]; then
  _V37_IMAGE_PATH_REMAP_RAW="${STEPCOUNT_IMAGE_PATH_REMAP_JSON}"
  _V37_CANONICAL_IMAGE_PATH_REMAP="$(${V37_PREFLIGHT_PYTHON:-python3} - "${REPO_DIR}" "${_V37_IMAGE_PATH_REMAP_RAW}" <<'PY'
import importlib.util
import json
import sys

module_path = f"{sys.argv[1]}/verl/utils/path_remap.py"
spec = importlib.util.spec_from_file_location("v37_path_remap", module_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load path remap implementation: {module_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

print(json.dumps(dict(module.parse_image_path_remap(sys.argv[2])), sort_keys=True, separators=(",", ":")))
PY
)" || _v37_error "invalid STEPCOUNT_IMAGE_PATH_REMAP_JSON"
  export STEPCOUNT_IMAGE_PATH_REMAP_JSON="${_V37_CANONICAL_IMAGE_PATH_REMAP}"
  unset _V37_IMAGE_PATH_REMAP_RAW _V37_CANONICAL_IMAGE_PATH_REMAP
fi

export V37_RUN_CLASS=${V37_RUN_CLASS:-}
export V37_DATA_MODE=${V37_DATA_MODE:-}
export V37_ARM=${V37_ARM:-}
export V37_SEED=${V37_SEED:-42}
if ! [[ "${V37_SEED}" =~ ^[0-9]+$ ]]; then
  _v37_error "V37_SEED must be a non-negative integer, got: ${V37_SEED}"
fi
case "${V37_RUN_CLASS}" in
  debug) _v37_mark_nonpromotable "debug_run_class" ;;
  canary) _v37_mark_nonpromotable "canary_requires_formal_gate" ;;
  formal) ;;
  *) _v37_error "V37_RUN_CLASS must be explicit: debug, canary, or formal" ;;
esac
if [[ "${V37_RUN_CLASS}" == "formal" ]]; then
  if [[ -n "${RAY_GPU_WAIT_TIMEOUT_SECONDS:-}" && "${RAY_GPU_WAIT_TIMEOUT_SECONDS}" != "900" ]]; then
    _v37_error "formal RAY_GPU_WAIT_TIMEOUT_SECONDS is locked to 900, got: ${RAY_GPU_WAIT_TIMEOUT_SECONDS}"
  fi
  export RAY_GPU_WAIT_TIMEOUT_SECONDS=900
  if [[ -n "${RAY_STATUS_TIMEOUT_SECONDS:-}" && "${RAY_STATUS_TIMEOUT_SECONDS}" != "10" ]]; then
    _v37_error "formal RAY_STATUS_TIMEOUT_SECONDS is locked to 10, got: ${RAY_STATUS_TIMEOUT_SECONDS}"
  fi
  export RAY_STATUS_TIMEOUT_SECONDS=10
  if [[ -n "${RAY_START_TIMEOUT_SECONDS:-}" && "${RAY_START_TIMEOUT_SECONDS}" != "60" ]]; then
    _v37_error "formal RAY_START_TIMEOUT_SECONDS is locked to 60, got: ${RAY_START_TIMEOUT_SECONDS}"
  fi
  export RAY_START_TIMEOUT_SECONDS=60
  if [[ -n "${RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS:-}" \
        && "${RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS}" != "900" ]]; then
    _v37_error "formal RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS is locked to 900, got: ${RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS}"
  fi
  export RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS=900
  if [[ -n "${V37_HF_MERGE_HOST_MEMORY_PREFLIGHT:-}" \
        && "${V37_HF_MERGE_HOST_MEMORY_PREFLIGHT}" != "error" ]]; then
    _v37_error "formal V37_HF_MERGE_HOST_MEMORY_PREFLIGHT is locked to error"
  fi
  if [[ -n "${V37_HF_MERGE_HOST_MEMORY_SAFETY_FACTOR:-}" \
        && "${V37_HF_MERGE_HOST_MEMORY_SAFETY_FACTOR}" != "4.0" ]]; then
    _v37_error "formal V37_HF_MERGE_HOST_MEMORY_SAFETY_FACTOR is locked to 4.0"
  fi
  export V37_HF_MERGE_HOST_MEMORY_PREFLIGHT=error
  export V37_HF_MERGE_HOST_MEMORY_SAFETY_FACTOR=4.0
else
  export V37_HF_MERGE_HOST_MEMORY_PREFLIGHT=${V37_HF_MERGE_HOST_MEMORY_PREFLIGHT:-off}
  export V37_HF_MERGE_HOST_MEMORY_SAFETY_FACTOR=${V37_HF_MERGE_HOST_MEMORY_SAFETY_FACTOR:-2.0}
fi
export V37_CONTINUATION_MODE=${V37_CONTINUATION_MODE:-0}
if _v37_is_true "${V37_CONTINUATION_MODE}"; then
  [[ "${V37_RUN_CLASS}" != "formal" ]] \
    || _v37_error "controlled continuation is non-promotable and forbidden for formal A/B"
  [[ "${V37_RUN_PURPOSE:-}" == "controlled_continuation" ]] \
    || _v37_error "continuation requires explicit V37_RUN_PURPOSE=controlled_continuation"
  _v37_mark_nonpromotable "controlled_continuation"
  export V37_RUN_PURPOSE=controlled_continuation
  export V37_RESUME_MODE=controlled_continuation
else
  export V37_RESUME_MODE=clean_start
  case "${V37_RUN_CLASS}" in
    formal) _v37_expected_purpose=formal_ab ;;
    canary) _v37_expected_purpose=canary_diagnostic ;;
    debug) _v37_expected_purpose=debug_mechanism ;;
  esac
  if [[ -n "${V37_RUN_PURPOSE:-}" && "${V37_RUN_PURPOSE}" != "${_v37_expected_purpose}" ]]; then
    _v37_error "${V37_RUN_CLASS} clean start requires V37_RUN_PURPOSE=${_v37_expected_purpose}"
  fi
  export V37_RUN_PURPOSE="${_v37_expected_purpose}"
fi
case "${V37_DATA_MODE}" in
  frontier_rl) ;;
  strict_winner_rft)
    _v37_error "strict_winner_rft is winner-only SFT/RFT data and is incompatible with this online RL entrypoint"
    ;;
  *) _v37_error "V37_DATA_MODE must be explicit: frontier_rl or strict_winner_rft" ;;
esac
if [[ "${V37_RUN_CLASS}" == "formal" ]] && _v37_is_true "${V37_ALLOW_FOCUSED10K_PILOT:-0}"; then
  _v37_error "formal runs cannot enable the focused10k fallback"
fi
if [[ "${V37_RUN_CLASS}" != "debug" ]] && _v37_is_true "${V37_ALLOW_BENCHMARK_DEV:-0}"; then
  _v37_error "benchmark development validation is debug-only"
fi
case "${V37_ALLOW_INDEPENDENT_VAL_OUTSIDE_TRAIN_RANGE:-0}" in
  0|1) ;;
  *) _v37_error "V37_ALLOW_INDEPENDENT_VAL_OUTSIDE_TRAIN_RANGE must be exactly 0 or 1" ;;
esac
if [[ "${V37_ALLOW_INDEPENDENT_VAL_OUTSIDE_TRAIN_RANGE:-0}" == "1" ]]; then
  [[ "${V37_RUN_CLASS}" == "debug" ]] \
    || _v37_error "V37_ALLOW_INDEPENDENT_VAL_OUTSIDE_TRAIN_RANGE=1 is debug-only; canary/formal refuse it"
  _v37_mark_nonpromotable "independent_validation_outside_train_range"
fi

# Core adaptive actor KL currently rejects Ulysses SP; keep CP pluggable but closed.
export V37_CP_SIZE=${V37_CP_SIZE:-1}
if ! [[ "${V37_CP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  _v37_error "V37_CP_SIZE must be a positive integer, got: ${V37_CP_SIZE}"
fi
if (( V37_CP_SIZE > 1 )); then
  _v37_error "adaptive actor KL is not validated with Ulysses sequence parallelism; V37_CP_SIZE must remain 1"
fi

# Both arms share outcome semantics. Only progress receives native action credit.
export PROCESS_REWARD_ENABLE=0
export BOK_CORRECTNESS_FIRST=1
export BOK_ALLWRONG_TERMINAL_ZERO=1
export BOK_CORRECTNESS_TASK_WEIGHT=0.25
export BOK_CORRECTNESS_QUALITY_WEIGHT=0.1
export BOK_CORRECTNESS_PARTIAL_SCALE=0.25
export BOK_WINNER_BOOST=0
export BOK_QUALITY_BONUS=0
if [[ -n "${V37_STRICT_POINT_PARSER_CONTRACT:-}" && "${V37_STRICT_POINT_PARSER_CONTRACT}" != "1" ]]; then
  _v37_error "V37 requires V37_STRICT_POINT_PARSER_CONTRACT=1 in both A/B arms"
fi
export V37_STRICT_POINT_PARSER_CONTRACT=1
export ACTION_EVENT_LEDGER_ENABLE=1
export V37_ACTION_PARSER_CONTRACT=1
export V37_ACTION_LEDGER_CONTRACT=1
case "${V37_ARM}" in
  baseline)
    export STEPCOUNT_RL_MODE=bok_grpo
    export ADV_ESTIMATOR=bok_grpo
    export ACTION_EVENT_REWARD_ENABLE=0
    export BOK_STEP_WEIGHT=0
    unset BOK_STEP_SIGNAL BOK_STEP_GATE BOK_STEP_MIN_GATE
    ;;
  progress)
    export STEPCOUNT_RL_MODE=bok_grpo_step
    export ADV_ESTIMATOR=bok_grpo_step
    export ACTION_EVENT_REWARD_ENABLE=1
    export BOK_STEP_SIGNAL=native_action_event
    export BOK_STEP_WEIGHT=${V37_STEP_WEIGHT:-0.1}
    export BOK_STEP_GATE=${V37_STEP_GATE:-answer_soft}
    export BOK_STEP_MIN_GATE=${V37_STEP_MIN_GATE:-0.2}
    ;;
  *) _v37_error "V37_ARM must be explicit: baseline or progress" ;;
esac
[[ "${ACTION_EVENT_LEDGER_ENABLE}" == "1" \
    && "${V37_ACTION_PARSER_CONTRACT}" == "1" \
    && "${V37_ACTION_LEDGER_CONTRACT}" == "1" ]] \
  || _v37_error "both arms require the shared read-only native action ledger"
if _v37_is_true "${ACTION_EVENT_REWARD_ENABLE}"; then
  [[ "${ADV_ESTIMATOR}" == "bok_grpo_step" \
      && "${BOK_STEP_SIGNAL}" == "native_action_event" \
      && "${V37_ACTION_PARSER_CONTRACT:-0}" == "1" \
      && "${V37_ACTION_LEDGER_CONTRACT:-0}" == "1" ]] \
    || _v37_error "native ACTION estimator requires bok_grpo_step plus parser+ledger contracts"
fi

# Adaptive actor KL is the optimizer term; reward-side KL is excluded by core.
# Current core makes this mode mutually exclusive with legacy use_kl_loss.
export DISABLE_KL=false
export USE_KL_LOSS=false
export ADAPTIVE_ACTOR_KL=true
export KL_TYPE=adaptive
export KL_COEF=0.08
export KL_TARGET=0.15
if [[ -n "${KL_HORIZON:-}" && "${KL_HORIZON}" != "50" ]]; then
  _v37_error "V37 adaptive actor KL horizon is locked to 50 executed optimizer updates"
fi
if [[ -n "${KL_HORIZON_UNIT:-}" && "${KL_HORIZON_UNIT}" != "executed_optimizer_updates" ]]; then
  _v37_error "V37 KL_HORIZON_UNIT is locked to executed_optimizer_updates"
fi
export KL_HORIZON=50
export KL_HORIZON_UNIT=executed_optimizer_updates
export KL_PENALTY=low_var_kl
if [[ -n "${V37_FALLBACK_LOGPROB_SIGN_OPT_IN:-}" && "${V37_FALLBACK_LOGPROB_SIGN_OPT_IN}" != "0" ]]; then
  _v37_error "fallback logprob sign opt-in is forbidden for V37 A/B and continuation"
fi
export V37_FALLBACK_LOGPROB_SIGN_OPT_IN=0
if [[ -n "${TORCH_LOGPROB_FALLBACK_MODE:-}" && "${TORCH_LOGPROB_FALLBACK_MODE}" != "error" ]]; then
  _v37_error "V37 requires TORCH_LOGPROB_FALLBACK_MODE=error"
fi
# V37 requires the FlashAttention CE kernel. This preserves the V36 legacy
# default while preventing a silent positive-CE fallback in formal training.
export TORCH_LOGPROB_FALLBACK_MODE=error

# V37-only exact-answer contract. Keep V36's permissive parser untouched, but
# fail closed here so ambiguous answer prose cannot enter the winner channel.
if [[ -n "${TRAJ_STRICT_ANSWER_INTEGER_PARSE:-}" && "${TRAJ_STRICT_ANSWER_INTEGER_PARSE}" != "1" ]]; then
  _v37_error "V37 requires TRAJ_STRICT_ANSWER_INTEGER_PARSE=1"
fi
export TRAJ_STRICT_ANSWER_INTEGER_PARSE=1
if [[ -n "${V37_RAW_SUCCESS_STRICT_WINNER:-}" && "${V37_RAW_SUCCESS_STRICT_WINNER}" != "1" ]]; then
  _v37_error "V37 requires V37_RAW_SUCCESS_STRICT_WINNER=1"
fi
export V37_RAW_SUCCESS_STRICT_WINNER=1
if [[ -n "${V37_WINNER_MODE:-}" && "${V37_WINNER_MODE}" != "outcome_success" ]]; then
  _v37_error "V37 requires V37_WINNER_MODE=outcome_success"
fi
export V37_WINNER_MODE=outcome_success
if [[ -n "${V37_REWARD_FAIL_CLOSED:-}" && "${V37_REWARD_FAIL_CLOSED}" != "1" ]]; then
  _v37_error "V37 requires V37_REWARD_FAIL_CLOSED=1"
fi
export V37_REWARD_FAIL_CLOSED=1

_v37_validate_frontier_manifest() {
  local manifest_path="$1"
  local required_class="$2"
  "${V37_PREFLIGHT_PYTHON:-python3}" - "${manifest_path}" "${required_class}" <<'PY'
import json
import hashlib
import math
import re
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
run_class = sys.argv[2]
try:
    def no_duplicates(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"contains duplicate JSON key: {key}")
            result[key] = item
        return result
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates,
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"non-finite constant: {item}")),
    )
except (OSError, json.JSONDecodeError, ValueError) as exc:
    raise SystemExit(f"[V37-strict-winner][ERROR] invalid frontier manifest: {exc}")

def fail(message):
    raise SystemExit(f"[V37-strict-winner][ERROR] frontier manifest {message}")

def known_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) and set(value) != {"0"}

if value.get("schema_version") != 3 or value.get("data_mode") != "frontier_rl":
    fail("must be schema_version=3 and data_mode=frontier_rl")
if value.get("publication") != "atomic_directory_v1":
    fail("must be an atomic_directory_v1 publication")
if value.get("run_class") != run_class:
    fail(f"run_class must equal {run_class}")
seeds = value.get("seeds")
if not isinstance(seeds, list) or len(seeds) != 2 or len({str(item) for item in seeds}) != 2:
    fail("must certify two distinct mining seeds")
if value.get("candidates_per_seed") != 32:
    fail("must certify M=32 candidates for each of two seeds")
if not known_sha(value.get("source_sha256")):
    fail("source_sha256 is missing or unknown")
audit_hashes = value.get("audit_sha256")
if not isinstance(audit_hashes, dict) or len(audit_hashes) != 2 or not all(known_sha(item) for item in audit_hashes.values()):
    fail("must contain two known audit/miner SHA256 values")
requested = value.get("requested_sample_count")
selected = value.get("selected_sample_count")
if not isinstance(requested, int) or requested <= 0 or selected != requested:
    fail("selection must be complete and nonempty")
if value.get("candidate_fields_injected") is not False:
    fail("must preserve prompt-only RL schema")
thresholds = value.get("classifier_thresholds")
expected_thresholds = {
    "winner_min": 3, "winner_max": 20, "process_min_quality": .55,
    "process_min_quality_range": .10, "process_max_duplicate_rate": .25,
    "process_max_cap_rate": .25,
}
if thresholds != expected_thresholds:
    fail("classifier thresholds are missing or not frozen")
if value.get("outcome_ratio") != .75 or value.get("process_ratio") != .25:
    fail("outcome/process composition must be 75/25")
mining = value.get("mining_generation_contract")
if not isinstance(mining, dict) or set(mining) != {
    "model_checkpoint_path", "model_checkpoint_sha256", "generation_config_by_seed",
    "sampling_config", "seed_difference_allowlist", "source_binding_contract",
    "source_binding_sha256",
}:
    fail("mining generation contract has an incomplete schema")
if (
    not isinstance(mining.get("model_checkpoint_path"), str)
    or not Path(mining["model_checkpoint_path"]).is_absolute()
    or not known_sha(mining.get("model_checkpoint_sha256"))
    or mining.get("source_binding_contract") != "source_row_image_rendered_prompt_v1"
    or not known_sha(mining.get("source_binding_sha256"))
):
    fail("mining model/source binding provenance is incomplete")
if mining.get("seed_difference_allowlist") != ["seed"]:
    fail("mining configs may differ only in the declared seed field")
generation = mining.get("generation_config_by_seed")
if not isinstance(generation, dict) or set(generation) != {str(item) for item in seeds}:
    fail("mining generation configs must cover both seeds")
normalized = []
for seed in seeds:
    config = generation[str(seed)]
    if not isinstance(config, dict) or str(config.get("seed")) != str(seed) or any(key in config for key in ("generation_seed", "declared_seed")):
        fail("mining generation config has an invalid seed declaration")
    normalized.append({key: item for key, item in config.items() if key != "seed"})
if normalized[0] != normalized[1]:
    fail("mining generation config differs across seeds outside seed")
sampling = mining.get("sampling_config")
if not isinstance(sampling, dict) or any(key in sampling for key in ("seed", "generation_seed", "declared_seed")):
    fail("sampling config must be identical and seed-independent")

# Verify the builder's atomic publication seal and all published input artifacts.
complete_path = path.parent / "_COMPLETE.json"
try:
    complete = json.loads(
        complete_path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates,
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"non-finite constant: {item}")),
    )
except (OSError, json.JSONDecodeError, ValueError) as exc:
    fail(f"atomic publication seal is missing or invalid: {exc}")
if complete.get("schema_version") != 1 or complete.get("publication") != "atomic_directory_v1":
    fail("atomic publication seal has the wrong contract")
sealed = complete.get("artifact_sha256")
declared = value.get("artifact_sha256")
if not isinstance(sealed, dict) or not isinstance(declared, dict):
    fail("artifact SHA256 maps are missing")
required_artifacts = ("frontier_rl.parquet", "selected_ids.json", "selection_manifest.json")
if complete.get("published_artifacts") != list(required_artifacts) or value.get("published_artifacts") != list(required_artifacts):
    fail("published artifact enumeration is incomplete or out of order")
if set(sealed) != set(required_artifacts):
    fail("atomic publication seal must enumerate exactly the published artifacts")
expected_entries = set(required_artifacts) | {"_COMPLETE.json"}
entries = list(path.parent.iterdir())
actual_entries = {item.name for item in entries}
invalid_entries = [
    item.name for item in entries
    if stat.S_ISLNK(os.lstat(item).st_mode) or not stat.S_ISREG(os.lstat(item).st_mode)
]
if actual_entries != expected_entries or invalid_entries:
    fail("publication directory contains unenumerated artifacts")
for name in required_artifacts:
    artifact = path.parent / name
    if not artifact.is_file() or not known_sha(sealed.get(name)):
        fail(f"published artifact is missing or unhashed: {name}")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != sealed[name]:
        fail(f"published artifact SHA256 mismatch: {name}")
    if name != "selection_manifest.json" and declared.get(name) != digest:
        fail(f"manifest artifact SHA256 mismatch: {name}")

if run_class == "formal":
    labels = ("2-10", "11-20", "21-30", "31-40", "41-50")
    ratios = (0.4, 0.1, 0.2, 0.2, 0.1)
    mapping = value.get("bucket_ratios")
    if not isinstance(mapping, dict) or any(not math.isclose(float(mapping.get(label, -1)), ratio, abs_tol=1e-12) for label, ratio in zip(labels, ratios)):
        fail("formal bucket ratios must be 40/10/20/20/10")
    raw = [requested * ratio for ratio in ratios]
    quotas = [math.floor(item) for item in raw]
    for index in sorted(range(5), key=lambda i: (-(raw[i] - quotas[i]), i))[: requested - sum(quotas)]:
        quotas[index] += 1
    selection = value.get("selection")
    if not isinstance(selection, dict):
        fail("formal selection quota evidence is missing")
    for label, quota in zip(labels, quotas):
        row = selection.get(label)
        if not isinstance(row, dict) or row.get("quota") != quota:
            fail(f"formal quota mismatch for {label}")
        if row.get("outcome-frontier", 0) + row.get("process-hard", 0) != quota:
            fail(f"formal quota is incomplete for {label}")
        category_quota = [math.floor(quota * .75), math.floor(quota * .25)]
        fractions = [quota * .75 - category_quota[0], quota * .25 - category_quota[1]]
        for index in sorted(range(2), key=lambda i: (-fractions[i], i))[:quota - sum(category_quota)]:
            category_quota[index] += 1
        if row.get("outcome-frontier") != category_quota[0] or row.get("process-hard") != category_quota[1]:
            fail(f"formal 75/25 category selection mismatch for {label}")
PY
}

export V37_FRONTIER_MANIFEST=${V37_FRONTIER_MANIFEST:-}
if [[ "${V37_RUN_CLASS}" != "debug" || -n "${V37_FRONTIER_MANIFEST}" ]]; then
  _v37_require_file "V37_FRONTIER_MANIFEST" "${V37_FRONTIER_MANIFEST}"
  V37_FRONTIER_MANIFEST="$(readlink -f "${V37_FRONTIER_MANIFEST}")"
  export V37_FRONTIER_MANIFEST
  _v37_validate_frontier_manifest "${V37_FRONTIER_MANIFEST}" "${V37_RUN_CLASS}"
  export V37_FRONTIER_MANIFEST_SHA256="$(_v37_sha256_file "${V37_FRONTIER_MANIFEST}")"
else
  unset V37_FRONTIER_MANIFEST_SHA256
fi

_V37_RESUME_CHECKPOINT=""
_V37_RESUME_CHECKPOINT_SHA256=""
_V37_RESUME_SEAL_SHA256=""
_V37_RESUME_SOURCE_MANIFEST=""
_V37_RESUME_GLOBAL_STEP=0
if _v37_is_true "${V37_CONTINUATION_MODE}"; then
  [[ -n "${V37_FRONTIER_MANIFEST_SHA256:-}" ]] \
    || _v37_error "controlled continuation requires a verified V37_FRONTIER_MANIFEST"
  [[ -d "${V37_RESUME_CHECKPOINT:-}" ]] \
    || _v37_error "controlled continuation requires V37_RESUME_CHECKPOINT directory"
  _v37_require_file "V37_RESUME_SEAL" "${V37_RESUME_SEAL:-}"
  _V37_RESUME_CHECKPOINT="$(readlink -f "${V37_RESUME_CHECKPOINT}")"
  _v37_resume_basename="$(basename "${_V37_RESUME_CHECKPOINT}")"
  [[ "${_v37_resume_basename}" =~ ^global_step_([1-9][0-9]*)$ ]] \
    || _v37_error "continuation checkpoint must end in global_step_N with N>0"
  _V37_RESUME_GLOBAL_STEP="${BASH_REMATCH[1]}"
  _V37_RESUME_CHECKPOINT_SHA256="$(_v37_sha256_tree "${_V37_RESUME_CHECKPOINT}")"
  _V37_RESUME_SEAL_SHA256="$(_v37_sha256_file "${V37_RESUME_SEAL}")"
  _V37_RESUME_SOURCE_MANIFEST="$(
    "${V37_PREFLIGHT_PYTHON:-python3}" "${REPO_DIR}/tools/v37_continuation.py" verify-seal \
      --seal "${V37_RESUME_SEAL}" \
      --checkpoint "${_V37_RESUME_CHECKPOINT}" \
      --checkpoint-sha256 "${_V37_RESUME_CHECKPOINT_SHA256}" \
      --world-size 8 \
      --run-class "${V37_RUN_CLASS}" \
      --arm "${V37_ARM}" \
      --seed "${V37_SEED}" \
      --frontier-manifest-sha256 "${V37_FRONTIER_MANIFEST_SHA256}"
  )" || _v37_error "controlled continuation provenance verification failed"
  export V37_RESUME_SEAL="$(readlink -f "${V37_RESUME_SEAL}")"
  if [[ -n "${V37_EXPECTED_RESUME_CHECKPOINT_PATH:-}" \
        && "${V37_EXPECTED_RESUME_CHECKPOINT_PATH}" != "${_V37_RESUME_CHECKPOINT}" ]]; then
    _v37_error "ambient V37_EXPECTED_RESUME_CHECKPOINT_PATH differs from the sealed checkpoint"
  fi
  if [[ -n "${V37_EXPECTED_RESUME_CHECKPOINT_SHA256:-}" \
        && "${V37_EXPECTED_RESUME_CHECKPOINT_SHA256}" != "${_V37_RESUME_CHECKPOINT_SHA256}" ]]; then
    _v37_error "ambient V37_EXPECTED_RESUME_CHECKPOINT_SHA256 differs from the sealed checkpoint"
  fi
  export V37_EXPECTED_RESUME_CHECKPOINT_PATH="${_V37_RESUME_CHECKPOINT}"
  export V37_EXPECTED_RESUME_CHECKPOINT_SHA256="${_V37_RESUME_CHECKPOINT_SHA256}"
  export _V37_RESUME_CHECKPOINT _V37_RESUME_CHECKPOINT_SHA256 _V37_RESUME_SEAL_SHA256
  export _V37_RESUME_SOURCE_MANIFEST
else
  if [[ -n "${V37_EXPECTED_RESUME_CHECKPOINT_PATH:-}" \
        || -n "${V37_EXPECTED_RESUME_CHECKPOINT_SHA256:-}" ]]; then
    _v37_error "clean start forbids an ambient expected resume checkpoint binding"
  fi
  unset V37_EXPECTED_RESUME_CHECKPOINT_PATH V37_EXPECTED_RESUME_CHECKPOINT_SHA256
fi

# Freeze the single-node 8xH200 topology before contract-only exits so the
# lightweight contract probe exercises the same exact-Ray resource policy.
for _v37_topology_spec in \
  "V31_NNODES:1" \
  "V31_N_GPUS_PER_NODE:8" \
  "HOST_NUM:1" \
  "HOST_GPU_NUM:8" \
  "INDEX:0"; do
  _v37_topology_name="${_v37_topology_spec%%:*}"
  _v37_topology_expected="${_v37_topology_spec#*:}"
  _v37_topology_actual="${!_v37_topology_name-}"
  if [[ -n "${_v37_topology_actual}" && "${_v37_topology_actual}" != "${_v37_topology_expected}" ]]; then
    _v37_error "${_v37_topology_name} must be ${_v37_topology_expected} for this 1x8 pilot, got ${_v37_topology_actual}"
  fi
done
export V37_REQUIRE_EXACT_RAY_GPUS=1
export V31_NNODES=1
export V31_N_GPUS_PER_NODE=8
export HOST_NUM=1
export HOST_GPU_NUM=8
export INDEX=0

# Debug/canary diagnostics may tune these V37 names. Formal stays frozen to
# the conservative H200 profile; update micro-batch 8 is explicitly unsafe.
export V37_MICRO_BATCH_UPDATE=${V37_MICRO_BATCH_UPDATE:-4}
export V37_MICRO_BATCH_EXP=${V37_MICRO_BATCH_EXP:-8}
export V37_VLLM_NUM_GPU_BLOCKS=${V37_VLLM_NUM_GPU_BLOCKS:-20480}
export V37_GPU_MEM_UTIL=${V37_GPU_MEM_UTIL:-0.50}
export V37_MAX_NUM_BATCHED_TOKENS=${V37_MAX_NUM_BATCHED_TOKENS:-49152}
for _v37_resource_spec in \
  "V37_MICRO_BATCH_UPDATE:${V37_MICRO_BATCH_UPDATE}" \
  "V37_MICRO_BATCH_EXP:${V37_MICRO_BATCH_EXP}" \
  "V37_VLLM_NUM_GPU_BLOCKS:${V37_VLLM_NUM_GPU_BLOCKS}" \
  "V37_MAX_NUM_BATCHED_TOKENS:${V37_MAX_NUM_BATCHED_TOKENS}"; do
  _v37_resource_name="${_v37_resource_spec%%:*}"
  _v37_resource_value="${_v37_resource_spec#*:}"
  [[ "${_v37_resource_value}" =~ ^[1-9][0-9]*$ ]] \
    || _v37_error "${_v37_resource_name} must be a positive integer, got: ${_v37_resource_value}"
done
case "${V37_MICRO_BATCH_UPDATE}" in
  1|2|4) ;;
  *) _v37_error "V37_MICRO_BATCH_UPDATE must be 1, 2, or 4; micro8 already OOMed on this workload" ;;
esac
"${V37_PREFLIGHT_PYTHON:-python3}" - "${V37_GPU_MEM_UTIL}" <<'PY' \
  || _v37_error "V37_GPU_MEM_UTIL must be a finite decimal in (0, 1]"
import math
import sys
try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit(1) from exc
raise SystemExit(0 if math.isfinite(value) and 0 < value <= 1 else 1)
PY
if [[ "${V37_RUN_CLASS}" == "formal" ]]; then
  for _v37_formal_resource in \
    "V37_MICRO_BATCH_UPDATE:4" \
    "V37_MICRO_BATCH_EXP:8" \
    "V37_VLLM_NUM_GPU_BLOCKS:20480" \
    "V37_GPU_MEM_UTIL:0.50" \
    "V37_MAX_NUM_BATCHED_TOKENS:49152" \
    "V37_CP_SIZE:1"; do
    _v37_resource_name="${_v37_formal_resource%%:*}"
    _v37_resource_expected="${_v37_formal_resource#*:}"
    [[ "${!_v37_resource_name}" == "${_v37_resource_expected}" ]] \
      || _v37_error "formal ${_v37_resource_name} is locked to ${_v37_resource_expected}, got: ${!_v37_resource_name}"
  done
fi
for _v37_legacy_resource in \
  "V31_MICRO_BATCH_UPDATE:V37_MICRO_BATCH_UPDATE" \
  "V31_MICRO_BATCH_EXP:V37_MICRO_BATCH_EXP" \
  "EASYR1_VLLM_NUM_GPU_BLOCKS:V37_VLLM_NUM_GPU_BLOCKS" \
  "V31_GPU_MEM_UTIL:V37_GPU_MEM_UTIL" \
  "V31_MAX_NUM_BATCHED_TOKENS:V37_MAX_NUM_BATCHED_TOKENS"; do
  _v37_legacy_name="${_v37_legacy_resource%%:*}"
  _v37_v37_name="${_v37_legacy_resource#*:}"
  if [[ -n "${!_v37_legacy_name:-}" && "${!_v37_legacy_name}" != "${!_v37_v37_name}" ]]; then
    _v37_error "${_v37_legacy_name} conflicts with ${_v37_v37_name}; configure resources through V37_* only"
  fi
done
export V31_MICRO_BATCH_UPDATE="${V37_MICRO_BATCH_UPDATE}"
export V31_MICRO_BATCH_EXP="${V37_MICRO_BATCH_EXP}"
export EASYR1_VLLM_NUM_GPU_BLOCKS="${V37_VLLM_NUM_GPU_BLOCKS}"
export V31_GPU_MEM_UTIL="${V37_GPU_MEM_UTIL}"
export V31_MAX_NUM_BATCHED_TOKENS="${V37_MAX_NUM_BATCHED_TOKENS}"

# Contract-only mode exercises switch expansion and manifest policy without assets/training.
if _v37_is_true "${V37_CONTRACT_ONLY:-0}"; then
  printf '%s\n' \
    "run_class=${V37_RUN_CLASS}" \
    "data_mode=${V37_DATA_MODE}" \
    "arm=${V37_ARM}" \
    "adv_estimator=${ADV_ESTIMATOR}" \
    "action_event_ledger_enable=${ACTION_EVENT_LEDGER_ENABLE}" \
    "action_event_reward_enable=${ACTION_EVENT_REWARD_ENABLE}" \
    "action_parser_contract=${V37_ACTION_PARSER_CONTRACT:-0}" \
    "action_ledger_contract=${V37_ACTION_LEDGER_CONTRACT:-0}" \
    "legacy_process_reward_enable=${PROCESS_REWARD_ENABLE}" \
    "step_weight=${BOK_STEP_WEIGHT}" \
    "correctness_first=${BOK_CORRECTNESS_FIRST}" \
    "allwrong_terminal_zero=${BOK_ALLWRONG_TERMINAL_ZERO}" \
    "correctness_task_weight=${BOK_CORRECTNESS_TASK_WEIGHT}" \
    "correctness_quality_weight=${BOK_CORRECTNESS_QUALITY_WEIGHT}" \
    "correctness_partial_scale=${BOK_CORRECTNESS_PARTIAL_SCALE}" \
    "reward_fail_closed=${V37_REWARD_FAIL_CLOSED}" \
    "image_path_remap=${STEPCOUNT_IMAGE_PATH_REMAP_JSON:-<disabled>}" \
    "winner_boost=${BOK_WINNER_BOOST}" \
    "adaptive_actor_kl=${ADAPTIVE_ACTOR_KL}" \
    "use_kl_loss=${USE_KL_LOSS}" \
    "kl=${KL_PENALTY}/${KL_COEF}/${KL_TARGET}/${KL_HORIZON}" \
    "kl_horizon_unit=${KL_HORIZON_UNIT}" \
    "cp_size=${V37_CP_SIZE}" \
    "resources=${V31_MICRO_BATCH_UPDATE}/${V31_MICRO_BATCH_EXP}/${EASYR1_VLLM_NUM_GPU_BLOCKS}/${V31_GPU_MEM_UTIL}/${V31_MAX_NUM_BATCHED_TOKENS}" \
    "fallback_logprob_sign_opt_in=${V37_FALLBACK_LOGPROB_SIGN_OPT_IN}" \
    "torch_logprob_fallback_mode=${TORCH_LOGPROB_FALLBACK_MODE}" \
    "strict_answer_integer_parse=${TRAJ_STRICT_ANSWER_INTEGER_PARSE}" \
    "strict_raw_success_winner=${V37_RAW_SUCCESS_STRICT_WINNER}" \
    "winner_mode=${V37_WINNER_MODE}" \
    "strict_point_parser_contract=${V37_STRICT_POINT_PARSER_CONTRACT}" \
    "topology=${V31_NNODES}x${V31_N_GPUS_PER_NODE}" \
    "ray_exact=${V37_REQUIRE_EXACT_RAY_GPUS}" \
    "ray_gpu_wait_timeout_seconds=${RAY_GPU_WAIT_TIMEOUT_SECONDS:-<v32-default>}" \
    "ray_status_timeout_seconds=${RAY_STATUS_TIMEOUT_SECONDS:-<v32-default>}" \
    "ray_start_timeout_seconds=${RAY_START_TIMEOUT_SECONDS:-<legacy-unbounded>}" \
    "ray_placement_group_timeout_seconds=${RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS:-<legacy-unbounded>}" \
    "hf_merge_host_memory=${V37_HF_MERGE_HOST_MEMORY_PREFLIGHT}/${V37_HF_MERGE_HOST_MEMORY_SAFETY_FACTOR}" \
    "run_purpose=${V37_RUN_PURPOSE}" \
    "resume_mode=${V37_RESUME_MODE}" \
    "promotable_candidate=${_V37_PROMOTABLE}"
  exit 0
fi

# Safe audit mode: exercise the exact launcher-to-preflight contract without
# loading model/trainer assets or claiming a run directory.
if _v37_is_true "${V37_PREFLIGHT_ONLY:-0}"; then
  [[ -n "${V37_FRONTIER_MANIFEST:-}" ]] || _v37_error "V37_PREFLIGHT_ONLY requires V37_FRONTIER_MANIFEST"
  _v37_preflight_data="${V37_FRONTIER_DATA:-$(dirname "${V37_FRONTIER_MANIFEST}")/frontier_rl.parquet}"
  [[ -e "${_v37_preflight_data}" ]] || _v37_error "V37_FRONTIER_DATA is missing: ${_v37_preflight_data}"
  _v37_require_file "V37_STEPCOUNT_MASKS_METADATA" "${V37_STEPCOUNT_MASKS_METADATA:-}"
  [[ -d "${V37_STEPCOUNT_MASKS_DIR:-}" ]] || _v37_error "V37_STEPCOUNT_MASKS_DIR must name an existing directory"
  [[ -n "${STEPCOUNT_V37_VAL_DATA:-}" ]] || _v37_error "V37_PREFLIGHT_ONLY requires STEPCOUNT_V37_VAL_DATA"
  [[ -n "${V37_FORBIDDEN_DATA:-}" ]] || _v37_error "V37_PREFLIGHT_ONLY requires V37_FORBIDDEN_DATA"
  _v37_audit_args=(
    "${REPO_DIR}/tools/preflight_v37_training.py" "${_v37_preflight_data}"
    --metadata "${V37_STEPCOUNT_MASKS_METADATA}"
    --masks-dir "${V37_STEPCOUNT_MASKS_DIR}"
    --data-mode frontier_rl --run-class "${V37_RUN_CLASS}"
    --entrypoint-contract frontier_rl
    --training-entrypoint "${REPO_DIR}/examples/v32_sparse_0_10_stable_drfix.sh"
  )
  IFS=',' read -r -a _v37_audit_vals <<< "${STEPCOUNT_V37_VAL_DATA}"
  for _v37_audit_item in "${_v37_audit_vals[@]}"; do
    _v37_audit_path="${_v37_audit_item#*::}"
    _v37_audit_args+=(--val-data "${_v37_audit_path}")
    [[ "${V37_RUN_CLASS}" != "formal" ]] || _v37_audit_args+=(--expected-val-sha256 "$(_v37_sha256_tree "${_v37_audit_path}")")
  done
  IFS=',' read -r -a _v37_audit_forbidden <<< "${V37_FORBIDDEN_DATA}"
  for _v37_audit_path in "${_v37_audit_forbidden[@]}"; do
    _v37_audit_args+=(--forbidden-data "${_v37_audit_path}")
    [[ "${V37_RUN_CLASS}" != "formal" ]] || _v37_audit_args+=(--expected-forbidden-sha256 "$(_v37_sha256_tree "${_v37_audit_path}")")
  done
  if [[ -n "${V37_IMAGE_ROOTS:-}" ]]; then
    IFS=':' read -r -a _v37_audit_image_roots <<< "${V37_IMAGE_ROOTS}"
    for _v37_audit_path in "${_v37_audit_image_roots[@]}"; do
      _v37_audit_args+=(--image-root "${_v37_audit_path}")
    done
  fi
  if [[ "${V37_RUN_CLASS}" == "formal" ]]; then
    _v37_require_file "V37_METADATA_COVERAGE_REPORT" "${V37_METADATA_COVERAGE_REPORT:-}"
    _v37_require_file "V37_FILTERED_MANIFEST" "${V37_FILTERED_MANIFEST:-}"
    _v37_audit_args+=(
      --expected-ratios 0.40,0.10,0.20,0.20,0.10 --tolerance 0.05
      --require-val-all-buckets --phash-hamming-threshold 4
      --coverage-report "${V37_METADATA_COVERAGE_REPORT}"
      --filtered-manifest "${V37_FILTERED_MANIFEST}"
      --loader-model-path "${V37_LOADER_MODEL_PATH:-${STEPCOUNT_1M_RESUME_CKPT476_MODEL_PATH}}"
      --loader-max-prompt-length 12000 --loader-max-pixels 12845056
      --loader-min-pixels 262144
      --loader-filter-num-proc "${V31_FILTER_OVERLONG_NUM_PROC:-${EASYR1_FILTER_OVERLONG_NUM_PROC:-64}}"
      --loader-system-prompt "${REPO_DIR}/examples/format_prompt/StepCount_interleaved_system_prompt.txt"
      --expected-data-sha256 "$(_v37_sha256_tree "${_v37_preflight_data}")"
      --expected-metadata-sha256 "$(_v37_sha256_tree "${V37_STEPCOUNT_MASKS_METADATA}")"
      --expected-masks-sha256 "$(_v37_sha256_tree "${V37_STEPCOUNT_MASKS_DIR}")"
      --expected-training-entrypoint-sha256 "$(_v37_sha256_file "${REPO_DIR}/examples/v32_sparse_0_10_stable_drfix.sh")"
    )
  fi
  exec "${V37_PREFLIGHT_PYTHON:-python3}" "${_v37_audit_args[@]}"
fi

_V37_CANONICAL_DATA_ROOT="$(readlink -f "${STEPCOUNT_DATA_ROOT}" 2>/dev/null || true)"
_V37_CANONICAL_MODEL="$(readlink -f "${STEPCOUNT_1M_RESUME_CKPT476_MODEL_PATH}" 2>/dev/null || true)"
_V37_CANONICAL_FOCUSED="$(readlink -f "${STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA}" 2>/dev/null || true)"
_V37_CANONICAL_REWARD="${REPO_DIR}/examples/reward_function/StepCount_mask_reward.py:compute_score"
[[ -d "${_V37_CANONICAL_DATA_ROOT}" ]] \
  || _v37_error "STEPCOUNT_DATA_ROOT must name the mounted StepCount dataset root: ${STEPCOUNT_DATA_ROOT}"
[[ -d "${_V37_CANONICAL_MODEL}" ]] \
  || _v37_error "STEPCOUNT_1M_RESUME_CKPT476_MODEL_PATH must name checkpoint-476: ${STEPCOUNT_1M_RESUME_CKPT476_MODEL_PATH}"
_V37_BENCHMARK_ROOTS=(
  "${STEPCOUNT_PIXMO_CANONICAL_JSON}"
  "${STEPCOUNT_STEPCOUNT500_CANONICAL_JSON}"
  "${STEPCOUNT_COUNTQA_DATA}"
  "${STEPCOUNT_BIAS_DATA}"
  "${STEPCOUNT_DENSE_CANONICAL_JSON}"
  "${STEPCOUNT_EXTREME_CANONICAL_JSON}"
)
_V37_FORBIDDEN_DATA=("${_V37_BENCHMARK_ROOTS[@]}")

# A frontier prompt dataset is mandatory outside debug. The existing
# focused10k set can only be used for a short, explicitly acknowledged mechanism pilot.
_V37_FOCUSED_PILOT=0
_V37_FRONTIER_DATA=${V37_FRONTIER_DATA:-${STEPCOUNT_V37_BALANCED_DATA:-}}
if [[ -n "${_V37_FRONTIER_DATA}" ]]; then
  [[ -e "${_V37_FRONTIER_DATA}" ]] \
    || _v37_error "V37_FRONTIER_DATA must name existing parquet data: ${_V37_FRONTIER_DATA}"
  export STEPCOUNT_TRAIN_DATA="${_V37_FRONTIER_DATA}"
elif [[ "${V37_RUN_CLASS}" == "debug" ]] && _v37_is_true "${V37_ALLOW_FOCUSED10K_PILOT:-0}"; then
  [[ -d "${STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA}" ]] \
    || _v37_error "focused10k pilot data directory is missing: ${STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA}"
  export STEPCOUNT_TRAIN_DATA="${STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA}"
  _V37_FOCUSED_PILOT=1
  _v37_mark_nonpromotable "focused10k_distribution_mismatch"
else
  _v37_error "set V37_FRONTIER_DATA to build_v37_frontier_dataset.py output; focused10k is debug-only"
fi

_V37_DATA_REAL="$(readlink -f "${STEPCOUNT_TRAIN_DATA}")"
export STEPCOUNT_TRAIN_DATA="${_V37_DATA_REAL}"
_V37_FOCUSED_REAL="$(readlink -f "${STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA}" 2>/dev/null || true)"
if [[ "${_V37_DATA_REAL}" == "${_V37_FOCUSED_REAL}" ]]; then
  if [[ "${V37_RUN_CLASS}" != "debug" ]]; then
    _v37_error "focused10k fallback is forbidden for canary/formal runs"
  elif _v37_is_true "${V37_ALLOW_FOCUSED10K_PILOT:-0}"; then
    _V37_FOCUSED_PILOT=1
    _v37_mark_nonpromotable "focused10k_distribution_mismatch"
  else
    _v37_error "focused10k is distribution-mismatched and requires V37_ALLOW_FOCUSED10K_PILOT=1"
  fi
fi
if [[ "${V37_RUN_CLASS}" == "formal" ]]; then
  # Formal frontier data cannot be a renamed/reweighted focused10k fallback.
  _V37_FORBIDDEN_DATA+=("${_V37_CANONICAL_FOCUSED}")
  [[ "${V37_STEP_WEIGHT:-0.1}" == "0.1" ]] \
    || _v37_error "formal progress step weight is locked to 0.1"
  [[ "${V37_STEP_GATE:-answer_soft}" == "answer_soft" ]] \
    || _v37_error "formal progress step gate is locked to answer_soft"
  [[ "${V37_STEP_MIN_GATE:-0.2}" == "0.2" ]] \
    || _v37_error "formal progress step min gate is locked to 0.2"
fi
if [[ "${V37_RUN_CLASS}" != "debug" ]]; then
  [[ -n "${V37_FRONTIER_MANIFEST}" ]] || _v37_error "canary/formal requires V37_FRONTIER_MANIFEST"
  [[ "$(readlink -f "$(dirname "${V37_FRONTIER_MANIFEST}")")" == "$(readlink -f "${_V37_DATA_REAL}" 2>/dev/null || dirname "${_V37_DATA_REAL}")" \
     || "$(readlink -f "$(dirname "${V37_FRONTIER_MANIFEST}")/frontier_rl.parquet" 2>/dev/null || true)" == "${_V37_DATA_REAL}" ]] \
    || _v37_error "canary/formal frontier manifest must accompany the selected frontier_rl parquet/directory"
fi
for _v37_benchmark in "${_V37_BENCHMARK_ROOTS[@]}"; do
  if [[ -e "${_v37_benchmark}" ]]; then
    _v37_benchmark_real="$(readlink -f "${_v37_benchmark}")"
    if _v37_paths_overlap "${_V37_DATA_REAL}" "${_v37_benchmark_real}"; then
      _v37_error "training input must not equal, contain, or descend from benchmark data: ${STEPCOUNT_TRAIN_DATA}"
    fi
  fi
done

# Training-time validation must be an explicit, held-out local split. Final
# benchmarks are blocked by default so a frequent pilot val loop cannot become
# a benchmark-tuning loop.
[[ -n "${STEPCOUNT_V37_VAL_DATA:-}" ]] \
  || _v37_error "set STEPCOUNT_V37_VAL_DATA to an independent held-out local validation split"
_V37_VAL_SPECS=()
_V37_VAL_HAS_BENCHMARK=0
declare -A _V37_VAL_PATHS_SEEN=()
declare -A _V37_VAL_SUITES_SEEN=()
IFS=',' read -r -a _v37_raw_val_specs <<< "${STEPCOUNT_V37_VAL_DATA}"
_v37_val_idx=0
for _v37_spec in "${_v37_raw_val_specs[@]}"; do
  _v37_spec="${_v37_spec#"${_v37_spec%%[![:space:]]*}"}"
  _v37_spec="${_v37_spec%"${_v37_spec##*[![:space:]]}"}"
  [[ -n "${_v37_spec}" ]] || _v37_error "STEPCOUNT_V37_VAL_DATA contains an empty entry"
  if [[ "${_v37_spec}" == *"::"* ]]; then
    _v37_suite="${_v37_spec%%::*}"
    _v37_val_path="${_v37_spec#*::}"
  else
    _v37_suite="v37-heldout-${_v37_val_idx}"
    _v37_val_path="${_v37_spec}"
  fi
  [[ -n "${_v37_suite}" && -n "${_v37_val_path}" ]] \
    || _v37_error "invalid validation spec: ${_v37_spec}"
  [[ -e "${_v37_val_path}" ]] \
    || _v37_error "validation path does not exist: ${_v37_val_path}"
  if [[ -f "${_v37_val_path}" ]]; then
    [[ "${_v37_val_path}" == *.parquet ]] \
      || _v37_error "validation file must be parquet: ${_v37_val_path}"
  elif ! find "${_v37_val_path}" -type f -name '*.parquet' -print -quit | grep -q .; then
    _v37_error "validation directory contains no parquet files: ${_v37_val_path}"
  fi
  _v37_val_real="$(readlink -f "${_v37_val_path}")"
  [[ "${_v37_val_real}" != "${_V37_DATA_REAL}" ]] \
    || _v37_error "training and validation paths must differ: ${_v37_val_path}"
  [[ -z "${_V37_VAL_PATHS_SEEN[${_v37_val_real}]:-}" ]] \
    || _v37_error "duplicate validation path: ${_v37_val_path}"
  [[ -z "${_V37_VAL_SUITES_SEEN[${_v37_suite}]:-}" ]] \
    || _v37_error "duplicate validation suite name: ${_v37_suite}"
  _V37_VAL_PATHS_SEEN["${_v37_val_real}"]=1
  _V37_VAL_SUITES_SEEN["${_v37_suite}"]=1
  for _v37_benchmark in "${_V37_BENCHMARK_ROOTS[@]}"; do
    if [[ -e "${_v37_benchmark}" ]]; then
      _v37_benchmark_real="$(readlink -f "${_v37_benchmark}")"
      if _v37_path_is_within "${_v37_benchmark_real}" "${_v37_val_real}" \
        && [[ "${_v37_val_real}" != "${_v37_benchmark_real}" ]]; then
        _v37_error "validation path is too broad and contains benchmark data: ${_v37_val_path}"
      elif _v37_path_is_within "${_v37_val_real}" "${_v37_benchmark_real}"; then
        _V37_VAL_HAS_BENCHMARK=1
      fi
    fi
  done
  _V37_VAL_SPECS+=("${_v37_suite}::${_v37_val_real}")
  _v37_val_idx=$((_v37_val_idx + 1))
done
if [[ "${_V37_VAL_HAS_BENCHMARK}" == "1" \
      && "${V37_ALLOW_INDEPENDENT_VAL_OUTSIDE_TRAIN_RANGE:-0}" == "1" ]]; then
  _v37_error "independent outside-range validation must remain non-benchmark"
fi
if [[ "${_V37_VAL_HAS_BENCHMARK}" == "1" && "${V37_RUN_CLASS}" != "debug" ]]; then
  _v37_error "benchmark development validation is allowed only for non-promotable debug runs"
fi
if [[ "${_V37_VAL_HAS_BENCHMARK}" == "1" ]] && ! _v37_is_true "${V37_ALLOW_BENCHMARK_DEV:-0}"; then
  _v37_error "debug benchmark validation requires V37_ALLOW_BENCHMARK_DEV=1"
fi
if [[ "${_V37_VAL_HAS_BENCHMARK}" == "1" ]]; then
  _v37_mark_nonpromotable "benchmark_used_for_training_val"
  echo "[V37-strict-winner][WARNING] benchmark used during training; this run is debug-only and cannot be promoted." >&2
fi
export STEPCOUNT_VAL_DATA="$(IFS=,; echo "${_V37_VAL_SPECS[*]}")"

if [[ "${_V37_FOCUSED_PILOT}" == "1" ]]; then
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
  echo "[V37-strict-winner][WARNING] FOCUSED10K MECHANISM PILOT ONLY" >&2
  echo "[V37-strict-winner][WARNING] the focused selection has no 2-10 samples and heavily weights 11-30; it does NOT match the balanced V37 2-50 target." >&2
  echo "[V37-strict-winner][WARNING] results cannot be used for strict-winner promotion or full A/B conclusions." >&2
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
fi

# Formal A/B is always a clean SFT checkpoint-476 start.  A separately sealed,
# non-promotable controlled continuation may restore trainer state.
[[ -d "${STEPCOUNT_1M_RESUME_CKPT476_MODEL_PATH}" ]] \
  || _v37_error "clean SFT checkpoint-476 directory is missing: ${STEPCOUNT_1M_RESUME_CKPT476_MODEL_PATH}"
[[ "${V37_EXPECTED_INITIAL_MODEL_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] \
  || _v37_error "V37_EXPECTED_INITIAL_MODEL_SHA256 must bind the canonical checkpoint-476"
if [[ -n "${MODEL_PATH:-}" \
      && "$(readlink -f "${MODEL_PATH}" 2>/dev/null || true)" != "$(readlink -f "${STEPCOUNT_1M_RESUME_CKPT476_MODEL_PATH}")" ]]; then
  _v37_error "MODEL_PATH must be the clean SFT checkpoint-476, got: ${MODEL_PATH}"
fi
export MODEL_PATH="${_V37_CANONICAL_MODEL}"
export REWARD_FN_PATH="${_V37_CANONICAL_REWARD}"
if _v37_is_true "${V37_CONTINUATION_MODE}"; then
  if [[ -n "${V36_LOAD_CHECKPOINT_PATH:-}" && "$(readlink -f "${V36_LOAD_CHECKPOINT_PATH}" 2>/dev/null || true)" != "${_V37_RESUME_CHECKPOINT}" ]]; then
    _v37_error "ambient V36_LOAD_CHECKPOINT_PATH differs from the sealed continuation checkpoint"
  fi
  if [[ -n "${V32_LOAD_CHECKPOINT_PATH:-}" && "$(readlink -f "${V32_LOAD_CHECKPOINT_PATH}" 2>/dev/null || true)" != "${_V37_RESUME_CHECKPOINT}" ]]; then
    _v37_error "ambient V32_LOAD_CHECKPOINT_PATH differs from the sealed continuation checkpoint"
  fi
  export V36_LOAD_CHECKPOINT_PATH="${_V37_RESUME_CHECKPOINT}"
  export V32_LOAD_CHECKPOINT_PATH="${_V37_RESUME_CHECKPOINT}"
else
  if [[ -n "${V36_LOAD_CHECKPOINT_PATH:-}" || -n "${V32_LOAD_CHECKPOINT_PATH:-}" ]]; then
    _v37_error "clean-start A/B forbids resume; use the sealed V37_CONTINUATION_MODE entrypoint"
  fi
  unset V36_LOAD_CHECKPOINT_PATH V32_LOAD_CHECKPOINT_PATH
fi
_V37_CANONICAL_CONFIG="${REPO_DIR}/examples/config.yaml"
if [[ -n "${CONFIG_PATH:-}" \
      && "$(readlink -f "${CONFIG_PATH}" 2>/dev/null || true)" != "$(readlink -f "${_V37_CANONICAL_CONFIG}")" ]]; then
  _v37_error "custom CONFIG_PATH is forbidden for a clean controlled pilot: ${CONFIG_PATH}"
fi
export CONFIG_PATH="${_V37_CANONICAL_CONFIG}"
unset V36_ALLOW_V32_LOAD_CHECKPOINT
if _v37_is_true "${V32_DRY_RUN:-0}" && ! _v37_is_true "${V37_DRY_RUN:-0}"; then
  _v37_error "ambient V32_DRY_RUN would suppress training; unset it or use V37_DRY_RUN=1"
fi
if _v37_is_true "${V37_VALIDATE_DOWNSTREAM:-0}" && ! _v37_is_true "${V37_DRY_RUN:-0}"; then
  _v37_error "V37_VALIDATE_DOWNSTREAM=1 is valid only together with V37_DRY_RUN=1"
fi

export STEPCOUNT_HARDWARE_PROFILE=h200
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export V31_ROLLOUT_BATCH_SIZE=128
export V31_GLOBAL_BATCH_SIZE=128
export V31_VAL_BATCH_SIZE=128
export V31_FILTER_OVERLONG_NUM_PROC=64
export EASYR1_FILTER_OVERLONG_NUM_PROC=64
export V31_MAX_PROMPT_LENGTH=12000
export V31_MAX_RESPONSE_LENGTH=16384
export V31_MAX_MODEL_LEN=32768
export V31_MAX_PIXELS=12845056
export V31_MIN_PIXELS=262144
export V31_ENFORCE_EAGER=false
export EASYR1_ALLOW_ZERO_MM_LOGPROB=0
if [[ "${V37_RUN_CLASS}" == "formal" ]]; then
  export V37_FINALIZE_HF_CHECKPOINT=1
else
  export V37_FINALIZE_HF_CHECKPOINT=0
fi
export V36_ENABLE_ULYSSES_SP=0
export V31_ULYSSES_SEQUENCE_PARALLEL_SIZE="${V37_CP_SIZE}"

export ROLLOUT_N=16
export ROLLOUT_TEMPERATURE=0.7
export ROLLOUT_TOP_P=1.0
export ACTOR_LR=2e-7
export BOK_TAU_INIT=0.7
export BOK_TAU_FINAL=0.55
export BOK_TAU_SCHEDULE=cosine
# Correctness-first already gives rare winners positive sign; no extra boost.
export BOK_WINNER_BOOST=0
export BOK_QUALITY_BONUS=0
export BOK_DAPO_FILTER=0
export BOK_FILTER_ALL_CORRECT=1
export BOK_SMART_FILTER_THRESHOLD=0.955
export BOK_ALLWRONG_CAP=0.5
export BOK_ALLWRONG_NEG_ONLY=1
export BOK_EASY_THRESHOLD=0.50
export BOK_FALLBACK_MODE=drgrpo
export BOK_TAU=0.5
export BOK_CLIP=4.0
export BOK_UNIFORM_MIX=0.1
export BOK_ADV_NORMALIZE=0
export BOK_LOW_VAR_THRESHOLD=1e-5
export BOK_DAPO_AUTO_DISABLE_THRESHOLD=0.5
export BOK_MIN_BATCH_STD=0.1
export BOK_EASY_SCORE_THRESHOLD=0.5
export BOK_ALLWRONG_ANSWER_THRESHOLD=0.5
export BOK_EASY_SCALE=1.0
export BOK_PROGRESS_DUP_PENALTY=1.0
export BOK_LOGIT_CAP=12.0
export BOK_TAU_ADAPTIVE=1
export BOK_TAU_MIN=0.08
export BOK_STEP_CLIP=2.5
export BOK_HYBRID_CLIP=4.0
export BOK_STEP_LOW_VAR_THRESHOLD=1e-6
export BOK_STEP_REWARD_EPS=1e-12
export BOK_STEP_MAX_SEMANTIC_STEPS=64
export BOK_STEP_OUTCOME_WEIGHT=1.0
export BOK_STEP_EXCLUDE_FINAL=1
export VCRL_ENABLE=0
export POLICY_LOSS_IS_LEVEL=sequence_token

export TRAIN_VAL_FREQ=5
export TRAIN_SAVE_FREQ=10
export TRAIN_SAVE_LIMIT=3
export TRAINER_VAL_BEFORE_TRAIN=true
export TRAINER_VAL_ONLY=false
export INTERLEAVED_ADAPTIVE_MAX_TURNS=true
export INTERLEAVED_MAX_TURNS=53
export INTERLEAVED_POINT_TURN_USE_ANSWER_BUDGET=true
export INTERLEAVED_PER_TURN_MAX_TOKENS=512
export INTERLEAVED_ANSWER_TURN_MAX_TOKENS=8192
export INTERLEAVED_HISTORY_MODE=0
export INTERLEAVED_FIRST_TURN_PROMPT_FILE=""
export INTERLEAVED_PROCESS_PROMPT_FILE="${REPO_DIR}/examples/format_prompt/StepCount_interleaved_process_prompt.txt"
export SYSTEM_PROMPT_FILE="${REPO_DIR}/examples/format_prompt/StepCount_interleaved_system_prompt.txt"
export INTERLEAVED_PROCESS_PROMPT_SHA256="$(_v37_sha256_file "${INTERLEAVED_PROCESS_PROMPT_FILE}")"
export INTERLEAVED_DEBUG=0
export INTERLEAVED_DEBUG_PRINT_CHARS=80
# Existing trainer cap is GT + one answer turn + margin, so margin=2 is GT+3.
export INTERLEAVED_ADAPTIVE_MAX_TURNS_MARGIN=2

# Freeze the V36 reward semantics so baseline/progress differ only in process
# credit assignment. Correct answers retain partial reward even when trajectory
# quality is poor; structural and point defects remain separately penalized.
export ANSWER_WEIGHT=0.7
export POINT_WEIGHT=0.2
export TRAJECTORY_FORMAT_WEIGHT=0.1
export TRAJ_DENSE_CONTINUOUS_REWARD=1
export TRAJ_DENSE_CONTINUOUS_MIN_GT=11
export TRAJ_DENSE_CONTINUOUS_WRONG_CAP=0.05
export TRAJ_DENSE_CONTINUOUS_WITHIN1_CAP=0.12
export TRAJ_COVERAGE_PENALTY_ENABLE=1
export TRAJ_COVERAGE_PENALTY_MODE=mask
export TRAJ_COVERAGE_PENALTY_PER_MISS=0.1
export TRAJ_COVERAGE_PENALTY_CAP=0.3
export TRAJ_STOP_NONE_ENABLE=1
export TRAJ_EARLY_STOP_PENALTY=0.2
export TRAJ_STOP_NONE_BONUS=0.05
export TRAJ_RETURN_POINT_STEP_SCORES=1
export TRAJ_EXTRA_POINT_PENALTY_LAMBDA=1.0
export TRAJ_STRICT_TRAJECTORY_INTEGRITY=0
export TRAJ_STRICT_ANSWER_POINT_CONSISTENCY=0
export TRAJ_EVAL_REQUIRE_INTEGRITY=0
export TRAJ_HARD_REJECT_TAG_BALANCE=1
export TRAJ_HARD_REJECT_COUNT_NUMBER=0
export TRAJ_POINT_COUNT_NUMBER_CHECK=required
export TRAJ_COUNT_NUMBER_FORMAT_WEIGHT=0.50
export TRAJ_COUNT_NUMBER_POINT_PENALTY_WEIGHT=0.20
export TRAJ_ANSWER_GATE_MODE=soft
export TRAJ_ANSWER_GATE_THRESHOLD=0.4
export TRAJ_SOFT_GATE_BASE=0.7
export TRAJ_FORMAT_STRICT_KEY=1
export TRAJ_FORMAT_TYPO_CREDIT=0.0
export TRAJ_FORMAT_GRADED=1
export TRAJ_FORMAT_REJECTION=structural
export TRAJ_POINT_KEY_NORMALIZE=1
export TRAJ_POINT_STRICT_JSON=1
export TRAJ_NO_SEQUENCE_FALLBACK=stat_mask_sim
export TRAJ_MISS_DECAY_ENABLE=1
export TRAJ_CONSISTENCY_PENALTY=0.5
export TRAJ_EVAL_ANSWER_ONLY_ON_NO_MASK=1
export TRAJ_SOFT_ANSWER_DECAY=1
export TRAJ_ANSWER_DECAY_ALPHA=8.0
export TRAJ_ANSWER_DECAY_CAP=0.4
export TRAJ_UNDER_ALPHA_GT_SCALE=0.5
export TRAJ_UNDER_ALPHA_GT_THRESHOLD=5
export TRAJ_EXACT_ANSWER_PARTIAL_REWARD=0.25

export CLIP_RATIO_LOW=0.2
export CLIP_RATIO_HIGH=0.28
export CLIP_RATIO_DUAL=3.0
export GRAD_SPIKE_PROTECT=1
export GRAD_SPIKE_THRESHOLD=3.0
export GRAD_SPIKE_ABSOLUTE_CAP=0
export GRAD_SPIKE_COOLDOWN=0
export GRAD_NONFINITE_COOLDOWN=0
export GRAD_SPIKE_LR_FACTOR=1.0
export GRAD_NONFINITE_LR_FACTOR=1.0
export GRAD_SPIKE_BRAKE_MAX=1000000
export GRAD_NONFINITE_BRAKE_MAX=1000000
export GRAD_SPIKE_BRAKE_WINDOW=50
export GRAD_NONFINITE_BRAKE_WINDOW=40
export FP16_GRAD_UNDERFLOW_MONITOR=0

export TRAIN_OVERSAMPLE_NO_MASK_FACTOR=1
export TRAIN_HARD_OVERSAMPLE_FACTOR=1
export STEPCOUNT_TOTAL_EPOCHS=1
export TRAINER_PROJECT_NAME=easy_r1
export TRAINER_VAL_GENERATIONS_TO_LOG=3
export STEPCOUNT_MASK_REQUIRE=1
export STEPCOUNT_MASK_PREFILL_BY_TURN=1
export STEPCOUNT_FORCE_TRAJECTORY_FOR_NUMERIC_GT=1
export STEPCOUNT_MASK_CACHE=1
export STEPCOUNT_MASK_CACHE_MAX=8192
export REWARD_NUM_WORKERS=8
export STEPCOUNT_MASK_DEBUG=0
export EASYR1_REWARD_SAMPLE_DEBUG=0

export PYTHONHASHSEED="${V37_SEED}"
export V31_DATA_SEED="${V37_SEED}"
export V31_ROLLOUT_SEED="${V37_SEED}"

case "${V37_RUN_CLASS}" in
  debug) _V37_DEFAULT_STEPS=2 ;;
  canary) _V37_DEFAULT_STEPS=4 ;;
  formal) _V37_DEFAULT_STEPS=12 ;;
esac
export V37_PILOT_STEPS=${V37_PILOT_STEPS:-${_V37_DEFAULT_STEPS}}
if ! [[ "${V37_PILOT_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  _v37_error "V37_PILOT_STEPS must be a positive integer, got: ${V37_PILOT_STEPS}"
fi
if [[ "${V37_RUN_CLASS}" == "canary" ]] && (( V37_PILOT_STEPS < 2 || V37_PILOT_STEPS > 4 )); then
  _v37_error "canary is restricted to 2-4 optimizer updates"
fi
if [[ "${V37_RUN_CLASS}" == "formal" && "${V37_PILOT_STEPS}" != "12" ]]; then
  _v37_error "formal baseline/progress A/B is locked to exactly 12 updates"
fi
if (( V37_PILOT_STEPS <= _V37_RESUME_GLOBAL_STEP )); then
  _v37_error "V37_PILOT_STEPS is an absolute final step and must exceed continuation start ${_V37_RESUME_GLOBAL_STEP}"
fi
export V37_EVIDENCE_START_STEP="${_V37_RESUME_GLOBAL_STEP}"
export V36_MAX_STEPS="${V37_PILOT_STEPS}"
export BOK_TOTAL_STEPS="${V37_PILOT_STEPS}"

if [[ "${_V37_FOCUSED_PILOT}" == "1" ]]; then
  export STEPCOUNT_MASKS_METADATA=${V37_STEPCOUNT_MASKS_METADATA:-${STEPCOUNT_DENSE_11_50_MASKS_METADATA}}
  export STEPCOUNT_MASKS_DIR=${V37_STEPCOUNT_MASKS_DIR:-${STEPCOUNT_DENSE_11_50_MASKS_DIR}}
else
  [[ -n "${V37_STEPCOUNT_MASKS_METADATA:-}" ]] \
    || _v37_error "formal balanced data requires explicit combined V37_STEPCOUNT_MASKS_METADATA"
  [[ -n "${V37_STEPCOUNT_MASKS_DIR:-}" ]] \
    || _v37_error "formal balanced data requires explicit combined V37_STEPCOUNT_MASKS_DIR"
  export STEPCOUNT_MASKS_METADATA="${V37_STEPCOUNT_MASKS_METADATA}"
  export STEPCOUNT_MASKS_DIR="${V37_STEPCOUNT_MASKS_DIR}"
fi
_v37_require_file "STEPCOUNT_MASKS_METADATA" "${STEPCOUNT_MASKS_METADATA}"
[[ -d "${STEPCOUNT_MASKS_DIR}" ]] \
  || _v37_error "STEPCOUNT_MASKS_DIR must name an existing directory: ${STEPCOUNT_MASKS_DIR}"
STEPCOUNT_MASKS_METADATA="$(readlink -f "${STEPCOUNT_MASKS_METADATA}")"
STEPCOUNT_MASKS_DIR="$(readlink -f "${STEPCOUNT_MASKS_DIR}")"
export STEPCOUNT_MASKS_METADATA STEPCOUNT_MASKS_DIR
if [[ "${V37_RUN_CLASS}" == "formal" && -z "${V37_METADATA_COVERAGE_REPORT:-}" ]]; then
  _v37_error "formal frontier data requires V37_METADATA_COVERAGE_REPORT"
fi
if [[ -n "${V37_METADATA_COVERAGE_REPORT:-}" ]]; then
  _v37_require_file "V37_METADATA_COVERAGE_REPORT" "${V37_METADATA_COVERAGE_REPORT}"
  export V37_METADATA_COVERAGE_REPORT="$(readlink -f "${V37_METADATA_COVERAGE_REPORT}")"
else
  unset V37_METADATA_COVERAGE_REPORT
fi
export V36_STEPCOUNT_MASKS_METADATA="${STEPCOUNT_MASKS_METADATA}"
export V36_STEPCOUNT_MASKS_DIR="${STEPCOUNT_MASKS_DIR}"

export V37_FILTERED_MANIFEST=${V37_FILTERED_MANIFEST:-}
if [[ "${V37_RUN_CLASS}" == "formal" ]]; then
  _v37_require_file "V37_FILTERED_MANIFEST" "${V37_FILTERED_MANIFEST}"
  V37_FILTERED_MANIFEST="$(readlink -f "${V37_FILTERED_MANIFEST}")"
  export V37_FILTERED_MANIFEST
fi

_V37_TIMESTAMP=${V37_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
export V31_EXPERIMENT_NAME=${V31_EXPERIMENT_NAME:-StepCount-7B_v37_${V37_RUN_CLASS}_frontier_${V37_ARM}_seed${V37_SEED}_ckpt476_8gpu_step${V37_PILOT_STEPS}_${_V37_TIMESTAMP}}
export V36_SAVE_ROOT=${V36_SAVE_ROOT:-${EASYR1_CHECKPOINT_ROOT}}
export V36_SAVE_ROOT="$(_v37_absolute_path "${V36_SAVE_ROOT}")"
export V31_SAVE_CHECKPOINT_PATH="$(_v37_absolute_path "${V31_SAVE_CHECKPOINT_PATH:-${V36_SAVE_ROOT}/${V31_EXPERIMENT_NAME}}")"

_V37_FORMAL_RATIOS="0.40,0.10,0.20,0.20,0.10"
_V37_RATIO_TOLERANCE="${V37_RATIO_TOLERANCE:-0.05}"
if [[ "${V37_RUN_CLASS}" == "formal" ]]; then
  if [[ -n "${V37_EXPECTED_RATIOS:-}" && "${V37_EXPECTED_RATIOS}" != "${_V37_FORMAL_RATIOS}" ]]; then
    _v37_error "formal V37 ratios are locked to ${_V37_FORMAL_RATIOS}"
  fi
  [[ "${_V37_RATIO_TOLERANCE}" == "0.05" ]] \
    || _v37_error "formal V37 ratio tolerance is locked to 0.05"
fi

_V37_PREFLIGHT_PYTHON=${V37_PREFLIGHT_PYTHON:-python3}
command -v "${_V37_PREFLIGHT_PYTHON}" >/dev/null 2>&1 \
  || _v37_error "preflight Python executable not found: ${_V37_PREFLIGHT_PYTHON}"
_V37_PYTHON_PROBE="$("${_V37_PREFLIGHT_PYTHON}" -c 'import sys; from pathlib import Path; print("V37_PYTHON_OK:" + str(Path(sys.executable).resolve()))' 2>/dev/null || true)"
[[ "${_V37_PYTHON_PROBE}" == V37_PYTHON_OK:* ]] \
  || _v37_error "V37_PREFLIGHT_PYTHON must be a working Python interpreter, got: ${_V37_PREFLIGHT_PYTHON}"
_V37_TRAIN_PYTHON_PROBE="$(python3 -c 'import sys; from pathlib import Path; print("V37_PYTHON_OK:" + str(Path(sys.executable).resolve()))' 2>/dev/null || true)"
[[ "${_V37_TRAIN_PYTHON_PROBE}" == "${_V37_PYTHON_PROBE}" ]] \
  || _v37_error "V37 preflight and training must use the same python3 interpreter"
"${_V37_PREFLIGHT_PYTHON}" -c 'import pyarrow.parquet; import PIL.Image' >/dev/null 2>&1 \
  || _v37_error "preflight Python must provide pyarrow and Pillow: ${_V37_PREFLIGHT_PYTHON}"
if [[ "${V37_RUN_CLASS}" == "formal" ]]; then
  "${_V37_PREFLIGHT_PYTHON}" - "${REPO_DIR}" <<'PY'
import sys

repo = sys.argv[1]
if repo not in sys.path:
    sys.path.insert(0, repo)
import flash_attn  # noqa: F401
import ray  # noqa: F401
import tensordict  # noqa: F401
import torch
import vllm  # noqa: F401
from verl.workers.actor.dp_actor import resolve_logprob_function

if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(
        f"[V37-strict-winner][ERROR] formal requires exactly 8 visible CUDA GPUs; "
        f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
    )
names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
if any("H200" not in name.upper() for name in names):
    raise SystemExit(f"[V37-strict-winner][ERROR] formal requires 8 H200 GPUs; found={names}")
resolve_logprob_function("error")
print(f"[V37-strict-winner] runtime probe passed: torch={torch.__version__} GPUs={names}")
PY
fi
if [[ "${V37_RUN_CLASS}" == "formal" ]]; then
  for _v37_ab_key in V37_AB_RUN_ID V37_AB_PLAN_PATH V37_AB_PLAN_SHA256 V37_AB_CELL_ID; do
    [[ -n "${!_v37_ab_key:-}" ]] || _v37_error "formal run requires ${_v37_ab_key}; use run_v37_strict_ab.sh"
  done
  export V37_AB_RUN_ID V37_AB_PLAN_PATH V37_AB_PLAN_SHA256 V37_AB_CELL_ID
  "${_V37_PREFLIGHT_PYTHON}" - \
    "${REPO_DIR}" "${V37_AB_PLAN_PATH}" "${V37_AB_PLAN_SHA256}" "${V37_AB_RUN_ID}" \
    "${V37_AB_CELL_ID}" "${V37_SEED}" "${V37_ARM}" "${V31_SAVE_CHECKPOINT_PATH}" <<'PY'
import sys
from pathlib import Path

repo, plan_path, plan_sha, run_id, cell_id, seed_text, arm, save_path = sys.argv[1:]
sys.path.insert(0, repo)
from tools import v37_gate

try:
    seed = int(seed_text)
    plan = v37_gate.validate_ab_plan(
        plan_path, plan_sha, run_id, require_target_dirs=False,
    )
except (ValueError, v37_gate.GateError) as exc:
    raise SystemExit(f"[V37-strict-winner][ERROR] invalid A/B plan: {exc}") from exc
if seed not in plan["seeds"] or cell_id != f"seed{seed}-{arm}":
    raise SystemExit("[V37-strict-winner][ERROR] A/B cell identity mismatch")
matching = [cell for cell in plan["cells"] if cell["cell_id"] == cell_id]
if len(matching) != 1 or Path(matching[0]["target_dir"]) != Path(save_path):
    raise SystemExit("[V37-strict-winner][ERROR] save path differs from preregistered A/B target")
PY
else
  for _v37_ab_key in V37_AB_RUN_ID V37_AB_PLAN_PATH V37_AB_PLAN_SHA256 V37_AB_CELL_ID; do
    [[ -z "${!_v37_ab_key:-}" ]] || _v37_error "${_v37_ab_key} is formal-only"
  done
fi
_V37_PREFLIGHT_TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/v37-preflight.XXXXXX")"
_V37_PREFLIGHT_JSON="${_V37_PREFLIGHT_TMPDIR}/v37_preflight.json"
_V37_RUN_DIR_CLAIMED=0
_v37_cleanup_preflight() {
  if [[ "${_V37_RUN_DIR_CLAIMED}" == "0" ]]; then
    rm -rf -- "${_V37_PREFLIGHT_TMPDIR}"
  fi
}
trap _v37_cleanup_preflight EXIT
_V37_PREFLIGHT_ARGS=(
  "${REPO_DIR}/tools/preflight_v37_training.py"
  "${STEPCOUNT_TRAIN_DATA}"
  --metadata "${STEPCOUNT_MASKS_METADATA}"
  --masks-dir "${STEPCOUNT_MASKS_DIR}"
  --tolerance "${_V37_RATIO_TOLERANCE}"
  --batch-size 256
  --json-out "${_V37_PREFLIGHT_JSON}"
  --data-mode "${V37_DATA_MODE}"
  --run-class "${V37_RUN_CLASS}"
  --entrypoint-contract frontier_rl
  --training-entrypoint "${REPO_DIR}/examples/v32_sparse_0_10_stable_drfix.sh"
)
for _v37_val_spec in "${_V37_VAL_SPECS[@]}"; do
  _v37_val_path="${_v37_val_spec#*::}"
  _V37_PREFLIGHT_ARGS+=(--val-data "${_v37_val_path}")
  if [[ "${V37_RUN_CLASS}" == "formal" ]]; then
    _V37_PREFLIGHT_ARGS+=(--expected-val-sha256 "$(_v37_sha256_tree "${_v37_val_path}")")
  fi
done
for _v37_forbidden_data in "${_V37_FORBIDDEN_DATA[@]}"; do
  [[ -e "${_v37_forbidden_data}" ]] \
    || _v37_error "canonical benchmark source is missing: ${_v37_forbidden_data}"
  _V37_PREFLIGHT_ARGS+=(--forbidden-data "${_v37_forbidden_data}")
  if [[ "${V37_RUN_CLASS}" == "formal" ]]; then
    _V37_PREFLIGHT_ARGS+=(--expected-forbidden-sha256 "$(_v37_sha256_tree "${_v37_forbidden_data}")")
  fi
done
if [[ "${V37_ALLOW_INDEPENDENT_VAL_OUTSIDE_TRAIN_RANGE:-0}" == "1" ]]; then
  _V37_PREFLIGHT_ARGS+=(--allow-val-answer-outside-train-range)
elif [[ "${_V37_VAL_HAS_BENCHMARK}" == "1" ]]; then
  _V37_PREFLIGHT_ARGS+=(--allow-val-without-image-path --allow-val-answer-outside-train-range)
elif [[ "${V37_RUN_CLASS}" != "debug" ]]; then
  _V37_PREFLIGHT_ARGS+=(--require-val-all-buckets)
fi
_V37_IMAGE_ROOTS=()
if [[ -n "${V37_IMAGE_ROOTS:-}" ]]; then
  IFS=':' read -r -a _V37_IMAGE_ROOTS <<< "${V37_IMAGE_ROOTS}"
else
  for _v37_image_root in "${STEPCOUNT_DATA_ROOT}/grouped_data/images"/*; do
    [[ -d "${_v37_image_root}" ]] && _V37_IMAGE_ROOTS+=("${_v37_image_root}")
  done
fi
_V37_CANONICAL_IMAGE_ROOTS=()
for _v37_image_root in "${_V37_IMAGE_ROOTS[@]}"; do
  [[ -d "${_v37_image_root}" ]] \
    || _v37_error "image root must name an existing directory: ${_v37_image_root}"
  _v37_image_root="$(readlink -f "${_v37_image_root}" 2>/dev/null || true)"
  [[ -n "${_v37_image_root}" ]] \
    || _v37_error "image root does not resolve to an existing path"
  _V37_CANONICAL_IMAGE_ROOTS+=("${_v37_image_root}")
  _V37_PREFLIGHT_ARGS+=(--image-root "${_v37_image_root}")
done
_V37_IMAGE_ROOTS=("${_V37_CANONICAL_IMAGE_ROOTS[@]}")
export V37_IMAGE_ROOTS="$(IFS=:; printf '%s' "${_V37_IMAGE_ROOTS[*]}")"
if [[ "${V37_RUN_CLASS}" == "formal" ]]; then
  _V37_PREFLIGHT_ARGS+=(--expected-ratios "${_V37_FORMAL_RATIOS}")
  _V37_PREFLIGHT_ARGS+=(--coverage-report "${V37_METADATA_COVERAGE_REPORT}")
  _V37_PREFLIGHT_ARGS+=(--filtered-manifest "${V37_FILTERED_MANIFEST}")
  _V37_PREFLIGHT_ARGS+=(--loader-model-path "${MODEL_PATH}")
  _V37_PREFLIGHT_ARGS+=(--loader-max-prompt-length "${V31_MAX_PROMPT_LENGTH}")
  _V37_PREFLIGHT_ARGS+=(--loader-max-pixels "${V31_MAX_PIXELS}" --loader-min-pixels "${V31_MIN_PIXELS}")
  # Exact loader contract: preflight and training share the same overlong
  # filtering parallelism instead of silently auditing a num_proc=1 variant.
  _V37_PREFLIGHT_ARGS+=(--loader-filter-num-proc "${V31_FILTER_OVERLONG_NUM_PROC}" --loader-system-prompt "${SYSTEM_PROMPT_FILE}")
  _V37_PREFLIGHT_ARGS+=(--expected-data-sha256 "$(_v37_sha256_tree "${STEPCOUNT_TRAIN_DATA}")")
  _V37_PREFLIGHT_ARGS+=(--expected-metadata-sha256 "$(_v37_sha256_tree "${STEPCOUNT_MASKS_METADATA}")")
  _V37_PREFLIGHT_ARGS+=(--expected-masks-sha256 "$(_v37_sha256_tree "${STEPCOUNT_MASKS_DIR}")")
  _V37_PREFLIGHT_ARGS+=(--expected-training-entrypoint-sha256 "$(_v37_sha256_file "${REPO_DIR}/examples/v32_sparse_0_10_stable_drfix.sh")")
elif [[ -n "${V37_EXPECTED_RATIOS:-}" ]]; then
  _V37_PREFLIGHT_ARGS+=(--expected-ratios "${V37_EXPECTED_RATIOS}")
fi

echo "[V37-strict-winner] running parquet/data/mask preflight..."
"${_V37_PREFLIGHT_PYTHON}" "${_V37_PREFLIGHT_ARGS[@]}"

# A production formal run must be reproducible from one immutable Git commit.
# Development probes keep working on a dirty tree because they never publish a
# promotable training artifact.
if [[ "${V37_RUN_CLASS}" == "formal" ]] && ! _v37_is_true "${V37_DRY_RUN:-0}"; then
  _v37_git_status="$(_v37_git status --porcelain=v1 -uall)" \
    || _v37_error "formal training could not inspect the Git worktree"
  [[ -z "${_v37_git_status}" ]] \
    || _v37_error "formal training requires a clean committed Git worktree"
  unset _v37_git_status
fi

# Claim the run directory only after every fail-closed preflight has passed.
mkdir -p "${V36_SAVE_ROOT}"
[[ ! -L "${V31_SAVE_CHECKPOINT_PATH}" ]] \
  || _v37_error "save path must not be a symlink: ${V31_SAVE_CHECKPOINT_PATH}"
if ! mkdir "${V31_SAVE_CHECKPOINT_PATH}" 2>/dev/null; then
  _v37_error "save path must not already exist; atomic claim failed: ${V31_SAVE_CHECKPOINT_PATH}"
fi
mv -- "${_V37_PREFLIGHT_JSON}" "${V31_SAVE_CHECKPOINT_PATH}/v37_preflight.json"
_V37_PREFLIGHT_JSON="${V31_SAVE_CHECKPOINT_PATH}/v37_preflight.json"
rm -rf -- "${_V37_PREFLIGHT_TMPDIR}"
_V37_RUN_DIR_CLAIMED=1
trap - EXIT

# Materialize the opt-in config in the run artifact; V36 defaults remain untouched.
_V37_SOURCE_CONFIG="${CONFIG_PATH}"
_V37_RUNTIME_CONFIG="${V31_SAVE_CHECKPOINT_PATH}/v37_config.yaml"
"${_V37_PREFLIGHT_PYTHON}" - "${_V37_SOURCE_CONFIG}" "${_V37_RUNTIME_CONFIG}" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
text = source.read_text(encoding="utf-8")
text, count = re.subn(
    r"(?m)^(\s*adaptive_actor_kl:)\s*(?:false|true)\s*$",
    r"\1 true",
    text,
)
if count != 1:
    raise SystemExit("V37 config requires exactly one algorithm.adaptive_actor_kl key")
temporary = target.with_name(f".{target.name}.tmp")
temporary.write_text(text, encoding="utf-8")
temporary.replace(target)
PY
export CONFIG_PATH="${_V37_RUNTIME_CONFIG}"
if _v37_is_true "${V37_DRY_RUN:-0}" && _v37_is_true "${V37_VALIDATE_DOWNSTREAM:-0}"; then
  export V32_DRY_RUN=1
fi

export _V37_PROMOTABLE
export _V37_NONPROMOTABLE_TEXT="$(IFS=,; echo "${_V37_NONPROMOTABLE_REASONS[*]}")"
export _V37_PREFLIGHT_JSON
export _V37_CONFIG_PATH="${CONFIG_PATH}"
export _V37_REWARD_FILE="${REWARD_FN_PATH%%:*}"
export _V37_LAUNCHER_FILE="$(readlink -f "${BASH_SOURCE[0]}")"
export _V37_PREFLIGHT_FILE="${REPO_DIR}/tools/preflight_v37_training.py"
export _V37_FRONTIER_BUILDER_FILE="${REPO_DIR}/tools/build_v37_frontier_dataset.py"
export _V37_CORE_ALGOS_FILE="${REPO_DIR}/verl/trainer/core_algos.py"
export _V37_TRAINER_FILE="${REPO_DIR}/verl/trainer/ray_trainer.py"
export _V37_REWARD_MANAGER_FILE="${REPO_DIR}/verl/workers/reward/function.py"
export _V37_ACTION_LEDGER_FILE="${REPO_DIR}/verl/utils/action_ledger.py"
export _V37_AB_LAUNCHER_FILE="${REPO_DIR}/examples/rl_launch/run_v37_strict_ab.sh"
export _V37_ACTOR_FILE="${REPO_DIR}/verl/workers/actor/dp_actor.py"
export _V37_ACTOR_CONFIG_FILE="${REPO_DIR}/verl/workers/actor/config.py"
export _V37_CHECKPOINT_MANAGER_FILE="${REPO_DIR}/verl/utils/checkpoint/checkpoint_manager.py"
export _V37_DATASET_FILE="${REPO_DIR}/verl/utils/dataset.py"
export _V37_PATH_REMAP_FILE="${REPO_DIR}/verl/utils/path_remap.py"
export _V37_POSTPROCESS_FILE="${REPO_DIR}/tools/v37_postprocess.py"
export _V37_EVAL_PRODUCER_FILE="${REPO_DIR}/tools/v37_eval_producer.py"
export _V37_PAIRED_UNIVERSE_FILE="${REPO_DIR}/tools/v37_paired_universe.py"
export _V37_CONTINUATION_FILE="${REPO_DIR}/tools/v37_continuation.py"
export _V37_MODEL_MERGER_FILE="${REPO_DIR}/scripts/model_merger.py"
export _V37_RAY_ENVIRONMENT_FILE="${REPO_DIR}/verl/utils/ray_environment.py"
export _V37_TRAINING_EVIDENCE_FILE="${REPO_DIR}/verl/utils/v37_training_evidence.py"
export _V37_FSDP_CHECKPOINT_MANAGER_FILE="${REPO_DIR}/verl/utils/checkpoint/fsdp_checkpoint_manager.py"
export _V37_FSDP_WORKERS_FILE="${REPO_DIR}/verl/workers/fsdp_workers.py"
export _V37_GATE_FILE="${REPO_DIR}/tools/v37_gate.py"
export _V37_BENCHMARK_EVIDENCE_FILE="${REPO_DIR}/tools/v37_benchmark_evidence.py"
export _V37_ROLLOUT_FILE="${REPO_DIR}/verl/workers/rollout/vllm_rollout_spmd.py"
export _V37_SHARDING_MANAGER_FILE="${REPO_DIR}/verl/workers/sharding_manager/fsdp_vllm.py"
export _V37_SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE}"
export _V37_PROCESS_PROMPT_FILE="${INTERLEAVED_PROCESS_PROMPT_FILE}"
export _V37_TORCH_FUNCTIONAL_FILE="${REPO_DIR}/verl/utils/torch_functional.py"
export _V37_TRAINER_CONFIG_FILE="${REPO_DIR}/verl/trainer/config.py"
export _V37_TRAINER_MAIN_FILE="${REPO_DIR}/verl/trainer/main.py"
export _V37_RAY_WORKER_BASE_FILE="${REPO_DIR}/verl/single_controller/ray/base.py"
export _V37_LOCAL_PATH_ENV_FILE="${REPO_DIR}/examples/local_path_env.sh"
export _V37_REQUIREMENTS_FILE="${REPO_DIR}/requirements.txt"
export _V37_BASE_LAUNCHER_FILE="${REPO_DIR}/examples/v36_dense_11_50_full_from_1m_ckpt476.sh"
export _V37_TRAINING_ENTRY_FILE="${REPO_DIR}/examples/v32_sparse_0_10_stable_drfix.sh"
export _V37_METADATA_COVERAGE_REPORT="${V37_METADATA_COVERAGE_REPORT:-}"
export _V37_FILTERED_MANIFEST="${V37_FILTERED_MANIFEST:-}"
export _V37_FRONTIER_MANIFEST="${V37_FRONTIER_MANIFEST:-}"
export _V37_SOURCE_CONFIG
export _V37_GIT_COMMIT="$(_v37_git rev-parse HEAD 2>/dev/null || echo unknown)"
export _V37_GIT_STATUS_SHA256="$(_v37_git status --porcelain=v1 -uall | sha256sum | awk '{print $1}')"
export _V37_GIT_DIFF_SHA256="$(_v37_git diff --no-ext-diff --no-textconv --binary HEAD | sha256sum | awk '{print $1}')"
if [[ "${V37_RUN_CLASS}" == "formal" ]] && ! _v37_is_true "${V37_DRY_RUN:-0}"; then
  _V37_EMPTY_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
  [[ "${_V37_GIT_STATUS_SHA256}" == "${_V37_EMPTY_SHA256}" \
      && "${_V37_GIT_DIFF_SHA256}" == "${_V37_EMPTY_SHA256}" ]] \
    || _v37_error "formal Git worktree changed after preflight; refusing manifest publication"
  unset _V37_EMPTY_SHA256
fi
export V37_RUN_MANIFEST="${V31_SAVE_CHECKPOINT_PATH}/v37_run_manifest.json"
_V37_EXPECTED_TRAINING_EVIDENCE="${V31_SAVE_CHECKPOINT_PATH}/v37_training_evidence.json"
if [[ -n "${V37_TRAINING_EVIDENCE_PATH:-}" \
      && "$(_v37_absolute_path "${V37_TRAINING_EVIDENCE_PATH}")" != "${_V37_EXPECTED_TRAINING_EVIDENCE}" ]]; then
  _v37_error "V37_TRAINING_EVIDENCE_PATH must stay inside the claimed run directory"
fi
export V37_TRAINING_EVIDENCE_PATH="${_V37_EXPECTED_TRAINING_EVIDENCE}"
export V37_TRAINING_EVIDENCE_REQUIRED=1
_V37_EXPECTED_EFFECTIVE_ENVIRONMENT="${V31_SAVE_CHECKPOINT_PATH}/v37_effective_environment.json"
if [[ -n "${V37_EFFECTIVE_ENVIRONMENT_PATH:-}" \
      && "$(_v37_absolute_path "${V37_EFFECTIVE_ENVIRONMENT_PATH}")" != "${_V37_EXPECTED_EFFECTIVE_ENVIRONMENT}" ]]; then
  _v37_error "V37_EFFECTIVE_ENVIRONMENT_PATH must stay inside the claimed run directory"
fi
export V37_EFFECTIVE_ENVIRONMENT_PATH="${_V37_EXPECTED_EFFECTIVE_ENVIRONMENT}"
"${_V37_PREFLIGHT_PYTHON}" - <<'PY'
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


digest_cache = {}


def strict_json_load(path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


def digest(path_text):
    if not path_text:
        return None
    raw_path = Path(path_text)
    reject_symlinks(raw_path, "file hash target")
    path = raw_path.resolve()
    cache_key = str(path)
    if cache_key in digest_cache:
        return digest_cache[cache_key]
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while hashing: {path}")
    value = h.hexdigest()
    digest_cache[cache_key] = value
    return value


def reject_symlinks(path, label):
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise RuntimeError(f"{label} does not exist: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{label} contains a symlink component: {current}")
    path = absolute
    if not path.is_dir():
        return
    for root, directories, filenames in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in directories + filenames:
            candidate = root_path / name
            if candidate.is_symlink():
                raise RuntimeError(f"{label} contains a symlink: {candidate}")


def digest_tree(path_text, *, parquet_only=False):
    raw_root = Path(path_text)
    reject_symlinks(raw_root, "snapshot target")
    root = raw_root.resolve()
    if root.is_file():
        files = [root]
        base = root.parent
    else:
        files = sorted(path for path in root.rglob("*") if path.is_file())
        base = root
    if parquet_only:
        files = [path for path in files if path.suffix.lower() == ".parquet"]
    if not files:
        raise RuntimeError(f"no files available for content snapshot: {root}")
    entries = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(base).as_posix()
        size = path.stat().st_size
        sha256 = digest(str(path))
        aggregate.update(relative.encode("utf-8") + b"\0")
        aggregate.update(str(size).encode("ascii") + b"\0")
        aggregate.update(sha256.encode("ascii") + b"\n")
        entries.append({"path": relative, "size": size, "sha256": sha256})
        total_bytes += size
    return {
        "root": str(root),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "sha256": aggregate.hexdigest(),
        "files": entries,
    }


def gate_tree_digest(path_text):
    """Match tools/v37_gate.py tree_sha256 for live artifact verification."""
    raw_root = Path(path_text)
    reject_symlinks(raw_root, "hash target")
    root = raw_root.resolve()
    if root.is_file():
        return digest(str(root))
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"empty tree cannot be audited: {root}")
    entries = [
        {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": digest(str(path))}
        for path in files
    ]
    encoded = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validation_snapshots(spec_text):
    snapshots = []
    for item in spec_text.split(","):
        suite, separator, path_text = item.partition("::")
        if not separator:
            path_text = suite
            suite = "default"
        snapshots.append({
            "suite": suite,
            "dataset": digest_tree(path_text, parquet_only=True),
        })
    return snapshots


preflight_path = Path(os.environ["_V37_PREFLIGHT_JSON"])
preflight = strict_json_load(preflight_path)
input_snapshots = {
    "model": digest_tree(os.environ["MODEL_PATH"]),
    "training_data": digest_tree(os.environ["STEPCOUNT_TRAIN_DATA"], parquet_only=True),
    "validation_data": validation_snapshots(os.environ["STEPCOUNT_VAL_DATA"]),
    "mask_metadata": {
        "path": os.environ["STEPCOUNT_MASKS_METADATA"],
        "sha256": digest(os.environ["STEPCOUNT_MASKS_METADATA"]),
    },
}
expected_initial_model_sha256 = os.environ.get("V37_EXPECTED_INITIAL_MODEL_SHA256")
if os.environ.get("V37_RUN_CLASS") == "formal" and (
    input_snapshots["model"]["sha256"] != expected_initial_model_sha256
):
    raise RuntimeError(
        "formal initial model differs from V37_EXPECTED_INITIAL_MODEL_SHA256"
    )
audited_prefixes = (
    "ACTION_", "ACTOR_", "ADAPTIVE_", "ANSWER_", "BOK_", "CLIP_", "EASYR1_", "GRAD_", "INTERLEAVED_",
    "KL_", "POINT_", "POLICY_", "PROCESS_", "REWARD_", "ROLLOUT_", "STEPCOUNT_",
    "TRAIN_", "TRAINER_", "TRAJECTORY_", "TRAJ_", "V31_", "V32_", "V36_", "V37_", "VCRL_",
    "CUBLAS_", "CUDA_", "FLASH_", "FSDP_", "MKL_", "NCCL_", "OMP_", "PYTORCH_",
    "RAY_", "TOKENIZERS_", "TORCH_", "TRANSFORMERS_", "VLLM_", "XFORMERS_",
    "FI_", "GLOO_", "MASTER_", "NVIDIA_", "OMPI_", "PMI_", "PMIX_", "TRITON_", "UCX_",
)
def sensitive_name(key):
    upper = key.upper()
    return (
        any(fragment in upper for fragment in (
            "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "API_KEY", "ACCESS_KEY",
            "PRIVATE_KEY", "AUTH", "BEARER", "COOKIE", "SESSION",
        ))
        or upper.endswith("_TOKEN")
        or "_TOKEN_" in upper
    )

def sensitive_json(value):
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return False
    def visit(item):
        if isinstance(item, dict):
            return any(sensitive_name(str(key)) or visit(child) for key, child in item.items())
        if isinstance(item, list):
            return any(visit(child) for child in item)
        return False
    return visit(payload)

audited_candidates = {
    key
    for key in os.environ
    if key.startswith(audited_prefixes)
}
sensitive_candidates = sorted(
    key for key in audited_candidates
    if sensitive_name(key) or (key.endswith("_JSON") and sensitive_json(os.environ[key]))
)
if sensitive_candidates:
    raise RuntimeError(
        "V37 refuses to persist sensitive environment variable names: "
        + ",".join(sensitive_candidates)
    )
audited_environment = {
    key: value
    for key, value in sorted(os.environ.items())
    if key not in {"V37_EXECUTION_ENVIRONMENT_PATH", "V37_EXECUTION_ENVIRONMENT_SHA256"}
    and (key.startswith(audited_prefixes)
    or key in {
        "CC", "CHIEF_IP", "CONFIG_PATH", "CONDA_PREFIX", "CUDA_HOME", "CXX", "DISABLE_KL",
        "HOST_GPU_NUM", "HOST_NUM", "INDEX", "LD_LIBRARY_PATH", "LD_PRELOAD", "MAX_STEPS",
        "HF_DATASETS_OFFLINE", "HF_HOME", "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_OFFLINE", "MODEL_PATH",
        "HOME", "HOSTNAME", "LANG", "LC_ALL", "LOCAL_RANK", "LOGNAME", "PATH", "PWD", "PYTHONHASHSEED",
        "PYTHONIOENCODING", "PYTHONNOUSERSITE", "PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONWARNINGS",
        "RANK", "SHELL", "SYSTEM_PROMPT_FILE", "TERM", "TMP", "TEMP", "TMPDIR", "TZ", "USE_KL_LOSS",
        "USER", "VIRTUAL_ENV",
        "WANDB_DIR", "WANDB_ENTITY", "WANDB_MODE", "WANDB_PROJECT", "WORLD_SIZE", "XDG_CACHE_HOME",
    })
}
execution_environment_path = Path(os.environ["V37_RUN_MANIFEST"]).with_name("v37_execution_environment.json")
execution_environment_hash = hashlib.sha256(json.dumps(
    audited_environment, sort_keys=True, separators=(",", ":"), allow_nan=False,
).encode("utf-8")).hexdigest()
audited_environment.update({
    "V37_EXECUTION_ENVIRONMENT_PATH": str(execution_environment_path.resolve()),
    "V37_EXECUTION_ENVIRONMENT_SHA256": execution_environment_hash,
})
execution_payload = {
    "schema_version": 1,
    "hash_contract": "canonical_environment_without_self_reference_v1",
    "environment_sha256": execution_environment_hash,
    "environment": audited_environment,
}
execution_temporary = execution_environment_path.with_name(
    f".{execution_environment_path.name}.tmp.{os.getpid()}"
)
execution_temporary.write_text(
    json.dumps(execution_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
execution_temporary.replace(execution_environment_path)
package_names = ("flash-attn", "numpy", "pyarrow", "ray", "tensordict", "torch", "transformers", "vllm")


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


package_versions = {name: package_version(name) for name in package_names}
missing_package_versions = [name for name, version in package_versions.items() if version is None]
if missing_package_versions:
    raise RuntimeError(f"formal runtime package versions are unavailable: {missing_package_versions}")
runtime_environment = {
    "python_executable": str(Path(sys.executable).resolve()),
    "python_executable_sha256": digest(str(Path(sys.executable).resolve())),
    "python_version": platform.python_version(),
    "packages": package_versions,
}
config = {
    "arm": os.environ["V37_ARM"],
    "estimator": os.environ["ADV_ESTIMATOR"],
    "process_reward": os.environ["ACTION_EVENT_REWARD_ENABLE"],
    "step_signal": os.environ.get("BOK_STEP_SIGNAL"),
    "step_weight": os.environ["BOK_STEP_WEIGHT"],
    "answer_gate": {"mode": os.environ.get("BOK_STEP_GATE"), "minimum": os.environ.get("BOK_STEP_MIN_GATE")},
    "rollout_n": int(os.environ["ROLLOUT_N"]),
    "rollout_temperature": os.environ["ROLLOUT_TEMPERATURE"],
    "actor_lr": os.environ["ACTOR_LR"],
    "kl": {
        "adaptive_actor_kl": os.environ["ADAPTIVE_ACTOR_KL"],
        "use_kl_loss": os.environ["USE_KL_LOSS"],
        "type": os.environ["KL_TYPE"],
        "penalty": os.environ["KL_PENALTY"],
        "init_beta": os.environ["KL_COEF"],
        "target": os.environ["KL_TARGET"],
        "horizon": os.environ["KL_HORIZON"],
        "horizon_unit": os.environ["KL_HORIZON_UNIT"],
        "loss_reduction": "response_token_mean",
        "selector_includes_kl": False,
    },
    "correctness_first": os.environ["BOK_CORRECTNESS_FIRST"],
    "allwrong_terminal_zero": os.environ["BOK_ALLWRONG_TERMINAL_ZERO"],
    "correctness_secondary": {
        "task_weight": os.environ["BOK_CORRECTNESS_TASK_WEIGHT"],
        "quality_weight": os.environ["BOK_CORRECTNESS_QUALITY_WEIGHT"],
        "partial_scale": os.environ["BOK_CORRECTNESS_PARTIAL_SCALE"],
    },
    "reward_fail_closed": os.environ["V37_REWARD_FAIL_CLOSED"],
    "legacy_process_reward_enable": os.environ["PROCESS_REWARD_ENABLE"],
    "bok_tau": {"init": os.environ["BOK_TAU_INIT"], "final": os.environ["BOK_TAU_FINAL"], "schedule": os.environ["BOK_TAU_SCHEDULE"]},
    "winner_boost": os.environ["BOK_WINNER_BOOST"],
    "dapo_filter": os.environ["BOK_DAPO_FILTER"],
    "allwrong_cap": os.environ["BOK_ALLWRONG_CAP"],
    "allwrong_negative_only": os.environ["BOK_ALLWRONG_NEG_ONLY"],
    "filter_all_correct": os.environ["BOK_FILTER_ALL_CORRECT"],
    "smart_filter_threshold": os.environ["BOK_SMART_FILTER_THRESHOLD"],
    "pilot_steps": int(os.environ["V37_PILOT_STEPS"]),
    "adaptive_turn_formula": "min(INTERLEAVED_MAX_TURNS, GT_answer + 1 + 2)",
    "topology": "1x8-H200",
    "micro_update": int(os.environ["V31_MICRO_BATCH_UPDATE"]),
    "micro_experience": int(os.environ["V31_MICRO_BATCH_EXP"]),
    "vllm_blocks": int(os.environ["EASYR1_VLLM_NUM_GPU_BLOCKS"]),
    "gpu_memory_utilization": os.environ["V31_GPU_MEM_UTIL"],
    "filter_overlong_num_proc": int(os.environ["V31_FILTER_OVERLONG_NUM_PROC"]),
    "cp_size": int(os.environ["V37_CP_SIZE"]),
    "hf_merge": {
        "host_memory_preflight": os.environ["V37_HF_MERGE_HOST_MEMORY_PREFLIGHT"],
        "host_memory_safety_factor": os.environ["V37_HF_MERGE_HOST_MEMORY_SAFETY_FACTOR"],
    },
    "world_size": int(os.environ["V31_NNODES"]) * int(os.environ["V31_N_GPUS_PER_NODE"]),
    "fallback_logprob_sign_opt_in": os.environ["V37_FALLBACK_LOGPROB_SIGN_OPT_IN"] == "1",
    "torch_logprob_fallback_mode": os.environ["TORCH_LOGPROB_FALLBACK_MODE"],
}
continuation_spec = importlib.util.spec_from_file_location(
    "v37_continuation_manifest", os.environ["_V37_CONTINUATION_FILE"]
)
if continuation_spec is None or continuation_spec.loader is None:
    raise RuntimeError("cannot load V37 continuation verifier")
continuation_module = importlib.util.module_from_spec(continuation_spec)
continuation_spec.loader.exec_module(continuation_module)
continuation_config_hash = continuation_module.continuation_config_sha256(config)
frontier_manifest_path = os.environ.get("_V37_FRONTIER_MANIFEST")
frontier_manifest = (
    strict_json_load(frontier_manifest_path)
    if frontier_manifest_path else None
)
ab_preregistration = (
    {
        "schema_version": 1,
        "ab_run_id": os.environ["V37_AB_RUN_ID"],
        "plan_path": os.environ["V37_AB_PLAN_PATH"],
        "plan_sha256": os.environ["V37_AB_PLAN_SHA256"],
        "cell_id": os.environ["V37_AB_CELL_ID"],
    }
    if os.environ["V37_RUN_CLASS"] == "formal" else None
)
manifest = {
    "manifest_version": 3,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_class": os.environ["V37_RUN_CLASS"],
    "data_mode": os.environ["V37_DATA_MODE"],
    "arm": os.environ["V37_ARM"],
    "seed": int(os.environ["V37_SEED"]),
    "promotable_candidate": os.environ["_V37_PROMOTABLE"] == "1",
    "run_purpose": os.environ["V37_RUN_PURPOSE"],
    "resume_mode": os.environ["V37_RESUME_MODE"],
    "ab_preregistration": ab_preregistration,
    "resume_evidence": None,
    "promotability": {
        "candidate": os.environ["_V37_PROMOTABLE"] == "1",
        "requires_final_gate": True,
        "run_class": os.environ["V37_RUN_CLASS"],
    },
    "nonpromotable_reasons": [item for item in os.environ.get("_V37_NONPROMOTABLE_TEXT", "").split(",") if item],
    "git_commit": os.environ["_V37_GIT_COMMIT"],
    "git_status_sha256": os.environ["_V37_GIT_STATUS_SHA256"],
    "git_diff_sha256": os.environ["_V37_GIT_DIFF_SHA256"],
    "config_path": os.environ["_V37_CONFIG_PATH"],
    "model_path": os.environ["MODEL_PATH"],
    "train_data": os.environ["STEPCOUNT_TRAIN_DATA"],
    "validation_data": os.environ["STEPCOUNT_VAL_DATA"],
    "mask_metadata": os.environ["STEPCOUNT_MASKS_METADATA"],
    "masks_dir": os.environ["STEPCOUNT_MASKS_DIR"],
    "metadata_coverage_report": os.environ.get("_V37_METADATA_COVERAGE_REPORT") or None,
    "metadata_coverage_report_sha256": digest(os.environ.get("_V37_METADATA_COVERAGE_REPORT")),
    "filtered_manifest": os.environ.get("_V37_FILTERED_MANIFEST") or None,
    "filtered_manifest_sha256": digest(os.environ.get("_V37_FILTERED_MANIFEST")),
    "dataset_manifest": frontier_manifest_path or None,
    "dataset_manifest_sha256": digest(frontier_manifest_path),
    "miner_manifest_hashes": frontier_manifest.get("audit_sha256") if frontier_manifest else None,
    "source_dataset_sha256": frontier_manifest.get("source_sha256") if frontier_manifest else None,
    "input_snapshots": input_snapshots,
    "preflight_report": str(preflight_path),
    "preflight_report_sha256": digest(str(preflight_path)),
    "preflight_summary": {
        "sample_count": preflight["sample_count"],
        "answer_buckets": preflight["answer_buckets"],
        "train_validation_overlap_count": preflight["train_validation_overlap_count"],
        "train_benchmark_overlap_count": preflight["train_benchmark_overlap_count"],
        "forbidden_sample_count": preflight["forbidden_sample_count"],
        "forbidden_identity_sha256": preflight["forbidden_identity_sha256"],
        "invalid_embedded_image_count": preflight["invalid_embedded_image_count"],
        "validation_rows_without_image_path": preflight["validation_rows_without_image_path"],
        "validation_missing_answer_count": preflight["validation_missing_answer_count"],
        "validation_invalid_answer_count": preflight["validation_invalid_answer_count"],
        "validation_invalid_image_count": preflight["validation_invalid_image_count"],
        "validation_answer_buckets": preflight["validation_answer_buckets"],
        "training_identity_sha256": preflight["training_identity_sha256"],
        "validation_identity_sha256": preflight["validation_identity_sha256"],
        "formal_contract": preflight.get("formal_contract"),
        "formal_input_snapshots": preflight.get("formal_input_snapshots"),
    },
    "dataset_sha256": input_snapshots["training_data"]["sha256"],
    "mask_tree_path": os.environ["STEPCOUNT_MASKS_DIR"],
    "mask_tree_sha256": (
        preflight.get("formal_contract", {}).get("mask_tree_sha256")
        or gate_tree_digest(os.environ["STEPCOUNT_MASKS_DIR"])
    ),
    "metadata_sha256": input_snapshots["mask_metadata"]["sha256"],
    "paired_eval_data_sha256": hashlib.sha256(json.dumps(
        [
            {"suite": item["suite"], "sha256": item["dataset"]["sha256"]}
            for item in input_snapshots["validation_data"]
        ],
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest(),
    "input_hashes": {
        "train": input_snapshots["training_data"]["sha256"],
        **{
            f"validation_{index}": item["dataset"]["sha256"]
            for index, item in enumerate(input_snapshots["validation_data"])
        },
        **({"source": frontier_manifest["source_sha256"]} if frontier_manifest else {}),
    },
    "mechanisms": {
        "adv_estimator": os.environ["ADV_ESTIMATOR"],
        "action_event_ledger_enable": os.environ["ACTION_EVENT_LEDGER_ENABLE"],
        "action_event_reward_enable": os.environ["ACTION_EVENT_REWARD_ENABLE"],
        "action_parser_contract": os.environ.get("V37_ACTION_PARSER_CONTRACT", "0"),
        "action_ledger_contract": os.environ.get("V37_ACTION_LEDGER_CONTRACT", "0"),
        "legacy_process_reward_enable": os.environ["PROCESS_REWARD_ENABLE"],
        "adaptive_actor_kl": os.environ["ADAPTIVE_ACTOR_KL"],
        "use_kl_loss": os.environ["USE_KL_LOSS"],
        "correctness_first": os.environ["BOK_CORRECTNESS_FIRST"],
        "allwrong_terminal_zero": os.environ["BOK_ALLWRONG_TERMINAL_ZERO"],
        "reward_fail_closed": os.environ["V37_REWARD_FAIL_CLOSED"],
        "winner_mode": os.environ["V37_WINNER_MODE"],
        "winner_boost": os.environ["BOK_WINNER_BOOST"],
        "cp_size": int(os.environ["V37_CP_SIZE"]),
    },
    "mechanism_config": {
        "schema_version": 3,
        "arm": os.environ["V37_ARM"],
        "estimator": os.environ["ADV_ESTIMATOR"],
        "native_action_enabled": os.environ["ACTION_EVENT_REWARD_ENABLE"] == "1",
        "native_action_parser_contract": "native_action_parser_v1",
        "native_action_ledger_contract": "native_action_ledger_v2",
        "legacy_process_reward_enabled": os.environ["PROCESS_REWARD_ENABLE"] == "1",
        "step_signal": os.environ.get("BOK_STEP_SIGNAL"),
        "step_weight": float(os.environ["BOK_STEP_WEIGHT"]),
        "answer_gate": {
            "mode": os.environ.get("BOK_STEP_GATE"),
            "minimum": float(os.environ["BOK_STEP_MIN_GATE"]) if os.environ.get("BOK_STEP_MIN_GATE") else None,
        },
        "correctness_first": os.environ["BOK_CORRECTNESS_FIRST"] == "1",
        "allwrong_terminal_zero": os.environ["BOK_ALLWRONG_TERMINAL_ZERO"] == "1",
        "correctness_secondary": {
            "task_weight": float(os.environ["BOK_CORRECTNESS_TASK_WEIGHT"]),
            "quality_weight": float(os.environ["BOK_CORRECTNESS_QUALITY_WEIGHT"]),
            "partial_scale": float(os.environ["BOK_CORRECTNESS_PARTIAL_SCALE"]),
            "all_answer_wrong_terminal_zero": True,
            "exact_answer_partial_enabled": True,
        },
        "reward_fail_closed": os.environ["V37_REWARD_FAIL_CLOSED"] == "1",
        "adaptive_actor_kl": {
            "enabled": os.environ["ADAPTIVE_ACTOR_KL"].lower() == "true",
            "type": os.environ["KL_TYPE"], "penalty": os.environ["KL_PENALTY"],
            "init_beta": float(os.environ["KL_COEF"]), "target": float(os.environ["KL_TARGET"]),
            "horizon": int(os.environ["KL_HORIZON"]),
            "horizon_unit": os.environ["KL_HORIZON_UNIT"],
            "loss_reduction": "response_token_mean", "selector_includes_kl": False,
        },
        "legacy_kl": {
            "use_kl_loss": os.environ["USE_KL_LOSS"].lower() == "true",
            "reward_kl_enabled": False,
        },
        "cp_size": int(os.environ["V37_CP_SIZE"]),
        "fallback_logprob_sign_opt_in": os.environ["V37_FALLBACK_LOGPROB_SIGN_OPT_IN"] == "1",
        "torch_logprob_fallback_mode": os.environ["TORCH_LOGPROB_FALLBACK_MODE"],
        "strict_answer_integer_parse": os.environ["TRAJ_STRICT_ANSWER_INTEGER_PARSE"] == "1",
        "strict_raw_success_winner": os.environ["V37_RAW_SUCCESS_STRICT_WINNER"] == "1",
        "winner_mode": os.environ["V37_WINNER_MODE"],
        "strict_point_parser_contract": "strict_point_slots_v2",
    },
    "mining_contract": {
        "seeds": frontier_manifest.get("seeds") if frontier_manifest else None,
        "candidates_per_seed": frontier_manifest.get("candidates_per_seed") if frontier_manifest else None,
        "classifier_thresholds": frontier_manifest.get("classifier_thresholds") if frontier_manifest else None,
        "outcome_ratio": frontier_manifest.get("outcome_ratio") if frontier_manifest else None,
        "process_ratio": frontier_manifest.get("process_ratio") if frontier_manifest else None,
        "bucket_ratios": frontier_manifest.get("bucket_ratios") if frontier_manifest else None,
        "model_checkpoint_path": (
            frontier_manifest.get("mining_generation_contract", {}).get("model_checkpoint_path")
            if frontier_manifest else None
        ),
        "model_checkpoint_sha256": (
            frontier_manifest.get("mining_generation_contract", {}).get("model_checkpoint_sha256")
            if frontier_manifest else None
        ),
        "source_binding_contract": (
            frontier_manifest.get("mining_generation_contract", {}).get("source_binding_contract")
            if frontier_manifest else None
        ),
        "source_binding_sha256": (
            frontier_manifest.get("mining_generation_contract", {}).get("source_binding_sha256")
            if frontier_manifest else None
        ),
        "generation_config_by_seed": (
            frontier_manifest.get("mining_generation_contract", {}).get("generation_config_by_seed")
            if frontier_manifest else None
        ),
        "sampling_config": (
            frontier_manifest.get("mining_generation_contract", {}).get("sampling_config")
            if frontier_manifest else None
        ),
        "seed_difference_allowlist": (
            frontier_manifest.get("mining_generation_contract", {}).get("seed_difference_allowlist")
            if frontier_manifest else None
        ),
    },
    "config": config,
    "continuation_config_sha256": continuation_config_hash,
    "audited_environment": audited_environment,
    "runtime_environment": runtime_environment,
    "execution_environment_schema": "v37_explicit_environment_v1",
    "execution_environment_path": str(execution_environment_path.resolve()),
    "execution_environment_sha256": execution_environment_hash,
    "effective_environment_path": os.environ["V37_EFFECTIVE_ENVIRONMENT_PATH"],
    "effective_environment_sha256": None,
    "training_evidence_path": os.environ["V37_TRAINING_EVIDENCE_PATH"],
    "training_evidence_sha256": None,
    "config_file_sha256": digest(os.environ["_V37_CONFIG_PATH"]),
    "source_config_sha256": digest(os.environ["_V37_SOURCE_CONFIG"]),
    "reward_function": os.environ["REWARD_FN_PATH"],
    "reward_function_sha256": digest(os.environ["_V37_REWARD_FILE"]),
    "implementation_sha256": {
        "ab_launcher": digest(os.environ["_V37_AB_LAUNCHER_FILE"]),
        "action_ledger": digest(os.environ["_V37_ACTION_LEDGER_FILE"]),
        "actor": digest(os.environ["_V37_ACTOR_FILE"]),
        "actor_config": digest(os.environ["_V37_ACTOR_CONFIG_FILE"]),
        "launcher": digest(os.environ["_V37_LAUNCHER_FILE"]),
        "preflight": digest(os.environ["_V37_PREFLIGHT_FILE"]),
        "frontier_builder": digest(os.environ["_V37_FRONTIER_BUILDER_FILE"]),
        "core_algos": digest(os.environ["_V37_CORE_ALGOS_FILE"]),
        "trainer": digest(os.environ["_V37_TRAINER_FILE"]),
        "reward_manager": digest(os.environ["_V37_REWARD_MANAGER_FILE"]),
        "checkpoint_manager": digest(os.environ["_V37_CHECKPOINT_MANAGER_FILE"]),
        "dataset": digest(os.environ["_V37_DATASET_FILE"]),
        "path_remap": digest(os.environ["_V37_PATH_REMAP_FILE"]),
        "postprocess": digest(os.environ["_V37_POSTPROCESS_FILE"]),
        "eval_producer": digest(os.environ["_V37_EVAL_PRODUCER_FILE"]),
        "paired_universe": digest(os.environ["_V37_PAIRED_UNIVERSE_FILE"]),
        "continuation": digest(os.environ["_V37_CONTINUATION_FILE"]),
        "model_merger": digest(os.environ["_V37_MODEL_MERGER_FILE"]),
        "ray_environment": digest(os.environ["_V37_RAY_ENVIRONMENT_FILE"]),
        "training_evidence": digest(os.environ["_V37_TRAINING_EVIDENCE_FILE"]),
        "fsdp_checkpoint_manager": digest(os.environ["_V37_FSDP_CHECKPOINT_MANAGER_FILE"]),
        "fsdp_workers": digest(os.environ["_V37_FSDP_WORKERS_FILE"]),
        "gate": digest(os.environ["_V37_GATE_FILE"]),
        "reward_function": digest(os.environ["_V37_REWARD_FILE"]),
        "rollout": digest(os.environ["_V37_ROLLOUT_FILE"]),
        "sharding_manager": digest(os.environ["_V37_SHARDING_MANAGER_FILE"]),
        "source_config": digest(os.environ["_V37_SOURCE_CONFIG"]),
        "system_prompt": digest(os.environ["_V37_SYSTEM_PROMPT_FILE"]),
        "process_prompt": digest(os.environ["_V37_PROCESS_PROMPT_FILE"]),
        "torch_functional": digest(os.environ["_V37_TORCH_FUNCTIONAL_FILE"]),
        "trainer_config": digest(os.environ["_V37_TRAINER_CONFIG_FILE"]),
        "trainer_main": digest(os.environ["_V37_TRAINER_MAIN_FILE"]),
        "ray_worker_base": digest(os.environ["_V37_RAY_WORKER_BASE_FILE"]),
        "local_path_env": digest(os.environ["_V37_LOCAL_PATH_ENV_FILE"]),
        "requirements": digest(os.environ["_V37_REQUIREMENTS_FILE"]),
        "base_launcher": digest(os.environ["_V37_BASE_LAUNCHER_FILE"]),
        "benchmark_evidence": digest(os.environ["_V37_BENCHMARK_EVIDENCE_FILE"]),
        "training_entry": digest(os.environ["_V37_TRAINING_ENTRY_FILE"]),
    },
}
if os.environ["V37_RESUME_MODE"] == "controlled_continuation":
    manifest["resume_evidence"] = continuation_module.validate_continuation(
        os.environ["V37_RESUME_SEAL"],
        os.environ["_V37_RESUME_CHECKPOINT"],
        manifest,
        expected_seal_sha256=os.environ["_V37_RESUME_SEAL_SHA256"],
        expected_checkpoint_sha256=os.environ["_V37_RESUME_CHECKPOINT_SHA256"],
        expected_source_manifest_path=os.environ["_V37_RESUME_SOURCE_MANIFEST"],
        expected_frontier_manifest_sha256=os.environ["V37_FRONTIER_MANIFEST_SHA256"],
    )
encoded_config = json.dumps(
    {"selected": config, "environment": audited_environment},
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode()
manifest["config_sha256"] = hashlib.sha256(encoded_config).hexdigest()
identity_payload = {
    "arm": manifest["arm"], "seed": manifest["seed"],
    "run_directory": str(Path(os.environ["V37_RUN_MANIFEST"]).resolve().parent),
    "manifest_path": str(Path(os.environ["V37_RUN_MANIFEST"]).resolve()),
    "config_sha256": manifest["config_sha256"],
    "runtime_config_sha256": manifest["config_file_sha256"],
    "source_config_sha256": manifest["source_config_sha256"],
    "runtime_environment": manifest["runtime_environment"],
    "execution_environment_sha256": manifest["execution_environment_sha256"],
    "ab_preregistration": manifest["ab_preregistration"],
    "resume_evidence": manifest["resume_evidence"],
    "git_commit": manifest["git_commit"],
    "git_status_sha256": manifest["git_status_sha256"],
    "git_diff_sha256": manifest["git_diff_sha256"],
    "implementation_sha256": manifest["implementation_sha256"],
    "initial_model_sha256": input_snapshots["model"]["sha256"],
    "dataset_sha256": manifest["dataset_sha256"],
}
manifest["evidence_run_id"] = hashlib.sha256(json.dumps(
    identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
).encode()).hexdigest()
target = Path(os.environ["V37_RUN_MANIFEST"])
temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
temporary.replace(target)
PY

echo "================================================================"
echo "[V37-strict-winner] class=${V37_RUN_CLASS} mode=${V37_DATA_MODE} arm=${V37_ARM} adv=${ADV_ESTIMATOR} action_ledger=${ACTION_EVENT_LEDGER_ENABLE} native_action_reward=${ACTION_EVENT_REWARD_ENABLE} legacy_process=${PROCESS_REWARD_ENABLE} step_weight=${BOK_STEP_WEIGHT}"
echo "[V37-strict-winner] resume_mode=${V37_RESUME_MODE} model=${MODEL_PATH} resume=${_V37_RESUME_CHECKPOINT:-<none>}"
echo "[V37-strict-winner] data=${STEPCOUNT_TRAIN_DATA} focused10k_pilot=${_V37_FOCUSED_PILOT}"
echo "[V37-strict-winner] val=${STEPCOUNT_VAL_DATA} benchmark_dev=${_V37_VAL_HAS_BENCHMARK}"
echo "[V37-strict-winner] promotable_candidate=${_V37_PROMOTABLE} reasons=${_V37_NONPROMOTABLE_TEXT:-none}"
echo "[V37-strict-winner] masks_meta=${STEPCOUNT_MASKS_METADATA} masks_dir=${STEPCOUNT_MASKS_DIR}"
echo "[V37-strict-winner] image_roots=${_V37_IMAGE_ROOTS[*]:-<dataset-relative-only>}"
echo "[V37-strict-winner] topology=${V31_NNODES}x${V31_N_GPUS_PER_NODE} profile=${STEPCOUNT_HARDWARE_PROFILE} micro=${V31_MICRO_BATCH_UPDATE}/${V31_MICRO_BATCH_EXP} blocks=${EASYR1_VLLM_NUM_GPU_BLOCKS} gpu_mem=${V31_GPU_MEM_UTIL} ulysses_sp=${V31_ULYSSES_SEQUENCE_PARALLEL_SIZE}"
echo "[V37-strict-winner] seed=${V37_SEED} rollout_n=${ROLLOUT_N} lr=${ACTOR_LR} actor_kl=${ADAPTIVE_ACTOR_KL} use_kl_loss=${USE_KL_LOSS} kl=${KL_TYPE}/${KL_PENALTY}/${KL_COEF} target=${KL_TARGET} horizon=${KL_HORIZON}/${KL_HORIZON_UNIT} tau=${BOK_TAU_INIT}->${BOK_TAU_FINAL} winner_boost=${BOK_WINNER_BOOST}"
echo "[V37-strict-winner] pilot_steps=${V36_MAX_STEPS} bok_total_steps=${BOK_TOTAL_STEPS} adaptive_turns=GT+3 val_freq=${TRAIN_VAL_FREQ} save_freq=${TRAIN_SAVE_FREQ} save_limit=${TRAIN_SAVE_LIMIT}"
echo "[V37-strict-winner] experiment=${V31_EXPERIMENT_NAME} save=${V31_SAVE_CHECKPOINT_PATH}"
echo "[V37-strict-winner] preflight=${_V37_PREFLIGHT_JSON} manifest=${V37_RUN_MANIFEST}"
echo "================================================================"

if _v37_is_true "${V37_DRY_RUN:-0}"; then
  if _v37_is_true "${V37_VALIDATE_DOWNSTREAM:-0}"; then
    echo "[V37-strict-winner] DRY RUN: preflight passed; validating downstream V36/V32 command expansion."
  else
    echo "[V37-strict-winner] DRY RUN: preflight passed; training was not started."
    exit 0
  fi
fi

cd "${REPO_DIR}"
if ! "${_V37_PREFLIGHT_PYTHON}" - "${V31_SAVE_CHECKPOINT_PATH}/v37_execution_environment.json" "${SCRIPT_DIR}/v36_dense_11_50_full_from_1m_ckpt476.sh" "${REPO_DIR}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

environment_path, training_script, repo = map(Path, sys.argv[1:])
payload = json.loads(
    environment_path.read_text(encoding="utf-8"),
    object_pairs_hook=lambda pairs: (
        (_ for _ in ()).throw(ValueError("duplicate JSON key"))
        if len({key for key, _ in pairs}) != len(pairs) else dict(pairs)
    ),
    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")),
)
environment = payload.get("environment")
if not isinstance(environment, dict) or any(
    not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items()
):
    raise SystemExit("invalid V37 execution environment schema")
hash_input = {
    key: value for key, value in environment.items()
    if key not in {"V37_EXECUTION_ENVIRONMENT_PATH", "V37_EXECUTION_ENVIRONMENT_SHA256"}
}
actual_hash = hashlib.sha256(json.dumps(
    hash_input, sort_keys=True, separators=(",", ":"), allow_nan=False,
).encode("utf-8")).hexdigest()
if (
    payload.get("schema_version") != 1
    or payload.get("hash_contract") != "canonical_environment_without_self_reference_v1"
    or payload.get("environment_sha256") != actual_hash
    or environment.get("V37_EXECUTION_ENVIRONMENT_SHA256") != actual_hash
    or environment.get("V37_EXECUTION_ENVIRONMENT_PATH") != str(environment_path.resolve())
    or "BASH_ENV" in environment or "ENV" in environment
    or any(key.startswith("BASH_FUNC_") for key in environment)
):
    raise SystemExit("V37 execution environment seal mismatch")
completed = subprocess.run(["bash", str(training_script)], cwd=repo, env=environment, check=False)
raise SystemExit(completed.returncode)
PY
then
  _v37_error "downstream V36/V32 training entrypoint failed"
fi
if _v37_is_true "${V37_DRY_RUN:-0}"; then
  exit 0
fi

# Bind the successful run manifest to the exact checkpoint used by final eval.
export _V37_FINAL_CHECKPOINT="${V31_SAVE_CHECKPOINT_PATH}/global_step_${V37_PILOT_STEPS}"
[[ -d "${_V37_FINAL_CHECKPOINT}" ]] \
  || _v37_error "successful training did not publish expected checkpoint: ${_V37_FINAL_CHECKPOINT}"
"${_V37_PREFLIGHT_PYTHON}" - <<'PY'
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path


def reject_symlinks(path):
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise SystemExit(f"final checkpoint contains a symlink component: {current}")
    for root, directories, filenames in os.walk(absolute, followlinks=False):
        for name in directories + filenames:
            candidate = Path(root) / name
            if candidate.is_symlink():
                raise SystemExit(f"final checkpoint contains a symlink: {candidate}")
    return absolute


checkpoint = reject_symlinks(Path(os.environ["_V37_FINAL_CHECKPOINT"]))
if os.environ.get("V37_FINALIZE_HF_CHECKPOINT") == "1":
    eval_model = checkpoint / "actor" / "huggingface"
    weight_files = sorted(eval_model.glob("*.safetensors")) if eval_model.is_dir() else []
    if not weight_files or any(not item.is_file() for item in weight_files):
        raise SystemExit(f"formal final checkpoint has no merged HF safetensors: {eval_model}")
files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
if not files:
    raise SystemExit(f"final checkpoint is empty: {checkpoint}")
entries = []
for path in files:
    file_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            file_digest.update(chunk)
    entries.append({
        "path": path.relative_to(checkpoint).as_posix(),
        "size": path.stat().st_size,
        "sha256": file_digest.hexdigest(),
    })
checkpoint_sha256 = hashlib.sha256(json.dumps(
    entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")).hexdigest()

effective_path = reject_symlinks(Path(os.environ["V37_EFFECTIVE_ENVIRONMENT_PATH"]))
if not effective_path.is_file():
    raise SystemExit(f"effective environment artifact is missing: {effective_path}")
effective_bytes = effective_path.read_bytes()
effective = json.loads(
    effective_bytes,
    object_pairs_hook=lambda pairs: (
        (_ for _ in ()).throw(ValueError("duplicate JSON key"))
        if len({key for key, _ in pairs}) != len(pairs) else dict(pairs)
    ),
    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")),
)
effective_environment = effective.get("environment")
if not isinstance(effective_environment, dict) or any(
    not isinstance(key, str) or not isinstance(value, str)
    for key, value in effective_environment.items()
):
    raise SystemExit("effective environment artifact schema mismatch")
effective_hash = hashlib.sha256(json.dumps(
    effective_environment, sort_keys=True, separators=(",", ":"), allow_nan=False,
).encode()).hexdigest()
if (
    effective.get("schema_version") != 1
    or effective.get("contract") != "v37_effective_pre_trainer_environment_v1"
    or effective.get("environment_sha256") != effective_hash
):
    raise SystemExit("effective environment artifact hash mismatch")

training_path = reject_symlinks(Path(os.environ["V37_TRAINING_EVIDENCE_PATH"]))
module_path = Path(os.environ["_V37_TRAINING_EVIDENCE_FILE"])
spec = importlib.util.spec_from_file_location("v37_training_evidence", module_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load V37 training evidence verifier")
training_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_module)
try:
    training_evidence = training_module.verify_file(training_path)
except training_module.TrainingEvidenceError as exc:
    raise SystemExit(f"invalid V37 training evidence: {exc}") from exc

target = Path(os.environ["V37_RUN_MANIFEST"])
manifest = json.loads(
    target.read_text(encoding="utf-8"),
    object_pairs_hook=lambda pairs: (
        (_ for _ in ()).throw(ValueError("duplicate JSON key"))
        if len({key for key, _ in pairs}) != len(pairs) else dict(pairs)
    ),
    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")),
)
manifest["final_checkpoint_id"] = f"{manifest['arm']}-seed{manifest['seed']}-{checkpoint.name}"
manifest["final_checkpoint_path"] = str(checkpoint)
manifest["final_checkpoint_sha256"] = checkpoint_sha256
for field in (
    "evidence_run_id", "execution_environment_sha256", "final_checkpoint_id",
    "final_checkpoint_path", "final_checkpoint_sha256",
):
    if training_evidence.get(field) != manifest.get(field):
        raise SystemExit(f"training evidence disagrees with run manifest: {field}")
if manifest.get("training_evidence_path") != str(training_path):
    raise SystemExit("run manifest training evidence path mismatch")
manifest["training_evidence_sha256"] = hashlib.sha256(training_path.read_bytes()).hexdigest()
if manifest.get("effective_environment_path") != str(effective_path):
    raise SystemExit("run manifest effective environment path mismatch")
manifest["effective_environment_sha256"] = hashlib.sha256(effective_bytes).hexdigest()
temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
temporary.replace(target)
PY
echo "[V37-strict-winner] finalized checkpoint provenance: ${_V37_FINAL_CHECKPOINT}"
