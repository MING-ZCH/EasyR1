#!/usr/bin/env python3
"""Fail-closed V37 canary/final A/B gate.

The gate computes paired deltas from sample rows in the baseline and progress
evaluation artifacts.  A formal final gate accepts only path+SHA256 artifacts;
embedded objects are intentionally limited to non-formal canary diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from tools import v37_benchmark_evidence as benchmark_evidence
except ModuleNotFoundError:  # Direct execution from the tools/ directory.
    import v37_benchmark_evidence as benchmark_evidence
try:
    from tools import v37_eval_producer as eval_producer
except ModuleNotFoundError:  # Direct execution from the tools/ directory.
    import v37_eval_producer as eval_producer
try:
    from tools import v37_continuation as continuation
except ModuleNotFoundError:  # Direct execution from the tools/ directory.
    import v37_continuation as continuation

import importlib.util

_training_spec = importlib.util.spec_from_file_location(
    "v37_training_evidence", Path(__file__).resolve().parents[1] / "verl/utils/v37_training_evidence.py"
)
if _training_spec is None or _training_spec.loader is None:
    raise ImportError("cannot load the dependency-light V37 training evidence verifier")
training_evidence = importlib.util.module_from_spec(_training_spec)
_training_spec.loader.exec_module(training_evidence)


class GateError(ValueError):
    """Evidence is missing, malformed, mismatched, or fails a hard gate."""


SHA256_RE = re.compile(r"[0-9a-f]{64}")
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
FORMAL_INITIAL_MODEL_SHA256 = "f9e6b1e8031bdbc509d34249745cdcf75af85c320d6b88918444b1abb4f580a3"
INTEGER_ANSWER_RE = re.compile(r"[+-]?\d+")
POINT_OPEN_RE = re.compile(r"<point>")
POINT_CLOSE_RE = re.compile(r"</point>")
POINT_BLOCK_CAPTURE_RE = re.compile(r"<point>(.*?)</point>", re.DOTALL)
ANSWER_OPEN_RE = re.compile(r"<answer>")
ANSWER_CLOSE_RE = re.compile(r"</answer>")
ANSWER_BLOCK_RE = re.compile(r"<answer>\s*([+-]?\d+)\s*</answer>", re.DOTALL)
FORMAL_PAIRED_SCHEMA_VERSION = 2
FORMAL_PAIRED_ARTIFACT_TYPE = "v37_paired_eval_evidence"
PAIRED_UNIVERSE_SCHEMA_VERSION = 1
PAIRED_UNIVERSE_ARTIFACT_TYPE = "v37_paired_sample_universe"
PAIRED_UNIVERSE_COLUMNS = ("sample_id", "prompt_id", "answer")
BUCKET_RANGES = {
    "2-10": (2, 10), "11-20": (11, 20), "21-30": (21, 30),
    "31-40": (31, 40), "41-50": (41, 50),
}
BUCKETS = ("2-10", "11-20", "21-30", "31-40", "41-50")
ARMS = ("baseline", "progress")
ARM_ALLOWLIST = {
    "arm", "estimator", "native_action_enabled", "process_reward",
    "step_signal", "step_weight", "answer_gate",
}
ARM_ENV_ALLOWLIST = {
    "ACTION_EVENT_REWARD_ENABLE", "ADV_ESTIMATOR", "BOK_STEP_GATE",
    "BOK_STEP_MIN_GATE", "BOK_STEP_SIGNAL", "BOK_STEP_WEIGHT",
    "STEPCOUNT_RL_MODE", "V37_ARM",
}
RUN_ENV_ALLOWLIST = {
    "CONFIG_PATH", "V31_EXPERIMENT_NAME", "V31_SAVE_CHECKPOINT_PATH", "V37_RUN_MANIFEST",
    "V37_EXECUTION_ENVIRONMENT_PATH", "V37_EXECUTION_ENVIRONMENT_SHA256",
    "V37_EFFECTIVE_ENVIRONMENT_PATH", "V37_TRAINING_EVIDENCE_PATH",
    "V37_AB_CELL_ID",
}
SEED_ENV_ALLOWLIST = {"PYTHONHASHSEED", "V31_DATA_SEED", "V31_ROLLOUT_SEED", "V37_SEED"}
IMPLEMENTATION_PATHS = {
    "ab_launcher": "examples/rl_launch/run_v37_strict_ab.sh",
    "action_ledger": "verl/utils/action_ledger.py",
    "actor": "verl/workers/actor/dp_actor.py",
    "actor_config": "verl/workers/actor/config.py",
    "base_launcher": "examples/v36_dense_11_50_full_from_1m_ckpt476.sh",
    "benchmark_evidence": "tools/v37_benchmark_evidence.py",
    "checkpoint_manager": "verl/utils/checkpoint/checkpoint_manager.py",
    "continuation": "tools/v37_continuation.py",
    "core_algos": "verl/trainer/core_algos.py",
    "dataset": "verl/utils/dataset.py",
    "eval_producer": "tools/v37_eval_producer.py",
    "path_remap": "verl/utils/path_remap.py",
    "paired_universe": "tools/v37_paired_universe.py",
    "postprocess": "tools/v37_postprocess.py",
    "process_prompt": "examples/format_prompt/StepCount_interleaved_process_prompt.txt",
    "ray_environment": "verl/utils/ray_environment.py",
    "training_evidence": "verl/utils/v37_training_evidence.py",
    "fsdp_checkpoint_manager": "verl/utils/checkpoint/fsdp_checkpoint_manager.py",
    "frontier_builder": "tools/build_v37_frontier_dataset.py",
    "fsdp_workers": "verl/workers/fsdp_workers.py",
    "gate": "tools/v37_gate.py",
    "launcher": "examples/v37_strict_winner_step_rl_pilot.sh",
    "local_path_env": "examples/local_path_env.sh",
    "model_merger": "scripts/model_merger.py",
    "preflight": "tools/preflight_v37_training.py",
    "reward_manager": "verl/workers/reward/function.py",
    "reward_function": "examples/reward_function/StepCount_mask_reward.py",
    "rollout": "verl/workers/rollout/vllm_rollout_spmd.py",
    "sharding_manager": "verl/workers/sharding_manager/fsdp_vllm.py",
    "source_config": "examples/config.yaml",
    "system_prompt": "examples/format_prompt/StepCount_interleaved_system_prompt.txt",
    "torch_functional": "verl/utils/torch_functional.py",
    "trainer": "verl/trainer/ray_trainer.py",
    "trainer_main": "verl/trainer/main.py",
    "trainer_config": "verl/trainer/config.py",
    "training_entry": "examples/v32_sparse_0_10_stable_drfix.sh",
    "ray_worker_base": "verl/single_controller/ray/base.py",
    "requirements": "requirements.txt",
}
SEED_CONFIG_FIELDS = {"seed", "random_seed", "data_seed", "rollout_seed"}
REQUIRED_PAIRED_METRICS = {
    "heldout_five_bucket_macro",
    "answer_exact",
    "format_compliance",
    "unique_valid_hit",
    "duplicate",
    "cap_turn_exceeded",
    "early_stop",
}
HIGHER_BETTER = {
    "heldout_five_bucket_macro", "answer_exact", "format_compliance",
    "unique_valid_hit",
}
LOWER_BETTER = {
    "duplicate", "cap_turn_exceeded", "early_stop",
}
COUNTER_FIELDS = {
    "skipped_updates", "nonfinite_count", "oom_count", "progress_degraded",
    "cap_turn_exceeded", "early_stop_count", "correctness_sign_error_count",
}
MIN_FIELDS = {"event_mapping_coverage", "runtime_mask_coverage", "frontier_mixed_group_ratio", "kl_recoverable"}
MAX_FIELDS = {"outcome_frontier_allwrong_ratio"}
FORMAL_OPTIMIZER_STEPS = 12
FORMAL_FRONTIER_MIXED_MIN = 0.70
FORMAL_OUTCOME_ALLWRONG_MAX = 0.25
FORMAL_FILTER_OVERLONG_NUM_PROC = 64
PIXMO_TOTAL = 529
PIXMO_CORRECT_MIN = 440
STEPCOUNT_TOTAL = 500
STEPCOUNT_CORRECT_MIN = 90
FORMAL_BENCHMARK_DATASET_SHA256 = {
    "pixmo": "9f58c7eabcf0e09e94a9584cd7a4ba2e94541b116daf35e50ff8299991b94da1",
    "stepcount": "f839edcf49b73c721395f1765a237ce45a12c15e3f07843d73e48716630fb836",
}
FORMAL_BENCHMARK_ORDERED_IDS_SHA256 = {
    "pixmo": "693bcc82f4a2c6c5e7b548ded47da89b51b2f2cfc2aaeeeb171e04ee6a66a8b0",
    "stepcount": "cff39f9d4393378815e44f3c754e7c1e8ffa265e9275118111689ec050912e4e",
}
FORMAL_BUCKET_RATIOS = {"2-10": .40, "11-20": .10, "21-30": .20, "31-40": .20, "41-50": .10}
FORMAL_CLASSIFIER_THRESHOLDS = {
    "winner_min": 3,
    "winner_max": 20,
    "process_min_quality": .55,
    "process_min_quality_range": .10,
    "process_max_duplicate_rate": .25,
    "process_max_cap_rate": .25,
}
FROZEN_PAIRED_METRICS: dict[str, dict[str, Any]] = {
    "heldout_five_bucket_macro": {"direction": "noninferiority", "beneficial_direction": "increase", "margin": 0.0},
    "answer_exact": {"direction": "noninferiority", "beneficial_direction": "increase", "margin": 0.0},
    "format_compliance": {"direction": "noninferiority", "beneficial_direction": "increase", "margin": .01},
    "unique_valid_hit": {"direction": "increase"},
    "duplicate": {"direction": "noninferiority", "beneficial_direction": "decrease", "margin": 0.0},
    "cap_turn_exceeded": {"direction": "noninferiority", "beneficial_direction": "decrease", "margin": 0.0},
    "early_stop": {"direction": "noninferiority", "beneficial_direction": "decrease", "margin": .01},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_nonsymlink_path(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise GateError(f"{label} does not exist: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise GateError(f"{label} contains a symlink component: {current}")
    return absolute


def _reject_symlinks(path: Path, label: str) -> None:
    path = _absolute_nonsymlink_path(path, label)
    if not path.is_dir():
        return
    for root, directories, filenames in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in directories + filenames:
            candidate = root_path / name
            if candidate.is_symlink():
                raise GateError(f"{label} contains a symlink: {candidate}")


def tree_sha256(path: Path) -> str:
    path = _absolute_nonsymlink_path(path, "hash target")
    _reject_symlinks(path, "hash target")
    if path.is_file():
        return sha256(path)
    if not path.is_dir():
        raise GateError(f"hash target does not exist: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise GateError(f"hash target is empty: {path}")
    entries = [
        {
            "path": item.relative_to(path).as_posix(),
            "size": item.stat().st_size,
            "sha256": sha256(item),
        }
        for item in files
    ]
    return hashlib.sha256(canonical(entries).encode("utf-8")).hexdigest()


def content_snapshot(path: Path, *, parquet_only: bool = False) -> dict[str, Any]:
    """Reproduce the launcher's immutable content snapshot for formal inputs."""
    path = _absolute_nonsymlink_path(path, "snapshot target")
    _reject_symlinks(path, "snapshot target")
    root = path.resolve()
    if root.is_file():
        files = [root]
        base = root.parent
    elif root.is_dir():
        files = sorted(item for item in root.rglob("*") if item.is_file())
        base = root
    else:
        raise GateError(f"snapshot target does not exist: {path}")
    if parquet_only:
        files = [item for item in files if item.suffix.lower() == ".parquet"]
    if not files:
        raise GateError(f"snapshot target has no auditable files: {root}")
    entries: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for item in files:
        if item.is_symlink():
            raise GateError(f"snapshot file must not be a symlink: {item}")
        relative = item.relative_to(base).as_posix()
        size = item.stat().st_size
        file_hash = sha256(item)
        aggregate.update(relative.encode("utf-8") + b"\0")
        aggregate.update(str(size).encode("ascii") + b"\0")
        aggregate.update(file_hash.encode("ascii") + b"\n")
        entries.append({"path": relative, "size": size, "sha256": file_hash})
        total_bytes += size
    return {
        "root": str(root), "file_count": len(entries), "total_bytes": total_bytes,
        "sha256": aggregate.hexdigest(), "files": entries,
    }


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise GateError(f"{name} is non-finite")
    return number


def integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise GateError(f"{name} must be an integer >= {minimum}")
    return value


def known_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value.lower()) or set(value.lower()) == {"0"}:
        raise GateError(f"{name} must be a known SHA256")
    return value.lower()


def canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise GateError("value is not canonical JSON") from exc


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(text: str) -> Any:
    def reject_constant(value: str) -> Any:
        raise GateError(f"non-finite JSON constant: {value}")

    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def _open_nonsymlink_file(path: Path) -> int:
    """Open an absolute path component-by-component without following symlinks."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    if not parts or len(parts) == 1:
        raise GateError(f"artifact path has no file component: {path}")
    common_flags = getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute.anchor, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        for index, component in enumerate(parts[1:]):
            leaf = index == len(parts[1:]) - 1
            flags = os.O_RDONLY | common_flags
            if not leaf:
                flags |= getattr(os, "O_DIRECTORY", 0)
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as exc:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise GateError(f"cannot open non-symlink artifact path {path}: {exc}") from exc


def _read_stable_regular_file(path: Path) -> bytes:
    """Read one regular-file snapshot without following or switching inodes."""
    descriptor = _open_nonsymlink_file(path)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GateError(f"artifact must be a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise GateError(f"artifact mutated while being read: {path}")
    try:
        current_descriptor = _open_nonsymlink_file(path)
        try:
            current = os.fstat(current_descriptor)
        finally:
            os.close(current_descriptor)
    except GateError as exc:
        raise GateError(f"artifact path changed while being read: {path}") from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or any(getattr(current, field) != getattr(after, field) for field in stable_fields)
    ):
        raise GateError(f"artifact path changed while being read: {path}")
    return b"".join(chunks)


def _decode_rows(raw: bytes, path: Path) -> tuple[Any, bool]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateError(f"{path}: artifact is not valid UTF-8") from exc
    try:
        return _decode_json(text), False
    except (json.JSONDecodeError, GateError):
        rows: list[Any] = []
        records: set[str] = set()
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = _decode_json(line)
                encoded = canonical(row)
                if encoded in records:
                    raise GateError(f"{path}:{number}: duplicate JSONL record")
                records.add(encoded)
                rows.append(row)
            except (json.JSONDecodeError, GateError) as exc:
                raise GateError(f"{path}:{number}: invalid JSON") from exc
        if not rows:
            raise GateError(f"{path}: empty JSON/JSONL artifact")
        return rows, True


def _load_rows(path: Path) -> tuple[Any, bool]:
    return _decode_rows(_read_stable_regular_file(path), path)


def _load_hashed_rows(path: Path, expected_sha256: str, label: str) -> tuple[Any, bool]:
    """Verify and decode the exact same byte snapshot."""
    raw = _read_stable_regular_file(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise GateError(f"{label} hash mismatch")
    return _decode_rows(raw, path)


def _aggregate_log_rows(rows: Sequence[Any], source: Path) -> dict[str, Any]:
    """Aggregate cumulative/worst log evidence without allowing later resets."""
    aggregate: dict[str, Any] = {}
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise GateError(f"{source}:{number}: JSONL log row must be an object")
        for key, value in row.items():
            if key == "optimizer_steps":
                current = integer(value, f"{source}:{number}:optimizer_steps")
                aggregate[key] = max(aggregate.get(key, 0), current)
            elif key in COUNTER_FIELDS:
                current = integer(value, f"{source}:{number}:{key}")
                aggregate[key] = max(aggregate.get(key, 0), current)
            elif key in MIN_FIELDS:
                current = finite(value, f"{source}:{number}:{key}")
                aggregate[key] = min(aggregate.get(key, current), current)
            elif key in MAX_FIELDS:
                current = finite(value, f"{source}:{number}:{key}")
                aggregate[key] = max(aggregate.get(key, current), current)
            elif key == "selector_kl_contribution":
                current = finite(value, f"{source}:{number}:{key}")
                previous = aggregate.get(key, 0.0)
                aggregate[key] = current if abs(current) >= abs(previous) else previous
            elif key not in aggregate:
                aggregate[key] = value
            elif canonical(aggregate[key]) != canonical(value):
                raise GateError(f"{source}:{number}: ambiguous JSONL field {key}; aggregation policy is undeclared")
    return aggregate


def artifact(
    run: Mapping[str, Any], key: str, *, require_path: bool,
    seen_paths: set[str] | None = None, seen_hashes: set[str] | None = None,
) -> dict[str, Any]:
    value = run.get(key)
    if isinstance(value, dict):
        if require_path:
            raise GateError(f"formal run requires {key} artifact path plus SHA256")
        return value
    if not isinstance(value, str) or not value.strip():
        raise GateError(f"run missing {key} artifact path")
    path = _absolute_nonsymlink_path(Path(value), f"{key} artifact")
    if not path.is_file():
        raise GateError(f"missing artifact {path}")
    expected = known_sha(run.get(f"{key}_sha256"), f"{key}_sha256")
    resolved = str(path)
    if seen_paths is not None:
        if resolved in seen_paths:
            raise GateError(f"shared/reused evidence artifact path: {resolved}")
        seen_paths.add(resolved)
    if seen_hashes is not None:
        if expected in seen_hashes:
            raise GateError(f"shared/reused self-reported evidence hash for {key}")
        seen_hashes.add(expected)
    decoded, is_jsonl = _load_hashed_rows(path, expected, f"tampered {key} artifact")
    if key == "log" and is_jsonl:
        decoded = _aggregate_log_rows(decoded, path)
    if not isinstance(decoded, dict):
        raise GateError(f"{key} artifact must decode to an object")
    return decoded


def config_without_arm(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in ARM_ALLOWLIST}


def config_without_seed_or_arm(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in config.items()
        if key not in ARM_ALLOWLIST and key not in SEED_CONFIG_FIELDS
    }


def _exact_object(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        missing = sorted(keys - set(value)) if isinstance(value, dict) else sorted(keys)
        extra = sorted(set(value) - keys) if isinstance(value, dict) else []
        raise GateError(f"{label} schema mismatch; missing={missing}, extra={extra}")
    return value


def _validate_config_seal(manifest: Mapping[str, Any], label: str, seed: Any) -> Mapping[str, str]:
    config = manifest.get("config")
    environment = manifest.get("audited_environment")
    if not isinstance(config, dict) or not isinstance(environment, dict):
        raise GateError(f"{label}: config/audited_environment missing")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items()):
        raise GateError(f"{label}: audited_environment must contain string key/value pairs")
    encoded = json.dumps(
        {"selected": config, "environment": environment},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    expected = known_sha(manifest.get("config_sha256"), f"{label}:config_sha256")
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise GateError(f"{label}: config_sha256 does not seal config plus audited_environment")
    continuation_hash = manifest.get("continuation_config_sha256")
    if continuation_hash is not None or manifest.get("run_class") == "formal":
        expected_continuation_hash = known_sha(
            continuation_hash, f"{label}:continuation_config_sha256"
        )
        try:
            actual_continuation_hash = continuation.continuation_config_sha256(config)
        except continuation.ContinuationError as exc:
            raise GateError(f"{label}: invalid continuation config identity: {exc}") from exc
        if expected_continuation_hash != actual_continuation_hash:
            raise GateError(f"{label}: continuation_config_sha256 does not seal numerical config")
    expected_seed = str(seed)
    for key in ("PYTHONHASHSEED", "V31_DATA_SEED", "V31_ROLLOUT_SEED", "V37_SEED"):
        if environment.get(key) != expected_seed:
            raise GateError(f"{label}: audited_environment.{key} must equal manifest seed {expected_seed}")
    return environment


def _validate_arm_seed_bindings(
    manifest: Mapping[str, Any], arm: str, seed: Any, label: str,
) -> None:
    """Validate every ignored A/B or seed field before normalization."""
    config = manifest.get("config")
    environment = manifest.get("audited_environment")
    mechanism = manifest.get("mechanism_config")
    if not isinstance(config, dict) or not isinstance(environment, dict) or not isinstance(mechanism, dict):
        raise GateError(f"{label}: arm/seed binding inputs are missing")

    def bound_number(value: Any, name: str) -> float:
        if isinstance(value, str):
            if value != value.strip() or not value:
                raise GateError(f"{name} must be canonical numeric text")
            try:
                number = float(value)
            except ValueError as exc:
                raise GateError(f"{name} must be numeric") from exc
            if not math.isfinite(number):
                raise GateError(f"{name} is non-finite")
            return number
        return finite(value, name)

    progress = arm == "progress"
    expected_config = {
        "arm": arm,
        "estimator": "bok_grpo_step" if progress else "bok_grpo",
        "process_reward": "1" if progress else "0",
        "step_signal": "native_action_event" if progress else None,
        "step_weight": "0.1" if progress else "0",
        "answer_gate": {
            "mode": "answer_soft" if progress else None,
            "minimum": "0.2" if progress else None,
        },
    }
    for field, expected in expected_config.items():
        actual = config.get(field)
        if field == "step_weight":
            if bound_number(actual, f"{label}:config.{field}") != float(expected):
                raise GateError(f"{label}: config.{field} is not bound to arm {arm}")
        elif field == "answer_gate":
            if not isinstance(actual, dict) or set(actual) != {"mode", "minimum"}:
                raise GateError(f"{label}: config.answer_gate has an invalid schema")
            if actual.get("mode") != expected["mode"]:
                raise GateError(f"{label}: config.answer_gate.mode is not bound to arm {arm}")
            minimum = actual.get("minimum")
            if expected["minimum"] is None:
                if minimum is not None:
                    raise GateError(f"{label}: baseline config.answer_gate.minimum must be null")
            elif bound_number(minimum, f"{label}:config.answer_gate.minimum") != float(expected["minimum"]):
                raise GateError(f"{label}: config.answer_gate.minimum is not bound to arm {arm}")
        elif actual != expected:
            raise GateError(f"{label}: config.{field} is not bound to arm {arm}")

    optional_mechanism_fields = {
        "native_action_enabled": mechanism.get("native_action_enabled"),
        "native_action_parser_contract": mechanism.get("native_action_parser_contract"),
        "native_action_ledger_contract": mechanism.get("native_action_ledger_contract"),
    }
    for field, expected in optional_mechanism_fields.items():
        if field in config and canonical(config[field]) != canonical(expected):
            raise GateError(f"{label}: config.{field} disagrees with mechanism_config")

    for field in SEED_CONFIG_FIELDS:
        if field in config and (type(config[field]) is not int or config[field] != seed):
            raise GateError(f"{label}: config.{field} must equal manifest seed {seed}")

    expected_environment = {
        "V37_ARM": arm,
        "ADV_ESTIMATOR": expected_config["estimator"],
        "ACTION_EVENT_REWARD_ENABLE": expected_config["process_reward"],
        "STEPCOUNT_RL_MODE": expected_config["estimator"],
        "BOK_STEP_WEIGHT": expected_config["step_weight"],
        "PROCESS_REWARD_ENABLE": "0",
        "BOK_CORRECTNESS_TASK_WEIGHT": "0.25",
        "BOK_CORRECTNESS_QUALITY_WEIGHT": "0.1",
        "BOK_CORRECTNESS_PARTIAL_SCALE": "0.25",
        "V37_STRICT_POINT_PARSER_CONTRACT": "1",
        "STEPCOUNT_MASK_REQUIRE": "1",
        "TRAJ_STRICT_ANSWER_INTEGER_PARSE": "1",
        "V37_RAW_SUCCESS_STRICT_WINNER": "1",
        "V37_WINNER_MODE": "outcome_success",
        "V37_REWARD_FAIL_CLOSED": "1",
        "ACTION_EVENT_LEDGER_ENABLE": "1",
        "V37_ACTION_PARSER_CONTRACT": "1",
        "V37_ACTION_LEDGER_CONTRACT": "1",
    }
    if progress:
        expected_environment.update({
            "BOK_STEP_SIGNAL": "native_action_event",
            "BOK_STEP_GATE": "answer_soft",
            "BOK_STEP_MIN_GATE": "0.2",
        })
    for field, expected in expected_environment.items():
        actual = environment.get(field)
        if field in {"BOK_STEP_WEIGHT", "BOK_STEP_MIN_GATE"} and actual is not None:
            if bound_number(actual, f"{label}:audited_environment.{field}") == float(expected):
                continue
        if actual != expected:
            raise GateError(f"{label}: audited_environment.{field} must equal {expected!r}")
    if not progress:
        for field in (
            "BOK_STEP_SIGNAL", "BOK_STEP_GATE", "BOK_STEP_MIN_GATE",
        ):
            if field in environment:
                raise GateError(f"{label}: baseline audited_environment must omit {field}")


def _normalized_environment(environment: Mapping[str, str], ignored: set[str]) -> dict[str, str]:
    return {key: value for key, value in environment.items() if key not in ignored}


def _validate_evidence_identity(
    manifest: Mapping[str, Any], log: Mapping[str, Any], evaluation: Mapping[str, Any], label: str,
    *, actual_manifest_path: Path | None = None,
) -> str:
    expected = known_sha(manifest.get("evidence_run_id"), f"{label}:evidence_run_id")
    environment = manifest.get("audited_environment", {})
    snapshots = manifest.get("input_snapshots", {})
    model_snapshot = snapshots.get("model", {}) if isinstance(snapshots, dict) else {}
    run_manifest_path = environment.get("V37_RUN_MANIFEST") if isinstance(environment, dict) else None
    if not isinstance(run_manifest_path, str) or not isinstance(model_snapshot, dict):
        raise GateError(f"{label}: evidence identity inputs are missing")
    declared_manifest_path = Path(run_manifest_path).resolve()
    if actual_manifest_path is not None:
        actual_manifest_path = actual_manifest_path.resolve()
        if declared_manifest_path != actual_manifest_path:
            raise GateError(f"{label}: V37_RUN_MANIFEST does not name the actual manifest artifact")
        declared_save_path = environment.get("V31_SAVE_CHECKPOINT_PATH")
        if not isinstance(declared_save_path, str) or Path(declared_save_path).resolve() != actual_manifest_path.parent:
            raise GateError(f"{label}: V31_SAVE_CHECKPOINT_PATH does not match the manifest directory")
        declared_config_path = environment.get("CONFIG_PATH")
        if not isinstance(declared_config_path, str) or Path(declared_config_path).resolve() != Path(
            str(manifest.get("config_path", ""))
        ).resolve():
            raise GateError(f"{label}: CONFIG_PATH does not match manifest.config_path")
        if actual_manifest_path.name != "v37_run_manifest.json":
            raise GateError(f"{label}: formal manifest must be named v37_run_manifest.json")
        runtime_config_path = _absolute_nonsymlink_path(
            Path(str(manifest.get("config_path", ""))), f"{label}:runtime config",
        )
        preflight_path = _absolute_nonsymlink_path(
            Path(str(manifest.get("preflight_report", ""))), f"{label}:preflight report",
        )
        if runtime_config_path.parent != actual_manifest_path.parent or runtime_config_path.name != "v37_config.yaml":
            raise GateError(f"{label}: runtime config is outside the formal run layout")
        if preflight_path.parent != actual_manifest_path.parent or preflight_path.name != "v37_preflight.json":
            raise GateError(f"{label}: preflight report is outside the formal run layout")
    identity_payload = {
        "arm": manifest.get("arm"), "seed": manifest.get("seed"),
        "run_directory": str(declared_manifest_path.parent),
        "manifest_path": str(declared_manifest_path),
        "config_sha256": manifest.get("config_sha256"),
        "runtime_config_sha256": manifest.get("config_file_sha256"),
        "source_config_sha256": manifest.get("source_config_sha256"),
        "runtime_environment": manifest.get("runtime_environment"),
        "execution_environment_sha256": manifest.get("execution_environment_sha256"),
        "ab_preregistration": manifest.get("ab_preregistration"),
        "resume_evidence": manifest.get("resume_evidence"),
        "git_commit": manifest.get("git_commit"),
        "git_status_sha256": manifest.get("git_status_sha256"),
        "git_diff_sha256": manifest.get("git_diff_sha256"),
        "implementation_sha256": manifest.get("implementation_sha256"),
        "initial_model_sha256": model_snapshot.get("sha256"),
        "dataset_sha256": manifest.get("dataset_sha256"),
    }
    recomputed = hashlib.sha256(json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    if recomputed != expected:
        raise GateError(f"{label}: evidence_run_id is not derived from the frozen run identity")
    if log.get("evidence_run_id") != expected or evaluation.get("evidence_run_id") != expected:
        raise GateError(f"{label}: log/eval evidence_run_id does not match the run manifest")
    execution_hash = manifest.get("execution_environment_sha256")
    if actual_manifest_path is not None and (
        log.get("execution_environment_sha256") != execution_hash
        or evaluation.get("execution_environment_sha256") != execution_hash
    ):
        raise GateError(f"{label}: log/eval execution environment hash mismatch")
    return expected


def _live_git_identity(repo: Path) -> dict[str, str]:
    git_environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    git_environment["LC_ALL"] = "C"
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=git_environment,
        ).stdout.decode("ascii").strip()
        git_status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-uall"], cwd=repo, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=git_environment,
        ).stdout
        git_diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD"],
            cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=git_environment,
        ).stdout
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise GateError("cannot recompute live Git identity") from exc
    return {
        "git_commit": git_commit,
        "git_status_sha256": hashlib.sha256(git_status).hexdigest(),
        "git_diff_sha256": hashlib.sha256(git_diff).hexdigest(),
    }


def _validate_implementation_snapshot(manifest: Mapping[str, Any], label: str) -> None:
    repo = Path(__file__).resolve().parents[1]
    try:
        live_git = _live_git_identity(repo)
    except GateError as exc:
        raise GateError(f"{label}: cannot recompute live Git identity") from exc
    git_commit = manifest.get("git_commit")
    if not isinstance(git_commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", git_commit):
        raise GateError(f"{label}: git_commit is not a full Git object ID")
    known_sha(manifest.get("git_status_sha256"), f"{label}:git_status_sha256")
    known_sha(manifest.get("git_diff_sha256"), f"{label}:git_diff_sha256")
    if manifest.get("run_class") == "formal" and (
        manifest.get("git_status_sha256") != EMPTY_SHA256
        or manifest.get("git_diff_sha256") != EMPTY_SHA256
    ):
        raise GateError(f"{label}: formal Git identity must describe a clean committed worktree")
    if any(manifest.get(field) != value for field, value in live_git.items()):
        raise GateError(f"{label}: live Git identity differs from the training manifest")
    raw = _exact_object(
        manifest.get("implementation_sha256"), set(IMPLEMENTATION_PATHS),
        f"{label}:implementation_sha256",
    )
    for name, relative in IMPLEMENTATION_PATHS.items():
        expected = known_sha(raw.get(name), f"{label}:implementation_sha256.{name}")
        path = _absolute_nonsymlink_path(repo / relative, f"{label}:implementation file")
        if not path.is_file() or sha256(path) != expected:
            raise GateError(f"{label}: implementation file changed or is missing: {relative}")
    environment = manifest.get("audited_environment")
    expected_process_path = str((repo / IMPLEMENTATION_PATHS["process_prompt"]).resolve())
    if (
        not isinstance(environment, dict)
        or environment.get("INTERLEAVED_PROCESS_PROMPT_FILE") != expected_process_path
        or environment.get("INTERLEAVED_PROCESS_PROMPT_SHA256") != raw["process_prompt"]
    ):
        raise GateError(f"{label}: process prompt runtime binding mismatch")
    if manifest.get("reward_function_sha256") != raw["reward_function"]:
        raise GateError(f"{label}: reward_function_sha256 disagrees with implementation snapshot")
    reward_spec = manifest.get("reward_function")
    if not isinstance(reward_spec, str) or str(Path(reward_spec.split(":", 1)[0]).resolve()) != str(
        (repo / IMPLEMENTATION_PATHS["reward_function"]).resolve()
    ):
        raise GateError(f"{label}: reward function path is not the frozen V37 implementation")
    if manifest.get("source_config_sha256") != raw["source_config"]:
        raise GateError(f"{label}: source_config_sha256 disagrees with implementation snapshot")
    runtime_config = _absolute_nonsymlink_path(
        Path(str(manifest.get("config_path", ""))), f"{label}:runtime config",
    )
    runtime_hash = known_sha(manifest.get("config_file_sha256"), f"{label}:config_file_sha256")
    if runtime_config.is_symlink() or not runtime_config.is_file() or sha256(runtime_config) != runtime_hash:
        raise GateError(f"{label}: runtime config changed or is missing")
    source_text = (repo / IMPLEMENTATION_PATHS["source_config"]).read_text(encoding="utf-8")
    expected_text, substitutions = re.subn(
        r"(?m)^(\s*adaptive_actor_kl:)\s*(?:false|true)\s*$",
        r"\1 true",
        source_text,
    )
    if substitutions != 1 or hashlib.sha256(expected_text.encode("utf-8")).hexdigest() != runtime_hash:
        raise GateError(f"{label}: runtime config is not the deterministic V37 source-config transform")


def _validate_runtime_environment(manifest: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    runtime = _exact_object(
        manifest.get("runtime_environment"),
        {"python_executable", "python_executable_sha256", "python_version", "packages"},
        f"{label}:runtime_environment",
    )
    if not isinstance(runtime["python_executable"], str) or not Path(runtime["python_executable"]).is_absolute():
        raise GateError(f"{label}: runtime python_executable must be absolute")
    executable = _absolute_nonsymlink_path(
        Path(runtime["python_executable"]), f"{label}:runtime python executable",
    )
    executable_hash = known_sha(
        runtime["python_executable_sha256"], f"{label}:runtime python executable SHA256",
    )
    if not executable.is_file() or sha256(executable) != executable_hash:
        raise GateError(f"{label}: runtime python executable hash mismatch")
    if not isinstance(runtime["python_version"], str) or not runtime["python_version"]:
        raise GateError(f"{label}: runtime python_version missing")
    packages = runtime["packages"]
    required_packages = {
        "flash-attn", "numpy", "pyarrow", "ray", "tensordict", "torch", "transformers", "vllm",
    }
    if not isinstance(packages, dict) or set(packages) != required_packages:
        raise GateError(f"{label}: runtime package version schema mismatch")
    if any(not isinstance(value, str) or not value for value in packages.values()):
        raise GateError(f"{label}: all frozen runtime package versions must be known strings")
    return runtime


def _validate_execution_environment(manifest: Mapping[str, Any], label: str) -> None:
    if manifest.get("execution_environment_schema") != "v37_explicit_environment_v1":
        raise GateError(f"{label}: explicit execution environment schema missing")
    path = _absolute_nonsymlink_path(
        Path(str(manifest.get("execution_environment_path", ""))),
        f"{label}:execution environment",
    )
    manifest_path = Path(manifest["audited_environment"]["V37_RUN_MANIFEST"])
    if path.parent != manifest_path.parent or path.name != "v37_execution_environment.json":
        raise GateError(f"{label}: execution environment is outside the formal run layout")
    payload, is_jsonl = _load_rows(path)
    expected_payload_keys = {"schema_version", "hash_contract", "environment_sha256", "environment"}
    if is_jsonl or not isinstance(payload, dict) or set(payload) != expected_payload_keys:
        raise GateError(f"{label}: execution environment artifact schema mismatch")
    environment = payload.get("environment")
    if not isinstance(environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items()
    ):
        raise GateError(f"{label}: execution environment must contain string key/value pairs")
    hash_input = {
        key: value for key, value in environment.items()
        if key not in {"V37_EXECUTION_ENVIRONMENT_PATH", "V37_EXECUTION_ENVIRONMENT_SHA256"}
    }
    recomputed = hashlib.sha256(canonical(hash_input).encode("utf-8")).hexdigest()
    expected = known_sha(manifest.get("execution_environment_sha256"), f"{label}:execution environment SHA256")
    if (
        payload.get("schema_version") != 1
        or payload.get("hash_contract") != "canonical_environment_without_self_reference_v1"
        or payload.get("environment_sha256") != recomputed
        or expected != recomputed
        or environment.get("V37_EXECUTION_ENVIRONMENT_SHA256") != recomputed
        or environment.get("V37_EXECUTION_ENVIRONMENT_PATH") != str(path)
        or canonical(environment) != canonical(manifest.get("audited_environment"))
        or "BASH_ENV" in environment or "ENV" in environment
        or any(key.startswith("BASH_FUNC_") for key in environment)
    ):
        raise GateError(f"{label}: execution environment seal mismatch")


def _validate_effective_environment(
    manifest: Mapping[str, Any], label: str, actual_manifest_path: Path,
) -> Mapping[str, str]:
    """Validate the environment observed after downstream defaults and Ray setup."""
    path = _absolute_nonsymlink_path(
        Path(str(manifest.get("effective_environment_path", ""))),
        f"{label}:effective environment",
    )
    expected_path = actual_manifest_path.with_name("v37_effective_environment.json")
    if path != expected_path or path.is_symlink() or not path.is_file():
        raise GateError(f"{label}: effective environment is outside the formal run layout")
    expected_file_hash = known_sha(
        manifest.get("effective_environment_sha256"),
        f"{label}:effective environment file SHA256",
    )
    payload, is_jsonl = _load_hashed_rows(
        path, expected_file_hash, f"{label}: effective environment artifact",
    )
    payload = _exact_object(
        payload,
        {"schema_version", "contract", "environment_sha256", "environment"},
        f"{label}:effective environment artifact",
    )
    environment = payload.get("environment")
    if is_jsonl or not isinstance(environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise GateError(f"{label}: effective environment must contain string key/value pairs")
    recomputed = hashlib.sha256(canonical(environment).encode("utf-8")).hexdigest()
    if (
        payload.get("schema_version") != 1
        or payload.get("contract") != "v37_effective_pre_trainer_environment_v1"
        or known_sha(
            payload.get("environment_sha256"),
            f"{label}:effective environment canonical SHA256",
        ) != recomputed
        or "BASH_ENV" in environment
        or "ENV" in environment
        or any(key.startswith("BASH_FUNC_") for key in environment)
    ):
        raise GateError(f"{label}: effective environment contract mismatch")

    arm = str(manifest.get("arm"))
    seed = manifest.get("seed")
    audited = manifest.get("audited_environment")
    if arm not in ARMS or type(seed) is not int or not isinstance(audited, dict):
        raise GateError(f"{label}: effective environment identity inputs are missing")
    expected = {
        "V37_RUN_CLASS": "formal",
        "V37_DATA_MODE": "frontier_rl",
        "V37_RUN_PURPOSE": "formal_ab",
        "V37_ARM": arm,
        "V37_SEED": str(seed),
        "V37_PILOT_STEPS": str(FORMAL_OPTIMIZER_STEPS),
        "V37_CONTINUATION_MODE": "0",
        "V37_EVIDENCE_START_STEP": "0",
        "V37_FINALIZE_HF_CHECKPOINT": "1",
        "V37_EXPECTED_INITIAL_MODEL_SHA256": FORMAL_INITIAL_MODEL_SHA256,
        "V37_REQUIRE_EXACT_RAY_GPUS": "1",
        "V31_NNODES": "1",
        "V31_N_GPUS_PER_NODE": "8",
        "HOST_NUM": "1",
        "HOST_GPU_NUM": "8",
        "INDEX": "0",
        "STEPCOUNT_HARDWARE_PROFILE": "h200",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "RAY_ADDRESS": "auto",
        "MODEL_PATH": str(manifest.get("model_path")),
        "STEPCOUNT_TRAIN_DATA": str(manifest.get("train_data")),
        "STEPCOUNT_VAL_DATA": str(manifest.get("validation_data")),
        "STEPCOUNT_MASKS_METADATA": str(manifest.get("mask_metadata")),
        "STEPCOUNT_MASKS_DIR": str(manifest.get("masks_dir")),
        "CONFIG_PATH": str(manifest.get("config_path")),
        "V37_RUN_MANIFEST": str(actual_manifest_path),
        "V37_EFFECTIVE_ENVIRONMENT_PATH": str(path),
        "ADAPTIVE_ACTOR_KL": "true",
        "USE_KL_LOSS": "false",
        "KL_TYPE": "adaptive",
        "KL_COEF": "0.08",
        "KL_TARGET": "0.15",
        "KL_HORIZON": "50",
        "KL_HORIZON_UNIT": "executed_optimizer_updates",
        "KL_PENALTY": "low_var_kl",
        "BOK_CORRECTNESS_FIRST": "1",
        "BOK_ALLWRONG_TERMINAL_ZERO": "1",
        "BOK_CORRECTNESS_TASK_WEIGHT": "0.25",
        "BOK_CORRECTNESS_QUALITY_WEIGHT": "0.1",
        "BOK_CORRECTNESS_PARTIAL_SCALE": "0.25",
        "PROCESS_REWARD_ENABLE": "0",
        "STEPCOUNT_MASK_REQUIRE": "1",
        "TRAJ_STRICT_ANSWER_INTEGER_PARSE": "1",
        "V37_RAW_SUCCESS_STRICT_WINNER": "1",
        "V37_WINNER_MODE": "outcome_success",
        "V37_STRICT_POINT_PARSER_CONTRACT": "1",
        "V37_REWARD_FAIL_CLOSED": "1",
        "INTERLEAVED_ADAPTIVE_MAX_TURNS": "true",
        "INTERLEAVED_ADAPTIVE_MAX_TURNS_MARGIN": "2",
        "INTERLEAVED_MAX_TURNS": "53",
        "ADV_ESTIMATOR": "bok_grpo_step" if arm == "progress" else "bok_grpo",
        "ACTION_EVENT_LEDGER_ENABLE": "1",
        "ACTION_EVENT_REWARD_ENABLE": "1" if arm == "progress" else "0",
        "V37_ACTION_PARSER_CONTRACT": "1",
        "V37_ACTION_LEDGER_CONTRACT": "1",
    }
    for key, value in expected.items():
        if environment.get(key) != value:
            raise GateError(f"{label}: effective environment {key} must equal {value!r}")

    # Every critical value already sealed by the wrapper must survive unchanged
    # through V36/V32. RAY_ADDRESS is deliberately created downstream.
    for key in set(expected) - {"RAY_ADDRESS"}:
        if audited.get(key) != environment.get(key):
            raise GateError(f"{label}: downstream changed sealed environment {key}")
    remap_key = "STEPCOUNT_IMAGE_PATH_REMAP_JSON"
    if audited.get(remap_key) != environment.get(remap_key):
        raise GateError(f"{label}: image path remap changed before trainer startup")
    for key in ("V37_ACTION_PARSER_CONTRACT", "V37_ACTION_LEDGER_CONTRACT"):
        if audited.get(key) != "1" or environment.get(key) != "1":
            raise GateError(f"{label}: effective environment lost shared ledger contract {key}")
    return dict(environment)


def validate_ab_plan(
    plan_path_value: os.PathLike[str] | str,
    plan_sha256_value: str,
    run_id: str,
    *,
    expected_seeds: Sequence[int] | None = None,
    require_target_dirs: bool = False,
) -> dict[str, Any]:
    """Validate the shared schema-v2 formal A/B plan contract."""
    plan_path = _absolute_nonsymlink_path(Path(plan_path_value), "A/B plan")
    if not plan_path.is_file() or plan_path.name != "ab_plan.json":
        raise GateError("A/B plan must be a non-symlink ab_plan.json file")
    plan_sha256 = known_sha(plan_sha256_value, "A/B plan SHA256")
    plan, is_jsonl = _load_hashed_rows(
        plan_path, plan_sha256, "A/B plan content changed after preregistration",
    )
    if is_jsonl:
        raise GateError("A/B plan must be one strict JSON object")
    plan = _exact_object(
        plan,
        {"schema_version", "ab_run_id", "plan_path", "seeds", "execution", "preregistered", "cells"},
        "A/B plan",
    )
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 2:
        raise GateError("A/B plan schema_version must be 2")
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"v37-ab-[0-9a-f]{32}", run_id) is None
        or plan["ab_run_id"] != run_id
        or plan["plan_path"] != str(plan_path)
    ):
        raise GateError("A/B plan identity mismatch")
    seeds = plan["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 2
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != 2
        or (expected_seeds is not None and seeds != list(expected_seeds))
    ):
        raise GateError("A/B plan seeds are invalid or differ from preregistration")
    expected_execution = {
        "run_class": "formal",
        "data_mode": "frontier_rl",
        "optimizer_steps": FORMAL_OPTIMIZER_STEPS,
        "nnodes": 1,
        "gpus_per_node": 8,
        "world_size": 8,
        "scheduling": "sequential",
    }
    if canonical(plan["execution"]) != canonical(expected_execution):
        raise GateError("A/B execution contract is not the frozen serial 1x8 protocol")
    preregistered = _exact_object(
        plan.get("preregistered"),
        {
            "schema_version", "complete", "missing", "input_snapshots",
            "implementation_sha256", "image_path_remap", "eval_contract",
        },
        "A/B preregistered contract",
    )
    if (
        preregistered.get("schema_version") != 1
        or preregistered.get("complete") is not True
        or preregistered.get("missing") != []
    ):
        raise GateError("formal A/B plan has incomplete preregistered inputs")
    eval_contract = _exact_object(
        preregistered.get("eval_contract"),
        {
            "paired_metrics", "benchmark_dataset_sha256", "benchmark_ordered_ids_sha256",
            "benchmark_thresholds", "pipeline", "paired_sample_universe",
        },
        "A/B eval contract",
    )
    expected_eval_contract = {
        "paired_metrics": FROZEN_PAIRED_METRICS,
        "benchmark_dataset_sha256": FORMAL_BENCHMARK_DATASET_SHA256,
        "benchmark_ordered_ids_sha256": FORMAL_BENCHMARK_ORDERED_IDS_SHA256,
        "benchmark_thresholds": {
            "pixmo_correct_min": PIXMO_CORRECT_MIN,
            "pixmo_total": PIXMO_TOTAL,
            "stepcount_correct_min": STEPCOUNT_CORRECT_MIN,
            "stepcount_total": STEPCOUNT_TOTAL,
        },
    }
    if canonical({key: eval_contract[key] for key in expected_eval_contract}) != canonical(expected_eval_contract):
        raise GateError("A/B eval protocol was not frozen before training")
    pipeline = _exact_object(
        eval_contract.get("pipeline"), {"producer", "recipe"}, "A/B eval pipeline",
    )
    _load_paired_sample_universe(
        eval_contract.get("paired_sample_universe"),
        preregistered.get("input_snapshots", {}).get("validation_data"),
        "A/B preregistration",
    )
    repo = Path(__file__).resolve().parents[1]
    expected_producer = (repo / IMPLEMENTATION_PATHS["eval_producer"]).resolve()
    for name, expected_path in (("producer", expected_producer), ("recipe", None)):
        binding = _exact_object(
            pipeline.get(name), {"path", "sha256"}, f"A/B eval pipeline {name}",
        )
        path_value = binding.get("path")
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise GateError(f"A/B eval pipeline {name} path must be absolute")
        bound_path = _absolute_nonsymlink_path(
            Path(path_value), f"A/B eval pipeline {name}",
        )
        if path_value != str(bound_path):
            raise GateError(f"A/B eval pipeline {name} path must be canonical")
        if not bound_path.is_file():
            raise GateError(f"A/B eval pipeline {name} must be a regular file")
        if expected_path is not None and bound_path != expected_path:
            raise GateError("A/B eval producer must be the repository implementation")
        if known_sha(binding.get("sha256"), f"A/B eval pipeline {name} SHA256") != sha256(bound_path):
            raise GateError(f"A/B eval pipeline {name} content changed after preregistration")
        if name == "producer" and not os.access(bound_path, os.X_OK):
            raise GateError("A/B eval producer must be executable")
        if name == "recipe":
            try:
                eval_producer.verify_recipe(bound_path)
            except eval_producer.EvalProducerError as exc:
                raise GateError(f"invalid preregistered formal eval recipe: {exc}") from exc
    input_snapshots = _exact_object(
        preregistered.get("input_snapshots"),
        {
            "model", "training_data", "validation_data", "mask_metadata", "mask_tree",
            "frontier_manifest", "coverage_report", "filtered_manifest",
        },
        "A/B input snapshots",
    )
    model_snapshot = _exact_object(
        input_snapshots.get("model"),
        {"root", "file_count", "total_bytes", "sha256", "files"},
        "A/B initial model snapshot",
    )
    if known_sha(
        model_snapshot.get("sha256"), "A/B initial model SHA256"
    ) != FORMAL_INITIAL_MODEL_SHA256:
        raise GateError("A/B initial model is not the canonical checkpoint-476")
    expected_cells = []
    for seed in seeds:
        for arm in ARMS:
            cell_id = f"seed{seed}-{arm}"
            expected_cells.append({
                "cell_id": cell_id,
                "seed": seed,
                "arm": arm,
                "target_dir": str((plan_path.parent / cell_id).absolute()),
            })
    cells = plan["cells"]
    if not isinstance(cells, list) or len(cells) != 4:
        raise GateError("A/B plan must contain exactly four cells")
    for index, cell in enumerate(cells):
        _exact_object(cell, {"cell_id", "seed", "arm", "target_dir"}, f"A/B cell {index}")
        if canonical(cell) != canonical(expected_cells[index]):
            raise GateError("A/B cell set/order/target differs from preregistration")
        if require_target_dirs:
            target = _absolute_nonsymlink_path(Path(cell["target_dir"]), "A/B target")
            if not target.is_dir():
                raise GateError("A/B target is not a directory")
    return dict(plan)


def _validate_ab_preregistration(
    manifest: Mapping[str, Any], label: str, seeds: Sequence[int], actual_manifest_path: Path,
) -> tuple[str, str, str]:
    registration = _exact_object(
        manifest.get("ab_preregistration"),
        {"schema_version", "ab_run_id", "plan_path", "plan_sha256", "cell_id"},
        f"{label}:ab_preregistration",
    )
    if type(registration["schema_version"]) is not int or registration["schema_version"] != 1:
        raise GateError(f"{label}: A/B preregistration schema_version must be 1")
    run_id = registration["ab_run_id"]
    if not isinstance(run_id, str) or re.fullmatch(r"v37-ab-[0-9a-f]{32}", run_id) is None:
        raise GateError(f"{label}: invalid A/B run id")
    plan_sha256 = known_sha(registration["plan_sha256"], f"{label}:A/B plan SHA256")
    plan = validate_ab_plan(
        registration["plan_path"],
        plan_sha256,
        run_id,
        expected_seeds=seeds,
        require_target_dirs=True,
    )
    plan_path = Path(plan["plan_path"])
    preregistered = plan["preregistered"]
    plan_inputs = preregistered["input_snapshots"]
    manifest_inputs = manifest.get("input_snapshots")
    if not isinstance(manifest_inputs, dict):
        raise GateError(f"{label}: manifest input snapshots missing")
    for field in ("model", "training_data", "validation_data", "mask_metadata"):
        if canonical(plan_inputs.get(field)) != canonical(manifest_inputs.get(field)):
            raise GateError(f"{label}: manifest {field} differs from preregistration")
    mask_tree = _exact_object(
        plan_inputs.get("mask_tree"), {"path", "sha256"}, f"{label}:A/B mask tree",
    )
    if (
        mask_tree.get("path") != manifest.get("mask_tree_path")
        or mask_tree.get("sha256") != manifest.get("mask_tree_sha256")
    ):
        raise GateError(f"{label}: mask tree differs from preregistration")
    for plan_name, manifest_path_name, manifest_hash_name in (
        ("frontier_manifest", "dataset_manifest", "dataset_manifest_sha256"),
        ("coverage_report", "metadata_coverage_report", "metadata_coverage_report_sha256"),
        ("filtered_manifest", "filtered_manifest", "filtered_manifest_sha256"),
    ):
        binding = _exact_object(
            plan_inputs.get(plan_name), {"path", "sha256"}, f"{label}:A/B {plan_name}",
        )
        if (
            binding.get("path") != manifest.get(manifest_path_name)
            or binding.get("sha256") != manifest.get(manifest_hash_name)
        ):
            raise GateError(f"{label}: {plan_name} differs from preregistration")
    if canonical(preregistered.get("implementation_sha256")) != canonical(
        manifest.get("implementation_sha256")
    ):
        raise GateError(f"{label}: implementation differs from preregistration")
    environment = manifest.get("audited_environment")
    expected_remap = preregistered.get("image_path_remap")
    actual_remap = environment.get("STEPCOUNT_IMAGE_PATH_REMAP_JSON") if isinstance(environment, dict) else None
    if actual_remap != expected_remap:
        raise GateError(f"{label}: image path remap differs from preregistration")
    cells = plan["cells"]
    arm = manifest.get("arm")
    seed = manifest.get("seed")
    cell_id = f"seed{seed}-{arm}"
    if registration["cell_id"] != cell_id:
        raise GateError(f"{label}: manifest A/B cell identity mismatch")
    matching = [cell for cell in cells if cell["cell_id"] == cell_id]
    if len(matching) != 1 or Path(matching[0]["target_dir"]).resolve() != actual_manifest_path.parent:
        raise GateError(f"{label}: manifest is outside its preregistered A/B target directory")
    environment = manifest.get("audited_environment")
    expected_environment = {
        "V37_AB_RUN_ID": run_id,
        "V37_AB_PLAN_PATH": str(plan_path),
        "V37_AB_PLAN_SHA256": plan_sha256,
        "V37_AB_CELL_ID": cell_id,
    }
    if not isinstance(environment, dict) or any(
        environment.get(key) != value for key, value in expected_environment.items()
    ):
        raise GateError(f"{label}: audited environment does not bind the A/B preregistration")
    return run_id, str(plan_path), plan_sha256


def _paired_eval_data_sha256(manifest: Mapping[str, Any], label: str) -> str:
    snapshots = manifest.get("input_snapshots")
    validation = snapshots.get("validation_data") if isinstance(snapshots, dict) else None
    if not isinstance(validation, list) or not validation:
        raise GateError(f"{label}: validation snapshots missing for paired eval binding")
    payload: list[dict[str, str]] = []
    suites: set[str] = set()
    for index, item in enumerate(validation):
        if not isinstance(item, dict) or set(item) != {"suite", "dataset"}:
            raise GateError(f"{label}: invalid validation snapshot {index}")
        suite = item.get("suite")
        dataset = item.get("dataset")
        if not isinstance(suite, str) or not suite or suite in suites or not isinstance(dataset, dict):
            raise GateError(f"{label}: invalid or duplicate validation suite {index}")
        suites.add(suite)
        payload.append({"suite": suite, "sha256": known_sha(dataset.get("sha256"), f"{label}:validation[{index}].sha256")})
    recomputed = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
    if known_sha(manifest.get("paired_eval_data_sha256"), f"{label}:paired_eval_data_sha256") != recomputed:
        raise GateError(f"{label}: paired_eval_data_sha256 is not derived from validation snapshots")
    return recomputed


def _validate_formal_input_snapshots(manifest: Mapping[str, Any], label: str) -> None:
    snapshots = _exact_object(
        manifest.get("input_snapshots"),
        {"model", "training_data", "validation_data", "mask_metadata"},
        f"{label}:input_snapshots",
    )
    for name, parquet_only in (("model", False), ("training_data", True)):
        expected = snapshots.get(name)
        if not isinstance(expected, dict) or not isinstance(expected.get("root"), str):
            raise GateError(f"{label}: invalid {name} content snapshot")
        actual = content_snapshot(Path(expected["root"]), parquet_only=parquet_only)
        if canonical(actual) != canonical(expected):
            raise GateError(f"{label}: {name} content snapshot mismatch")
    validation = snapshots.get("validation_data")
    if not isinstance(validation, list) or not validation:
        raise GateError(f"{label}: validation_data snapshots missing")
    for index, item in enumerate(validation):
        if not isinstance(item, dict) or set(item) != {"suite", "dataset"}:
            raise GateError(f"{label}: invalid validation snapshot {index}")
        expected = item["dataset"]
        if not isinstance(item["suite"], str) or not isinstance(expected, dict) or not isinstance(expected.get("root"), str):
            raise GateError(f"{label}: invalid validation snapshot {index}")
        if canonical(content_snapshot(Path(expected["root"]), parquet_only=True)) != canonical(expected):
            raise GateError(f"{label}: validation snapshot {index} mismatch")
    metadata = _exact_object(
        snapshots.get("mask_metadata"), {"path", "sha256"}, f"{label}:mask_metadata snapshot",
    )
    metadata_path = _absolute_nonsymlink_path(Path(metadata["path"]), f"{label}:metadata snapshot")
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise GateError(f"{label}: metadata snapshot path is not a regular non-symlink file")
    metadata_hash = known_sha(metadata.get("sha256"), f"{label}:mask_metadata.sha256")
    if sha256(metadata_path) != metadata_hash:
        raise GateError(f"{label}: metadata file changed after launch")

    model_root = str(Path(snapshots["model"]["root"]).resolve())
    train_root = str(Path(snapshots["training_data"]["root"]).resolve())
    if str(Path(manifest.get("model_path", "")).resolve()) != model_root:
        raise GateError(f"{label}: model_path does not match the initial model snapshot")
    if str(Path(manifest.get("train_data", "")).resolve()) != train_root:
        raise GateError(f"{label}: train_data does not match the training snapshot")
    if str(Path(manifest.get("mask_metadata", "")).resolve()) != str(metadata_path.resolve()):
        raise GateError(f"{label}: mask_metadata path does not match its snapshot")
    if manifest.get("dataset_sha256") != snapshots["training_data"]["sha256"]:
        raise GateError(f"{label}: dataset_sha256 does not match training_data snapshot")
    if manifest.get("metadata_sha256") != metadata_hash:
        raise GateError(f"{label}: metadata_sha256 does not match metadata snapshot")
    inputs = manifest.get("input_hashes")
    if not isinstance(inputs, dict) or inputs.get("train") != snapshots["training_data"]["sha256"]:
        raise GateError(f"{label}: input_hashes.train mismatch")
    for index, item in enumerate(validation):
        if inputs.get(f"validation_{index}") != item["dataset"]["sha256"]:
            raise GateError(f"{label}: input_hashes.validation_{index} mismatch")


def _validate_preflight_binding(manifest: Mapping[str, Any], label: str) -> None:
    path_text = manifest.get("preflight_report")
    if not isinstance(path_text, str) or not path_text:
        raise GateError(f"{label}: preflight_report path missing")
    path = _absolute_nonsymlink_path(Path(path_text), f"{label}:preflight report")
    if path.is_symlink() or not path.is_file():
        raise GateError(f"{label}: preflight_report must be a regular non-symlink file")
    expected = known_sha(manifest.get("preflight_report_sha256"), f"{label}:preflight_report_sha256")
    report, is_jsonl = _load_hashed_rows(path, expected, f"{label}: preflight_report")
    if is_jsonl or not isinstance(report, dict):
        raise GateError(f"{label}: preflight_report must be one JSON object")
    if report.get("ok") is not True or report.get("run_class") != "formal" or report.get("data_mode") != "frontier_rl":
        raise GateError(f"{label}: preflight_report is not a successful formal frontier audit")
    if report.get("errors") != []:
        raise GateError(f"{label}: preflight_report contains errors")
    formal = report.get("formal_contract")
    if not isinstance(formal, dict) or formal.get("errors") != []:
        raise GateError(f"{label}: formal preflight contract is missing or failed")
    external_images = formal.get("external_image_snapshots")
    try:
        from tools.preflight_v37_training import verify_external_image_snapshots
    except ModuleNotFoundError:  # Direct `python tools/v37_gate.py` execution.
        from preflight_v37_training import verify_external_image_snapshots
    external_errors = verify_external_image_snapshots(external_images)
    if external_errors:
        raise GateError(f"{label}: external image snapshot verification failed: {external_errors[0]}")
    loader = formal.get("loader_observation")
    if not isinstance(loader, dict) or loader.get("loader_contract") != "verl.utils.dataset.RLHFDataset:v1":
        raise GateError(f"{label}: real RLHFDataset loader observation missing")
    loader_config = loader.get("loader_config")
    config = manifest.get("config", {})
    if not isinstance(loader_config, dict) or loader_config.get("filter_overlong_num_proc") != config.get("filter_overlong_num_proc"):
        raise GateError(f"{label}: preflight/training filter_overlong_num_proc mismatch")
    if config.get("filter_overlong_num_proc") != FORMAL_FILTER_OVERLONG_NUM_PROC:
        raise GateError(
            f"{label}: formal filter_overlong_num_proc must equal {FORMAL_FILTER_OVERLONG_NUM_PROC}"
        )
    if loader.get("sample_count") != report.get("sample_count") or integer(loader.get("sample_count"), f"{label}:loader sample_count", minimum=1) < 1:
        raise GateError(f"{label}: loader observation sample count mismatch")
    frozen = manifest.get("preflight_summary", {}).get("formal_contract")
    if canonical(frozen) != canonical(formal):
        raise GateError(f"{label}: manifest preflight_summary does not match the sealed report")
    snapshots = report.get("formal_input_snapshots")
    if not isinstance(snapshots, dict):
        raise GateError(f"{label}: formal preflight input snapshots missing")
    if canonical(manifest.get("preflight_summary", {}).get("formal_input_snapshots")) != canonical(snapshots):
        raise GateError(f"{label}: manifest does not seal formal preflight input snapshots")
    for field in ("data", "metadata", "masks", "validation", "forbidden"):
        if field not in snapshots:
            raise GateError(f"{label}: formal preflight snapshot {field} missing")
    input_snapshots = manifest["input_snapshots"]
    training_root = Path(input_snapshots["training_data"]["root"])
    validation_roots = [Path(item["dataset"]["root"]) for item in input_snapshots["validation_data"]]
    actual_data_hash = tree_sha256(training_root)
    actual_validation_hashes = [tree_sha256(item) for item in validation_roots]
    forbidden_paths = report.get("forbidden_paths")
    if not isinstance(forbidden_paths, list) or not forbidden_paths:
        raise GateError(f"{label}: preflight forbidden_paths missing")
    actual_forbidden_hashes = [tree_sha256(Path(item)) for item in forbidden_paths]
    if report.get("data_path") != str(training_root.resolve()):
        raise GateError(f"{label}: preflight data_path does not match training snapshot")
    if report.get("validation_paths") != [str(item.resolve()) for item in validation_roots]:
        raise GateError(f"{label}: preflight validation paths do not match validation snapshots")
    expected_preflight_snapshots = {
        "data": actual_data_hash,
        "metadata": sha256(Path(manifest["mask_metadata"])),
        "masks": tree_sha256(Path(manifest["masks_dir"])),
        "validation": actual_validation_hashes,
        "forbidden": actual_forbidden_hashes,
    }
    if canonical(snapshots) != canonical(expected_preflight_snapshots):
        raise GateError(f"{label}: preflight input snapshots do not match current entities")
    if formal.get("dataset_sha256") != actual_data_hash:
        raise GateError(f"{label}: preflight dataset hash mismatch")
    if formal.get("metadata_sha256") != manifest.get("metadata_sha256"):
        raise GateError(f"{label}: preflight metadata hash mismatch")
    if formal.get("mask_tree_sha256") != manifest.get("mask_tree_sha256"):
        raise GateError(f"{label}: preflight mask tree hash mismatch")


def _validate_mechanism_config(manifest: Mapping[str, Any], arm: str, run_class: str, seed: Any) -> dict[str, Any]:
    label = f"{arm}/{seed}:mechanism_config"
    if manifest.get("manifest_version") != 3:
        raise GateError(f"{arm}/{seed}: manifest_version must be 3")
    if manifest.get("data_mode") != "frontier_rl":
        raise GateError(f"{arm}/{seed}: data_mode must be frontier_rl")
    expected_promotable = run_class == "formal"
    if manifest.get("promotable_candidate") is not expected_promotable:
        raise GateError(f"{arm}/{seed}: promotable_candidate mismatch")
    expected_purpose = "formal_ab" if run_class == "formal" else "canary_diagnostic"
    if manifest.get("run_purpose") != expected_purpose:
        raise GateError(f"{arm}/{seed}: run_purpose must be {expected_purpose}")
    if manifest.get("resume_mode") != "clean_start":
        raise GateError(f"{arm}/{seed}: A/B evidence must be a clean start")
    if manifest.get("resume_evidence") is not None:
        raise GateError(f"{arm}/{seed}: clean-start A/B must not carry continuation lineage")
    if run_class == "formal":
        selected_config = manifest.get("config")
        environment = manifest.get("audited_environment")
        expected_merge = {
            "host_memory_preflight": "error",
            "host_memory_safety_factor": "4.0",
        }
        if not isinstance(selected_config, dict) or selected_config.get("hf_merge") != expected_merge:
            raise GateError(f"{arm}/{seed}: formal HF merge memory policy is not frozen")
        expected_environment = {
            "V37_FINALIZE_HF_CHECKPOINT": "1",
            "V37_HF_MERGE_HOST_MEMORY_PREFLIGHT": "error",
            "V37_HF_MERGE_HOST_MEMORY_SAFETY_FACTOR": "4.0",
            "RAY_GPU_WAIT_TIMEOUT_SECONDS": "900",
            "RAY_STATUS_TIMEOUT_SECONDS": "10",
            "RAY_START_TIMEOUT_SECONDS": "60",
            "RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS": "900",
        }
        if not isinstance(environment, dict) or any(
            environment.get(key) != value for key, value in expected_environment.items()
        ):
            raise GateError(f"{arm}/{seed}: formal HF merge/Ray release policy is not frozen")
    keys = {
        "schema_version", "arm", "estimator", "native_action_enabled",
        "native_action_parser_contract", "native_action_ledger_contract",
        "legacy_process_reward_enabled", "step_signal", "step_weight", "answer_gate",
        "correctness_first", "allwrong_terminal_zero", "correctness_secondary",
        "reward_fail_closed", "adaptive_actor_kl",
        "legacy_kl", "cp_size", "fallback_logprob_sign_opt_in",
        "torch_logprob_fallback_mode", "strict_answer_integer_parse",
        "strict_raw_success_winner", "winner_mode", "strict_point_parser_contract",
    }
    raw = _exact_object(manifest.get("mechanism_config"), keys, label)
    expected_arm = {
        "baseline": {
            "arm": "baseline", "estimator": "bok_grpo", "native_action_enabled": False,
            "native_action_parser_contract": "native_action_parser_v1",
            "native_action_ledger_contract": "native_action_ledger_v2",
            "step_signal": None, "step_weight": 0.0,
            "answer_gate": {"mode": None, "minimum": None},
        },
        "progress": {
            "arm": "progress", "estimator": "bok_grpo_step", "native_action_enabled": True,
            "native_action_parser_contract": "native_action_parser_v1",
            "native_action_ledger_contract": "native_action_ledger_v2",
            "step_signal": "native_action_event", "step_weight": .1,
            "answer_gate": {"mode": "answer_soft", "minimum": .2},
        },
    }[arm]
    for field, expected in expected_arm.items():
        if canonical(raw.get(field)) != canonical(expected):
            raise GateError(f"{label}.{field} must equal {expected!r}")
    shared = {
        "schema_version": 3,
        "legacy_process_reward_enabled": False,
        "correctness_first": True,
        "allwrong_terminal_zero": True,
        "reward_fail_closed": True,
        "cp_size": 1,
        "fallback_logprob_sign_opt_in": False,
        "torch_logprob_fallback_mode": "error",
        "strict_answer_integer_parse": True,
        "strict_raw_success_winner": True,
        "winner_mode": "outcome_success",
        "strict_point_parser_contract": "strict_point_slots_v2",
    }
    for field, expected in shared.items():
        if raw.get(field) != expected:
            raise GateError(f"{label}.{field} must equal {expected!r}")
    correctness_secondary = _exact_object(
        raw.get("correctness_secondary"),
        {
            "task_weight", "quality_weight", "partial_scale",
            "all_answer_wrong_terminal_zero", "exact_answer_partial_enabled",
        },
        f"{label}.correctness_secondary",
    )
    expected_secondary = {
        "task_weight": .25,
        "quality_weight": .1,
        "partial_scale": .25,
        "all_answer_wrong_terminal_zero": True,
        "exact_answer_partial_enabled": True,
    }
    if canonical(correctness_secondary) != canonical(expected_secondary):
        raise GateError(f"{label}.correctness_secondary is not the frozen contract")
    adaptive = _exact_object(
        raw.get("adaptive_actor_kl"),
        {
            "enabled", "type", "penalty", "init_beta", "target", "horizon",
            "horizon_unit", "loss_reduction", "selector_includes_kl",
        },
        f"{label}.adaptive_actor_kl",
    )
    expected_adaptive = {
        "enabled": True, "type": "adaptive", "penalty": "low_var_kl",
        "init_beta": .08, "target": .15, "horizon": 50,
        "horizon_unit": "executed_optimizer_updates",
        "loss_reduction": "response_token_mean", "selector_includes_kl": False,
    }
    if canonical(adaptive) != canonical(expected_adaptive):
        raise GateError(f"{label}.adaptive_actor_kl values are not the frozen contract")
    legacy = _exact_object(raw.get("legacy_kl"), {"use_kl_loss", "reward_kl_enabled"}, f"{label}.legacy_kl")
    if legacy != {"use_kl_loss": False, "reward_kl_enabled": False}:
        raise GateError(f"{label}: legacy KL must be fully excluded")
    return dict(raw)


def _validate_mining_contract(manifest: Mapping[str, Any], label: str) -> None:
    raw = _exact_object(
        manifest.get("mining_contract"),
        {
            "seeds", "candidates_per_seed", "classifier_thresholds", "outcome_ratio",
            "process_ratio", "bucket_ratios", "model_checkpoint_path",
            "model_checkpoint_sha256", "source_binding_contract", "source_binding_sha256",
            "generation_config_by_seed", "sampling_config", "seed_difference_allowlist",
        },
        f"{label}:mining_contract",
    )
    checkpoint = _absolute_nonsymlink_path(
        Path(str(raw.get("model_checkpoint_path", ""))), f"{label}:mining checkpoint",
    )
    checkpoint_hash = known_sha(
        raw.get("model_checkpoint_sha256"), f"{label}:mining checkpoint SHA256",
    )
    if tree_sha256(checkpoint) != checkpoint_hash:
        raise GateError(f"{label}: mining checkpoint changed after frontier publication")
    if raw.get("source_binding_contract") != "source_row_image_rendered_prompt_v1":
        raise GateError(f"{label}: source binding contract is missing")
    known_sha(raw.get("source_binding_sha256"), f"{label}:source binding SHA256")
    if raw.get("candidates_per_seed") != 32:
        raise GateError(f"{label}: candidates_per_seed must be 32")
    if canonical(raw.get("classifier_thresholds")) != canonical(FORMAL_CLASSIFIER_THRESHOLDS):
        raise GateError(f"{label}: classifier thresholds are not frozen")
    if finite(raw.get("outcome_ratio"), f"{label}:outcome_ratio") != .75 or finite(raw.get("process_ratio"), f"{label}:process_ratio") != .25:
        raise GateError(f"{label}: outcome/process composition must be 75/25")
    if canonical(raw.get("bucket_ratios")) != canonical(FORMAL_BUCKET_RATIOS):
        raise GateError(f"{label}: selection ratios must be 40/10/20/20/10")
    if raw.get("seed_difference_allowlist") != ["seed"]:
        raise GateError(f"{label}: mining seed difference allowlist must be exactly ['seed']")
    seeds = raw.get("seeds")
    if (
        not isinstance(seeds, list) or len(seeds) != 2
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != 2
    ):
        raise GateError(f"{label}: exactly two distinct nonnegative integer mining seeds are required")
    generation = raw.get("generation_config_by_seed")
    if not isinstance(generation, dict) or set(generation) != {str(seed) for seed in seeds}:
        raise GateError(f"{label}: generation config must cover both preregistered seeds")
    normalized: list[str] = []
    for seed in seeds:
        config = generation[str(seed)]
        if not isinstance(config, dict) or set(config) & {"generation_seed", "declared_seed"} or canonical(config.get("seed")) != canonical(seed):
            raise GateError(f"{label}: invalid declared generation seed config")
        normalized.append(canonical({key: value for key, value in config.items() if key != "seed"}))
    if len(set(normalized)) != 1:
        raise GateError(f"{label}: generation config differs across seeds outside seed")
    sampling = raw.get("sampling_config")
    if not isinstance(sampling, dict) or set(sampling) & {"seed", "generation_seed", "declared_seed"}:
        raise GateError(f"{label}: sampling config must be seed-independent")


def _validate_rule(name: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise GateError(f"paired metric rule {name} must be an object")
    direction = raw.get("direction")
    expected_benefit = "increase" if name in HIGHER_BETTER else "decrease"
    if direction in {"increase", "decrease"}:
        if direction != expected_benefit:
            raise GateError(f"paired metric {name} has unsafe direction {direction}")
        if "margin" in raw:
            raise GateError(f"paired metric {name}: strict gain rule cannot declare a margin")
        return {"direction": direction}
    if direction != "noninferiority":
        raise GateError(f"paired metric {name} has unknown direction {direction!r}")
    beneficial = raw.get("beneficial_direction")
    if beneficial != expected_benefit:
        raise GateError(f"paired metric {name} has invalid beneficial_direction")
    margin = finite(raw.get("margin"), f"paired metric {name} margin")
    if margin < 0:
        raise GateError(f"paired metric {name} margin must be nonnegative")
    return {"direction": direction, "beneficial_direction": beneficial, "margin": margin}


def _validate_metric_rules(pre: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = pre.get("paired_metrics")
    if not isinstance(raw, dict):
        raise GateError("paired_metrics must be a preregistered object")
    missing = sorted(REQUIRED_PAIRED_METRICS - raw.keys())
    if missing:
        raise GateError(f"missing mandatory paired metric rules: {missing}")
    unknown = sorted(raw.keys() - REQUIRED_PAIRED_METRICS)
    if unknown:
        raise GateError(f"unknown paired metric rules: {unknown}")
    return {name: _validate_rule(name, raw[name]) for name in sorted(raw)}


def _formal_metric_rules(pre: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rules = _validate_metric_rules(pre)
    if canonical(rules) != canonical(FROZEN_PAIRED_METRICS):
        raise GateError("formal paired metric directions/margins are not the frozen contract")
    return rules


def _paired_source_identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
    ):
        raise GateError(f"{label} must be a canonical nonempty string")
    return value


def _paired_source_answer(value: Any, label: str) -> int:
    if type(value) is int:
        answer = value
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        answer = int(value.strip())
    else:
        raise GateError(f"{label} must be an explicit integer")
    if not 2 <= answer <= 50:
        raise GateError(f"{label} must be in dense-count range 2..50")
    return answer


def _paired_source_bucket(answer: int) -> str:
    for bucket, (low, high) in BUCKET_RANGES.items():
        if low <= answer <= high:
            return bucket
    raise GateError(f"paired source answer {answer} is outside the bucket contract")


def _recompute_paired_source_rows(data_path: Path) -> list[dict[str, Any]]:
    """Read authoritative IDs/GT from parquet instead of trusting universe caches."""
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise GateError(
            "pyarrow is required to independently verify the formal paired sample universe"
        ) from exc
    try:
        table = ds.dataset(data_path, format="parquet").to_table(
            columns=list(PAIRED_UNIVERSE_COLUMNS)
        )
    except Exception as exc:
        raise GateError(f"cannot read formal paired universe parquet source: {exc}") from exc
    if table.num_rows < 1:
        raise GateError("formal paired universe parquet source is empty")

    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for index in range(table.num_rows):
        sample_id = _paired_source_identifier(
            table["sample_id"][index].as_py(), f"paired source row {index} sample_id",
        )
        prompt_id = _paired_source_identifier(
            table["prompt_id"][index].as_py(), f"paired source row {index} prompt_id",
        )
        identity = (sample_id, prompt_id)
        if identity in identities:
            raise GateError(f"duplicate paired source identity: {identity}")
        identities.add(identity)
        ground_truth = _paired_source_answer(
            table["answer"][index].as_py(), f"paired source row {index} answer",
        )
        bucket = _paired_source_bucket(ground_truth)
        projection = {
            "sample_id": sample_id,
            "prompt_id": prompt_id,
            "bucket": bucket,
            "ground_truth": ground_truth,
        }
        rows.append({
            **projection,
            "source_row_sha256": hashlib.sha256(
                eval_producer._canonical_bytes(projection)
            ).hexdigest(),
        })
    rows.sort(key=lambda row: (row["sample_id"], row["prompt_id"]))
    return rows


def _load_paired_sample_universe(
    binding: Any,
    validation_snapshots: Any,
    label: str,
    *,
    verify_source_rows: bool = False,
) -> tuple[dict[tuple[str, str], dict[str, Any]], str, str]:
    binding = _exact_object(binding, {"path", "sha256"}, f"{label}:sample universe binding")
    path = _absolute_nonsymlink_path(Path(str(binding.get("path", ""))), f"{label}:sample universe")
    if not path.is_file():
        raise GateError(f"{label}: sample universe must be a regular file")
    binding_hash = known_sha(binding.get("sha256"), f"{label}:sample universe SHA256")
    payload, is_jsonl = _load_hashed_rows(
        path, binding_hash, f"{label}: sample universe changed after preregistration",
    )
    if is_jsonl:
        raise GateError(f"{label}: sample universe must be one strict JSON object")
    payload = _exact_object(
        payload,
        {
            "schema_version", "artifact_type", "data_path", "data_sha256",
            "sample_count", "bucket_counts", "ordered_rows_sha256", "rows",
        },
        f"{label}:sample universe",
    )
    if (
        payload.get("schema_version") != PAIRED_UNIVERSE_SCHEMA_VERSION
        or payload.get("artifact_type") != PAIRED_UNIVERSE_ARTIFACT_TYPE
    ):
        raise GateError(f"{label}: sample universe schema/type mismatch")
    data_path = _absolute_nonsymlink_path(
        Path(str(payload.get("data_path", ""))), f"{label}:sample universe data",
    )
    data_hash = known_sha(payload.get("data_sha256"), f"{label}:sample universe data SHA256")
    if content_snapshot(data_path, parquet_only=True).get("sha256") != data_hash:
        raise GateError(f"{label}: sample universe source data changed")
    if not isinstance(validation_snapshots, list):
        raise GateError(f"{label}: validation snapshots are missing")
    matching_validation = [
        item for item in validation_snapshots
        if isinstance(item, dict)
        and isinstance(item.get("dataset"), dict)
        and item["dataset"].get("root") == str(data_path)
        and item["dataset"].get("sha256") == data_hash
    ]
    if len(matching_validation) != 1:
        raise GateError(f"{label}: sample universe is not bound to exactly one validation dataset")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise GateError(f"{label}: sample universe rows must be nonempty")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    normalized_rows: list[dict[str, Any]] = []
    bucket_counts = {bucket: 0 for bucket in BUCKETS}
    for index, row in enumerate(rows):
        row = _exact_object(
            row,
            {"sample_id", "prompt_id", "bucket", "ground_truth", "source_row_sha256"},
            f"{label}:sample universe row {index}",
        )
        sample_id = row.get("sample_id")
        prompt_id = row.get("prompt_id")
        if not isinstance(sample_id, str) or not sample_id or not isinstance(prompt_id, str) or not prompt_id:
            raise GateError(f"{label}: sample universe identifiers must be nonempty strings")
        key = (sample_id, prompt_id)
        if key in indexed:
            raise GateError(f"{label}: duplicate sample universe identity {key}")
        bucket = row.get("bucket")
        ground_truth = integer(row.get("ground_truth"), f"{label}:universe ground_truth", minimum=1)
        if bucket not in BUCKET_RANGES or not (
            BUCKET_RANGES[bucket][0] <= ground_truth <= BUCKET_RANGES[bucket][1]
        ):
            raise GateError(f"{label}: sample universe GT/bucket mismatch")
        source_hash = known_sha(
            row.get("source_row_sha256"), f"{label}:universe source row SHA256",
        )
        source_projection = {
            "sample_id": sample_id, "prompt_id": prompt_id,
            "bucket": bucket, "ground_truth": ground_truth,
        }
        expected_source_hash = hashlib.sha256(
            eval_producer._canonical_bytes(source_projection)
        ).hexdigest()
        if source_hash != expected_source_hash:
            raise GateError(f"{label}: universe source row hash is not derived from identity/GT/bucket")
        normalized = {
            **source_projection,
            "source_row_sha256": source_hash,
        }
        indexed[key] = normalized
        normalized_rows.append(normalized)
        bucket_counts[bucket] += 1
    if integer(payload.get("sample_count"), f"{label}:sample_count", minimum=1) != len(rows):
        raise GateError(f"{label}: sample universe count mismatch")
    if payload.get("bucket_counts") != bucket_counts or any(value == 0 for value in bucket_counts.values()):
        raise GateError(f"{label}: sample universe must exactly cover all five buckets")
    ordered_hash = hashlib.sha256(canonical(normalized_rows).encode("utf-8")).hexdigest()
    if known_sha(payload.get("ordered_rows_sha256"), f"{label}:ordered rows SHA256") != ordered_hash:
        raise GateError(f"{label}: sample universe ordered row hash mismatch")
    sorted_rows = sorted(normalized_rows, key=lambda row: (row["sample_id"], row["prompt_id"]))
    if normalized_rows != sorted_rows:
        raise GateError(f"{label}: sample universe rows must be sorted by sample_id/prompt_id")
    if verify_source_rows:
        source_before = content_snapshot(data_path, parquet_only=True)
        authoritative_rows = _recompute_paired_source_rows(data_path)
        source_after = content_snapshot(data_path, parquet_only=True)
        if canonical(source_before) != canonical(source_after) or source_after.get("sha256") != data_hash:
            raise GateError(f"{label}: sample universe source data mutated during verification")
        if canonical(authoritative_rows) != canonical(normalized_rows):
            raise GateError(f"{label}: sample universe rows differ from authoritative parquet IDs/GT")
    return indexed, binding_hash, data_hash


def _strict_transcript_facts(
    transcript: Any, point_events: Any, label: str,
) -> tuple[bool, int | None, int, list[str]]:
    if not isinstance(transcript, list) or not transcript or any(
        not isinstance(turn, str) or "\x00" in turn for turn in transcript
    ):
        raise GateError(f"{label}: transcript must be a nonempty list of NUL-free strings")
    if not isinstance(point_events, list):
        raise GateError(f"{label}: point_events must be a list")

    structurally_valid = True
    answer_value: int | None = None
    point_turns = 0
    point_payloads: list[str] = []
    answer_turn: int | None = None
    for turn_index, turn in enumerate(transcript):
        point_open = len(POINT_OPEN_RE.findall(turn))
        point_close = len(POINT_CLOSE_RE.findall(turn))
        answer_open = len(ANSWER_OPEN_RE.findall(turn))
        answer_close = len(ANSWER_CLOSE_RE.findall(turn))
        point_action = point_open == point_close == 1 and answer_open == answer_close == 0
        answer_matches = ANSWER_BLOCK_RE.findall(turn)
        answer_action = (
            answer_open == answer_close == 1
            and point_open == point_close == 0
            and len(answer_matches) == 1
        )
        if point_action:
            point_turns += 1
            payloads = POINT_BLOCK_CAPTURE_RE.findall(turn)
            if len(payloads) != 1:
                structurally_valid = False
            else:
                point_payloads.append(payloads[0])
            if answer_turn is not None:
                structurally_valid = False
        elif answer_action:
            if answer_turn is not None:
                structurally_valid = False
            answer_turn = turn_index
            answer_value = int(answer_matches[0])
        else:
            structurally_valid = False
    if answer_turn is not None and answer_turn != len(transcript) - 1:
        structurally_valid = False
    if point_turns != len(point_events):
        structurally_valid = False
    return structurally_valid, answer_value, point_turns, point_payloads


def _recompute_formal_sample(
    row: Mapping[str, Any], label: str, *, ground_truth: int, bucket: str,
) -> dict[str, float]:
    expected_keys = {
        "sample_id", "prompt_id", "source_row_sha256", "predicted_answer",
        "transcript", "point_events", "termination_reason", "num_rounds",
        "configured_max_turns", "effective_max_turns", "metrics",
    }
    _exact_object(row, expected_keys, label)
    configured_max = integer(
        row.get("configured_max_turns"), f"{label}:configured_max_turns", minimum=1,
    )
    effective_max = integer(
        row.get("effective_max_turns"), f"{label}:effective_max_turns", minimum=1,
    )
    if effective_max != min(configured_max, ground_truth + 3):
        raise GateError(f"{label}: effective_max_turns is not min(configured, GT+3)")
    transcript = row.get("transcript")
    point_events = row.get("point_events")
    structurally_valid, parsed_answer, point_turns, point_payloads = _strict_transcript_facts(
        transcript, point_events, label,
    )
    assert isinstance(transcript, list) and isinstance(point_events, list)
    if integer(row.get("num_rounds"), f"{label}:num_rounds", minimum=1) != len(transcript):
        raise GateError(f"{label}: num_rounds does not equal transcript length")
    if len(transcript) > effective_max:
        raise GateError(f"{label}: transcript exceeds effective_max_turns")

    predicted = row.get("predicted_answer")
    if predicted is not None:
        if type(predicted) is int:
            predicted_value = predicted
        elif isinstance(predicted, str) and INTEGER_ANSWER_RE.fullmatch(predicted.strip()):
            predicted_value = int(predicted.strip())
        else:
            raise GateError(f"{label}: predicted_answer must be an explicit integer or null")
    else:
        predicted_value = None
    if predicted_value != parsed_answer:
        raise GateError(f"{label}: predicted_answer differs from the final transcript answer")

    seen_targets: set[str] = set()
    duplicate_count = 0
    sequential = True
    for event_index, event in enumerate(point_events, start=1):
        event = _exact_object(
            event, {"point_index", "point_payload_sha256", "matched_target_id"},
            f"{label}:point_event[{event_index}]",
        )
        if integer(event.get("point_index"), f"{label}:point_index", minimum=1) != event_index:
            sequential = False
        raw_payload = point_payloads[event_index - 1] if event_index <= len(point_payloads) else ""
        payload_hash = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
        if known_sha(
            event.get("point_payload_sha256"), f"{label}:point_payload_sha256"
        ) != payload_hash:
            raise GateError(f"{label}: point event is detached from transcript payload")
        payload_valid = True
        try:
            payload = json.loads(raw_payload, object_pairs_hook=_reject_duplicate_keys)
            if not isinstance(payload, dict):
                raise ValueError("point payload is not an object")
            count_number = integer(
                payload.get("count_number"), f"{label}:point payload count_number", minimum=1,
            )
            point_2d = payload.get("point_2d")
            if (
                not isinstance(point_2d, list) or len(point_2d) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point_2d)
                or any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1000 for value in point_2d)
            ):
                raise ValueError("point_2d is invalid")
            label_value = payload.get("label", "object")
            if not isinstance(label_value, str) or not label_value.strip() or "\x00" in label_value:
                raise ValueError("point label is invalid")
            if count_number != event_index:
                sequential = False
        except (GateError, TypeError, ValueError, json.JSONDecodeError):
            payload_valid = False
            sequential = False
        target_id = event.get("matched_target_id")
        if target_id is not None and (
            not isinstance(target_id, str) or not target_id.strip() or "\x00" in target_id
        ):
            raise GateError(f"{label}: matched_target_id must be a nonempty string or null")
        if not payload_valid and target_id is not None:
            raise GateError(f"{label}: malformed point payload cannot claim a target match")
        if isinstance(target_id, str):
            target_id = target_id.strip()
            if target_id in seen_targets:
                duplicate_count += 1
            else:
                seen_targets.add(target_id)

    termination = row.get("termination_reason")
    if termination not in {"answer", "cap", "malformed", "error"}:
        raise GateError(f"{label}: invalid termination_reason")
    if termination == "answer" and parsed_answer is None:
        raise GateError(f"{label}: answer termination lacks a strict final answer")
    if termination != "answer" and parsed_answer is not None:
        raise GateError(f"{label}: non-answer termination contains a strict final answer")
    if termination == "cap" and len(transcript) != effective_max:
        raise GateError(f"{label}: cap termination must occur exactly at effective_max_turns")
    if len(seen_targets) > ground_truth:
        raise GateError(f"{label}: unique matched targets exceed authoritative GT")
    known_sha(row.get("source_row_sha256"), f"{label}:source_row_sha256")

    denominator = max(len(point_events), 1)
    recomputed = {
        "answer_exact": float(parsed_answer == ground_truth),
        "format_compliance": float(structurally_valid and sequential),
        "unique_valid_hit": len(seen_targets) / float(ground_truth),
        "duplicate": duplicate_count / float(denominator),
        "cap_turn_exceeded": float(termination == "cap"),
        "early_stop": float(termination == "answer" and point_turns < ground_truth),
    }
    cached = row.get("metrics")
    if not isinstance(cached, dict) or set(cached) != set(recomputed):
        raise GateError(f"{label}: cached metrics schema mismatch")
    for name, expected in recomputed.items():
        actual = finite(cached.get(name), f"{label}:metrics.{name}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise GateError(f"{label}: cached metric {name} differs from gate recomputation")
    return recomputed


def _sample_rows(
    ev: Mapping[str, Any], label: str, *, require_recomputable: bool = False,
    sample_universe: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    sample_universe_sha256: str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if require_recomputable:
        formal_keys = {
            "schema_version", "artifact_type", "producer_invocation_nonce",
            "sample_universe_sha256",
            "data_sha256", "samples",
            "sample_set_sha256", "evidence_run_id", "execution_environment_sha256",
            "final_checkpoint_id", "final_checkpoint_path", "final_checkpoint_sha256",
        }
        _exact_object(ev, formal_keys, f"{label}:formal paired evidence")
        if (
            ev.get("schema_version") != FORMAL_PAIRED_SCHEMA_VERSION
            or ev.get("artifact_type") != FORMAL_PAIRED_ARTIFACT_TYPE
        ):
            raise GateError(f"{label}: formal paired evidence schema/type mismatch")
        nonce = ev.get("producer_invocation_nonce")
        if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
            raise GateError(f"{label}: invalid producer invocation nonce")
        if sample_universe is None or known_sha(
            ev.get("sample_universe_sha256"), f"{label}:sample universe SHA256",
        ) != sample_universe_sha256:
            raise GateError(f"{label}: paired evidence sample universe binding mismatch")
    rows = ev.get("samples")
    if not isinstance(rows, list) or not rows:
        raise GateError(f"{label}: eval.samples must be a nonempty list")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    sample_ids: set[str] = set()
    prompt_ids: set[str] = set()
    for number, row in enumerate(rows):
        if not isinstance(row, dict):
            raise GateError(f"{label}: sample {number} must be an object")
        sample_id = row.get("sample_id")
        prompt_id = row.get("prompt_id")
        if isinstance(sample_id, (dict, list, bool)) or not str(sample_id or "").strip():
            raise GateError(f"{label}: sample {number} has invalid sample_id")
        if isinstance(prompt_id, (dict, list, bool)) or not str(prompt_id or "").strip():
            raise GateError(f"{label}: sample {number} has invalid prompt_id")
        key = (str(sample_id).strip(), str(prompt_id).strip())
        if key[0] in sample_ids:
            raise GateError(f"{label}: duplicate sample_id {key[0]}")
        if key[1] in prompt_ids:
            raise GateError(f"{label}: duplicate prompt_id {key[1]}")
        sample_ids.add(key[0])
        prompt_ids.add(key[1])
        if require_recomputable:
            assert sample_universe is not None
            authoritative = sample_universe.get(key)
            if authoritative is None:
                raise GateError(f"{label}: sample {key} is outside the preregistered universe")
            if row.get("source_row_sha256") != authoritative["source_row_sha256"]:
                raise GateError(f"{label}: sample {key} source row binding mismatch")
            metrics = _recompute_formal_sample(
                row, f"{label}:sample {key}",
                ground_truth=authoritative["ground_truth"],
                bucket=authoritative["bucket"],
            )
            bucket = authoritative["bucket"]
        else:
            if row.get("bucket") not in BUCKETS:
                raise GateError(f"{label}: sample {key} has invalid/missing bucket")
            metrics = row.get("metrics")
            bucket = row["bucket"]
        if not isinstance(metrics, dict):
            raise GateError(f"{label}: sample {key} metrics must be an object")
        for name in REQUIRED_PAIRED_METRICS - {"heldout_five_bucket_macro"}:
            value = finite(metrics.get(name), f"{label}:{key}:{name}")
            if name in {"unique_valid_hit", "duplicate"}:
                if value < 0:
                    raise GateError(f"{label}: sample {key} metric {name} must be nonnegative")
            elif not 0 <= value <= 1:
                raise GateError(f"{label}: sample {key} metric {name} must be in [0,1]")
        indexed[key] = {**row, "bucket": bucket, "metrics": metrics}
    if require_recomputable and set(indexed) != set(sample_universe or {}):
        raise GateError(f"{label}: paired evidence does not exactly cover the sample universe")
    if {row["bucket"] for row in indexed.values()} != set(BUCKETS):
        raise GateError(f"{label}: held-out evaluation must cover all five buckets")
    return indexed


def _sample_set_sha256(rows: Mapping[tuple[str, str], dict[str, Any]]) -> str:
    frozen = [[sample_id, prompt_id, rows[(sample_id, prompt_id)]["bucket"]] for sample_id, prompt_id in sorted(rows)]
    return hashlib.sha256(canonical(frozen).encode("utf-8")).hexdigest()


def _mean_ci(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        raise GateError("cannot compute a paired metric from zero observations")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean, mean
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    radius = 1.96 * standard_error
    return mean, mean - radius, mean + radius


def _paired_values(
    name: str,
    baseline: Mapping[tuple[str, str], dict[str, Any]],
    progress: Mapping[tuple[str, str], dict[str, Any]],
) -> list[float]:
    if name == "heldout_five_bucket_macro":
        bucket_values: list[float] = []
        for bucket in BUCKETS:
            differences = [
                finite(progress[key]["metrics"].get("answer_exact"), f"{name}:progress")
                - finite(baseline[key]["metrics"].get("answer_exact"), f"{name}:baseline")
                for key in baseline if baseline[key]["bucket"] == bucket
            ]
            bucket_values.append(statistics.fmean(differences))
        return bucket_values
    return [
        finite(progress[key]["metrics"].get(name), f"progress:{key}:{name}")
        - finite(baseline[key]["metrics"].get(name), f"baseline:{key}:{name}")
        for key in baseline
    ]


def _apply_rule(name: str, rule: Mapping[str, Any], delta: float, low: float, high: float, label: str) -> None:
    direction = rule["direction"]
    if not low <= delta <= high:
        raise GateError(f"{label}: internally invalid CI for {name}")
    if direction == "increase" and not low > 0:
        raise GateError(f"{label}: {name} required gain CI does not exclude zero")
    if direction == "decrease" and not high < 0:
        raise GateError(f"{label}: {name} required reduction CI does not exclude zero")
    if direction == "noninferiority":
        margin = finite(rule["margin"], f"{name} margin")
        if rule["beneficial_direction"] == "increase" and low < -margin:
            raise GateError(f"{label}: {name} violates noninferiority margin")
        if rule["beneficial_direction"] == "decrease" and high > margin:
            raise GateError(f"{label}: {name} violates noninferiority margin")


def _verify_eval_pair(
    seed: Any,
    baseline: Mapping[str, Any],
    progress: Mapping[str, Any],
    rules: Mapping[str, Mapping[str, Any]],
    *,
    require_recomputable: bool = False,
    sample_universe: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    sample_universe_sha256: str | None = None,
) -> dict[str, dict[str, float]]:
    label = f"seed {seed}"
    comparable_fields = (
        ("data_sha256",)
        if require_recomputable
        else ("protocol_sha256", "data_sha256", "eval_config_sha256")
    )
    for field in comparable_fields:
        left = known_sha(baseline.get(field), f"baseline/{seed}:{field}")
        right = known_sha(progress.get(field), f"progress/{seed}:{field}")
        if left != right:
            raise GateError(f"{label}: eval {field} mismatch")
    baseline_rows = _sample_rows(
        baseline, f"baseline/{seed}", require_recomputable=require_recomputable,
        sample_universe=sample_universe, sample_universe_sha256=sample_universe_sha256,
    )
    progress_rows = _sample_rows(
        progress, f"progress/{seed}", require_recomputable=require_recomputable,
        sample_universe=sample_universe, sample_universe_sha256=sample_universe_sha256,
    )
    baseline_frozen = known_sha(baseline.get("sample_set_sha256"), f"baseline/{seed}:sample_set_sha256")
    progress_frozen = known_sha(progress.get("sample_set_sha256"), f"progress/{seed}:sample_set_sha256")
    if baseline_frozen != _sample_set_sha256(baseline_rows) or progress_frozen != _sample_set_sha256(progress_rows):
        raise GateError(f"{label}: frozen sample-set hash mismatch")
    if baseline_frozen != progress_frozen:
        raise GateError(f"{label}: baseline/progress frozen sample sets differ")
    if set(baseline_rows) != set(progress_rows):
        raise GateError(f"{label}: baseline/progress sample identities differ")
    for key in baseline_rows:
        if baseline_rows[key]["bucket"] != progress_rows[key]["bucket"]:
            raise GateError(f"{label}: sample bucket mismatch for {key}")
    computed: dict[str, dict[str, float]] = {}
    for name, rule in rules.items():
        values = _paired_values(name, baseline_rows, progress_rows)
        delta, low, high = _mean_ci(values)
        _apply_rule(name, rule, delta, low, high, label)
        computed[name] = {"delta": delta, "ci_low": low, "ci_high": high, "n": len(values)}
    return computed


def _validate_training_evidence_binding(
    manifest: Mapping[str, Any],
    log: Mapping[str, Any],
    log_path_value: str,
    log_sha256: str,
    label: str,
) -> None:
    try:
        training_evidence.verify_payload(dict(log))
    except training_evidence.TrainingEvidenceError as exc:
        raise GateError(f"{label}: invalid trainer-owned training evidence: {exc}") from exc
    if log.get("start_global_step") != 0 or log.get("final_global_step") != FORMAL_OPTIMIZER_STEPS:
        raise GateError(f"{label}: formal training evidence must cover clean-start steps 1-12")
    log_path = _absolute_nonsymlink_path(Path(log_path_value), f"{label}:training evidence")
    if manifest.get("training_evidence_path") != str(log_path):
        raise GateError(f"{label}: manifest training evidence path mismatch")
    if known_sha(
        manifest.get("training_evidence_sha256"), f"{label}:training_evidence_sha256"
    ) != known_sha(log_sha256, f"{label}:log_sha256"):
        raise GateError(f"{label}: manifest training evidence hash mismatch")
    for field in (
        "evidence_run_id",
        "execution_environment_sha256",
        "final_checkpoint_id",
        "final_checkpoint_path",
        "final_checkpoint_sha256",
    ):
        if log.get(field) != manifest.get(field):
            raise GateError(f"{label}: training evidence disagrees with manifest {field}")


def _validate_log(log: Mapping[str, Any], arm: str, seed: Any, pre: Mapping[str, Any]) -> None:
    label = f"{arm}/{seed}"
    expected_steps = integer(pre.get("optimizer_steps"), "preregistered.optimizer_steps", minimum=1)
    if integer(log.get("optimizer_steps"), f"{label}:optimizer_steps") != expected_steps:
        raise GateError(f"{label}: optimizer step mismatch")
    for field in ("skipped_updates", "nonfinite_count", "oom_count", "progress_degraded", "correctness_sign_error_count"):
        if integer(log.get(field), f"{label}:{field}") != 0:
            raise GateError(f"{label}: {field} nonzero")
    integer(log.get("cap_turn_exceeded"), f"{label}:cap_turn_exceeded")
    integer(log.get("early_stop_count"), f"{label}:early_stop_count")
    if finite(log.get("selector_kl_contribution"), f"{label}:selector_kl_contribution") != 0:
        raise GateError(f"{label}: selector KL not zero")
    if finite(log.get("event_mapping_coverage"), f"{label}:event_mapping_coverage") != 1:
        raise GateError(f"{label}: event mapping coverage below 100%")
    if finite(log.get("runtime_mask_coverage"), f"{label}:runtime_mask_coverage") != 1:
        raise GateError(f"{label}: runtime mask coverage below 100%")
    if finite(log.get("kl_recoverable"), f"{label}:kl_recoverable") != 1:
        raise GateError(f"{label}: KL recoverability not proven")
    if finite(log.get("frontier_mixed_group_ratio"), f"{label}:frontier_mixed_group_ratio") < finite(pre.get("frontier_mixed_min"), "frontier_mixed_min"):
        raise GateError(f"{label}: frontier mixed ratio gate failed")
    if finite(log.get("outcome_frontier_allwrong_ratio"), f"{label}:outcome_frontier_allwrong_ratio") > finite(pre.get("outcome_allwrong_max"), "outcome_allwrong_max"):
        raise GateError(f"{label}: outcome all-wrong ratio gate failed")


def _checkpoint_pass(
    manifest: Mapping[str, Any], ev: Mapping[str, Any], arm: str, seed: Any,
    checkpoint_paths: set[str], checkpoint_ids: set[str], checkpoint_hashes: set[str],
) -> None:
    final_id = manifest.get("final_checkpoint_id")
    final_hash = known_sha(manifest.get("final_checkpoint_sha256"), f"{arm}/{seed}:final_checkpoint_sha256")
    final_path = manifest.get("final_checkpoint_path")
    if not isinstance(final_id, str) or not final_id:
        raise GateError(f"{arm}/{seed}: final checkpoint identity missing")
    if not isinstance(final_path, str) or not final_path.strip():
        raise GateError(f"{arm}/{seed}: final checkpoint path missing")
    path = _absolute_nonsymlink_path(Path(final_path), f"{arm}/{seed}:final checkpoint")
    _reject_symlinks(path, f"{arm}/{seed}:final checkpoint")
    if not path.is_dir():
        raise GateError(f"{arm}/{seed}: final checkpoint must be an actual non-symlink directory")
    if arm not in final_id or f"seed{seed}" not in final_id:
        raise GateError(f"{arm}/{seed}: final checkpoint identity must be arm/seed-specific")
    if tree_sha256(path) != final_hash:
        raise GateError(f"{arm}/{seed}: actual checkpoint tree hash mismatch")
    resolved = str(path)
    if resolved in checkpoint_paths or final_id in checkpoint_ids or final_hash in checkpoint_hashes:
        raise GateError("checkpoint path/identity/hash reused across A/B runs")
    checkpoint_paths.add(resolved)
    checkpoint_ids.add(final_id)
    checkpoint_hashes.add(final_hash)
    if (
        ev.get("final_checkpoint_id") != final_id
        or ev.get("final_checkpoint_path") != str(path)
        or known_sha(ev.get("final_checkpoint_sha256"), f"{arm}/{seed}:eval final checkpoint hash") != final_hash
    ):
        raise GateError(f"{arm}/{seed}: final evaluation checkpoint provenance mismatch")


def _validate_eval_receipt(
    *,
    run: Mapping[str, Any],
    manifest: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    receipt: Mapping[str, Any],
    arm: str,
    seed: int,
) -> str:
    label = f"{arm}/{seed}:eval receipt"
    _exact_object(
        receipt,
        {
            "schema_version", "artifact_type", "cell_id", "arm", "seed",
            "invocation_nonce", "invocation_request_sha256", "manifest", "checkpoint", "eval_model", "recipe",
            "producer", "environment_sha256", "steps",
        },
        label,
    )
    registration = manifest.get("ab_preregistration")
    if not isinstance(registration, dict):
        raise GateError(f"{label}: A/B registration missing")
    nonce = receipt.get("invocation_nonce")
    if (
        receipt.get("schema_version") != eval_producer.RECEIPT_SCHEMA_VERSION
        or receipt.get("artifact_type") != eval_producer.RECEIPT_TYPE
        or receipt.get("cell_id") != registration.get("cell_id")
        or receipt.get("arm") != arm
        or receipt.get("seed") != seed
        or not isinstance(nonce, str)
        or re.fullmatch(r"[0-9a-f]{64}", nonce) is None
        or evaluation.get("producer_invocation_nonce") != nonce
    ):
        raise GateError(f"{label}: identity/nonce mismatch")

    manifest_path = _absolute_nonsymlink_path(Path(str(run["manifest"])), f"{label}:manifest")
    checkpoint = _absolute_nonsymlink_path(
        Path(str(manifest.get("final_checkpoint_path", ""))), f"{label}:checkpoint",
    )
    eval_model = _absolute_nonsymlink_path(
        checkpoint / "actor" / "huggingface", f"{label}:eval model",
    )
    expected_bindings = {
        "manifest": {
            "path": str(manifest_path),
            "sha256": known_sha(run.get("manifest_sha256"), f"{label}:manifest hash"),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": known_sha(
                manifest.get("final_checkpoint_sha256"), f"{label}:checkpoint hash",
            ),
        },
        "eval_model": {"path": str(eval_model), "sha256": tree_sha256(eval_model)},
    }
    for name, expected in expected_bindings.items():
        if receipt.get(name) != expected:
            raise GateError(f"{label}: {name} binding mismatch")

    plan = validate_ab_plan(
        registration.get("plan_path"),
        registration.get("plan_sha256"),
        registration.get("ab_run_id"),
        require_target_dirs=True,
    )
    pipeline = plan["preregistered"]["eval_contract"]["pipeline"]
    producer_binding = pipeline["producer"]
    recipe_binding = pipeline["recipe"]
    if receipt.get("producer") != producer_binding or receipt.get("recipe") != recipe_binding:
        raise GateError(f"{label}: producer/recipe differs from preregistration")
    request_hash = eval_producer.invocation_request_sha256(
        cell_id=str(registration["cell_id"]), arm=arm, seed=seed,
        invocation_nonce=nonce,
        manifest_path=str(manifest_path), manifest_sha256=expected_bindings["manifest"]["sha256"],
        checkpoint_path=str(checkpoint), checkpoint_sha256=expected_bindings["checkpoint"]["sha256"],
        eval_model_path=str(eval_model), eval_model_sha256=expected_bindings["eval_model"]["sha256"],
        recipe_path=recipe_binding["path"], recipe_sha256=recipe_binding["sha256"],
        producer_path=producer_binding["path"], producer_sha256=producer_binding["sha256"],
    )
    if known_sha(
        receipt.get("invocation_request_sha256"), f"{label}:invocation request SHA256"
    ) != request_hash:
        raise GateError(f"{label}: invocation request binding mismatch")
    recipe_path = _absolute_nonsymlink_path(Path(recipe_binding["path"]), f"{label}:recipe")
    recipe = eval_producer.verify_recipe(recipe_path)
    rendered_environment = {
        key: eval_producer._render(value, {
            "cell_id": str(registration["cell_id"]),
            "arm": arm,
            "seed": str(seed),
            "invocation_nonce": nonce,
            "manifest": str(manifest_path),
            "checkpoint": str(checkpoint),
            "eval_model": str(eval_model),
            "output": "",
            "output_dir": "",
        }, f"{label}:environment.{key}")
        for key, value in recipe["environment"].items()
    }
    environment_hash = hashlib.sha256(
        json.dumps(
            rendered_environment, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if known_sha(receipt.get("environment_sha256"), f"{label}:environment hash") != environment_hash:
        raise GateError(f"{label}: rendered environment hash mismatch")

    steps = receipt.get("steps")
    expected_step_names = [
        name for name in eval_producer.STEP_NAMES if arm in eval_producer.STEP_ARMS[name]
    ]
    if not isinstance(steps, list) or [
        item.get("name") for item in steps if isinstance(item, dict)
    ] != expected_step_names:
        raise GateError(f"{label}: executed step order mismatch")
    output_hashes = {
        "paired_eval": known_sha(run.get("eval_sha256"), f"{label}:paired output hash"),
    }
    if arm == "progress":
        output_hashes.update({
            "pixmo_benchmark": known_sha(
                run.get("pixmo_eval_sha256"), f"{label}:pixmo output hash",
            ),
            "stepcount_benchmark": known_sha(
                run.get("stepcount_eval_sha256"), f"{label}:stepcount output hash",
            ),
        })
    for item in steps:
        item = _exact_object(
            item,
            {
                "name", "program_path", "program_sha256", "argv", "argv_sha256",
                "output_path", "output_dir", "output_name", "output_sha256",
            },
            f"{label}:step",
        )
        name = item["name"]
        recipe_step = recipe["steps"][name]
        if (
            item.get("program_path") != recipe_step["program"]["path"]
            or item.get("program_sha256") != recipe_step["program"]["sha256"]
            or item.get("output_name") != recipe_step["output"]
            or item.get("output_sha256") != output_hashes[name]
        ):
            raise GateError(f"{label}: step {name} provenance mismatch")
        step_output = Path(str(item.get("output_path", "")))
        step_output_dir = Path(str(item.get("output_dir", "")))
        if (
            not step_output.is_absolute()
            or not step_output_dir.is_absolute()
            or step_output.parent != step_output_dir
            or step_output.name != recipe_step["output"]
            or step_output_dir.parent.parent != manifest_path.parent
            or not step_output_dir.parent.name.startswith(".v37_postprocess.")
            or not step_output_dir.name.startswith(f".{name}.")
        ):
            raise GateError(f"{label}: step {name} output staging path is invalid")
        bindings = {
            "cell_id": str(registration["cell_id"]), "arm": arm, "seed": str(seed),
            "manifest": str(manifest_path), "checkpoint": str(checkpoint),
            "eval_model": str(eval_model), "output": str(step_output),
            "output_dir": str(step_output_dir), "invocation_nonce": nonce,
        }
        expected_argv = [recipe_step["program"]["path"]] + [
            eval_producer._render(argument, bindings, f"{label}:{name}:argv")
            for argument in recipe_step["argv"]
        ]
        argv = item.get("argv")
        if not isinstance(argv, list) or argv != expected_argv:
            raise GateError(f"{label}: step {name} rendered argv differs from recipe")
        argv_hash = hashlib.sha256(eval_producer._canonical_bytes(argv)).hexdigest()
        if known_sha(item.get("argv_sha256"), f"{label}:{name}:argv hash") != argv_hash:
            raise GateError(f"{label}: step {name} argv hash mismatch")
    return nonce


def _benchmark_pass(
    manifest: Mapping[str, Any], evidence: Mapping[str, Any], suite: str, seed: Any,
) -> str:
    final_path = str(Path(manifest["final_checkpoint_path"]).resolve())
    final_hash = manifest["final_checkpoint_sha256"]
    expected_total, expected_correct = (
        (PIXMO_TOTAL, PIXMO_CORRECT_MIN) if suite == "pixmo" else (STEPCOUNT_TOTAL, STEPCOUNT_CORRECT_MIN)
    )
    expected_suite = "pixmo-test" if suite == "pixmo" else "stepcount-500"
    try:
        report = benchmark_evidence.verify_descriptor(evidence)
    except benchmark_evidence.EvidenceError as exc:
        raise GateError(f"progress/{seed}: invalid {suite} recomputable benchmark evidence: {exc}") from exc
    if report.get("suite") != expected_suite:
        raise GateError(f"progress/{seed}: benchmark suite mismatch for {suite}")
    artifacts = _exact_object(
        evidence.get("artifacts"), set(benchmark_evidence.ARTIFACT_KINDS),
        f"progress/{seed}:{suite}:artifacts",
    )
    dataset = artifacts.get("dataset")
    if not isinstance(dataset, dict) or known_sha(
        dataset.get("sha256"), f"progress/{seed}:{suite}:dataset.sha256",
    ) != FORMAL_BENCHMARK_DATASET_SHA256[suite]:
        raise GateError(f"progress/{seed}: {suite} is not the frozen canonical dataset")
    sample_cache = evidence.get("samples")
    if not isinstance(sample_cache, list) or any(
        not isinstance(sample, dict) or not isinstance(sample.get("id"), str)
        for sample in sample_cache
    ):
        raise GateError(f"progress/{seed}: {suite} ordered sample identities are missing")
    ordered_ids_sha256 = hashlib.sha256(canonical(
        [sample["id"] for sample in sample_cache]
    ).encode("utf-8")).hexdigest()
    if ordered_ids_sha256 != FORMAL_BENCHMARK_ORDERED_IDS_SHA256[suite]:
        raise GateError(f"progress/{seed}: {suite} ordered sample identities are not canonical")
    checkpoint = artifacts.get("checkpoint_tree")
    if not isinstance(checkpoint, dict):
        raise GateError(f"progress/{seed}: {suite} checkpoint tree binding missing")
    checkpoint_path = checkpoint.get("path")
    checkpoint_hash = known_sha(
        checkpoint.get("sha256"), f"progress/{seed}:{suite}:checkpoint_tree.sha256",
    )
    if not isinstance(checkpoint_path, str) or str(Path(checkpoint_path).resolve()) != final_path:
        raise GateError(f"progress/{seed}: {suite} benchmark checkpoint path mismatch")
    if checkpoint_hash != final_hash:
        raise GateError(f"progress/{seed}: {suite} benchmark checkpoint hash mismatch")
    eval_model = artifacts.get("eval_model")
    expected_eval_model = Path(final_path) / "actor" / "huggingface"
    if (
        not isinstance(eval_model, dict)
        or not isinstance(eval_model.get("path"), str)
        or Path(eval_model["path"]).resolve() != expected_eval_model
    ):
        raise GateError(
            f"progress/{seed}: {suite} eval model must be the checkpoint actor/huggingface tree"
        )
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise GateError(f"progress/{seed}: {suite} recomputed summary missing")
    total = integer(summary.get("total"), f"progress/{seed}:{suite}:total", minimum=1)
    correct = integer(summary.get("correct"), f"progress/{seed}:{suite}:correct")
    if total != expected_total:
        raise GateError(f"progress/{seed}: {suite} total must be exactly {expected_total}")
    if correct > total or correct < expected_correct:
        raise GateError(f"progress/{seed}: {suite} requires at least {expected_correct}/{expected_total}")
    # Checkpoint/model/results differ by seed.  Dataset, images, protocol and
    # evaluator implementations must remain identical across the two seeds.
    return canonical({
        "suite": evidence.get("suite"),
        "dataset": artifacts.get("dataset"),
        "external_image_manifest": artifacts.get("external_image_manifest"),
        "implementations": evidence.get("implementations"),
        "protocol": evidence.get("protocol"),
    })


def evaluate(spec: Any) -> dict[str, Any]:
    errors: list[str] = []
    computed: dict[str, Any] = {}
    try:
        canonical(spec)
        if not isinstance(spec, dict):
            raise GateError("gate input must be a JSON object")
        allowed_spec_keys = {"schema_version", "stage", "run_class", "preregistered", "runs", "ack"}
        if not set(spec) <= allowed_spec_keys:
            raise GateError(f"gate input has unknown fields: {sorted(set(spec) - allowed_spec_keys)}")
        if spec.get("schema_version") != 2:
            raise GateError("schema_version must be 2")
        stage = spec.get("stage")
        run_class = spec.get("run_class")
        if stage not in {"canary", "final"}:
            raise GateError("stage must be explicit: canary or final")
        if run_class not in {"debug", "canary", "formal"}:
            raise GateError("run_class must be explicit: debug, canary, or formal")
        if stage == "canary" and run_class != "canary":
            raise GateError("canary gate requires run_class=canary; debug is never promotable")
        if stage == "final" and run_class != "formal":
            raise GateError("final promotion requires run_class=formal")
        if run_class == "formal" and stage != "final":
            raise GateError("formal run_class requires stage=final")
        preregistered = spec.get("preregistered")
        preregistered_keys = {
            "seeds", "optimizer_steps", "confidence_level", "ci_method",
            "frontier_mixed_min", "outcome_allwrong_max", "paired_metrics",
        }
        if not isinstance(preregistered, dict) or set(preregistered) != preregistered_keys:
            raise GateError("missing preregistered object")
        seeds = preregistered.get("seeds")
        if (
            not isinstance(seeds, list) or len(seeds) != 2
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != 2
        ):
            raise GateError("exactly two distinct nonnegative integer preregistered seeds required")
        optimizer_steps = integer(preregistered.get("optimizer_steps"), "optimizer_steps", minimum=1)
        if run_class == "canary" and not 2 <= optimizer_steps <= 4:
            raise GateError("canary optimizer_steps must be in the frozen 2-4 update range")
        if finite(preregistered.get("confidence_level"), "confidence_level") != 0.95:
            raise GateError("confidence_level must be preregistered as 0.95")
        if preregistered.get("ci_method") != "paired_normal_95":
            raise GateError("ci_method must be preregistered as paired_normal_95")
        mixed_min = finite(preregistered.get("frontier_mixed_min"), "frontier_mixed_min")
        allwrong_max = finite(preregistered.get("outcome_allwrong_max"), "outcome_allwrong_max")
        if not 0 <= mixed_min <= 1 or not 0 <= allwrong_max <= 1:
            raise GateError("frontier ratio limits must be in [0,1]")
        if run_class == "formal":
            if optimizer_steps != FORMAL_OPTIMIZER_STEPS:
                raise GateError("formal optimizer_steps is hard-locked to 12")
            if mixed_min != FORMAL_FRONTIER_MIXED_MIN:
                raise GateError("formal frontier_mixed_min is hard-locked to 0.70")
            if allwrong_max != FORMAL_OUTCOME_ALLWRONG_MAX:
                raise GateError("formal outcome_allwrong_max is hard-locked to 0.25")
            metric_rules = _formal_metric_rules(preregistered)
        else:
            metric_rules = _validate_metric_rules(preregistered)
        runs = spec.get("runs")
        if not isinstance(runs, list):
            raise GateError("runs must be a list")
        require_paths = run_class == "formal"
        loaded: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
        evidence_paths: set[str] = set()
        evidence_hashes: set[str] = set()
        evidence_run_ids: set[str] = set()
        benchmark_evidence: dict[tuple[str, str], dict[str, Any]] = {}
        eval_receipts: dict[tuple[str, str], dict[str, Any]] = {}
        eval_invocation_nonces: set[str] = set()
        formal_ab_contracts: set[tuple[str, str, str]] = set()
        benchmark_contracts: dict[str, str] = {}
        effective_environments: dict[tuple[str, str], Mapping[str, str]] = {}
        formal_sample_universe: dict[tuple[str, str], dict[str, Any]] | None = None
        formal_sample_universe_sha256: str | None = None
        formal_sample_universe_data_sha256: str | None = None
        seed_keys = {canonical(seed) for seed in seeds}
        for run in runs:
            if not isinstance(run, dict):
                raise GateError("run must be an object")
            arm = run.get("arm")
            seed = run.get("seed")
            expected_run_keys = {"arm", "seed", "manifest", "log", "eval"}
            if require_paths:
                expected_run_keys |= {
                    "manifest_sha256", "log_sha256", "eval_sha256",
                    "eval_receipt", "eval_receipt_sha256",
                }
                if arm == "progress":
                    expected_run_keys |= {
                        "pixmo_eval", "pixmo_eval_sha256", "stepcount_eval", "stepcount_eval_sha256",
                    }
            if set(run) != expected_run_keys:
                raise GateError(
                    f"run artifact schema mismatch; missing={sorted(expected_run_keys - set(run))}, "
                    f"extra={sorted(set(run) - expected_run_keys)}"
                )
            if type(seed) is not int or seed < 0:
                raise GateError("run seed must be a nonnegative integer")
            key = (str(arm), canonical(seed))
            if arm not in ARMS or canonical(seed) not in seed_keys or key in loaded:
                raise GateError("unexpected or duplicate arm/seed")
            manifest = artifact(run, "manifest", require_path=require_paths, seen_paths=evidence_paths if require_paths else None, seen_hashes=evidence_hashes if require_paths else None)
            log = artifact(run, "log", require_path=require_paths, seen_paths=evidence_paths if require_paths else None, seen_hashes=evidence_hashes if require_paths else None)
            evaluation = artifact(run, "eval", require_path=require_paths, seen_paths=evidence_paths if require_paths else None, seen_hashes=evidence_hashes if require_paths else None)
            if require_paths:
                eval_receipts[key] = artifact(
                    run, "eval_receipt", require_path=True,
                    seen_paths=evidence_paths, seen_hashes=evidence_hashes,
                )
            if require_paths and arm == "progress":
                for suite in ("pixmo", "stepcount"):
                    benchmark_evidence[(seed_key := canonical(seed), suite)] = artifact(
                        run, f"{suite}_eval", require_path=True,
                        seen_paths=evidence_paths, seen_hashes=evidence_hashes,
                    )
            if manifest.get("arm") != arm or type(manifest.get("seed")) is not int or manifest.get("seed") != seed:
                raise GateError("manifest arm/seed mismatch")
            if manifest.get("run_class") != run_class:
                raise GateError(f"{arm}/{seed}: manifest run_class mismatch")
            config = manifest.get("config")
            if not isinstance(config, dict):
                raise GateError(f"{arm}/{seed}: manifest config must be an object")
            label = f"{arm}/{seed}"
            if integer(
                config.get("pilot_steps"), f"{label}:config.pilot_steps", minimum=1,
            ) != optimizer_steps:
                raise GateError(f"{label}: config.pilot_steps disagrees with preregistration")
            audited_environment = manifest.get("audited_environment")
            if not isinstance(audited_environment, dict):
                raise GateError(f"{label}: audited_environment must be an object")
            expected_steps_text = str(optimizer_steps)
            for step_key in ("V37_PILOT_STEPS", "V36_MAX_STEPS", "BOK_TOTAL_STEPS"):
                if audited_environment.get(step_key) != expected_steps_text:
                    raise GateError(
                        f"{label}: audited_environment.{step_key} disagrees with preregistration"
                    )
            if require_paths:
                _validate_training_evidence_binding(
                    manifest, log, run["log"], run["log_sha256"], label,
                )
            _validate_config_seal(manifest, label, seed)
            _validate_arm_seed_bindings(manifest, str(arm), seed, label)
            actual_manifest_path = (
                _absolute_nonsymlink_path(Path(run["manifest"]), f"{label}:manifest artifact")
                if require_paths else None
            )
            evidence_run_id = _validate_evidence_identity(
                manifest, log, evaluation, label, actual_manifest_path=actual_manifest_path,
            )
            if evidence_run_id in evidence_run_ids:
                raise GateError(f"{label}: evidence_run_id is reused across runs")
            evidence_run_ids.add(evidence_run_id)
            _validate_mechanism_config(manifest, str(arm), str(run_class), seed)
            _validate_mining_contract(manifest, f"{arm}/{seed}")
            input_hashes = manifest.get("input_hashes")
            if not isinstance(input_hashes, dict) or not input_hashes:
                raise GateError(f"{arm}/{seed}: manifest input_hashes missing")
            for name, value in input_hashes.items():
                known_sha(value, f"{arm}/{seed}:input_hashes.{name}")
            for field in ("dataset_sha256", "mask_tree_sha256", "metadata_sha256"):
                known_sha(manifest.get(field), f"{arm}/{seed}:{field}")
            if manifest.get("mask_tree_path") is not None:
                mask_path = Path(manifest["mask_tree_path"])
                if tree_sha256(mask_path) != manifest["mask_tree_sha256"]:
                    raise GateError("mask tree changed after manifest creation")
            if require_paths:
                _validate_formal_input_snapshots(manifest, label)
                _validate_preflight_binding(manifest, label)
                _validate_implementation_snapshot(manifest, label)
                _validate_runtime_environment(manifest, label)
                _validate_execution_environment(manifest, label)
                assert actual_manifest_path is not None
                effective_environments[key] = _validate_effective_environment(
                    manifest, label, actual_manifest_path,
                )
                formal_ab_contracts.add(
                    _validate_ab_preregistration(manifest, label, seeds, actual_manifest_path)
                )
                registration = manifest["ab_preregistration"]
                plan = validate_ab_plan(
                    registration["plan_path"], registration["plan_sha256"],
                    registration["ab_run_id"], expected_seeds=seeds,
                    require_target_dirs=True,
                )
                universe, universe_hash, universe_data_hash = _load_paired_sample_universe(
                    plan["preregistered"]["eval_contract"]["paired_sample_universe"],
                    manifest["input_snapshots"]["validation_data"],
                    label,
                    verify_source_rows=True,
                )
                if formal_sample_universe is None:
                    formal_sample_universe = universe
                    formal_sample_universe_sha256 = universe_hash
                    formal_sample_universe_data_sha256 = universe_data_hash
                elif (
                    canonical([
                        formal_sample_universe[key] for key in sorted(formal_sample_universe)
                    ]) != canonical([universe[key] for key in sorted(universe)])
                    or formal_sample_universe_sha256 != universe_hash
                    or formal_sample_universe_data_sha256 != universe_data_hash
                ):
                    raise GateError("formal runs do not share one preregistered sample universe")
                _paired_eval_data_sha256(manifest, label)
                invocation_nonce = _validate_eval_receipt(
                    run=run,
                    manifest=manifest,
                    evaluation=evaluation,
                    receipt=eval_receipts[key],
                    arm=str(arm),
                    seed=seed,
                )
                if invocation_nonce in eval_invocation_nonces:
                    raise GateError("formal eval invocation nonce is reused across cells")
                eval_invocation_nonces.add(invocation_nonce)
            _validate_log(log, str(arm), seed, preregistered)
            loaded[key] = (manifest, log, evaluation)
        expected = {(arm, canonical(seed)) for arm in ARMS for seed in seeds}
        if set(loaded) != expected:
            raise GateError("requires baseline/progress for both preregistered seeds")
        if require_paths and len(formal_ab_contracts) != 1:
            raise GateError("all four formal runs must share one A/B preregistration plan")

        reference_eval_contract: tuple[str, str, str, tuple[tuple[str, str], ...]] | None = None
        reference_seed_config: str | None = None
        reference_seed_environment: str | None = None
        reference_effective_environment: str | None = None
        reference_input_contract: str | None = None
        reference_mining_contract: str | None = None
        checkpoint_paths: set[str] = set()
        checkpoint_ids: set[str] = set()
        checkpoint_hashes: set[str] = set()
        for seed in seeds:
            seed_key = canonical(seed)
            baseline = loaded[("baseline", seed_key)]
            progress = loaded[("progress", seed_key)]
            if canonical(config_without_arm(baseline[0]["config"])) != canonical(config_without_arm(progress[0]["config"])):
                raise GateError(f"non-allowlisted A/B config difference for seed {seed}")
            baseline_environment = baseline[0]["audited_environment"]
            progress_environment = progress[0]["audited_environment"]
            ignored_ab_environment = ARM_ENV_ALLOWLIST | RUN_ENV_ALLOWLIST
            if canonical(_normalized_environment(baseline_environment, ignored_ab_environment)) != canonical(
                _normalized_environment(progress_environment, ignored_ab_environment)
            ):
                raise GateError(f"non-allowlisted A/B audited_environment difference for seed {seed}")
            baseline_mechanism = config_without_arm(baseline[0]["mechanism_config"])
            progress_mechanism = config_without_arm(progress[0]["mechanism_config"])
            if canonical(baseline_mechanism) != canonical(progress_mechanism):
                raise GateError(f"non-allowlisted A/B mechanism difference for seed {seed}")
            if require_paths:
                baseline_effective = effective_environments[("baseline", seed_key)]
                progress_effective = effective_environments[("progress", seed_key)]
                if canonical(_normalized_environment(baseline_effective, ignored_ab_environment)) != canonical(
                    _normalized_environment(progress_effective, ignored_ab_environment)
                ):
                    raise GateError(f"non-allowlisted A/B effective_environment difference for seed {seed}")
            for field in ("input_hashes", "dataset_sha256", "mask_tree_sha256", "metadata_sha256"):
                if canonical(baseline[0][field]) != canonical(progress[0][field]):
                    raise GateError(f"manifest {field} mismatch for seed {seed}")
            if require_paths:
                for field in (
                    "input_snapshots", "implementation_sha256", "config_file_sha256",
                    "source_config_sha256", "runtime_environment", "preflight_summary",
                    "git_commit", "git_status_sha256", "git_diff_sha256",
                ):
                    if canonical(baseline[0][field]) != canonical(progress[0][field]):
                        raise GateError(f"formal manifest {field} mismatch for seed {seed}")
            seed_config = canonical(config_without_seed_or_arm(baseline[0]["config"]))
            if reference_seed_config is None:
                reference_seed_config = seed_config
            elif seed_config != reference_seed_config:
                raise GateError("non-seed training config differs across preregistered seeds")
            seed_environment = canonical(_normalized_environment(
                baseline_environment, SEED_ENV_ALLOWLIST | RUN_ENV_ALLOWLIST,
            ))
            if reference_seed_environment is None:
                reference_seed_environment = seed_environment
            elif seed_environment != reference_seed_environment:
                raise GateError("non-seed audited_environment differs across preregistered seeds")
            if require_paths:
                effective_seed_environment = canonical(_normalized_environment(
                    effective_environments[("baseline", seed_key)],
                    SEED_ENV_ALLOWLIST | RUN_ENV_ALLOWLIST,
                ))
                if reference_effective_environment is None:
                    reference_effective_environment = effective_seed_environment
                elif effective_seed_environment != reference_effective_environment:
                    raise GateError("non-seed effective_environment differs across preregistered seeds")
            input_contract = canonical({
                field: baseline[0][field]
                for field in ("input_hashes", "dataset_sha256", "mask_tree_sha256", "metadata_sha256")
            })
            if require_paths:
                input_contract = canonical({
                    "hash_contract": json.loads(input_contract),
                    "initial_model": baseline[0]["input_snapshots"]["model"],
                    "training_data": baseline[0]["input_snapshots"]["training_data"],
                    "validation_data": baseline[0]["input_snapshots"]["validation_data"],
                    "mask_metadata": baseline[0]["input_snapshots"]["mask_metadata"],
                    "implementation_sha256": baseline[0]["implementation_sha256"],
                    "config_file_sha256": baseline[0]["config_file_sha256"],
                    "source_config_sha256": baseline[0]["source_config_sha256"],
                    "runtime_environment": baseline[0]["runtime_environment"],
                    "preflight_summary": baseline[0]["preflight_summary"],
                    "git_commit": baseline[0]["git_commit"],
                    "git_status_sha256": baseline[0]["git_status_sha256"],
                    "git_diff_sha256": baseline[0]["git_diff_sha256"],
                })
            if reference_input_contract is None:
                reference_input_contract = input_contract
            elif input_contract != reference_input_contract:
                raise GateError("training data/mask/input hashes differ across preregistered seeds")
            mining_contract = canonical(baseline[0]["mining_contract"])
            if canonical(progress[0]["mining_contract"]) != mining_contract:
                raise GateError(f"mining contract mismatch for seed {seed}")
            if reference_mining_contract is None:
                reference_mining_contract = mining_contract
            elif reference_mining_contract != mining_contract:
                raise GateError("mining contract differs across run seeds")

            computed[str(seed)] = _verify_eval_pair(
                seed, baseline[2], progress[2], metric_rules,
                require_recomputable=require_paths,
                sample_universe=formal_sample_universe,
                sample_universe_sha256=formal_sample_universe_sha256,
            )
            for arm_name, run_data in (("baseline", baseline), ("progress", progress)):
                expected_eval_data = (
                    _paired_eval_data_sha256(run_data[0], f"{arm_name}/{seed}")
                    if require_paths else known_sha(
                        run_data[0].get("paired_eval_data_sha256"),
                        f"{arm_name}/{seed}:paired_eval_data_sha256",
                    )
                )
                if known_sha(run_data[2].get("data_sha256"), f"{arm_name}/{seed}:eval data_sha256") != expected_eval_data:
                    raise GateError(f"{arm_name}/{seed}: eval data is not bound to the manifest validation contract")
            baseline_rows = _sample_rows(
                baseline[2], f"baseline/{seed}", require_recomputable=require_paths,
                sample_universe=formal_sample_universe,
                sample_universe_sha256=formal_sample_universe_sha256,
            )
            contract = (
                known_sha(baseline[2].get("data_sha256"), "data_sha256"),
                known_sha(baseline[2].get("sample_set_sha256"), "sample_set_sha256"),
                formal_sample_universe_sha256,
            ) if require_paths else (
                known_sha(baseline[2].get("protocol_sha256"), "protocol_sha256"),
                known_sha(baseline[2].get("data_sha256"), "data_sha256"),
                known_sha(baseline[2].get("eval_config_sha256"), "eval_config_sha256"),
                known_sha(baseline[2].get("sample_set_sha256"), "sample_set_sha256"),
            )
            if reference_eval_contract is None:
                reference_eval_contract = contract
            elif contract != reference_eval_contract:
                raise GateError("sample identity/protocol/data/eval config differs across seeds")
            if stage == "final":
                _checkpoint_pass(baseline[0], baseline[2], "baseline", seed, checkpoint_paths, checkpoint_ids, checkpoint_hashes)
                _checkpoint_pass(progress[0], progress[2], "progress", seed, checkpoint_paths, checkpoint_ids, checkpoint_hashes)
                for suite in ("pixmo", "stepcount"):
                    contract = _benchmark_pass(
                        progress[0], benchmark_evidence[(seed_key, suite)], suite, seed,
                    )
                    if suite in benchmark_contracts and benchmark_contracts[suite] != contract:
                        raise GateError(
                            f"{suite} dataset/images/protocol/implementation differ across seeds"
                        )
                    benchmark_contracts[suite] = contract
    except Exception as exc:  # Structured NO-GO is required even for malformed nested schemas.
        errors.append(str(exc) or exc.__class__.__name__)
    ack = bool(spec.get("ack")) if isinstance(spec, dict) else False
    return {
        "schema_version": 2,
        "decision": "GO" if not errors else "NO-GO",
        "hard_failure": bool(errors),
        "ack_requested": ack,
        "ack_overrode_failure": False,
        "computed_paired_metrics": computed,
        "errors": errors,
    }


def atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--ack", action="store_true", help="record ACK only; never overrides hard failures")
    args = parser.parse_args(argv)
    try:
        decoded, is_jsonl = _load_rows(args.input)
        if is_jsonl:
            raise GateError("gate input must be one JSON object, not JSONL")
        if isinstance(decoded, dict):
            decoded["ack"] = bool(args.ack or decoded.get("ack"))
        result = evaluate(decoded)
    except Exception as exc:
        result = {
            "schema_version": 2, "decision": "NO-GO", "hard_failure": True,
            "ack_requested": args.ack, "ack_overrode_failure": False,
            "computed_paired_metrics": {}, "errors": [str(exc) or exc.__class__.__name__],
        }
    if args.json_out:
        atomic(args.json_out, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
