#!/usr/bin/env python3
"""Validate a non-promotable V37 controlled-continuation provenance chain."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import secrets
import stat
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


SHA256_RE = re.compile(r"[0-9a-f]{64}")
SEAL_KEYS = {
    "schema_version",
    "purpose",
    "checkpoint_path",
    "checkpoint_tree_sha256",
    "world_size",
    "source_run_manifest_path",
    "source_run_manifest_sha256",
}
EXACT_BINDING_FIELDS = (
    "run_class",
    "data_mode",
    "arm",
    "seed",
    "git_commit",
    "git_status_sha256",
    "git_diff_sha256",
    "model_path",
    "train_data",
    "validation_data",
    "mask_metadata",
    "masks_dir",
    "dataset_manifest",
    "dataset_manifest_sha256",
    "miner_manifest_hashes",
    "source_dataset_sha256",
    "dataset_sha256",
    "metadata_sha256",
    "mask_tree_path",
    "mask_tree_sha256",
    "paired_eval_data_sha256",
    "metadata_coverage_report_sha256",
    "metadata_coverage_report",
    "filtered_manifest_sha256",
    "filtered_manifest",
    "input_hashes",
    "input_snapshots",
    "runtime_environment",
    "implementation_sha256",
    "config_file_sha256",
    "source_config_sha256",
    "mechanisms",
    "mechanism_config",
    "mining_contract",
    "preflight_summary",
)
CONTINUATION_ENV_ALLOWLIST = {
    "BOK_TOTAL_STEPS",
    "CONFIG_PATH",
    "MAX_STEPS",
    "V31_EXPERIMENT_NAME",
    "V31_SAVE_CHECKPOINT_PATH",
    "V32_LOAD_CHECKPOINT_PATH",
    "V36_LOAD_CHECKPOINT_PATH",
    "V36_MAX_STEPS",
    "V37_CONTINUATION_MODE",
    "V37_EFFECTIVE_ENVIRONMENT_PATH",
    "V37_EVIDENCE_START_STEP",
    "V37_EXPECTED_RESUME_CHECKPOINT_PATH",
    "V37_EXPECTED_RESUME_CHECKPOINT_SHA256",
    "V37_EXECUTION_ENVIRONMENT_PATH",
    "V37_EXECUTION_ENVIRONMENT_SHA256",
    "V37_PILOT_STEPS",
    "V37_RESUME_CHECKPOINT",
    "V37_RESUME_MODE",
    "V37_RESUME_SEAL",
    "V37_RUN_MANIFEST",
    "V37_RUN_PURPOSE",
    "V37_RUN_TIMESTAMP",
    "V37_TRAINING_EVIDENCE_PATH",
}

_TRAINING_EVIDENCE_PATH = (
    Path(__file__).resolve().parents[1] / "verl/utils/v37_training_evidence.py"
)
_TRAINING_SPEC = importlib.util.spec_from_file_location(
    "v37_continuation_training_evidence", _TRAINING_EVIDENCE_PATH
)
if _TRAINING_SPEC is None or _TRAINING_SPEC.loader is None:
    raise ImportError("cannot load V37 training evidence verifier")
training_evidence = importlib.util.module_from_spec(_TRAINING_SPEC)
_TRAINING_SPEC.loader.exec_module(training_evidence)


class ContinuationError(ValueError):
    """The continuation source does not match the requested run."""


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContinuationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContinuationError("continuation evidence is not canonical finite JSON") from exc


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContinuationError(f"{label} must be a lowercase full SHA256")
    return value


def _absolute_nonsymlink(path: os.PathLike[str] | str, label: str) -> Path:
    raw = Path(os.path.expanduser(os.fspath(path)))
    if not raw.is_absolute():
        raise ContinuationError(f"{label} must be absolute")
    absolute = Path(os.path.abspath(raw))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ContinuationError(f"{label} does not exist: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ContinuationError(f"{label} contains a symlink component: {current}")
    return absolute


def strict_json_file_with_hash(
    path: os.PathLike[str] | str, label: str,
) -> tuple[Path, dict[str, Any], str]:
    resolved = _absolute_nonsymlink(path, label)
    if not resolved.is_file():
        raise ContinuationError(f"{label} must be a regular file")
    try:
        raw = resolved.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ContinuationError(f"{label} contains non-finite constant {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"invalid {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContinuationError(f"{label} must be a JSON object")
    _canonical(payload)
    return resolved, payload, hashlib.sha256(raw).hexdigest()


def strict_json_file(path: os.PathLike[str] | str, label: str) -> tuple[Path, dict[str, Any]]:
    resolved, payload, _ = strict_json_file_with_hash(path, label)
    return resolved, payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def live_git_identity() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[1]
    git_environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    git_environment["LC_ALL"] = "C"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=git_environment,
        ).stdout.decode("ascii").strip()
        status_bytes = subprocess.run(
            ["git", "status", "--porcelain=v1", "-uall"], cwd=repo, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=git_environment,
        ).stdout
        diff_bytes = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD"],
            cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=git_environment,
        ).stdout
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise ContinuationError("cannot recompute live Git identity") from exc
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
        raise ContinuationError("live Git commit is not a full object ID")
    return {
        "git_commit": commit,
        "git_status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "git_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
    }


def tree_sha256(path: os.PathLike[str] | str) -> str:
    root = _absolute_nonsymlink(path, "continuation checkpoint")
    if not root.is_dir():
        raise ContinuationError("continuation checkpoint must be a directory")
    for walk_root, directories, filenames in os.walk(root, followlinks=False):
        for name in directories + filenames:
            if (Path(walk_root) / name).is_symlink():
                raise ContinuationError("continuation checkpoint contains a symlink")
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ContinuationError("continuation checkpoint is empty")
    entries = [{
        "path": item.relative_to(root).as_posix(),
        "size": item.stat().st_size,
        "sha256": file_sha256(item),
    } for item in files]
    return hashlib.sha256(_canonical(entries)).hexdigest()


def _atomic_publish_json(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> tuple[Path, str]:
    raw = Path(os.path.expanduser(os.fspath(path)))
    if not raw.is_absolute():
        raise ContinuationError("continuation seal output must be absolute")
    output = Path(os.path.abspath(raw))
    if output != raw:
        raise ContinuationError("continuation seal output must be canonical")
    parent = _absolute_nonsymlink(output.parent, "continuation seal output parent")
    if not parent.is_dir():
        raise ContinuationError("continuation seal output parent must be a directory")
    if os.path.lexists(output):
        raise ContinuationError("continuation seal output already exists")

    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary = parent / f".{output.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    try:
        fd = os.open(temporary, flags, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as exc:
            raise ContinuationError("continuation seal output was claimed concurrently") from exc
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        temporary.unlink(missing_ok=True)
    return output, hashlib.sha256(encoded).hexdigest()


def continuation_config_sha256(config: Mapping[str, Any]) -> str:
    if not isinstance(config, Mapping):
        raise ContinuationError("manifest config must be an object")
    selected = dict(config)
    pilot_steps = selected.pop("pilot_steps", None)
    if isinstance(pilot_steps, bool) or not isinstance(pilot_steps, int) or pilot_steps < 1:
        raise ContinuationError("manifest config.pilot_steps must be a positive integer")
    return hashlib.sha256(_canonical(selected)).hexdigest()


def verify_seal(
    seal_path: os.PathLike[str] | str,
    checkpoint_path: os.PathLike[str] | str,
    checkpoint_sha256: str,
    *,
    world_size: int,
    run_class: str,
    arm: str,
    seed: int,
    frontier_manifest_sha256: str,
) -> Path:
    _, seal, _ = strict_json_file_with_hash(seal_path, "continuation seal")
    if set(seal) != SEAL_KEYS:
        raise ContinuationError("continuation seal exact schema mismatch")
    if seal["schema_version"] != 1 or seal["purpose"] != "controlled_continuation":
        raise ContinuationError("continuation seal version/purpose mismatch")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ContinuationError("world_size must be a positive integer")
    if seal["world_size"] != world_size:
        raise ContinuationError("continuation seal world_size mismatch")
    checkpoint = _absolute_nonsymlink(checkpoint_path, "continuation checkpoint")
    checkpoint_hash = _sha256(checkpoint_sha256, "checkpoint SHA256")
    if tree_sha256(checkpoint) != checkpoint_hash:
        raise ContinuationError("continuation checkpoint live tree hash mismatch")
    if seal["checkpoint_path"] != str(checkpoint) or seal["checkpoint_tree_sha256"] != checkpoint_hash:
        raise ContinuationError("continuation checkpoint path/tree hash mismatch")

    source_raw = seal["source_run_manifest_path"]
    if not isinstance(source_raw, str):
        raise ContinuationError("source manifest path must be a string")
    source_path, source, source_hash = strict_json_file_with_hash(
        source_raw, "continuation source manifest"
    )
    if source_hash != _sha256(
        seal["source_run_manifest_sha256"], "source manifest SHA256"
    ):
        raise ContinuationError("continuation source manifest hash mismatch")
    if source.get("manifest_version") != 3 or source.get("data_mode") != "frontier_rl":
        raise ContinuationError("continuation source is not a V37 frontier manifest")
    if source.get("final_checkpoint_path") != str(checkpoint) or source.get(
        "final_checkpoint_sha256"
    ) != checkpoint_hash:
        raise ContinuationError("continuation source manifest does not seal this checkpoint")
    if not isinstance(run_class, str) or source.get("run_class") != run_class:
        raise ContinuationError("continuation source run_class mismatch")
    if type(seed) is not int or type(source.get("seed")) is not int:
        raise ContinuationError("continuation seed must be an exact integer")
    if _canonical(source.get("arm")) != _canonical(arm) or source.get("seed") != seed:
        raise ContinuationError("continuation source arm/seed mismatch")
    if source.get("dataset_manifest_sha256") != _sha256(
        frontier_manifest_sha256, "frontier manifest SHA256"
    ):
        raise ContinuationError("continuation source frontier manifest mismatch")
    config = source.get("config")
    if not isinstance(config, dict) or config.get("world_size") != world_size:
        raise ContinuationError("continuation source world_size mismatch")
    checkpoint_match = re.fullmatch(r"global_step_([1-9][0-9]*)", checkpoint.name)
    if checkpoint_match is None or config.get("pilot_steps") != int(checkpoint_match.group(1)):
        raise ContinuationError("continuation source final step does not match checkpoint")
    computed_config_hash = continuation_config_sha256(config)
    if source.get("continuation_config_sha256") != computed_config_hash:
        raise ContinuationError("continuation source config identity mismatch")
    return source_path


def _normalized_environment(manifest: Mapping[str, Any], label: str) -> dict[str, str]:
    environment = manifest.get("audited_environment")
    if not isinstance(environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ContinuationError(f"continuation {label} audited_environment is invalid")
    return {
        key: value for key, value in environment.items()
        if key not in CONTINUATION_ENV_ALLOWLIST
    }


def _step_environment(manifest: Mapping[str, Any], label: str) -> tuple[int, int]:
    config = manifest.get("config")
    environment = manifest.get("audited_environment")
    if not isinstance(config, dict) or not isinstance(environment, dict):
        raise ContinuationError(f"continuation {label} config/environment is invalid")
    steps = config.get("pilot_steps")
    if type(steps) is not int or steps < 1:
        raise ContinuationError(f"continuation {label} pilot_steps must be a positive integer")

    def integer_environment(name: str) -> int:
        raw = environment.get(name)
        if not isinstance(raw, str) or not re.fullmatch(r"0|[1-9][0-9]*", raw):
            raise ContinuationError(f"continuation {label} {name} must be canonical integer text")
        return int(raw)

    for name in ("V37_PILOT_STEPS", "V36_MAX_STEPS", "BOK_TOTAL_STEPS"):
        if integer_environment(name) != steps:
            raise ContinuationError(f"continuation {label} {name} disagrees with pilot_steps")
    return steps, integer_environment("V37_EVIDENCE_START_STEP")


def continuation_binding_payload(manifest: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ContinuationError(f"continuation {label} manifest must be an object")
    missing = [field for field in EXACT_BINDING_FIELDS if field not in manifest]
    if missing:
        raise ContinuationError(
            f"continuation {label} manifest lacks binding fields: {','.join(missing)}"
        )
    seed = manifest.get("seed")
    if type(seed) is not int or seed < 0:
        raise ContinuationError(f"continuation {label} seed must be an exact nonnegative integer")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ContinuationError(f"continuation {label} config must be an object")
    selected_config = dict(config)
    selected_config.pop("pilot_steps", None)
    return {
        "fields": {field: manifest.get(field) for field in EXACT_BINDING_FIELDS},
        "config_without_pilot_steps": selected_config,
        "audited_environment_without_resume_fields": _normalized_environment(manifest, label),
    }


def verify_current_manifest(source: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    """Compare complete manifests after current inputs/config have been snapshotted."""
    source_config = source.get("config") if isinstance(source, Mapping) else None
    current_config = current.get("config") if isinstance(current, Mapping) else None
    source_hash = continuation_config_sha256(source_config)  # type: ignore[arg-type]
    current_hash = continuation_config_sha256(current_config)  # type: ignore[arg-type]
    if source.get("continuation_config_sha256") != source_hash:
        raise ContinuationError("continuation source config identity is corrupt")
    if current.get("continuation_config_sha256") != current_hash or current_hash != source_hash:
        raise ContinuationError("continuation numerical config changed beyond pilot_steps")
    source_steps, source_start = _step_environment(source, "source")
    current_steps, current_start = _step_environment(current, "current")
    if current_steps <= source_steps:
        raise ContinuationError("continuation pilot_steps must extend the source final step")
    if source_start >= source_steps:
        raise ContinuationError("continuation source evidence start step is invalid")
    if current_start != source_steps:
        raise ContinuationError("continuation current evidence start must equal source final step")

    source_payload = continuation_binding_payload(source, "source")
    current_payload = continuation_binding_payload(current, "current")
    if _canonical(source_payload) != _canonical(current_payload):
        source_fields = source_payload["fields"]
        current_fields = current_payload["fields"]
        changed_fields = [
            field for field in EXACT_BINDING_FIELDS
            if _canonical(source_fields[field]) != _canonical(current_fields[field])
        ]
        if changed_fields:
            raise ContinuationError(
                "continuation source/current binding mismatch: " + ",".join(changed_fields)
            )
        if _canonical(source_payload["config_without_pilot_steps"]) != _canonical(
            current_payload["config_without_pilot_steps"]
        ):
            raise ContinuationError("continuation numerical config changed beyond pilot_steps")
        source_environment = source_payload["audited_environment_without_resume_fields"]
        current_environment = current_payload["audited_environment_without_resume_fields"]
        changed = sorted(set(source_environment) ^ set(current_environment) | {
            key for key in set(source_environment) & set(current_environment)
            if source_environment[key] != current_environment[key]
        })
        raise ContinuationError(
            "continuation audited environment changed outside the resume allowlist: "
            + ",".join(changed)
        )
    return hashlib.sha256(_canonical(source_payload)).hexdigest()


def _manifest_config_sha256(manifest: Mapping[str, Any]) -> str:
    config = manifest.get("config")
    environment = manifest.get("audited_environment")
    if not isinstance(config, dict) or not isinstance(environment, dict):
        raise ContinuationError("source manifest config/audited_environment is missing")
    encoded = json.dumps(
        {"selected": config, "environment": environment},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_evidence_run_id(manifest: Mapping[str, Any]) -> str:
    environment = manifest.get("audited_environment")
    snapshots = manifest.get("input_snapshots")
    if not isinstance(environment, dict) or not isinstance(snapshots, dict):
        raise ContinuationError("source manifest identity inputs are missing")
    model_snapshot = snapshots.get("model")
    if not isinstance(model_snapshot, dict):
        raise ContinuationError("source manifest model snapshot is missing")
    manifest_path = environment.get("V37_RUN_MANIFEST")
    if not isinstance(manifest_path, str) or not Path(manifest_path).is_absolute():
        raise ContinuationError("source manifest identity path is invalid")
    identity = {
        "arm": manifest.get("arm"),
        "seed": manifest.get("seed"),
        "run_directory": str(Path(manifest_path).resolve().parent),
        "manifest_path": str(Path(manifest_path).resolve()),
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
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_source_completion(
    source: Mapping[str, Any],
    checkpoint: Path,
    checkpoint_hash: str,
) -> tuple[Path, str, int]:
    config = source.get("config")
    if not isinstance(config, dict):
        raise ContinuationError("continuation source config is missing")
    source_steps = config.get("pilot_steps")
    if type(source_steps) is not int or source_steps < 1:
        raise ContinuationError("continuation source pilot_steps is invalid")
    if source.get("config_sha256") != _manifest_config_sha256(source):
        raise ContinuationError("continuation source config_sha256 is invalid")
    source_run_id = _sha256(source.get("evidence_run_id"), "source evidence_run_id")
    if source_run_id != _manifest_evidence_run_id(source):
        raise ContinuationError("continuation source evidence_run_id is invalid")

    evidence_path_value = source.get("training_evidence_path")
    if not isinstance(evidence_path_value, str):
        raise ContinuationError("continuation source training evidence path is missing")
    evidence_path, _, evidence_file_hash = strict_json_file_with_hash(
        evidence_path_value, "continuation source training evidence"
    )
    if evidence_file_hash != _sha256(
        source.get("training_evidence_sha256"), "source training evidence SHA256"
    ):
        raise ContinuationError("continuation source training evidence hash mismatch")
    try:
        evidence = training_evidence.verify_file(evidence_path)
    except (training_evidence.TrainingEvidenceError, OSError) as exc:
        raise ContinuationError(f"continuation source training evidence is invalid: {exc}") from exc
    if (
        evidence.get("completed") is not True
        or evidence.get("evidence_run_id") != source_run_id
        or evidence.get("execution_environment_sha256") != source.get("execution_environment_sha256")
        or evidence.get("final_global_step") != source_steps
        or evidence.get("final_checkpoint_path") != str(checkpoint)
        or evidence.get("final_checkpoint_sha256") != checkpoint_hash
    ):
        raise ContinuationError(
            "continuation source training evidence does not seal the final step/checkpoint"
        )
    return evidence_path, evidence_file_hash, source_steps


def create_seal(
    source_manifest_path: os.PathLike[str] | str,
    checkpoint_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
) -> tuple[Path, str]:
    """Atomically publish a seal only for a complete, locally reproducible source run."""
    source_path, source, source_hash = strict_json_file_with_hash(
        source_manifest_path, "continuation source manifest"
    )
    checkpoint = _absolute_nonsymlink(checkpoint_path, "continuation checkpoint")
    checkpoint_hash = tree_sha256(checkpoint)
    if source.get("manifest_version") != 3 or source.get("data_mode") != "frontier_rl":
        raise ContinuationError("continuation source is not a V37 frontier manifest")
    if source.get("run_class") not in {"debug", "canary"}:
        raise ContinuationError("only debug/canary runs may create continuation seals")
    if source.get("promotable_candidate") is not False:
        raise ContinuationError("continuation source must be explicitly non-promotable")
    environment = source.get("audited_environment")
    if not isinstance(environment, dict) or environment.get("V37_RUN_MANIFEST") != str(source_path):
        raise ContinuationError("continuation source manifest path identity mismatch")
    config = source.get("config")
    if not isinstance(config, dict) or type(config.get("world_size")) is not int:
        raise ContinuationError("continuation source world_size is invalid")
    if config["world_size"] < 1:
        raise ContinuationError("continuation source world_size must be positive")
    step_match = re.fullmatch(r"global_step_([1-9][0-9]*)", checkpoint.name)
    if step_match is None or config.get("pilot_steps") != int(step_match.group(1)):
        raise ContinuationError("continuation source final step does not match checkpoint")
    if (
        source.get("final_checkpoint_path") != str(checkpoint)
        or source.get("final_checkpoint_sha256") != checkpoint_hash
    ):
        raise ContinuationError("continuation source does not seal the live checkpoint")
    if any(source.get(field) != value for field, value in live_git_identity().items()):
        raise ContinuationError("continuation source manifest does not match live Git identity")
    frontier_value = source.get("dataset_manifest")
    if not isinstance(frontier_value, str):
        raise ContinuationError("continuation source frontier manifest path is missing")
    frontier = _absolute_nonsymlink(frontier_value, "continuation source frontier manifest")
    if not frontier.is_file() or source.get("dataset_manifest_sha256") != file_sha256(frontier):
        raise ContinuationError("continuation source frontier manifest live hash mismatch")
    evidence_path, _, _ = _verify_source_completion(source, checkpoint, checkpoint_hash)
    if evidence_path.parent != source_path.parent:
        raise ContinuationError("continuation source training evidence is outside the source run")

    seal = {
        "schema_version": 1,
        "purpose": "controlled_continuation",
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256": checkpoint_hash,
        "world_size": config["world_size"],
        "source_run_manifest_path": str(source_path),
        "source_run_manifest_sha256": source_hash,
    }
    if strict_json_file_with_hash(source_path, "continuation source manifest")[2] != source_hash:
        raise ContinuationError("continuation source manifest changed before seal publication")
    if tree_sha256(checkpoint) != checkpoint_hash:
        raise ContinuationError("continuation checkpoint changed before seal publication")
    return _atomic_publish_json(output_path, seal)


def validate_continuation(
    seal_path: os.PathLike[str] | str,
    checkpoint_path: os.PathLike[str] | str,
    current: Mapping[str, Any],
    *,
    expected_seal_sha256: str,
    expected_checkpoint_sha256: str,
    expected_source_manifest_path: os.PathLike[str] | str,
    expected_frontier_manifest_sha256: str,
) -> dict[str, Any]:
    """Re-read all live parents once, then bind them to the current manifest."""
    seal_file, seal, seal_hash = strict_json_file_with_hash(seal_path, "continuation seal")
    if seal_hash != _sha256(expected_seal_sha256, "expected continuation seal SHA256"):
        raise ContinuationError("continuation seal changed after early verification")
    if set(seal) != SEAL_KEYS or seal.get("schema_version") != 1 or seal.get(
        "purpose"
    ) != "controlled_continuation":
        raise ContinuationError("continuation seal exact schema/version mismatch")

    checkpoint = _absolute_nonsymlink(checkpoint_path, "continuation checkpoint")
    checkpoint_hash = tree_sha256(checkpoint)
    expected_checkpoint_hash = _sha256(
        expected_checkpoint_sha256, "expected continuation checkpoint SHA256"
    )
    if checkpoint_hash != expected_checkpoint_hash:
        raise ContinuationError("continuation checkpoint changed after early verification")
    if seal.get("checkpoint_path") != str(checkpoint) or seal.get(
        "checkpoint_tree_sha256"
    ) != checkpoint_hash:
        raise ContinuationError("continuation seal checkpoint binding mismatch")

    source_value = seal.get("source_run_manifest_path")
    if not isinstance(source_value, str):
        raise ContinuationError("continuation seal source path is invalid")
    source_path, source, source_hash = strict_json_file_with_hash(
        source_value, "continuation source manifest"
    )
    expected_source = _absolute_nonsymlink(
        expected_source_manifest_path, "expected continuation source manifest"
    )
    if source_path != expected_source:
        raise ContinuationError("continuation seal source path changed after early verification")
    if source_hash != _sha256(
        seal.get("source_run_manifest_sha256"), "sealed source manifest SHA256"
    ):
        raise ContinuationError("continuation source manifest changed after early verification")
    source_environment = source.get("audited_environment")
    if not isinstance(source_environment, dict) or source_environment.get(
        "V37_RUN_MANIFEST"
    ) != str(source_path):
        raise ContinuationError("continuation source manifest path identity mismatch")

    config = current.get("config")
    if not isinstance(config, dict) or type(config.get("world_size")) is not int:
        raise ContinuationError("continuation current world_size is invalid")
    if seal.get("world_size") != config["world_size"]:
        raise ContinuationError("continuation seal/current world_size mismatch")
    if source.get("manifest_version") != 3 or source.get("data_mode") != "frontier_rl":
        raise ContinuationError("continuation source is not a V37 frontier manifest")
    live_git = live_git_identity()
    if any(current.get(field) != value for field, value in live_git.items()):
        raise ContinuationError("continuation current manifest does not match live Git identity")
    if source.get("final_checkpoint_path") != str(checkpoint) or source.get(
        "final_checkpoint_sha256"
    ) != checkpoint_hash:
        raise ContinuationError("continuation source does not seal the live checkpoint")

    frontier_value = current.get("dataset_manifest")
    if not isinstance(frontier_value, str):
        raise ContinuationError("continuation current frontier manifest path is missing")
    frontier_path = _absolute_nonsymlink(frontier_value, "current frontier manifest")
    frontier_hash = file_sha256(frontier_path)
    expected_frontier_hash = _sha256(
        expected_frontier_manifest_sha256, "expected frontier manifest SHA256"
    )
    if (
        frontier_hash != expected_frontier_hash
        or current.get("dataset_manifest_sha256") != frontier_hash
        or source.get("dataset_manifest_sha256") != frontier_hash
    ):
        raise ContinuationError("continuation frontier manifest live hash mismatch")

    evidence_path, evidence_hash, source_steps = _verify_source_completion(
        source, checkpoint, checkpoint_hash
    )
    if evidence_path.parent != source_path.parent:
        raise ContinuationError("continuation source training evidence is outside the source run")
    binding_hash = verify_current_manifest(source, current)
    target_steps = current["config"]["pilot_steps"]
    return {
        "schema_version": 2,
        "contract": "v37_controlled_continuation_v2",
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256": checkpoint_hash,
        "seal_path": str(seal_file),
        "seal_sha256": seal_hash,
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": source_hash,
        "source_evidence_run_id": source["evidence_run_id"],
        "source_training_evidence_path": str(evidence_path),
        "source_training_evidence_sha256": evidence_hash,
        "source_global_step": source_steps,
        "target_global_step": target_steps,
        "binding_sha256": binding_hash,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-seal")
    verify.add_argument("--seal", type=Path, required=True)
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument("--checkpoint-sha256", required=True)
    verify.add_argument("--world-size", type=int, required=True)
    verify.add_argument("--run-class", choices=("debug", "canary"), required=True)
    verify.add_argument("--arm", required=True)
    verify.add_argument("--seed", type=int, required=True)
    verify.add_argument("--frontier-manifest-sha256", required=True)
    create = subparsers.add_parser("create-seal")
    create.add_argument("--source-manifest", type=Path, required=True)
    create.add_argument("--checkpoint", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "verify-seal":
        print(verify_seal(
            args.seal,
            args.checkpoint,
            args.checkpoint_sha256,
            world_size=args.world_size,
            run_class=args.run_class,
            arm=args.arm,
            seed=args.seed,
            frontier_manifest_sha256=args.frontier_manifest_sha256,
        ))
    elif args.command == "create-seal":
        path, digest = create_seal(args.source_manifest, args.checkpoint, args.output)
        print(json.dumps({"seal_path": str(path), "seal_sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except ContinuationError as exc:
        raise SystemExit(f"[V37-continuation][ERROR] {exc}") from exc
