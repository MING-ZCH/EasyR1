#!/usr/bin/env python3
"""Assemble and execute the V37 four-cell formal evidence gate.

The A/B launcher owns training and publishes ``ab_evidence_index.json``.  This
tool owns the next boundary: it optionally invokes one preregistered producer
per cell, discovers only fixed-name evidence artifacts, constructs the exact
gate input, runs the gate, and atomically publishes a final evidence index.

The producer interface is intentionally argv-only (never ``shell=True``):

    PRODUCER --cell-id ID --arm ARM --seed N --manifest PATH
             --checkpoint PATH --output-dir PATH

It must write ``paired_eval.json`` for every cell and both benchmark descriptor
files for progress cells.  ``v37_training_evidence.json`` is trainer-owned and
must already exist in the cell directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from tools import v37_gate as gate
except ModuleNotFoundError:  # Direct execution from tools/.
    import v37_gate as gate
try:
    from tools import v37_eval_producer as eval_producer
except ModuleNotFoundError:  # Direct execution from tools/.
    import v37_eval_producer as eval_producer


SCHEMA_VERSION = 1
INDEX_TYPE = "v37_formal_postprocess_index"
TRAINING_EVIDENCE_NAME = "v37_training_evidence.json"
PAIRED_EVIDENCE_NAME = "paired_eval.json"
PIXMO_EVIDENCE_NAME = "pixmo_benchmark_evidence.json"
STEPCOUNT_EVIDENCE_NAME = "stepcount_benchmark_evidence.json"
PRODUCER_RECEIPT_NAME = "producer_receipt.json"


class PostprocessError(ValueError):
    """Post-training evidence is missing, mutable, ambiguous, or invalid."""


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PostprocessError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise PostprocessError(f"non-finite JSON constant: {value}")


def _finite_json(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PostprocessError(f"{label} contains a non-finite number")
    if isinstance(value, list):
        for item in value:
            _finite_json(item, label)
    elif isinstance(value, dict):
        for item in value.values():
            _finite_json(item, label)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PostprocessError("value is not finite canonical JSON") from exc


def _absolute(path: os.PathLike[str] | str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _nonsymlink(path: os.PathLike[str] | str, label: str) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise PostprocessError(f"{label} does not exist: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise PostprocessError(f"{label} contains a symlink component: {current}")
    return absolute


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    if not path.is_dir():
        raise PostprocessError(f"checkpoint is not a directory: {path}")
    for root, directories, filenames in os.walk(path, followlinks=False):
        for name in directories + filenames:
            candidate = Path(root) / name
            if candidate.is_symlink():
                raise PostprocessError(f"checkpoint contains a symlink: {candidate}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise PostprocessError(f"checkpoint is empty: {path}")
    entries = []
    for item in files:
        if item.is_symlink():
            raise PostprocessError(f"checkpoint contains a symlink: {item}")
        entries.append({
            "path": item.relative_to(path).as_posix(),
            "size": item.stat().st_size,
            "sha256": _file_sha256(item),
        })
    return hashlib.sha256(_canonical_bytes(entries)).hexdigest()


def _load_json(path: os.PathLike[str] | str, label: str) -> tuple[Path, dict[str, Any], str]:
    absolute = _nonsymlink(path, label)
    metadata_before = os.stat(absolute)
    if not stat.S_ISREG(metadata_before.st_mode):
        raise PostprocessError(f"{label} must be a regular file: {absolute}")
    raw = absolute.read_bytes()
    metadata_after = os.stat(absolute)
    state_before = (
        metadata_before.st_dev, metadata_before.st_ino, metadata_before.st_size,
        metadata_before.st_mtime_ns, metadata_before.st_ctime_ns,
    )
    state_after = (
        metadata_after.st_dev, metadata_after.st_ino, metadata_after.st_size,
        metadata_after.st_mtime_ns, metadata_after.st_ctime_ns,
    )
    if state_before != state_after:
        raise PostprocessError(f"{label} changed while reading: {absolute}")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostprocessError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PostprocessError(f"{label} must be a JSON object")
    _finite_json(value, label)
    return absolute, value, hashlib.sha256(raw).hexdigest()


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise PostprocessError(
            f"{label} schema mismatch; missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
        )
    return value


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _file_sha256(path)}


def _preregistered_pipeline(plan: Mapping[str, Any]) -> tuple[Path, str, Path, str]:
    try:
        pipeline = plan["preregistered"]["eval_contract"]["pipeline"]
        producer_binding = pipeline["producer"]
        recipe_binding = pipeline["recipe"]
    except (KeyError, TypeError) as exc:
        raise PostprocessError("A/B plan lacks a preregistered eval pipeline") from exc
    producer = _nonsymlink(producer_binding.get("path"), "preregistered eval producer")
    recipe = _nonsymlink(recipe_binding.get("path"), "preregistered eval recipe")
    if (
        not producer.is_file()
        or _file_sha256(producer) != producer_binding.get("sha256")
        or not recipe.is_file()
        or _file_sha256(recipe) != recipe_binding.get("sha256")
    ):
        raise PostprocessError("preregistered eval pipeline content binding changed")
    return producer, producer_binding["sha256"], recipe, recipe_binding["sha256"]


def _validate_ab_index(index_path: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    _, index, _ = _load_json(index_path, "A/B evidence index")
    _exact(index, {"plan", "manifests"}, "A/B evidence index")
    plan_binding = _exact(index["plan"], {"path", "sha256"}, "A/B plan binding")
    plan_path, plan, plan_hash = _load_json(plan_binding["path"], "A/B plan")
    if plan_hash != plan_binding["sha256"]:
        raise PostprocessError("A/B plan hash differs from the launcher index")
    _exact(
        plan,
        {"schema_version", "ab_run_id", "plan_path", "seeds", "execution", "preregistered", "cells"},
        "A/B plan",
    )
    if plan.get("schema_version") != 2 or plan.get("plan_path") != str(plan_path):
        raise PostprocessError("A/B plan identity/schema mismatch")
    try:
        plan = gate.validate_ab_plan(
            plan_path, plan_hash, plan.get("ab_run_id"), require_target_dirs=False,
        )
    except gate.GateError as exc:
        raise PostprocessError(f"invalid shared A/B plan contract: {exc}") from exc
    preregistered = plan.get("preregistered")
    if not isinstance(preregistered, dict) or preregistered.get("complete") is not True:
        raise PostprocessError("formal postprocess requires a complete preregistration")
    execution = plan.get("execution")
    if execution != {
        "run_class": "formal", "data_mode": "frontier_rl", "optimizer_steps": 12,
        "nnodes": 1, "gpus_per_node": 8, "world_size": 8, "scheduling": "sequential",
    }:
        raise PostprocessError("A/B plan execution contract is not the frozen formal contract")
    cells = plan.get("cells")
    manifests = index.get("manifests")
    if not isinstance(cells, list) or len(cells) != 4:
        raise PostprocessError("A/B plan must contain four cells")
    if not isinstance(manifests, list) or len(manifests) != 4:
        raise PostprocessError("A/B index must contain four manifests")

    validated: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for number, (cell, binding) in enumerate(zip(cells, manifests, strict=True)):
        _exact(cell, {"cell_id", "seed", "arm", "target_dir"}, f"A/B cell {number}")
        _exact(binding, {"path", "sha256"}, f"manifest binding {number}")
        target_dir = _nonsymlink(cell["target_dir"], f"cell {cell['cell_id']} target")
        if not target_dir.is_dir():
            raise PostprocessError(f"cell target is not a directory: {target_dir}")
        manifest_path, manifest, manifest_hash = _load_json(
            binding["path"], f"cell {cell['cell_id']} manifest"
        )
        expected_manifest = target_dir / "v37_run_manifest.json"
        if manifest_path != expected_manifest:
            raise PostprocessError(f"cell {cell['cell_id']} manifest is outside its target directory")
        if manifest_hash != binding["sha256"]:
            raise PostprocessError(f"cell {cell['cell_id']} manifest hash differs from launcher index")
        if str(manifest_path) in seen_paths:
            raise PostprocessError("manifest path is reused across cells")
        seen_paths.add(str(manifest_path))
        if (
            manifest.get("arm") != cell["arm"]
            or manifest.get("seed") != cell["seed"]
            or manifest.get("run_class") != "formal"
            or manifest.get("data_mode") != "frontier_rl"
        ):
            raise PostprocessError(f"cell {cell['cell_id']} manifest identity mismatch")
        registration = manifest.get("ab_preregistration")
        if not isinstance(registration, dict) or (
            registration.get("ab_run_id") != plan["ab_run_id"]
            or registration.get("plan_path") != str(plan_path)
            or registration.get("plan_sha256") != plan_hash
            or registration.get("cell_id") != cell["cell_id"]
        ):
            raise PostprocessError(f"cell {cell['cell_id']} manifest preregistration mismatch")
        checkpoint_value = manifest.get("final_checkpoint_path")
        if not isinstance(checkpoint_value, str) or not checkpoint_value or not Path(checkpoint_value).is_absolute():
            raise PostprocessError(f"cell {cell['cell_id']} final checkpoint path is invalid")
        checkpoint = _nonsymlink(
            checkpoint_value, f"cell {cell['cell_id']} checkpoint"
        )
        checkpoint_hash = _tree_sha256(checkpoint)
        if checkpoint_hash != manifest.get("final_checkpoint_sha256"):
            raise PostprocessError(f"cell {cell['cell_id']} checkpoint hash mismatch")
        validated.append({
            **dict(cell),
            "target_dir": target_dir,
            "manifest_path": manifest_path,
            "manifest_hash": manifest_hash,
            "manifest": manifest,
            "checkpoint": checkpoint,
            "checkpoint_hash": checkpoint_hash,
        })
    return plan_path, plan, validated


def _expected_producer_outputs(arm: str) -> set[str]:
    outputs = {PAIRED_EVIDENCE_NAME, PRODUCER_RECEIPT_NAME}
    if arm == "progress":
        outputs.update({PIXMO_EVIDENCE_NAME, STEPCOUNT_EVIDENCE_NAME})
    return outputs


def _remove_staging(path: Path) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


def _validate_producer_receipt(
    receipt_path: Path,
    *,
    cell: Mapping[str, Any],
    producer: Path,
    producer_sha256: str,
    recipe: Path,
    recipe_sha256: str,
    invocation_nonce: str,
    staging: Path,
) -> None:
    _, receipt, _ = _load_json(receipt_path, f"cell {cell['cell_id']} producer receipt")
    _exact(
        receipt,
        {
            "schema_version", "artifact_type", "cell_id", "arm", "seed",
            "invocation_nonce", "invocation_request_sha256", "manifest", "checkpoint", "eval_model", "recipe",
            "producer", "environment_sha256", "steps",
        },
        f"cell {cell['cell_id']} producer receipt",
    )
    if (
        receipt.get("schema_version") != eval_producer.RECEIPT_SCHEMA_VERSION
        or receipt.get("artifact_type") != eval_producer.RECEIPT_TYPE
        or receipt.get("cell_id") != cell["cell_id"]
        or receipt.get("arm") != cell["arm"]
        or receipt.get("seed") != cell["seed"]
        or receipt.get("invocation_nonce") != invocation_nonce
    ):
        raise PostprocessError(f"producer receipt identity mismatch for {cell['cell_id']}")
    expected_bindings = {
        "manifest": {"path": str(cell["manifest_path"]), "sha256": cell["manifest_hash"]},
        "checkpoint": {"path": str(cell["checkpoint"]), "sha256": cell["checkpoint_hash"]},
        "eval_model": {
            "path": str(Path(cell["checkpoint"]) / "actor" / "huggingface"),
            "sha256": _tree_sha256(Path(cell["checkpoint"]) / "actor" / "huggingface"),
        },
        "recipe": {"path": str(recipe), "sha256": recipe_sha256},
        "producer": {"path": str(producer), "sha256": producer_sha256},
    }
    for name, expected in expected_bindings.items():
        if receipt.get(name) != expected:
            raise PostprocessError(f"producer receipt {name} binding mismatch for {cell['cell_id']}")
    request_hash = eval_producer.invocation_request_sha256(
        cell_id=str(cell["cell_id"]), arm=str(cell["arm"]), seed=int(cell["seed"]),
        invocation_nonce=invocation_nonce,
        manifest_path=str(cell["manifest_path"]), manifest_sha256=cell["manifest_hash"],
        checkpoint_path=str(cell["checkpoint"]), checkpoint_sha256=cell["checkpoint_hash"],
        eval_model_path=expected_bindings["eval_model"]["path"],
        eval_model_sha256=expected_bindings["eval_model"]["sha256"],
        recipe_path=str(recipe), recipe_sha256=recipe_sha256,
        producer_path=str(producer), producer_sha256=producer_sha256,
    )
    if receipt.get("invocation_request_sha256") != request_hash:
        raise PostprocessError(f"producer invocation request mismatch for {cell['cell_id']}")
    environment_hash = receipt.get("environment_sha256")
    if not isinstance(environment_hash, str) or len(environment_hash) != 64:
        raise PostprocessError(f"producer receipt environment hash is invalid for {cell['cell_id']}")
    steps = receipt.get("steps")
    expected_names = [
        name for name in eval_producer.STEP_NAMES
        if cell["arm"] in eval_producer.STEP_ARMS[name]
    ]
    if not isinstance(steps, list) or [item.get("name") for item in steps if isinstance(item, dict)] != expected_names:
        raise PostprocessError(f"producer receipt step order mismatch for {cell['cell_id']}")
    for step in steps:
        _exact(
            step,
            {
                "name", "program_path", "program_sha256", "argv", "argv_sha256",
                "output_path", "output_dir", "output_name", "output_sha256",
            },
            f"cell {cell['cell_id']} producer receipt step",
        )
        output = staging / str(step["output_name"])
        if not output.is_file() or _file_sha256(output) != step.get("output_sha256"):
            raise PostprocessError(f"producer receipt output binding mismatch for {cell['cell_id']}")
        step_output = Path(str(step["output_path"]))
        step_output_dir = Path(str(step["output_dir"]))
        argv = step.get("argv")
        if (
            not step_output.is_absolute()
            or not step_output_dir.is_absolute()
            or step_output.parent != step_output_dir
            or step_output.name != step["output_name"]
            or step_output_dir.parent != staging
            or not step_output_dir.name.startswith(f".{step['name']}.")
            or not isinstance(argv, list)
            or any(not isinstance(item, str) for item in argv)
            or hashlib.sha256(_canonical_bytes(argv)).hexdigest() != step.get("argv_sha256")
        ):
            raise PostprocessError(f"producer receipt argv/output path mismatch for {cell['cell_id']}")
        for name in ("program_sha256", "argv_sha256", "output_sha256"):
            value = step.get(name)
            if not isinstance(value, str) or len(value) != 64:
                raise PostprocessError(f"producer receipt {name} is invalid for {cell['cell_id']}")
    _, paired, _ = _load_json(
        staging / PAIRED_EVIDENCE_NAME, f"cell {cell['cell_id']} paired evidence",
    )
    if paired.get("producer_invocation_nonce") != invocation_nonce:
        raise PostprocessError(f"paired evidence invocation nonce mismatch for {cell['cell_id']}")


def _run_producer(
    producer: Path,
    producer_sha256: str,
    recipe: Path,
    recipe_sha256: str,
    cells: Sequence[Mapping[str, Any]],
) -> None:
    producer = _nonsymlink(producer, "postprocess producer")
    if not producer.is_file() or not os.access(producer, os.X_OK):
        raise PostprocessError(f"postprocess producer must be an executable regular file: {producer}")
    recipe_path = _nonsymlink(recipe, "postprocess recipe")
    if not recipe_path.is_file():
        raise PostprocessError(f"postprocess recipe must be a regular file: {recipe_path}")
    environment = {
        key: os.environ[key]
        for key in (
            "PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "HOME", "TMPDIR",
            "XDG_CACHE_HOME", "CUDA_CACHE_PATH",
        )
        if key in os.environ
    }
    environment.update({
        "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "0",
        "WANDB_MODE": "offline", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
    })
    producer_raw = gate._read_stable_regular_file(producer)
    recipe_raw = gate._read_stable_regular_file(recipe_path)
    if (
        hashlib.sha256(producer_raw).hexdigest() != producer_sha256
        or hashlib.sha256(recipe_raw).hexdigest() != recipe_sha256
    ):
        raise PostprocessError("preregistered producer/recipe changed before execution snapshot")
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for cell in cells:
            target_dir = Path(cell["target_dir"])
            evidence_dir = target_dir / "v37_postprocess"
            if evidence_dir.exists() or evidence_dir.is_symlink():
                raise PostprocessError(f"producer output directory already exists: {evidence_dir}")
            staging = Path(tempfile.mkdtemp(prefix=".v37_postprocess.", dir=target_dir))
            staged.append((staging, evidence_dir))
            execution_dir = Path(tempfile.mkdtemp(prefix=".v37_execution.", dir=target_dir))
            producer_snapshot = execution_dir / "producer.snapshot.py"
            recipe_snapshot = execution_dir / "recipe.snapshot.json"
            try:
                for path, raw, mode in (
                    (producer_snapshot, producer_raw, 0o500),
                    (recipe_snapshot, recipe_raw, 0o400),
                ):
                    descriptor = os.open(
                        path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                        mode,
                    )
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(raw)
                        handle.flush()
                        os.fsync(handle.fileno())
                execution_environment = dict(environment)
                execution_environment.update({
                    "V37_PREREGISTERED_PRODUCER_PATH": str(producer),
                    "V37_PREREGISTERED_PRODUCER_SHA256": producer_sha256,
                    "V37_PREREGISTERED_RECIPE_PATH": str(recipe_path),
                    "V37_PREREGISTERED_RECIPE_SHA256": recipe_sha256,
                })
                command = [
                    sys.executable,
                    str(producer_snapshot),
                    "--cell-id", str(cell["cell_id"]),
                    "--arm", str(cell["arm"]),
                    "--seed", str(cell["seed"]),
                    "--manifest", str(cell["manifest_path"]),
                    "--checkpoint", str(cell["checkpoint"]),
                    "--output-dir", str(staging),
                    "--invocation-nonce", (invocation_nonce := secrets.token_hex(32)),
                    "--recipe", str(recipe_snapshot),
                ]
                completed = subprocess.run(
                    command, cwd=target_dir, env=execution_environment, check=False,
                )
            finally:
                _remove_staging(execution_dir)
            if completed.returncode != 0:
                raise PostprocessError(
                    f"producer failed for {cell['cell_id']} with exit code {completed.returncode}"
                )
            _, _, manifest_hash = _load_json(
                cell["manifest_path"], f"cell {cell['cell_id']} manifest"
            )
            if manifest_hash != cell["manifest_hash"]:
                raise PostprocessError(f"producer changed manifest for {cell['cell_id']}")
            if _tree_sha256(Path(cell["checkpoint"])) != cell["checkpoint_hash"]:
                raise PostprocessError(f"producer changed checkpoint for {cell['cell_id']}")
            metadata = os.lstat(staging)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise PostprocessError(f"producer replaced its staging directory for {cell['cell_id']}")
            children = list(staging.iterdir())
            expected = _expected_producer_outputs(str(cell["arm"]))
            if {child.name for child in children} != expected:
                raise PostprocessError(
                    f"producer output schema mismatch for {cell['cell_id']}; "
                    f"expected={sorted(expected)}, actual={sorted(child.name for child in children)}"
                )
            for child in children:
                child_metadata = os.lstat(child)
                if stat.S_ISLNK(child_metadata.st_mode) or not stat.S_ISREG(child_metadata.st_mode):
                    raise PostprocessError(
                        f"producer output must be a non-symlink regular file: {child}"
                    )
                _load_json(child, f"cell {cell['cell_id']} producer output {child.name}")
                descriptor = os.open(child, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            _validate_producer_receipt(
                staging / PRODUCER_RECEIPT_NAME,
                cell=cell,
                producer=producer,
                producer_sha256=producer_sha256,
                recipe=recipe_path,
                recipe_sha256=recipe_sha256,
                invocation_nonce=invocation_nonce,
                staging=staging,
            )
            staging_fd = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(staging_fd)
            finally:
                os.close(staging_fd)

        for staging, evidence_dir in staged:
            if evidence_dir.exists() or evidence_dir.is_symlink():
                raise PostprocessError(f"producer output directory appeared concurrently: {evidence_dir}")
            os.replace(staging, evidence_dir)
            published.append(evidence_dir)
        for target_dir in {path.parent for path in published}:
            target_fd = os.open(target_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(target_fd)
            finally:
                os.close(target_fd)
    except BaseException:
        for evidence_dir in reversed(published):
            _remove_staging(evidence_dir)
        raise
    finally:
        for staging, _ in staged:
            _remove_staging(staging)


def _artifact(path: Path, label: str) -> tuple[Path, dict[str, Any], str]:
    return _load_json(path, label)


def assemble(
    *,
    ab_index: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    producer: os.PathLike[str] | str | None = None,
    recipe: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Validate inputs, optionally produce evidence, run the gate, and publish."""
    index_path = _nonsymlink(ab_index, "A/B evidence index")
    plan_path, plan, cells = _validate_ab_index(index_path)
    output = _absolute(output_dir)
    if output.exists() or output.is_symlink():
        raise PostprocessError(f"postprocess output directory must not exist: {output}")
    parent = _nonsymlink(output.parent, "postprocess output parent")
    if not parent.is_dir():
        raise PostprocessError(f"postprocess output parent is not a directory: {parent}")

    (
        preregistered_producer,
        preregistered_producer_sha256,
        preregistered_recipe,
        preregistered_recipe_sha256,
    ) = _preregistered_pipeline(plan)
    if producer is None or recipe is None:
        raise PostprocessError(
            "formal postprocess must execute the preregistered producer and recipe"
        )
    selected_producer = _nonsymlink(producer, "selected eval producer")
    selected_recipe = _nonsymlink(recipe, "selected eval recipe")
    if selected_producer != preregistered_producer or selected_recipe != preregistered_recipe:
        raise PostprocessError("selected eval producer/recipe differs from A/B preregistration")
    _run_producer(
        selected_producer,
        preregistered_producer_sha256,
        selected_recipe,
        preregistered_recipe_sha256,
        cells,
    )

    run_specs: list[dict[str, Any]] = []
    source_bindings: list[dict[str, Any]] = []
    for cell in cells:
        cell_id = str(cell["cell_id"])
        evidence_dir = Path(cell["target_dir"]) / "v37_postprocess"
        training_path, _, training_hash = _artifact(
            Path(cell["target_dir"]) / TRAINING_EVIDENCE_NAME,
            f"{cell_id} training evidence",
        )
        paired_path, _, paired_hash = _artifact(
            evidence_dir / PAIRED_EVIDENCE_NAME, f"{cell_id} paired evidence"
        )
        receipt_path, _, receipt_hash = _artifact(
            evidence_dir / PRODUCER_RECEIPT_NAME, f"{cell_id} producer receipt"
        )
        run = {
            "arm": cell["arm"],
            "seed": cell["seed"],
            "manifest": str(cell["manifest_path"]),
            "manifest_sha256": cell["manifest_hash"],
            "log": str(training_path),
            "log_sha256": training_hash,
            "eval": str(paired_path),
            "eval_sha256": paired_hash,
            "eval_receipt": str(receipt_path),
            "eval_receipt_sha256": receipt_hash,
        }
        source = {
            "cell_id": cell_id,
            "manifest": _binding(Path(cell["manifest_path"])),
            "training_evidence": _binding(training_path),
            "paired_evidence": _binding(paired_path),
            "producer_receipt": _binding(receipt_path),
        }
        if cell["arm"] == "progress":
            pixmo_path, _, pixmo_hash = _artifact(
                evidence_dir / PIXMO_EVIDENCE_NAME, f"{cell_id} pixmo evidence"
            )
            stepcount_path, _, stepcount_hash = _artifact(
                evidence_dir / STEPCOUNT_EVIDENCE_NAME, f"{cell_id} stepcount evidence"
            )
            run.update({
                "pixmo_eval": str(pixmo_path), "pixmo_eval_sha256": pixmo_hash,
                "stepcount_eval": str(stepcount_path), "stepcount_eval_sha256": stepcount_hash,
            })
            source.update({
                "pixmo_evidence": _binding(pixmo_path),
                "stepcount_evidence": _binding(stepcount_path),
            })
        run_specs.append(run)
        source_bindings.append(source)

    gate_spec = {
        "schema_version": 2,
        "stage": "final",
        "run_class": "formal",
        "preregistered": {
            "seeds": plan["seeds"],
            "optimizer_steps": gate.FORMAL_OPTIMIZER_STEPS,
            "confidence_level": 0.95,
            "ci_method": "paired_normal_95",
            "frontier_mixed_min": gate.FORMAL_FRONTIER_MIXED_MIN,
            "outcome_allwrong_max": gate.FORMAL_OUTCOME_ALLWRONG_MAX,
            "paired_metrics": gate.FROZEN_PAIRED_METRICS,
        },
        "runs": run_specs,
        "ack": False,
    }
    result = gate.evaluate(gate_spec)

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        spec_path = temporary / "v37_final_gate_spec.json"
        result_path = temporary / "v37_final_gate_result.json"
        index_output_path = temporary / "v37_final_evidence_index.json"
        _write_new(spec_path, gate_spec)
        _write_new(result_path, result)
        final_index = {
            "schema_version": SCHEMA_VERSION,
            "index_type": INDEX_TYPE,
            "ab_index": _binding(index_path),
            "ab_plan": _binding(plan_path),
            "sources": source_bindings,
            "gate_spec": {
                "path": str(output / spec_path.name),
                "sha256": _file_sha256(spec_path),
            },
            "gate_result": {
                "path": str(output / result_path.name),
                "sha256": _file_sha256(result_path),
            },
            "decision": result.get("decision"),
        }
        _write_new(index_output_path, final_index)
        directory_fd = os.open(temporary, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(temporary, output)
        parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return {
        "decision": result.get("decision"),
        "output_dir": str(output),
        "gate_result": result,
    }


def _write_new(path: Path, value: Any) -> None:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--producer")
    parser.add_argument("--recipe")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = assemble(
            ab_index=args.ab_index,
            output_dir=args.output_dir,
            producer=args.producer,
            recipe=args.recipe,
        )
    except (OSError, PostprocessError) as exc:
        print(f"[V37-postprocess][ERROR] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
