#!/usr/bin/env python3
"""Run a preregistered, argv-only V37 formal evaluation recipe.

This tool is intentionally an orchestration boundary, not an evaluator.  Each
recipe step binds an executable by SHA256 and must natively emit the strict
JSON artifact consumed by ``v37_gate.py``.  No shell expansion, post-hoc field
backfill, or ambient training environment is allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import string
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
RECIPE_TYPE = "v37_formal_eval_recipe"
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_TYPE = "v37_eval_producer_receipt"
RECEIPT_NAME = "producer_receipt.json"
STEP_NAMES = ("paired_eval", "pixmo_benchmark", "stepcount_benchmark")
STEP_OUTPUTS = {
    "paired_eval": "paired_eval.json",
    "pixmo_benchmark": "pixmo_benchmark_evidence.json",
    "stepcount_benchmark": "stepcount_benchmark_evidence.json",
}
STEP_ARMS = {
    "paired_eval": ["baseline", "progress"],
    "pixmo_benchmark": ["progress"],
    "stepcount_benchmark": ["progress"],
}
PLACEHOLDERS = {
    "cell_id", "arm", "seed", "manifest", "checkpoint", "eval_model",
    "output", "output_dir", "invocation_nonce",
}
REQUIRED_ENVIRONMENT = {
    "WANDB_MODE": "offline",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONHASHSEED": "{seed}",
}
OPTIONAL_ENVIRONMENT_KEYS = {
    "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "HOME", "XDG_CACHE_HOME",
    "HF_HOME", "TRANSFORMERS_CACHE", "CUDA_CACHE_PATH",
    "CUDA_DEVICE_MAX_CONNECTIONS", "TOKENIZERS_PARALLELISM",
    "CUDA_VISIBLE_DEVICES",
}
ALLOWED_ENVIRONMENT_KEYS = set(REQUIRED_ENVIRONMENT) | OPTIONAL_ENVIRONMENT_KEYS
SENSITIVE_NAME = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE|ACCESS_KEY|API_KEY|AUTH|BEARER|COOKIE|SESSION)",
    re.IGNORECASE,
)
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class EvalProducerError(ValueError):
    """The preregistered formal eval recipe or one of its outputs is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvalProducerError("value is not finite canonical JSON") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_regular_bytes(path: Path, label: str) -> bytes:
    """Read one non-symlink file from a stable inode for execution snapshotting."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvalProducerError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise EvalProducerError(f"{label} changed while being snapshotted")
    return b"".join(chunks)


def _write_execution_snapshot(directory: Path, name: str, raw: bytes) -> Path:
    path = directory / name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o500)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return path


def _tree_sha256(path: Path) -> str:
    if not path.is_dir():
        raise EvalProducerError(f"checkpoint is not a directory: {path}")
    for root, directories, filenames in os.walk(path, followlinks=False):
        for name in directories + filenames:
            candidate = Path(root) / name
            if candidate.is_symlink():
                raise EvalProducerError(f"checkpoint contains a symlink: {candidate}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise EvalProducerError(f"checkpoint is empty: {path}")
    entries = []
    for item in files:
        if item.is_symlink():
            raise EvalProducerError(f"checkpoint contains a symlink: {item}")
        entries.append({
            "path": item.relative_to(path).as_posix(),
            "size": item.stat().st_size,
            "sha256": _sha256(item),
        })
    return hashlib.sha256(_canonical_bytes(entries)).hexdigest()


def _absolute_nonsymlink(path: os.PathLike[str] | str, label: str) -> Path:
    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise EvalProducerError(f"{label} does not exist: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise EvalProducerError(f"{label} contains a symlink component: {current}")
    return absolute


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    path = _absolute_nonsymlink(path, label)
    if not path.is_file():
        raise EvalProducerError(f"{label} must be a regular file: {path}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvalProducerError(f"{label} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                EvalProducerError(f"{label} contains non-finite constant {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvalProducerError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvalProducerError(f"{label} must be a JSON object")
    _canonical_bytes(value)
    return value


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise EvalProducerError(
            f"{label} schema mismatch; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _validate_template(value: str, label: str) -> list[str]:
    if not isinstance(value, str) or "\x00" in value:
        raise EvalProducerError(f"{label} must be a NUL-free string")
    fields: list[str] = []
    try:
        parsed = string.Formatter().parse(value)
        for _, field, format_spec, conversion in parsed:
            if field is None:
                continue
            if field not in PLACEHOLDERS or format_spec or conversion:
                raise EvalProducerError(f"{label} uses forbidden placeholder {{{field}}}")
            fields.append(field)
    except EvalProducerError:
        raise
    except ValueError as exc:
        raise EvalProducerError(f"{label} has invalid braces") from exc
    return fields


def _validate_recipe(path: Path) -> dict[str, Any]:
    recipe = _exact(
        _strict_json(path, "formal eval recipe"),
        {"schema_version", "recipe_type", "environment", "steps"},
        "formal eval recipe",
    )
    if recipe["schema_version"] != SCHEMA_VERSION or recipe["recipe_type"] != RECIPE_TYPE:
        raise EvalProducerError("formal eval recipe version/type mismatch")
    environment = recipe["environment"]
    if not isinstance(environment, dict) or not environment:
        raise EvalProducerError("recipe environment must be a non-empty object")
    for key, value in environment.items():
        if not isinstance(key, str) or ENV_NAME.fullmatch(key) is None:
            raise EvalProducerError(f"invalid recipe environment key: {key!r}")
        if SENSITIVE_NAME.search(key):
            raise EvalProducerError(f"sensitive environment key is forbidden: {key}")
        if key not in ALLOWED_ENVIRONMENT_KEYS:
            raise EvalProducerError(f"environment key is not in the formal allowlist: {key}")
        fields = _validate_template(value, f"environment.{key}")
        if not set(fields) <= {"cell_id", "arm", "seed", "invocation_nonce"}:
            raise EvalProducerError(
                f"environment.{key} uses a path-dependent placeholder"
            )
    for key, expected in REQUIRED_ENVIRONMENT.items():
        if environment.get(key) != expected:
            raise EvalProducerError(f"environment.{key} must equal {expected!r}")
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(visible, str):
        raise EvalProducerError("environment.CUDA_VISIBLE_DEVICES is required")
    devices = visible.split(",")
    if (
        len(devices) != 8
        or len(set(devices)) != 8
        or any(not item.isdigit() for item in devices)
    ):
        raise EvalProducerError("CUDA_VISIBLE_DEVICES must name exactly eight distinct GPUs")
    if not isinstance(environment.get("PATH"), str) or not environment["PATH"]:
        raise EvalProducerError("recipe environment must define PATH")

    steps = _exact(recipe["steps"], set(STEP_NAMES), "formal eval recipe steps")
    for name in STEP_NAMES:
        step = _exact(steps[name], {"arms", "program", "argv", "output"}, f"step {name}")
        if step["arms"] != STEP_ARMS[name] or step["output"] != STEP_OUTPUTS[name]:
            raise EvalProducerError(f"step {name} arm/output contract mismatch")
        program = _exact(step["program"], {"path", "sha256"}, f"step {name} program")
        executable = _absolute_nonsymlink(program["path"], f"step {name} program")
        if program["path"] != str(executable) or not Path(program["path"]).is_absolute():
            raise EvalProducerError(f"step {name} program path must be canonical absolute")
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise EvalProducerError(f"step {name} program must be executable: {executable}")
        if not isinstance(program["sha256"], str) or program["sha256"] != _sha256(executable):
            raise EvalProducerError(f"step {name} program SHA256 mismatch")
        argv = step["argv"]
        if not isinstance(argv, list) or not argv:
            raise EvalProducerError(f"step {name} argv must be a non-empty list")
        fields: list[str] = []
        for index, argument in enumerate(argv):
            fields.extend(_validate_template(argument, f"step {name} argv[{index}]") )
        for required in (
            "manifest", "checkpoint", "eval_model", "output", "invocation_nonce",
        ):
            if fields.count(required) != 1:
                raise EvalProducerError(
                    f"step {name} argv must reference {{{required}}} exactly once"
                )
    return dict(recipe)


def verify_recipe(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Strictly validate a recipe and every executable content binding."""
    return _validate_recipe(_absolute_nonsymlink(path, "formal eval recipe"))


def _render(value: str, bindings: Mapping[str, str], label: str) -> str:
    _validate_template(value, label)
    try:
        return value.format_map(bindings)
    except KeyError as exc:  # Defensive; template validation owns the public error.
        raise EvalProducerError(f"{label} has unknown placeholder") from exc


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
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


def invocation_request_sha256(
    *,
    cell_id: str,
    arm: str,
    seed: int,
    invocation_nonce: str,
    manifest_path: str,
    manifest_sha256: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    eval_model_path: str,
    eval_model_sha256: str,
    recipe_path: str,
    recipe_sha256: str,
    producer_path: str,
    producer_sha256: str,
) -> str:
    payload = {
        "cell_id": cell_id, "arm": arm, "seed": seed,
        "invocation_nonce": invocation_nonce,
        "manifest": {"path": manifest_path, "sha256": manifest_sha256},
        "checkpoint": {"path": checkpoint_path, "sha256": checkpoint_sha256},
        "eval_model": {"path": eval_model_path, "sha256": eval_model_sha256},
        "recipe": {"path": recipe_path, "sha256": recipe_sha256},
        "producer": {"path": producer_path, "sha256": producer_sha256},
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def produce(
    *,
    cell_id: str,
    arm: str,
    seed: int,
    manifest: os.PathLike[str] | str,
    checkpoint: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    recipe: os.PathLike[str] | str,
    invocation_nonce: str,
) -> dict[str, Any]:
    if not isinstance(cell_id, str) or not cell_id or arm not in {"baseline", "progress"}:
        raise EvalProducerError("cell identity/arm is invalid")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvalProducerError("seed must be a nonnegative integer")
    if not isinstance(invocation_nonce, str) or re.fullmatch(r"[0-9a-f]{64}", invocation_nonce) is None:
        raise EvalProducerError("invocation nonce must be exactly 64 lowercase hex characters")
    manifest_path = _absolute_nonsymlink(manifest, "run manifest")
    checkpoint_path = _absolute_nonsymlink(checkpoint, "checkpoint")
    recipe_execution_path = _absolute_nonsymlink(recipe, "formal eval recipe")
    output_path = _absolute_nonsymlink(output_dir, "producer output directory")
    if not output_path.is_dir() or any(output_path.iterdir()):
        raise EvalProducerError("producer output directory must be an existing empty directory")
    if manifest_path.parent != output_path.parent:
        raise EvalProducerError("producer output must be staged inside the manifest run directory")
    manifest_value = _strict_json(manifest_path, "run manifest")
    if manifest_value.get("arm") != arm or manifest_value.get("seed") != seed:
        raise EvalProducerError("run manifest arm/seed differs from producer invocation")
    if manifest_value.get("final_checkpoint_path") != str(checkpoint_path):
        raise EvalProducerError("run manifest checkpoint differs from producer invocation")
    checkpoint_hash = _tree_sha256(checkpoint_path)
    if manifest_value.get("final_checkpoint_sha256") != checkpoint_hash:
        raise EvalProducerError("run manifest checkpoint SHA256 mismatch")
    eval_model = _absolute_nonsymlink(checkpoint_path / "actor" / "huggingface", "eval model")
    if not eval_model.is_dir():
        raise EvalProducerError("eval model must be checkpoint/actor/huggingface")
    recipe_value = verify_recipe(recipe_execution_path)
    manifest_hash = _sha256(manifest_path)
    recipe_hash = _sha256(recipe_execution_path)
    eval_model_hash = _tree_sha256(eval_model)
    producer_execution_path = Path(__file__).resolve()
    producer_hash = _sha256(producer_execution_path)
    producer_path = Path(os.environ.get(
        "V37_PREREGISTERED_PRODUCER_PATH", str(producer_execution_path)
    ))
    recipe_path = Path(os.environ.get(
        "V37_PREREGISTERED_RECIPE_PATH", str(recipe_execution_path)
    ))
    expected_producer_hash = os.environ.get("V37_PREREGISTERED_PRODUCER_SHA256", producer_hash)
    expected_recipe_hash = os.environ.get("V37_PREREGISTERED_RECIPE_SHA256", recipe_hash)
    if producer_hash != expected_producer_hash or recipe_hash != expected_recipe_hash:
        raise EvalProducerError("execution snapshot differs from the preregistered producer/recipe")
    if not producer_path.is_absolute() or not recipe_path.is_absolute():
        raise EvalProducerError("preregistered producer/recipe identity paths must be absolute")
    invocation_request_hash = invocation_request_sha256(
        cell_id=cell_id, arm=arm, seed=seed, invocation_nonce=invocation_nonce,
        manifest_path=str(manifest_path), manifest_sha256=manifest_hash,
        checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
        eval_model_path=str(eval_model), eval_model_sha256=eval_model_hash,
        recipe_path=str(recipe_path), recipe_sha256=recipe_hash,
        producer_path=str(producer_path), producer_sha256=producer_hash,
    )
    bindings = {
        "cell_id": cell_id,
        "arm": arm,
        "seed": str(seed),
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "eval_model": str(eval_model),
        "output": "",  # Replaced per step.
        "output_dir": "",  # Replaced per step.
        "invocation_nonce": invocation_nonce,
        "invocation_request_sha256": invocation_request_hash,
    }
    environment = {
        key: _render(value, bindings, f"environment.{key}")
        for key, value in recipe_value["environment"].items()
    }
    executed: list[str] = []
    receipt_steps: list[dict[str, Any]] = []
    for name in STEP_NAMES:
        step = recipe_value["steps"][name]
        if arm not in step["arms"]:
            continue
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=output_path))
        try:
            program_path = _absolute_nonsymlink(
                step["program"]["path"], f"step {name} program"
            )
            program_raw = _stable_regular_bytes(program_path, f"step {name} program")
            program_hash = hashlib.sha256(program_raw).hexdigest()
            if program_hash != step["program"]["sha256"]:
                raise EvalProducerError(f"step {name} program changed before execution")
            execution_path = _write_execution_snapshot(
                temporary, f".{name}.program.snapshot", program_raw,
            )
            step_output = temporary / step["output"]
            step_bindings = dict(bindings)
            step_bindings.update(output=str(step_output), output_dir=str(temporary))
            command = [step["program"]["path"]] + [
                _render(argument, step_bindings, f"step {name} argv")
                for argument in step["argv"]
            ]
            execution_command = [str(execution_path), *command[1:]]
            completed = subprocess.run(
                execution_command, cwd=manifest_path.parent, env=environment, check=False,
            )
            if completed.returncode != 0:
                raise EvalProducerError(f"step {name} failed with exit code {completed.returncode}")
            output_value = _strict_json(step_output, f"step {name} output")
            if name == "paired_eval" and output_value.get("producer_invocation_nonce") != invocation_nonce:
                raise EvalProducerError("paired eval output did not echo the producer invocation nonce")
            destination = output_path / step["output"]
            if destination.exists() or destination.is_symlink():
                raise EvalProducerError(f"step {name} output already exists")
            os.link(step_output, destination)
            executed.append(name)
            receipt_steps.append({
                "name": name,
                "program_path": step["program"]["path"],
                "program_sha256": program_hash,
                "argv": command,
                "argv_sha256": hashlib.sha256(_canonical_bytes(command)).hexdigest(),
                "output_path": str(step_output),
                "output_dir": str(temporary),
                "output_name": step["output"],
                "output_sha256": _sha256(destination),
            })
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    if _sha256(manifest_path) != manifest_hash:
        raise EvalProducerError("evaluation changed the run manifest")
    if _tree_sha256(checkpoint_path) != checkpoint_hash:
        raise EvalProducerError("evaluation changed the checkpoint")
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "artifact_type": RECEIPT_TYPE,
        "cell_id": cell_id,
        "arm": arm,
        "seed": seed,
        "invocation_nonce": invocation_nonce,
        "invocation_request_sha256": invocation_request_hash,
        "manifest": {"path": str(manifest_path), "sha256": manifest_hash},
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_hash},
        "eval_model": {"path": str(eval_model), "sha256": eval_model_hash},
        "recipe": {"path": str(recipe_path), "sha256": recipe_hash},
        "producer": {
            "path": str(producer_path),
            "sha256": producer_hash,
        },
        "environment_sha256": hashlib.sha256(_canonical_bytes(environment)).hexdigest(),
        "steps": receipt_steps,
    }
    _write_new_json(output_path / RECEIPT_NAME, receipt)
    expected = {
        STEP_OUTPUTS[name] for name in STEP_NAMES if arm in STEP_ARMS[name]
    } | {RECEIPT_NAME}
    if {item.name for item in output_path.iterdir()} != expected:
        raise EvalProducerError("producer final output schema mismatch")
    return {
        "cell_id": cell_id, "arm": arm, "seed": seed, "steps": executed,
        "receipt": RECEIPT_NAME,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--arm", required=True, choices=("baseline", "progress"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--invocation-nonce", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = produce(
            cell_id=args.cell_id,
            arm=args.arm,
            seed=args.seed,
            manifest=args.manifest,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            recipe=args.recipe,
            invocation_nonce=args.invocation_nonce,
        )
    except (OSError, EvalProducerError) as exc:
        print(f"[V37-eval-producer][ERROR] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
