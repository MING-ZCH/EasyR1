#!/usr/bin/env python3
"""Build and verify non-self-reported V37 final benchmark evidence.

The descriptor is intentionally a cache over immutable, path-bound artifacts.
Scores are always recomputed from the canonical dataset and raw transcript.
Protocol facts that cannot be recovered from a transcript must be present in
both the run manifest and each result row; legacy artifacts are never upgraded
by guessing evaluator defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 2
DESCRIPTOR_TYPE = "v37_final_benchmark_evidence"
RUN_MANIFEST_TYPE = "v37_final_benchmark_run"
TREE_HASH_ALGORITHM = "v37-canonical-tree-manifest-sha256-v2"

SUITE_SPECS: dict[str, dict[str, int]] = {
    "pixmo-test": {"total": 529, "task_cap": 13},
    "stepcount-500": {"total": 500, "task_cap": 53},
}

ARTIFACT_KINDS = {
    "checkpoint_tree": "tree",
    "eval_model": "tree",
    "dataset": "file",
    "external_image_manifest": "file",
    "raw_results": "file",
    "eval_run_manifest": "file",
}
RUN_ARTIFACT_KEYS = tuple(key for key in ARTIFACT_KINDS if key != "eval_run_manifest")
IMPLEMENTATION_KEYS = ("evaluator", "validator", "prompt", "requirements")
PORTABLE_IMAGE_SPLITS = ("train", "validation", "forbidden")
PORTABLE_IMAGE_SNAPSHOT_CONTRACT = "external_image_resolved_path_sha256_v1"
PORTABLE_EMBEDDED_BYTES_CONTRACT = "covered_by_source_parquet_sha256"

SHA256_RE = re.compile(r"[0-9a-f]{64}")
INTEGER_RE = re.compile(r"[+-]?\d+")
COMPLETE_TAG_RE = re.compile(r"</(point|answer)\s*>", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<answer\s*>\s*([+-]?\d+)\s*</answer\s*>", re.IGNORECASE | re.DOTALL)
IMAGE_PLACEHOLDER_RE = re.compile(r"^\s*<image>\s*", re.IGNORECASE)


class EvidenceError(ValueError):
    """An evidence artifact is missing, ambiguous, mutable, or inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceError("value is not finite canonical JSON") from exc


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA256 of the module's canonical JSON representation."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def dataset_row_sha256(row: Mapping[str, Any]) -> str:
    """Hash one complete canonical dataset row, including unknown fields."""
    if not isinstance(row, Mapping):
        raise EvidenceError("dataset row must be an object")
    return canonical_json_sha256(dict(row))


def frozen_protocol(suite: str) -> dict[str, Any]:
    """Return a fresh copy of the only promotable final-eval protocol."""
    spec = SUITE_SPECS.get(suite)
    if spec is None:
        raise EvidenceError(f"unsupported suite: {suite!r}")
    return {
        "precision": "bfloat16",
        "decoding": {
            "mode": "greedy",
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
        },
        "round_limit": {
            "mode": "adaptive_ground_truth_plus",
            "task_cap": spec["task_cap"],
            "extra_rounds": 3,
            "formula": "min(task_cap,ground_truth+3)",
        },
        "answer": {
            "mode": "explicit_tagged_integer",
            "tag": "answer",
            "integer_pattern": "[+-]?\\d+",
        },
        "point_count_fallback": False,
        "stop": {
            "after_first_complete_tag": True,
            "complete_tags": ["point", "answer"],
        },
    }


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(data: bytes, label: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError(f"{label} is not UTF-8 JSON") from exc

    def reject_constant(value: str) -> Any:
        raise EvidenceError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"{label} is invalid JSON: {exc}") from exc
    _validate_finite_json(value, label)
    return value


def _validate_finite_json(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceError(f"{label} contains a non-finite number")
    if isinstance(value, list):
        for item in value:
            _validate_finite_json(item, label)
    elif isinstance(value, dict):
        for item in value.values():
            _validate_finite_json(item, label)


def _strict_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise EvidenceError(f"{label} keys mismatch; missing={missing}, extra={extra}")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise EvidenceError(f"{label} must be a boolean")
    return value


def _strict_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise EvidenceError(f"{label} must be an integer{suffix}")
    return value


def _known_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise EvidenceError(f"{label} must be a lowercase full SHA256")
    return value


def _same_json(left: Any, right: Any) -> bool:
    return _canonical_bytes(left) == _canonical_bytes(right)


def _path_state(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _absolute_path(path: os.PathLike[str] | str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _check_nonsymlink_components(
    path: os.PathLike[str] | str,
    label: str,
    *,
    allow_missing_leaf: bool = False,
) -> Path:
    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        try:
            metadata = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError) as exc:
            if allow_missing_leaf and index == len(parts) - 1:
                return absolute
            raise EvidenceError(f"{label} does not exist: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise EvidenceError(f"{label} contains a symlink component: {current}")
    return absolute


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    kind: str
    sha256: str
    size: int
    file_count: int
    total_bytes: int
    data: bytes | None
    root_state: tuple[int, int, int, int, int, int]
    entries: tuple[tuple[str, str, tuple[int, int, int, int, int, int]], ...]

    def reference(self) -> dict[str, Any]:
        if self.kind == "file":
            return {
                "path": str(self.path),
                "kind": "file",
                "sha256": self.sha256,
                "size": self.size,
            }
        return {
            "path": str(self.path),
            "kind": "tree",
            "hash_algorithm": TREE_HASH_ALGORITHM,
            "sha256": self.sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }


def _snapshot_file(
    path: Path,
    label: str,
    *,
    load_bytes: bool,
    secondary_digest: Any | None = None,
) -> _Snapshot:
    path = _check_nonsymlink_components(path, label)
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(f"cannot open {label}: {path}: {exc}") from exc
    chunks: list[bytes] | None = [] if load_bytes else None
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceError(f"{label} must be a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if secondary_digest is not None:
                secondary_digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _path_state(before) != _path_state(after):
        raise EvidenceError(f"{label} mutated while being read: {path}")
    final = os.lstat(path)
    if stat.S_ISLNK(final.st_mode) or _path_state(final) != _path_state(after):
        raise EvidenceError(f"{label} path changed while being read: {path}")
    _check_nonsymlink_components(path, label)
    return _Snapshot(
        path=path,
        kind="file",
        sha256=digest.hexdigest(),
        size=after.st_size,
        file_count=1,
        total_bytes=after.st_size,
        data=None if chunks is None else b"".join(chunks),
        root_state=_path_state(after),
        entries=(),
    )


def _scan_tree(path: Path, label: str) -> tuple[
    tuple[int, int, int, int, int, int],
    tuple[tuple[str, str, tuple[int, int, int, int, int, int]], ...],
]:
    root_metadata = os.lstat(path)
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise EvidenceError(f"{label} must be a non-symlink directory: {path}")
    entries: list[tuple[str, str, tuple[int, int, int, int, int, int]]] = []

    def visit(directory: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            raise EvidenceError(f"cannot scan {label}: {directory}: {exc}") from exc
        for child in children:
            child_path = directory / child.name
            metadata = child.stat(follow_symlinks=False)
            relative = child_path.relative_to(path).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise EvidenceError(f"{label} contains a symlink: {child_path}")
            if stat.S_ISDIR(metadata.st_mode):
                entries.append((relative, "dir", _path_state(metadata)))
                visit(child_path)
            elif stat.S_ISREG(metadata.st_mode):
                entries.append((relative, "file", _path_state(metadata)))
            else:
                raise EvidenceError(f"{label} contains a special file: {child_path}")

    visit(path)
    return _path_state(root_metadata), tuple(entries)


def _snapshot_tree(path: Path, label: str) -> _Snapshot:
    path = _check_nonsymlink_components(path, label)
    root_before, entries_before = _scan_tree(path, label)
    file_entries = [entry for entry in entries_before if entry[1] == "file"]
    if not file_entries:
        raise EvidenceError(f"{label} tree is empty: {path}")
    # This intentionally matches tools/v37_gate.py and the V37 launcher:
    # canonical relative path + size + per-file SHA256 entries.
    manifest_entries: list[dict[str, Any]] = []
    total_bytes = 0
    for relative, _, state_before in file_entries:
        snapshot = _snapshot_file(
            path / relative,
            f"{label}/{relative}",
            load_bytes=False,
            secondary_digest=None,
        )
        if snapshot.root_state != state_before:
            raise EvidenceError(f"{label} entry changed during tree hash: {relative}")
        manifest_entries.append({
            "path": relative,
            "size": snapshot.size,
            "sha256": snapshot.sha256,
        })
        total_bytes += snapshot.size
    root_after, entries_after = _scan_tree(path, label)
    if root_before != root_after or entries_before != entries_after:
        raise EvidenceError(f"{label} tree mutated while being hashed: {path}")
    _check_nonsymlink_components(path, label)
    return _Snapshot(
        path=path,
        kind="tree",
        sha256=hashlib.sha256(_canonical_bytes(manifest_entries)).hexdigest(),
        size=total_bytes,
        file_count=len(file_entries),
        total_bytes=total_bytes,
        data=None,
        root_state=root_after,
        entries=entries_after,
    )


class _SnapshotCache:
    def __init__(self) -> None:
        self._snapshots: dict[tuple[str, str], _Snapshot] = {}

    def capture(
        self,
        path: os.PathLike[str] | str,
        kind: str,
        label: str,
        *,
        load_bytes: bool = False,
    ) -> _Snapshot:
        absolute = _absolute_path(path)
        key = (str(absolute), kind)
        cached = self._snapshots.get(key)
        if cached is not None and (not load_bytes or cached.data is not None):
            return cached
        if kind == "file":
            current = _snapshot_file(absolute, label, load_bytes=load_bytes)
        elif kind == "tree":
            if load_bytes:
                raise EvidenceError(f"cannot load a tree as bytes: {label}")
            current = _snapshot_tree(absolute, label)
        else:
            raise EvidenceError(f"unknown artifact kind for {label}: {kind}")
        if cached is not None and cached.sha256 != current.sha256:
            raise EvidenceError(f"{label} changed between reads")
        self._snapshots[key] = current
        return current

    def assert_stable(self) -> None:
        for snapshot in self._snapshots.values():
            label = f"snapshot {snapshot.path}"
            _check_nonsymlink_components(snapshot.path, label)
            if snapshot.kind == "file":
                metadata = os.lstat(snapshot.path)
                if stat.S_ISLNK(metadata.st_mode) or _path_state(metadata) != snapshot.root_state:
                    raise EvidenceError(f"artifact changed after verification: {snapshot.path}")
            else:
                root, entries = _scan_tree(snapshot.path, label)
                if root != snapshot.root_state or entries != snapshot.entries:
                    raise EvidenceError(f"artifact tree changed after verification: {snapshot.path}")


def sha256_file(path: os.PathLike[str] | str) -> str:
    """Hash every byte of a regular non-symlink file."""
    return _snapshot_file(_absolute_path(path), "hash target", load_bytes=False).sha256


def tree_sha256(path: os.PathLike[str] | str) -> str:
    """Hash every regular file byte and relative name in a directory tree."""
    return _snapshot_tree(_absolute_path(path), "hash target").sha256


def artifact_binding(path: os.PathLike[str] | str, *, kind: str) -> dict[str, str]:
    """Create the compact path+SHA256 binding used by an eval run manifest."""
    if kind not in {"file", "tree"}:
        raise EvidenceError(f"artifact binding kind must be file or tree, got {kind!r}")
    snapshot = (
        _snapshot_file(_absolute_path(path), "artifact binding", load_bytes=False)
        if kind == "file"
        else _snapshot_tree(_absolute_path(path), "artifact binding")
    )
    return {"path": str(snapshot.path), "sha256": snapshot.sha256}


def _parse_integer(value: Any, label: str, *, allow_missing: bool = False) -> int | None:
    if value is None and allow_missing:
        return None
    if type(value) is int:
        return value
    if isinstance(value, str) and INTEGER_RE.fullmatch(value.strip()):
        return int(value.strip())
    if allow_missing and value == "":
        return None
    raise EvidenceError(f"{label} must be an explicit integer")


def _sample_id(value: Any, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise EvidenceError(f"{label} must be a string or integer ID")
    identifier = str(value)
    if not identifier:
        raise EvidenceError(f"{label} must not be empty")
    return identifier


def _single_field(row: Mapping[str, Any], names: Sequence[str], label: str) -> tuple[str, Any]:
    present = [(name, row[name]) for name in names if name in row]
    if not present:
        raise EvidenceError(f"{label} missing one of {list(names)}")
    first_name, first_value = present[0]
    for name, value in present[1:]:
        if not _same_json(value, first_value):
            raise EvidenceError(f"{label} has conflicting {first_name}/{name} fields")
    return first_name, first_value


def _rows_from_container(value: Any, label: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be a JSON array or object containing rows")
    names = [name for name in ("samples", "results", "rows", "data") if name in value]
    if len(names) != 1 or not isinstance(value[names[0]], list):
        raise EvidenceError(f"{label} must contain exactly one list field: samples/results/rows/data")
    return value[names[0]]


@dataclass(frozen=True)
class _DatasetSample:
    index: int
    identifier: str
    answer: int
    question: str
    image_path: Path
    row_sha256: str
    raw: Mapping[str, Any]


def _dataset_image_path(row: Mapping[str, Any], dataset_path: Path, label: str) -> Path:
    if "image_path" in row:
        value = row["image_path"]
    elif "image" in row and isinstance(row["image"], str):
        value = row["image"]
    elif "images" in row and isinstance(row["images"], list) and len(row["images"]) == 1:
        image = row["images"][0]
        value = image.get("path") if isinstance(image, dict) else image
    else:
        raise EvidenceError(f"{label} missing a single external image path")
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{label} image path must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = dataset_path.parent / path
    return _absolute_path(path)


def _load_dataset(value: Any, dataset_path: Path, suite: str) -> list[_DatasetSample]:
    rows = _rows_from_container(value, "canonical dataset")
    expected_total = SUITE_SPECS[suite]["total"]
    if len(rows) != expected_total:
        raise EvidenceError(
            f"{suite} canonical dataset must contain exactly {expected_total} rows, got {len(rows)}"
        )
    samples: list[_DatasetSample] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        label = f"canonical dataset row {index}"
        if not isinstance(row, dict):
            raise EvidenceError(f"{label} must be an object")
        _, raw_id = _single_field(row, ("id", "sample_id", "prompt_id"), label)
        identifier = _sample_id(raw_id, f"{label} id")
        if identifier in seen:
            raise EvidenceError(f"duplicate canonical dataset ID: {identifier}")
        seen.add(identifier)
        _, raw_answer = _single_field(row, ("answer", "ground_truth", "correct_answer"), label)
        answer = _parse_integer(raw_answer, f"{label} answer")
        assert answer is not None
        if answer < 0:
            raise EvidenceError(f"{label} answer must be non-negative")
        _, raw_question = _single_field(row, ("question", "problem"), label)
        if not isinstance(raw_question, str) or not raw_question:
            raise EvidenceError(f"{label} question must be a non-empty string")
        samples.append(
            _DatasetSample(
                index=index,
                identifier=identifier,
                answer=answer,
                question=raw_question,
                image_path=_dataset_image_path(row, dataset_path, label),
                row_sha256=dataset_row_sha256(row),
                raw=row,
            )
        )
    return samples


def _load_image_manifest(
    value: Any,
    suite: str,
    samples: Sequence[_DatasetSample],
    cache: _SnapshotCache,
) -> list[str]:
    if isinstance(value, dict) and value.get("schema_version") == 1:
        return _load_portable_image_snapshot(value, samples, cache)
    manifest = _strict_keys(value, {"schema_version", "suite", "images"}, "image manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise EvidenceError("image manifest schema_version must be 2")
    if manifest["suite"] != suite:
        raise EvidenceError("image manifest suite mismatch")
    images = manifest["images"]
    if not isinstance(images, list) or len(images) != len(samples):
        raise EvidenceError(f"image manifest must contain exactly {len(samples)} ordered entries")
    hashes: list[str] = []
    for index, (entry, sample) in enumerate(zip(images, samples)):
        item = _strict_keys(entry, {"id", "path", "sha256"}, f"image manifest entry {index}")
        identifier = _sample_id(item["id"], f"image manifest entry {index} id")
        if identifier != sample.identifier:
            raise EvidenceError(f"image manifest order/ID mismatch at row {index}")
        if not isinstance(item["path"], str) or not Path(item["path"]).is_absolute():
            raise EvidenceError(f"image manifest entry {index} path must be absolute")
        image_path = _absolute_path(item["path"])
        if str(image_path) != item["path"] or image_path != sample.image_path:
            raise EvidenceError(f"image manifest path mismatch at row {index}")
        expected_hash = _known_sha256(item["sha256"], f"image manifest entry {index} sha256")
        snapshot = cache.capture(image_path, "file", f"source image {sample.identifier}")
        if snapshot.sha256 != expected_hash:
            raise EvidenceError(f"source image hash mismatch for {sample.identifier}")
        hashes.append(expected_hash)
    return hashes


def _load_portable_image_snapshot(
    value: Mapping[str, Any],
    samples: Sequence[_DatasetSample],
    cache: _SnapshotCache,
) -> list[str]:
    expected_keys = {
        "schema_version",
        "snapshot_contract",
        "embedded_bytes_contract",
        *PORTABLE_IMAGE_SPLITS,
        "aggregate_sha256",
    }
    snapshot = _strict_keys(value, expected_keys, "portable image snapshot")
    if _strict_int(snapshot["schema_version"], "portable image snapshot schema_version") != 1:
        raise EvidenceError("portable image snapshot schema_version must be 1")
    if snapshot["snapshot_contract"] != PORTABLE_IMAGE_SNAPSHOT_CONTRACT:
        raise EvidenceError("portable image snapshot contract mismatch")
    if snapshot["embedded_bytes_contract"] != PORTABLE_EMBEDDED_BYTES_CONTRACT:
        raise EvidenceError("portable image embedded-bytes contract mismatch")
    by_path: dict[str, str] = {}
    group_hashes: dict[str, str] = {}
    for split in PORTABLE_IMAGE_SPLITS:
        group = _strict_keys(
            snapshot[split], {"file_count", "files", "aggregate_sha256"}, f"image snapshot {split}"
        )
        files = group["files"]
        if not isinstance(files, list):
            raise EvidenceError(f"image snapshot {split}.files must be a list")
        if _strict_int(group["file_count"], f"image snapshot {split}.file_count") != len(files):
            raise EvidenceError(f"image snapshot {split} file count mismatch")
        previous = ""
        normalized: list[dict[str, str]] = []
        for index, item in enumerate(files):
            record = _strict_keys(
                item, {"resolved_path", "sha256"}, f"image snapshot {split} file {index}"
            )
            path_text = record["resolved_path"]
            if (
                not isinstance(path_text, str)
                or not Path(path_text).is_absolute()
                or str(_absolute_path(path_text)) != path_text
                or path_text <= previous
            ):
                raise EvidenceError(f"image snapshot {split} paths must be normalized and sorted")
            previous = path_text
            image_hash = _known_sha256(record["sha256"], f"image snapshot {split} file {index}")
            if path_text in by_path and by_path[path_text] != image_hash:
                raise EvidenceError(f"conflicting portable image hashes for {path_text}")
            by_path[path_text] = image_hash
            normalized.append({"resolved_path": path_text, "sha256": image_hash})
        expected_group_hash = canonical_json_sha256(normalized)
        if group["aggregate_sha256"] != expected_group_hash:
            raise EvidenceError(f"image snapshot {split} aggregate mismatch")
        group_hashes[split] = expected_group_hash
    if snapshot["aggregate_sha256"] != canonical_json_sha256(group_hashes):
        raise EvidenceError("portable image snapshot aggregate mismatch")

    hashes: list[str] = []
    for sample in samples:
        path_text = str(sample.image_path)
        expected_hash = by_path.get(path_text)
        if expected_hash is None:
            raise EvidenceError(f"portable image snapshot omits {sample.identifier}: {path_text}")
        actual = cache.capture(sample.image_path, "file", f"source image {sample.identifier}")
        if actual.sha256 != expected_hash:
            raise EvidenceError(f"source image hash mismatch for {sample.identifier}")
        hashes.append(expected_hash)
    return hashes


def _compact_binding(reference: Mapping[str, Any]) -> dict[str, str]:
    return {"path": str(reference["path"]), "sha256": str(reference["sha256"])}


def _verify_compact_binding(value: Any, expected: Mapping[str, Any], label: str) -> None:
    binding = _strict_keys(value, {"path", "sha256"}, label)
    if not isinstance(binding["path"], str) or not Path(binding["path"]).is_absolute():
        raise EvidenceError(f"{label}.path must be absolute")
    _known_sha256(binding["sha256"], f"{label}.sha256")
    if not _same_json(binding, _compact_binding(expected)):
        raise EvidenceError(f"{label} does not bind the descriptor artifact")


def _verify_run_manifest(
    value: Any,
    suite: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    implementations: Mapping[str, Mapping[str, Any]],
) -> None:
    manifest = _strict_keys(
        value,
        {"schema_version", "manifest_type", "suite", "artifacts", "implementations", "protocol"},
        "eval run manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise EvidenceError("eval run manifest schema_version must be 2")
    if manifest["manifest_type"] != RUN_MANIFEST_TYPE:
        raise EvidenceError(f"eval run manifest type must be {RUN_MANIFEST_TYPE!r}")
    if manifest["suite"] != suite:
        raise EvidenceError("eval run manifest suite mismatch")
    if not _same_json(manifest["protocol"], frozen_protocol(suite)):
        raise EvidenceError("eval run manifest does not contain the frozen V37 protocol")
    run_artifacts = _strict_keys(manifest["artifacts"], set(RUN_ARTIFACT_KEYS), "run artifacts")
    for key in RUN_ARTIFACT_KEYS:
        _verify_compact_binding(run_artifacts[key], artifacts[key], f"run artifacts.{key}")
    run_implementations = _strict_keys(
        manifest["implementations"], set(IMPLEMENTATION_KEYS), "run implementations"
    )
    for key in IMPLEMENTATION_KEYS:
        _verify_compact_binding(
            run_implementations[key], implementations[key], f"run implementations.{key}"
        )


def _required_row_field(row: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in row:
        raise EvidenceError(
            f"{label} missing protocol evidence field {key}; refusing to infer evaluator defaults"
        )
    return row[key]


def _verify_row_protocol(row: Mapping[str, Any], suite: str, label: str) -> None:
    expected = frozen_protocol(suite)
    if "protocol" in row:
        if not _same_json(row["protocol"], expected):
            raise EvidenceError(f"{label}.protocol is not the frozen V37 protocol")
        return

    dtype = _required_row_field(row, "model_dtype", label)
    if not isinstance(dtype, str) or dtype.lower() not in {"bf16", "bfloat16"}:
        raise EvidenceError(f"{label}.model_dtype must be BF16")
    if _strict_bool(_required_row_field(row, "do_sample", label), f"{label}.do_sample"):
        raise EvidenceError(f"{label}.do_sample must be false")
    if _strict_int(
        _required_row_field(row, "num_beams", label), f"{label}.num_beams"
    ) != 1:
        raise EvidenceError(f"{label}.num_beams must be 1")
    if _strict_int(
        _required_row_field(row, "num_return_sequences", label),
        f"{label}.num_return_sequences",
    ) != 1:
        raise EvidenceError(f"{label}.num_return_sequences must be 1")
    if not _strict_bool(
        _required_row_field(row, "adaptive_max_rounds", label),
        f"{label}.adaptive_max_rounds",
    ):
        raise EvidenceError(f"{label}.adaptive_max_rounds must be true")
    if (
        _strict_int(
            _required_row_field(row, "adaptive_max_rounds_extra", label),
            f"{label}.adaptive_max_rounds_extra",
        )
        != 3
    ):
        raise EvidenceError(f"{label}.adaptive_max_rounds_extra must be 3")
    task_cap = SUITE_SPECS[suite]["task_cap"]
    if (
        _strict_int(
            _required_row_field(row, "max_rounds_config", label),
            f"{label}.max_rounds_config",
        )
        != task_cap
    ):
        raise EvidenceError(f"{label}.max_rounds_config must be {task_cap}")
    if not _strict_bool(
        _required_row_field(row, "require_explicit_answer", label),
        f"{label}.require_explicit_answer",
    ):
        raise EvidenceError(f"{label}.require_explicit_answer must be true")
    if _strict_bool(
        _required_row_field(row, "point_count_fallback_enabled", label),
        f"{label}.point_count_fallback_enabled",
    ):
        raise EvidenceError(f"{label}.point_count_fallback_enabled must be false")
    if not _strict_bool(
        _required_row_field(row, "stop_after_first_complete_tag", label),
        f"{label}.stop_after_first_complete_tag",
    ):
        raise EvidenceError(f"{label}.stop_after_first_complete_tag must be true")


def validate_raw_results_protocol(raw_results: os.PathLike[str] | str, suite: str) -> None:
    """Validate only explicit per-row protocol evidence, useful for migrations."""
    if suite not in SUITE_SPECS:
        raise EvidenceError(f"unsupported suite: {suite!r}")
    snapshot = _snapshot_file(_absolute_path(raw_results), "raw results", load_bytes=True)
    assert snapshot.data is not None
    rows = _rows_from_container(_decode_json(snapshot.data, "raw results"), "raw results")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise EvidenceError(f"raw result row {index} must be an object")
        _verify_row_protocol(row, suite, f"raw result row {index}")


def _message_role(message: Mapping[str, Any], label: str) -> str:
    role = message.get("role")
    if role in {"human", "user"}:
        return "human"
    if role in {"model", "assistant"}:
        return "model"
    if role == "system":
        return "system"
    raise EvidenceError(f"{label} has invalid role {role!r}")


def _message_content(message: Mapping[str, Any], label: str) -> str:
    content = message.get("content")
    if not isinstance(content, str):
        raise EvidenceError(f"{label}.content must be a string")
    return content


def _response_event(response: str, label: str) -> tuple[str | None, int | None]:
    closing = COMPLETE_TAG_RE.search(response)
    if closing is None:
        return None, None
    if response[closing.end() :].strip():
        raise EvidenceError(f"{label} contains content after the first complete tag")
    event = closing.group(1).lower()
    opening = re.search(rf"<{event}\s*>", response[: closing.start()], re.IGNORECASE)
    if opening is None:
        raise EvidenceError(f"{label} closes {event} without an opening tag")
    if event == "answer":
        answer_match = ANSWER_TAG_RE.search(response)
        if answer_match is None or answer_match.end() != closing.end():
            raise EvidenceError(f"{label} answer must contain only an explicit integer")
        return event, int(answer_match.group(1))
    return event, None


def _transcript_from_row(row: Mapping[str, Any], label: str) -> list[Any]:
    present = [(key, row[key]) for key in ("transcript", "output", "messages") if key in row]
    if not present:
        raise EvidenceError(f"{label} missing transcript/output; rounds cannot be independently verified")
    first_name, first = present[0]
    for name, value in present[1:]:
        if not _same_json(value, first):
            raise EvidenceError(f"{label} has conflicting {first_name}/{name} transcripts")
    if not isinstance(first, list):
        raise EvidenceError(f"{label}.{first_name} must be a list")
    return first


@dataclass(frozen=True)
class _TranscriptResult:
    response: str
    prediction: int | None
    rounds: int
    last_event: str | None


def _verify_transcript(
    row: Mapping[str, Any],
    sample: _DatasetSample,
    raw_results_path: Path,
    prompt_text: str,
    label: str,
) -> _TranscriptResult:
    transcript = _transcript_from_row(row, label)
    if len(transcript) < 3 or not all(isinstance(item, dict) for item in transcript):
        raise EvidenceError(f"{label} transcript is incomplete")
    first = transcript[0]
    assert isinstance(first, dict)
    if _message_role(first, f"{label} transcript[0]") != "system":
        raise EvidenceError(f"{label} transcript must start with the bound system prompt")
    if _message_content(first, f"{label} transcript[0]") != prompt_text.rstrip("\r\n"):
        raise EvidenceError(f"{label} system prompt does not match the bound prompt file")

    cursor = 1
    rounds = 0
    last_response = ""
    last_event: str | None = None
    prediction: int | None = None
    while cursor < len(transcript):
        human = transcript[cursor]
        assert isinstance(human, dict)
        human_label = f"{label} transcript[{cursor}]"
        if _message_role(human, human_label) == "system":
            if cursor != len(transcript) - 1 or prediction is not None:
                raise EvidenceError(f"{label} has an invalid terminal system diagnostic")
            cursor += 1
            break
        if _message_role(human, human_label) != "human":
            raise EvidenceError(f"{label} transcript must alternate human/model messages")
        human_content = _message_content(human, human_label)
        if rounds == 0:
            expected_question = IMAGE_PLACEHOLDER_RE.sub("", sample.question, count=1)
            if human_content != expected_question:
                raise EvidenceError(f"{label} first human question differs from the canonical dataset")
            image_value = human.get("image", human.get("image_path"))
            if not isinstance(image_value, str) or not image_value:
                raise EvidenceError(f"{label} first human message lacks the canonical image path")
            transcript_image = Path(image_value)
            if not transcript_image.is_absolute():
                transcript_image = raw_results_path.parent / transcript_image
            transcript_image = _absolute_path(transcript_image)
            if transcript_image != sample.image_path:
                raise EvidenceError(f"{label} transcript image differs from the canonical dataset")
        cursor += 1
        if cursor >= len(transcript):
            raise EvidenceError(f"{label} transcript ends before a model response")
        model = transcript[cursor]
        assert isinstance(model, dict)
        model_label = f"{label} transcript[{cursor}]"
        if _message_role(model, model_label) != "model":
            raise EvidenceError(f"{label} transcript must alternate human/model messages")
        rounds += 1
        if "round" in model and _strict_int(model["round"], f"{model_label}.round") != rounds:
            raise EvidenceError(f"{model_label}.round is not sequential")
        last_response = _message_content(model, model_label)
        last_event, round_prediction = _response_event(last_response, model_label)
        if last_event == "answer":
            prediction = round_prediction
            if cursor != len(transcript) - 1:
                raise EvidenceError(f"{label} continued after an explicit answer")
        cursor += 1
    if cursor != len(transcript):
        raise EvidenceError(f"{label} transcript contains trailing messages")
    if not last_response:
        raise EvidenceError(f"{label} transcript has no model response")
    if "final_response" in row:
        final = row["final_response"]
        if isinstance(final, dict):
            final = final.get("content", final.get("text"))
        if not isinstance(final, str) or final != last_response:
            raise EvidenceError(f"{label}.final_response differs from the final transcript response")
    return _TranscriptResult(last_response, prediction, rounds, last_event)


def _cached_prediction(value: Any, label: str) -> int | None:
    return _parse_integer(value, label, allow_missing=True)


def _require_cache(row: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in row:
        raise EvidenceError(f"{label} missing required cache field {key}")
    return row[key]


def _verify_result_provenance(
    row: Mapping[str, Any],
    sample: _DatasetSample,
    image_sha256: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    label: str,
) -> None:
    required = {
        "dataset_sha256": artifacts["dataset"]["sha256"],
        "dataset_row_sha256": sample.row_sha256,
        "source_image_sha256": image_sha256,
        "checkpoint_sha256": artifacts["checkpoint_tree"]["sha256"],
        "model_path": artifacts["eval_model"]["path"],
        "eval_model_sha256": artifacts["eval_model"]["sha256"],
    }
    for key, expected in required.items():
        if key not in row:
            raise EvidenceError(f"{label} missing provenance field {key}")
        if row[key] != expected:
            raise EvidenceError(f"{label}.{key} provenance mismatch")


def _recompute_samples(
    suite: str,
    samples: Sequence[_DatasetSample],
    image_hashes: Sequence[str],
    raw_value: Any,
    raw_results_path: Path,
    prompt_text: str,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = _rows_from_container(raw_value, "raw results")
    expected_total = SUITE_SPECS[suite]["total"]
    if len(rows) != expected_total:
        raise EvidenceError(f"{suite} raw results must contain exactly {expected_total} rows, got {len(rows)}")
    evidence_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    correct_count = 0
    task_cap = SUITE_SPECS[suite]["task_cap"]
    for index, (row, sample, image_sha256) in enumerate(zip(rows, samples, image_hashes)):
        label = f"raw result row {index}"
        if not isinstance(row, dict):
            raise EvidenceError(f"{label} must be an object")
        identifier = _sample_id(_require_cache(row, "id", label), f"{label}.id")
        if identifier in seen:
            raise EvidenceError(f"duplicate raw result ID: {identifier}")
        seen.add(identifier)
        if identifier != sample.identifier:
            raise EvidenceError(f"raw result order/ID mismatch at row {index}")
        raw_gt = _parse_integer(_require_cache(row, "correct_answer", label), f"{label}.correct_answer")
        if raw_gt != sample.answer:
            raise EvidenceError(f"{label}.correct_answer differs from the canonical dataset")
        if _require_cache(row, "question", label) != sample.question:
            raise EvidenceError(f"{label}.question differs from the canonical dataset")
        _verify_row_protocol(row, suite, label)
        _verify_result_provenance(row, sample, image_sha256, artifacts, label)
        transcript = _verify_transcript(row, sample, raw_results_path, prompt_text, label)
        effective_cap = min(task_cap, sample.answer + 3)
        if effective_cap < 1:
            effective_cap = 1
        if transcript.rounds < 1 or transcript.rounds > effective_cap:
            raise EvidenceError(f"{label} transcript exceeds its adaptive turn cap")
        if (
            _strict_int(
                _require_cache(row, "effective_max_rounds", label),
                f"{label}.effective_max_rounds",
            )
            != effective_cap
        ):
            raise EvidenceError(f"{label}.effective_max_rounds mismatch")
        if (
            _strict_int(_require_cache(row, "num_rounds", label), f"{label}.num_rounds")
            != transcript.rounds
        ):
            raise EvidenceError(f"{label}.num_rounds differs from transcript")
        used_fallback = _strict_bool(
            _require_cache(row, "used_point_count_fallback", label),
            f"{label}.used_point_count_fallback",
        )
        if used_fallback:
            raise EvidenceError(f"{label} used forbidden point-count fallback")

        prediction = transcript.prediction
        correct = prediction is not None and prediction == sample.answer
        expected_termination = "explicit_answer" if prediction is not None else "max_rounds"
        if prediction is None and transcript.rounds != effective_cap:
            raise EvidenceError(f"{label} terminated without an answer before its turn cap")
        if _require_cache(row, "termination_reason", label) != expected_termination:
            raise EvidenceError(f"{label}.termination_reason mismatch")
        if _require_cache(row, "last_response_event", label) != transcript.last_event:
            raise EvidenceError(f"{label}.last_response_event mismatch")
        explicit = _strict_bool(
            _require_cache(row, "explicit_answer_detected", label),
            f"{label}.explicit_answer_detected",
        )
        if explicit != (prediction is not None):
            raise EvidenceError(f"{label}.explicit_answer_detected is forged or stale")
        cached_prediction = _cached_prediction(
            _require_cache(row, "predicted_answer", label), f"{label}.predicted_answer"
        )
        if cached_prediction != prediction:
            raise EvidenceError(f"{label}.predicted_answer is forged or stale")
        cached_correct = _strict_bool(
            _require_cache(row, "is_correct", label), f"{label}.is_correct"
        )
        if cached_correct != correct:
            raise EvidenceError(f"{label}.is_correct is forged or stale")
        correct_count += int(correct)
        evidence_rows.append(
            {
                "index": index,
                "id": sample.identifier,
                "dataset_row_sha256": sample.row_sha256,
                "image_sha256": image_sha256,
                "predicted_answer": prediction,
                "correct": correct,
                "rounds": transcript.rounds,
                "effective_max_rounds": effective_cap,
                "termination_reason": expected_termination,
            }
        )
    summary = {
        "total": expected_total,
        "correct": correct_count,
        "incorrect": expected_total - correct_count,
    }
    return evidence_rows, summary


def _reference_from_descriptor(value: Any, kind: str, label: str) -> Mapping[str, Any]:
    expected_keys = (
        {"path", "kind", "sha256", "size"}
        if kind == "file"
        else {"path", "kind", "hash_algorithm", "sha256", "file_count", "total_bytes"}
    )
    reference = _strict_keys(value, expected_keys, label)
    if reference["kind"] != kind:
        raise EvidenceError(f"{label}.kind must be {kind}")
    path = reference["path"]
    if not isinstance(path, str) or not Path(path).is_absolute() or str(_absolute_path(path)) != path:
        raise EvidenceError(f"{label}.path must be a normalized absolute path")
    _known_sha256(reference["sha256"], f"{label}.sha256")
    if kind == "file":
        _strict_int(reference["size"], f"{label}.size", minimum=0)
    else:
        if reference["hash_algorithm"] != TREE_HASH_ALGORITHM:
            raise EvidenceError(f"{label}.hash_algorithm mismatch")
        _strict_int(reference["file_count"], f"{label}.file_count", minimum=1)
        _strict_int(reference["total_bytes"], f"{label}.total_bytes", minimum=0)
    return reference


def _capture_inputs(
    cache: _SnapshotCache,
    artifact_paths: Mapping[str, os.PathLike[str] | str],
    implementation_paths: Mapping[str, os.PathLike[str] | str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, bytes]]:
    artifacts: dict[str, dict[str, Any]] = {}
    implementations: dict[str, dict[str, Any]] = {}
    json_data: dict[str, bytes] = {}
    for key, kind in ARTIFACT_KINDS.items():
        snapshot = cache.capture(
            artifact_paths[key],
            kind,
            key.replace("_", " "),
            load_bytes=kind == "file" and key in {
                "dataset", "external_image_manifest", "raw_results", "eval_run_manifest"
            },
        )
        artifacts[key] = snapshot.reference()
        if snapshot.data is not None:
            json_data[key] = snapshot.data
    for key in IMPLEMENTATION_KEYS:
        snapshot = cache.capture(
            implementation_paths[key],
            "file",
            f"{key} implementation",
            load_bytes=key == "prompt",
        )
        implementations[key] = snapshot.reference()
        if snapshot.data is not None:
            json_data[key] = snapshot.data
    return artifacts, implementations, json_data


def _compute_descriptor_payload(
    suite: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    implementations: Mapping[str, Mapping[str, Any]],
    json_data: Mapping[str, bytes],
    cache: _SnapshotCache,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    dataset_value = _decode_json(json_data["dataset"], "canonical dataset")
    dataset_path = Path(str(artifacts["dataset"]["path"]))
    samples = _load_dataset(dataset_value, dataset_path, suite)
    image_value = _decode_json(json_data["external_image_manifest"], "image manifest")
    image_hashes = _load_image_manifest(image_value, suite, samples, cache)
    run_manifest = _decode_json(json_data["eval_run_manifest"], "eval run manifest")
    _verify_run_manifest(run_manifest, suite, artifacts, implementations)
    try:
        prompt_text = json_data["prompt"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError("prompt implementation is not UTF-8") from exc
    raw_value = _decode_json(json_data["raw_results"], "raw results")
    return _recompute_samples(
        suite,
        samples,
        image_hashes,
        raw_value,
        Path(str(artifacts["raw_results"]["path"])),
        prompt_text,
        artifacts,
    )


def _write_json_atomic(path: os.PathLike[str] | str, value: Any) -> Path:
    output = _absolute_path(path)
    parent = _check_nonsymlink_components(output.parent, "descriptor output parent")
    if not parent.is_dir():
        raise EvidenceError(f"descriptor output parent is not a directory: {parent}")
    if output.exists() or output.is_symlink():
        _check_nonsymlink_components(output, "descriptor output")
        if not output.is_file():
            raise EvidenceError(f"descriptor output is not a regular file: {output}")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _check_nonsymlink_components(parent, "descriptor output parent")
        os.replace(temporary, output)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def build_descriptor(
    *,
    suite: str,
    checkpoint_tree: os.PathLike[str] | str,
    eval_model: os.PathLike[str] | str,
    dataset: os.PathLike[str] | str,
    external_image_manifest: os.PathLike[str] | str,
    raw_results: os.PathLike[str] | str,
    eval_run_manifest: os.PathLike[str] | str,
    evaluator: os.PathLike[str] | str,
    validator: os.PathLike[str] | str,
    prompt: os.PathLike[str] | str,
    requirements: os.PathLike[str] | str,
    output: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Build a strict descriptor after independently validating all artifacts."""
    if suite not in SUITE_SPECS:
        raise EvidenceError(f"unsupported suite: {suite!r}")
    artifact_paths = {
        "checkpoint_tree": checkpoint_tree,
        "eval_model": eval_model,
        "dataset": dataset,
        "external_image_manifest": external_image_manifest,
        "raw_results": raw_results,
        "eval_run_manifest": eval_run_manifest,
    }
    implementation_paths = {
        "evaluator": evaluator,
        "validator": validator,
        "prompt": prompt,
        "requirements": requirements,
    }
    cache = _SnapshotCache()
    artifacts, implementations, json_data = _capture_inputs(
        cache, artifact_paths, implementation_paths
    )
    evidence_rows, summary = _compute_descriptor_payload(
        suite, artifacts, implementations, json_data, cache
    )
    descriptor = {
        "schema_version": SCHEMA_VERSION,
        "descriptor_type": DESCRIPTOR_TYPE,
        "suite": suite,
        "artifacts": artifacts,
        "implementations": implementations,
        "protocol": frozen_protocol(suite),
        "samples": evidence_rows,
        "summary": summary,
    }
    cache.assert_stable()
    if output is not None:
        _write_json_atomic(output, descriptor)
        cache.assert_stable()
    return descriptor


build_evidence_descriptor = build_descriptor
build_benchmark_evidence = build_descriptor


def _verify_descriptor_object(descriptor: Any) -> dict[str, Any]:
    if isinstance(descriptor, dict) and descriptor.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError(
            "formal benchmark evidence requires schema_version=2; schema v1 is rejected"
        )
    top = _strict_keys(
        descriptor,
        {
            "schema_version",
            "descriptor_type",
            "suite",
            "artifacts",
            "implementations",
            "protocol",
            "samples",
            "summary",
        },
        "benchmark evidence descriptor",
    )
    if top["schema_version"] != SCHEMA_VERSION:
        raise EvidenceError("formal benchmark evidence requires schema_version=2; schema v1 is rejected")
    if top["descriptor_type"] != DESCRIPTOR_TYPE:
        raise EvidenceError(f"descriptor_type must be {DESCRIPTOR_TYPE!r}")
    suite = top["suite"]
    if not isinstance(suite, str) or suite not in SUITE_SPECS:
        raise EvidenceError(f"unsupported suite: {suite!r}")
    if not _same_json(top["protocol"], frozen_protocol(suite)):
        raise EvidenceError("descriptor protocol is not the frozen V37 protocol")
    artifact_values = _strict_keys(top["artifacts"], set(ARTIFACT_KINDS), "descriptor artifacts")
    implementation_values = _strict_keys(
        top["implementations"], set(IMPLEMENTATION_KEYS), "descriptor implementations"
    )
    cache = _SnapshotCache()
    artifacts: dict[str, dict[str, Any]] = {}
    implementations: dict[str, dict[str, Any]] = {}
    json_data: dict[str, bytes] = {}
    for key, kind in ARTIFACT_KINDS.items():
        expected = _reference_from_descriptor(artifact_values[key], kind, f"artifacts.{key}")
        snapshot = cache.capture(
            expected["path"],
            kind,
            key.replace("_", " "),
            load_bytes=kind == "file" and key in {
                "dataset", "external_image_manifest", "raw_results", "eval_run_manifest"
            },
        )
        actual = snapshot.reference()
        if not _same_json(actual, expected):
            raise EvidenceError(f"artifacts.{key} content drift")
        artifacts[key] = actual
        if snapshot.data is not None:
            json_data[key] = snapshot.data
    for key in IMPLEMENTATION_KEYS:
        expected = _reference_from_descriptor(
            implementation_values[key], "file", f"implementations.{key}"
        )
        snapshot = cache.capture(
            expected["path"],
            "file",
            f"{key} implementation",
            load_bytes=key == "prompt",
        )
        actual = snapshot.reference()
        if not _same_json(actual, expected):
            raise EvidenceError(f"implementations.{key} content drift")
        implementations[key] = actual
        if snapshot.data is not None:
            json_data[key] = snapshot.data
    evidence_rows, summary = _compute_descriptor_payload(
        suite, artifacts, implementations, json_data, cache
    )
    if not _same_json(top["samples"], evidence_rows):
        raise EvidenceError("descriptor sample cache differs from independently recomputed evidence")
    if not _same_json(top["summary"], summary):
        raise EvidenceError("descriptor summary cache differs from independently recomputed score")
    cache.assert_stable()
    return {
        "valid": True,
        "schema_version": SCHEMA_VERSION,
        "suite": suite,
        "summary": summary,
    }


def verify_descriptor(descriptor: Mapping[str, Any] | os.PathLike[str] | str) -> dict[str, Any]:
    """Verify a descriptor object or strict JSON descriptor file."""
    if isinstance(descriptor, (str, os.PathLike)):
        snapshot = _snapshot_file(_absolute_path(descriptor), "benchmark descriptor", load_bytes=True)
        assert snapshot.data is not None
        value = _decode_json(snapshot.data, "benchmark descriptor")
        report = _verify_descriptor_object(value)
        metadata = os.lstat(snapshot.path)
        if _path_state(metadata) != snapshot.root_state:
            raise EvidenceError("benchmark descriptor changed during verification")
        return report
    return _verify_descriptor_object(descriptor)


verify_evidence_descriptor = verify_descriptor
verify_benchmark_evidence = verify_descriptor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify strict V37 final benchmark evidence schema v2."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="validate artifacts and write a schema v2 descriptor")
    build.add_argument("--suite", required=True, choices=sorted(SUITE_SPECS))
    build.add_argument("--checkpoint-tree", "--checkpoint", dest="checkpoint_tree", required=True)
    build.add_argument("--eval-model", required=True)
    build.add_argument("--dataset", required=True)
    build.add_argument(
        "--external-image-manifest", "--image-manifest", dest="external_image_manifest", required=True
    )
    build.add_argument("--raw-results", "--results", dest="raw_results", required=True)
    build.add_argument("--eval-run-manifest", "--run-manifest", dest="eval_run_manifest", required=True)
    build.add_argument("--evaluator", required=True)
    build.add_argument("--validator", default=str(Path(__file__).absolute()))
    build.add_argument("--prompt", required=True)
    build.add_argument("--requirements", required=True)
    build.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify", help="recompute and verify a schema v2 descriptor")
    verify.add_argument("descriptor", nargs="?")
    verify.add_argument("--descriptor", dest="descriptor_option")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            descriptor = build_descriptor(
                suite=args.suite,
                checkpoint_tree=args.checkpoint_tree,
                eval_model=args.eval_model,
                dataset=args.dataset,
                external_image_manifest=args.external_image_manifest,
                raw_results=args.raw_results,
                eval_run_manifest=args.eval_run_manifest,
                evaluator=args.evaluator,
                validator=args.validator,
                prompt=args.prompt,
                requirements=args.requirements,
                output=args.output,
            )
            report = {
                "valid": True,
                "schema_version": SCHEMA_VERSION,
                "suite": descriptor["suite"],
                "descriptor": str(_absolute_path(args.output)),
                "summary": descriptor["summary"],
            }
        else:
            descriptor_path = args.descriptor_option or args.descriptor
            if not descriptor_path:
                raise EvidenceError("verify requires a descriptor path")
            report = verify_descriptor(descriptor_path)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except (EvidenceError, OSError) as exc:
        print(f"[V37-benchmark-evidence][ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
