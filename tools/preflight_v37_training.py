#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail-fast validation for V37 StepCount parquet training data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import importlib.util

_remap_spec = importlib.util.spec_from_file_location(
    "v37_path_remap", Path(__file__).resolve().parents[1] / "verl/utils/path_remap.py"
)
if _remap_spec is None or _remap_spec.loader is None:
    raise ImportError("cannot load the dependency-light V37 path remap module")
_remap_module = importlib.util.module_from_spec(_remap_spec)
_remap_spec.loader.exec_module(_remap_module)
parse_image_path_remap = _remap_module.parse_image_path_remap
remap_image_path = _remap_module.remap_image_path


BUCKETS = ((2, 10), (11, 20), (21, 30), (31, 40), (41, 50))
BUCKET_LABELS = tuple(f"{low}-{high}" for low, high in BUCKETS)
CANDIDATE_ONLY_COLUMNS = {
    "response", "responses", "trajectory", "trajectories", "sampled_response",
    "candidate_id", "raw_success", "trajectory_quality", "process", "termination",
    "declared_seed", "response_sha256", "candidate_content_sha256",
}
FORMAL_RATIOS = dict(zip(BUCKET_LABELS, (0.40, 0.10, 0.20, 0.20, 0.10)))
FORMAL_RATIO_TOLERANCE = 0.05
FORMAL_PHASH_HAMMING_THRESHOLD = 4
FORMAL_FILTER_OVERLONG_NUM_PROC = 64
EXTERNAL_IMAGE_SNAPSHOT_SCHEMA_VERSION = 1
EXTERNAL_IMAGE_SNAPSHOT_CONTRACT = "external_image_resolved_path_sha256_v1"
EXTERNAL_IMAGE_EMBEDDED_BYTES_CONTRACT = "covered_by_source_parquet_sha256"
EXTERNAL_IMAGE_SPLITS = ("train", "validation", "forbidden")
FORMAL_CLASSIFIER_THRESHOLDS = {
    "winner_min": 3, "winner_max": 20, "process_min_quality": .55,
    "process_min_quality_range": .10, "process_max_duplicate_rate": .25,
    "process_max_cap_rate": .25,
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")),
    )


def _parse_answer(value: Any) -> Optional[int]:
    """Parse the numeric or JSON answer formats used by StepCount datasets."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, dict):
        for key in ("count_number", "final_count", "answer", "N", "total_count"):
            if key in value:
                parsed = _parse_answer(value[key])
                if parsed is not None:
                    return parsed
        for key in ("point_gts", "point_trajectory", "trajectory_points", "points"):
            points = value.get(key)
            if isinstance(points, list) and points:
                return len(points)
        return None
    if isinstance(value, (list, tuple)):
        return None

    text = str(value).strip()
    if not text:
        return None
    try:
        payload = _json_loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        payload = None
    if payload is not None and payload != text:
        return _parse_answer(payload)
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def _parse_answer_strict(value: Any) -> Optional[int]:
    """Formal/canary parser: integer evidence only, without JSON/string coercion."""
    if type(value) is int:
        return value
    if isinstance(value, dict):
        for key in ("count_number", "final_count", "answer", "N", "total_count"):
            if key in value:
                return _parse_answer_strict(value[key])
        for key in ("point_gts", "point_trajectory", "trajectory_points", "points"):
            points = value.get(key)
            if isinstance(points, list) and points:
                return len(points)
    return None


def _parse_expected_ratios(value: Optional[str]) -> Optional[dict[str, float]]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        raise ValueError("--expected-ratios cannot be empty")

    parsed: Any
    try:
        parsed = _json_loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        unknown = sorted(set(parsed) - set(BUCKET_LABELS))
        missing = sorted(set(BUCKET_LABELS) - set(parsed))
        if unknown or missing:
            raise ValueError(
                f"expected ratio keys must be {', '.join(BUCKET_LABELS)}; missing={missing}, unknown={unknown}"
            )
        ratios = {label: float(parsed[label]) for label in BUCKET_LABELS}
    else:
        if isinstance(parsed, list):
            parts = parsed
        elif "=" in text:
            pairs = {}
            for item in text.split(","):
                key, separator, raw = item.partition("=")
                if not separator:
                    raise ValueError("ratio mappings must use bucket=value")
                pairs[key.strip()] = raw.strip()
            return _parse_expected_ratios(json.dumps(pairs))
        else:
            parts = [part.strip() for part in text.split(",")]
        if len(parts) != len(BUCKET_LABELS):
            raise ValueError(f"expected exactly {len(BUCKET_LABELS)} ratios in bucket order")
        ratios = {label: float(raw) for label, raw in zip(BUCKET_LABELS, parts)}

    if any(not math.isfinite(ratio) or ratio < 0 or ratio > 1 for ratio in ratios.values()):
        raise ValueError("expected ratios must be finite values between 0 and 1")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"expected ratios must sum to 1.0, got {sum(ratios.values()):.8f}")
    return ratios


def _parquet_files(data_path: Path) -> list[Path]:
    if data_path.is_file() and data_path.suffix.lower() == ".parquet":
        return [data_path]
    if data_path.is_dir():
        return sorted(path for path in data_path.rglob("*.parquet") if path.is_file())
    return []


def _image_items(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _image_path(value: Any) -> Optional[str]:
    if isinstance(value, (str, os.PathLike)):
        text = os.fspath(value).strip()
        return text or None
    if isinstance(value, dict):
        raw = next((value.get(key) for key in ("path", "image_path", "file_name") if value.get(key) is not None), None)
        if isinstance(raw, (str, os.PathLike)):
            text = os.fspath(raw).strip()
            return text or None
    return None


def _embedded_image_bytes(value: Any) -> Optional[bytes]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return payload or None
    if not isinstance(value, dict):
        return None
    payload = value.get("bytes")
    if isinstance(payload, (bytes, bytearray, memoryview)):
        payload = bytes(payload)
        return payload if payload else None
    return None


def _embedded_image_is_decodable(payload: bytes) -> bool:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - repository requirements include Pillow
        raise RuntimeError("Pillow is required to verify embedded image payloads") from exc
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if image.width < 1 or image.height < 1:
                return False
        return True
    except Exception:
        return False


def _image_file_is_decodable(path: Path) -> bool:
    try:
        return _embedded_image_is_decodable(path.read_bytes())
    except OSError:
        return False


def _sequence_identifier(raw_path: str) -> str:
    basename = re.sub(r"\s+", "", Path(raw_path).name)
    stem = Path(basename).stem
    stem = re.sub(r"_k\d+$", "", stem)
    stem = re.sub(r"_\d+$", "", stem)
    return stem


def _resolve_existing_path(raw_path: str, roots: tuple[Path, ...]) -> Optional[Path]:
    path = Path(os.path.expandvars(os.path.expanduser(
        remap_image_path(raw_path, parse_image_path_remap())
    )))
    candidates = (path,) if path.is_absolute() else tuple(root / path for root in roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _first_symlink_component(path: Path) -> Optional[Path]:
    """Return the first symlink traversed by path without resolving it."""
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        if part == ".":
            continue
        if part == "..":
            current = current.parent
            continue
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except (FileNotFoundError, NotADirectoryError):
            continue
        if stat.S_ISLNK(mode):
            return current
    return None


def _resolve_external_image_path(raw_path: str, roots: tuple[Path, ...]) -> Path:
    """Resolve one external image path while rejecting every symlink candidate."""
    path = Path(
        os.path.expandvars(
            os.path.expanduser(remap_image_path(raw_path, parse_image_path_remap()))
        )
    )
    candidates = (path,) if path.is_absolute() else tuple(root / path for root in roots)
    for candidate in candidates:
        symlink = _first_symlink_component(candidate)
        if symlink is not None:
            raise ValueError(f"external image path contains a symlink component: {symlink}")
        try:
            metadata = os.lstat(candidate)
        except (FileNotFoundError, NotADirectoryError):
            continue
        if stat.S_ISREG(metadata.st_mode):
            resolved = candidate.resolve(strict=True)
            symlink = _first_symlink_component(candidate)
            if symlink is not None:
                raise ValueError(
                    f"external image path became a symlink while resolving: {symlink}"
                )
            after = os.lstat(candidate)
            resolved_metadata = os.lstat(resolved)
            expected_identity = (metadata.st_dev, metadata.st_ino)
            if (
                (after.st_dev, after.st_ino) != expected_identity
                or (resolved_metadata.st_dev, resolved_metadata.st_ino) != expected_identity
            ):
                raise ValueError(f"external image file changed while resolving: {candidate}")
            return resolved
    raise ValueError(f"external image path is missing or not a regular file: {raw_path}")


def _regular_file_sha256(path: Path) -> str:
    """Hash a stable regular file descriptor and reject symlink/race evidence."""
    symlink = _first_symlink_component(path)
    if symlink is not None:
        raise ValueError(f"external image path contains a symlink component: {symlink}")
    try:
        before = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError(f"external image file is missing: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"external image path is not a regular file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"external image file cannot be opened safely: {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"external image file changed while opening: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            finished = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    try:
        after = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError(f"external image file disappeared while hashing: {path}") from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(opened, field) != getattr(finished, field) for field in stable_fields) or any(
        getattr(finished, field) != getattr(after, field) for field in stable_fields
    ):
        raise ValueError(f"external image file changed while hashing: {path}")
    symlink = _first_symlink_component(path)
    if symlink is not None:
        raise ValueError(f"external image path became a symlink while hashing: {symlink}")
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _external_image_group(records: list[dict[str, str]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: item["resolved_path"])
    return {
        "file_count": len(ordered),
        "files": ordered,
        "aggregate_sha256": _canonical_sha256(ordered),
    }


def _external_image_snapshot_from_paths(
    paths_by_split: dict[str, set[Path]],
) -> tuple[dict[str, Any], list[str]]:
    groups: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for split in EXTERNAL_IMAGE_SPLITS:
        records: list[dict[str, str]] = []
        for path in sorted(paths_by_split.get(split, set()), key=lambda item: str(item)):
            try:
                records.append({
                    "resolved_path": str(path),
                    "sha256": _regular_file_sha256(path),
                })
            except (OSError, ValueError) as exc:
                errors.append(f"{split}: {exc}")
        groups[split] = _external_image_group(records)
    aggregate_payload = {
        split: groups[split]["aggregate_sha256"] for split in EXTERNAL_IMAGE_SPLITS
    }
    return {
        "schema_version": EXTERNAL_IMAGE_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_contract": EXTERNAL_IMAGE_SNAPSHOT_CONTRACT,
        "embedded_bytes_contract": EXTERNAL_IMAGE_EMBEDDED_BYTES_CONTRACT,
        **groups,
        "aggregate_sha256": _canonical_sha256(aggregate_payload),
    }, sorted(set(errors))


def _parquet_external_image_references(
    data_path: Path, pq: Any, batch_size: int, image_roots: tuple[Path, ...]
) -> Iterable[tuple[str, tuple[Path, ...], str]]:
    data_root = data_path if data_path.is_dir() else data_path.parent
    for parquet_path in _parquet_files(data_path):
        parquet = pq.ParquetFile(parquet_path)
        available = set(parquet.schema_arrow.names)
        columns = [name for name in ("images", "image", "image_path", "path") if name in available]
        row_number = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            values = batch.to_pydict()
            for index in range(batch.num_rows):
                raw = values.get("images", [None] * batch.num_rows)[index]
                if raw is None:
                    raw = next(
                        (values[name][index] for name in ("image", "image_path", "path") if name in values),
                        None,
                    )
                for item_number, item in enumerate(_image_items(raw)):
                    if _embedded_image_bytes(item) is not None:
                        continue
                    raw_path = _image_path(item)
                    if raw_path is not None:
                        yield (
                            raw_path,
                            (data_root, parquet_path.parent, *image_roots),
                            f"{parquet_path}:{row_number}[{item_number}]",
                        )
                row_number += 1


def _json_external_image_references(
    json_path: Path, image_roots: tuple[Path, ...]
) -> Iterable[tuple[str, tuple[Path, ...], str]]:
    payload = _json_loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = next(
            (payload[key] for key in ("data", "samples", "rows") if isinstance(payload.get(key), list)),
            None,
        )
    if not isinstance(payload, list):
        raise ValueError(f"benchmark JSON must contain a list of samples: {json_path}")
    roots = (json_path.parent, *image_roots)
    for row_number, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        raw = row.get("images")
        if raw is None:
            raw = next((row.get(name) for name in ("image", "image_path", "path") if row.get(name) is not None), None)
        for item_number, item in enumerate(_image_items(raw)):
            if _embedded_image_bytes(item) is not None:
                continue
            raw_path = _image_path(item)
            if raw_path is not None:
                yield raw_path, roots, f"{json_path}:{row_number}[{item_number}]"


def _collect_external_image_snapshots(
    data_path: Path,
    val_paths: tuple[Path, ...],
    forbidden_paths: tuple[Path, ...],
    pq: Any,
    batch_size: int,
    image_roots: tuple[Path, ...],
) -> tuple[dict[str, Any], list[str]]:
    """Freeze external image contents; embedded bytes remain sealed by parquet."""
    paths_by_split = {split: set() for split in EXTERNAL_IMAGE_SPLITS}
    errors: list[str] = []
    sources = {
        "train": (data_path,),
        "validation": val_paths,
        "forbidden": forbidden_paths,
    }
    for split in EXTERNAL_IMAGE_SPLITS:
        for source in sources[split]:
            try:
                if _parquet_files(source):
                    references = _parquet_external_image_references(
                        source, pq, batch_size, image_roots
                    )
                elif split == "forbidden" and source.is_file() and source.suffix.lower() == ".json":
                    references = _json_external_image_references(source, image_roots)
                else:
                    raise ValueError(f"external image snapshot source is not readable: {source}")
                for raw_path, roots, location in references:
                    try:
                        paths_by_split[split].add(
                            _resolve_external_image_path(raw_path, roots)
                        )
                    except (OSError, ValueError) as exc:
                        errors.append(f"{split} {location}: {exc}")
            except (OSError, ValueError) as exc:
                errors.append(f"{split} {source}: {exc}")
    snapshot, snapshot_errors = _external_image_snapshot_from_paths(paths_by_split)
    return snapshot, sorted(set(errors + snapshot_errors))


def verify_external_image_snapshots(snapshot: Any) -> list[str]:
    """Recompute a report snapshot from its resolved paths for gate consumers."""
    expected_keys = {
        "schema_version", "snapshot_contract", "embedded_bytes_contract",
        *EXTERNAL_IMAGE_SPLITS, "aggregate_sha256",
    }
    if not isinstance(snapshot, dict) or set(snapshot) != expected_keys:
        return ["external image snapshot schema mismatch"]
    if (
        snapshot.get("schema_version") != EXTERNAL_IMAGE_SNAPSHOT_SCHEMA_VERSION
        or snapshot.get("snapshot_contract") != EXTERNAL_IMAGE_SNAPSHOT_CONTRACT
        or snapshot.get("embedded_bytes_contract") != EXTERNAL_IMAGE_EMBEDDED_BYTES_CONTRACT
    ):
        return ["external image snapshot contract mismatch"]

    paths_by_split: dict[str, set[Path]] = {split: set() for split in EXTERNAL_IMAGE_SPLITS}
    errors: list[str] = []
    for split in EXTERNAL_IMAGE_SPLITS:
        group = snapshot.get(split)
        if not isinstance(group, dict) or set(group) != {"file_count", "files", "aggregate_sha256"}:
            errors.append(f"{split}: external image group schema mismatch")
            continue
        files = group.get("files")
        if type(group.get("file_count")) is not int or not isinstance(files, list) or group["file_count"] != len(files):
            errors.append(f"{split}: external image file count mismatch")
            continue
        previous = ""
        for item in files:
            if not isinstance(item, dict) or set(item) != {"resolved_path", "sha256"}:
                errors.append(f"{split}: external image file record schema mismatch")
                continue
            path_text = item.get("resolved_path")
            expected_hash = item.get("sha256")
            if (
                not isinstance(path_text, str)
                or not Path(path_text).is_absolute()
                or path_text <= previous
                or not isinstance(expected_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
            ):
                errors.append(f"{split}: invalid external image file record")
                continue
            previous = path_text
            paths_by_split[split].add(Path(path_text))
    if errors:
        return sorted(set(errors))

    actual, actual_errors = _external_image_snapshot_from_paths(paths_by_split)
    errors.extend(actual_errors)
    if actual != snapshot:
        errors.append("external image snapshot does not match current files")
    return sorted(set(errors))


@lru_cache(maxsize=None)
def _file_sha256(path_text: str) -> str:
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_keys(item: Any, roots: tuple[Path, ...] = ()) -> set[str]:
    keys: set[str] = set()
    path = _image_path(item)
    if path:
        sequence_id = _sequence_identifier(path)
        if sequence_id:
            keys.add(f"sequence:{sequence_id}")
        resolved = _resolve_existing_path(path, roots)
        if resolved is not None:
            keys.add(f"image_sha256:{_file_sha256(str(resolved))}")
    payload = _embedded_image_bytes(item)
    if payload:
        keys.add(f"image_sha256:{hashlib.sha256(payload).hexdigest()}")
    return keys


def _canonical_id(value: Any) -> Optional[str]:
    """Normalize numeric and textual IDs so cross-field matches collide."""
    if value is None or isinstance(value, (bool, dict, list, tuple)):
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        value = int(value) if value.is_integer() else value
    text = str(value).strip()
    return text or None


def _add_identifier(keys: set[str], value: Any) -> None:
    canonical = _canonical_id(value)
    if canonical is not None:
        keys.add(f"canonical_id:{canonical}")


def _collect_identity_keys(
    data_path: Path, pq: Any, batch_size: int, image_roots: tuple[Path, ...] = ()
) -> tuple[set[str], int, int, int, list[str]]:
    files = _parquet_files(data_path)
    keys: set[str] = set()
    row_count = 0
    rows_without_identity = 0
    rows_without_image_path = 0
    seen_ids = {"sample_id": set(), "prompt_id": set()}
    data_root = data_path if data_path.is_dir() else data_path.parent
    for parquet_path in files:
        parquet_file = pq.ParquetFile(parquet_path)
        available = set(parquet_file.schema_arrow.names)
        columns = [name for name in ("images", "image", "image_path", "path", "prompt_id", "id", "sample_id", "sequence_id") if name in available]
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            values = batch.to_pydict()
            for row_index in range(batch.num_rows):
                row_count += 1
                row_keys: set[str] = set()
                row_has_image_path = False
                raw_images = values.get("images", [None] * batch.num_rows)[row_index]
                if raw_images is None:
                    raw_images = next((values[name][row_index] for name in ("image", "image_path", "path") if name in values), None)
                for item in _image_items(raw_images):
                    row_has_image_path = row_has_image_path or _image_path(item) is not None
                    row_keys.update(
                        _identity_keys(item, (data_root, parquet_path.parent, *image_roots))
                    )
                for column in ("prompt_id", "sample_id", "sequence_id", "id"):
                    raw_identifier = values.get(column, [None] * batch.num_rows)[row_index]
                    _add_identifier(row_keys, raw_identifier)
                    if column in seen_ids:
                        identifier = _canonical_id(raw_identifier)
                        if identifier is not None and identifier in seen_ids[column]:
                            raise ValueError(f"duplicate {column} in {data_path}: {identifier}")
                        if identifier is not None:
                            seen_ids[column].add(identifier)
                if not row_keys:
                    rows_without_identity += 1
                if not row_has_image_path:
                    rows_without_image_path += 1
                keys.update(row_keys)
    return keys, row_count, rows_without_identity, rows_without_image_path, [str(path) for path in files]


def _path_exists(raw_path: str, roots: tuple[Path, ...]) -> bool:
    return _resolve_existing_path(raw_path, roots) is not None


def _collect_json_identity_keys(
    json_path: Path, image_roots: tuple[Path, ...]
) -> tuple[set[str], int, int]:
    payload = _json_loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("data", "samples", "rows"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"benchmark JSON must contain a list of samples: {json_path}")

    keys: set[str] = set()
    rows_without_identity = 0
    records: set[str] = set()
    seen_ids = {"sample_id": set(), "prompt_id": set()}
    roots = (json_path.parent, *image_roots)
    for row in payload:
        row_keys: set[str] = set()
        if isinstance(row, dict):
            record = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if record in records:
                raise ValueError(f"duplicate JSON record in {json_path}")
            records.add(record)
            raw_images = row.get("images")
            if raw_images is None:
                for field in ("image_path", "image", "path"):
                    if row.get(field) is not None:
                        raw_images = row[field]
                        break
            for item in _image_items(raw_images):
                row_keys.update(_identity_keys(item, roots))
            for column in ("prompt_id", "sample_id", "sequence_id", "id"):
                _add_identifier(row_keys, row.get(column))
                if column in seen_ids:
                    identifier = _canonical_id(row.get(column))
                    if identifier is not None and identifier in seen_ids[column]:
                        raise ValueError(f"duplicate {column} in {json_path}: {identifier}")
                    if identifier is not None:
                        seen_ids[column].add(identifier)
        if not row_keys:
            rows_without_identity += 1
        keys.update(row_keys)
    return keys, len(payload), rows_without_identity


def _collect_forbidden_identity_keys(
    paths: tuple[Path, ...], pq: Any, batch_size: int, image_roots: tuple[Path, ...]
) -> tuple[set[str], int, int, list[str]]:
    keys: set[str] = set()
    row_count = 0
    rows_without_identity = 0
    sources: list[str] = []
    for path in paths:
        parquet_files = _parquet_files(path)
        if parquet_files:
            path_keys, rows, no_identity, _, files = _collect_identity_keys(
                path, pq, batch_size, image_roots
            )
            keys.update(path_keys)
            row_count += rows
            rows_without_identity += no_identity
            sources.extend(files)
        elif path.is_file() and path.suffix.lower() == ".json":
            path_keys, rows, no_identity = _collect_json_identity_keys(path, image_roots)
            keys.update(path_keys)
            row_count += rows
            rows_without_identity += no_identity
            sources.append(str(path))
        else:
            raise ValueError(f"forbidden benchmark data is not readable parquet/JSON: {path}")
    return keys, row_count, rows_without_identity, sources


def _tree_sha256(path: Path) -> str:
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError(f"cannot hash missing tree: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"cannot hash symlinked tree component: {current}")
    if absolute.is_dir():
        for root, directories, filenames in os.walk(absolute, followlinks=False):
            for name in directories + filenames:
                candidate = Path(root) / name
                if candidate.is_symlink():
                    raise ValueError(f"cannot hash tree containing symlink: {candidate}")
    path = absolute
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    if path.is_file():
        return _file_sha256(str(path))
    if not files:
        raise ValueError(f"cannot hash empty tree: {path}")
    entries = [
        {
            "path": item.relative_to(path).as_posix(),
            "size": item.stat().st_size,
            "sha256": _file_sha256(str(item)),
        }
        for item in files
    ]
    encoded = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _phash(payload: bytes) -> str:
    """A real 32x32 DCT perceptual hash, implemented with Pillow + stdlib."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for formal perceptual hashing") from exc
    with Image.open(io.BytesIO(payload)) as image:
        pixels = list(image.convert("L").resize((32, 32)).getdata())
    # Low-frequency 8x8 two-dimensional DCT.  This intentionally avoids a fake
    # digest fallback when the optional imagehash package is unavailable.
    coeffs = []
    for u in range(8):
        for v in range(8):
            total = 0.0
            for x in range(32):
                cx = math.cos((2 * x + 1) * u * math.pi / 64)
                for y in range(32):
                    total += pixels[x * 32 + y] * cx * math.cos((2 * y + 1) * v * math.pi / 64)
            coeffs.append(total)
    median = sorted(coeffs[1:])[len(coeffs[1:]) // 2]
    return f"{sum((value > median) << index for index, value in enumerate(coeffs)):016x}"


def _phash_distance(left: str, right: str) -> int:
    """Return the 64-bit perceptual-hash Hamming distance."""
    if not re.fullmatch(r"[0-9a-f]{16}", left) or not re.fullmatch(r"[0-9a-f]{16}", right):
        raise ValueError("invalid perceptual hash")
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _mapping_key(value: str) -> str:
    """Normalize metadata/image keys without losing basename collision evidence."""
    normalized = unicodedata.normalize("NFKC", value).replace("\\", "/").strip()
    return Path(normalized).name.casefold()


def _metadata_entries(payload: Any) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    if isinstance(payload, dict) and all(isinstance(value, str) for value in payload.values()):
        return [(str(key), value) for key, value in payload.items()]
    if isinstance(payload, dict):
        payload = next((payload[key] for key in ("data", "samples", "rows", "metadata") if isinstance(payload.get(key), list)), payload)
    if not isinstance(payload, list):
        raise ValueError("mask metadata must be a list or basename-to-mask mapping")
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"mask metadata row {index} must be an object")
        image = next((row.get(key) for key in ("image", "image_path", "path", "basename", "file_name") if row.get(key)), None)
        mask = next((row.get(key) for key in ("mask", "mask_path", "masks", "segmentation") if row.get(key)), None)
        if isinstance(mask, list) and len(mask) == 1:
            mask = mask[0]
        if not isinstance(image, str) or not isinstance(mask, str):
            raise ValueError(f"mask metadata row {index} lacks image/mask path")
        entries.append((Path(image).name, mask))
    return entries


def _decode_image_evidence(payload: bytes, label: str) -> dict[str, Any]:
    try:
        from PIL import Image
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            width, height = image.size
        if width < 1 or height < 1:
            raise ValueError("zero-sized image")
        return {
            "label": label,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "phash": _phash(payload),
            "width": width,
            "height": height,
        }
    except Exception as exc:
        raise ValueError(f"unreadable image {label}: {exc}") from exc


def _collect_image_evidence(
    path: Path,
    pq: Any,
    batch_size: int,
    image_roots: tuple[Path, ...],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect one SHA+pHash record per image and fail closed on unreadable rows."""
    evidence: list[dict[str, Any]] = []
    errors: list[str] = []
    parquet_files = _parquet_files(path)
    if parquet_files:
        for parquet_path in parquet_files:
            parquet = pq.ParquetFile(parquet_path)
            available = set(parquet.schema_arrow.names)
            wanted = [name for name in ("images", "image", "image_path", "path", "prompt_id", "sample_id", "sequence_id", "id") if name in available]
            for batch in parquet.iter_batches(batch_size=batch_size, columns=wanted):
                values = batch.to_pydict()
                for index in range(batch.num_rows):
                    identity = next(
                        (str(values[name][index]) for name in ("prompt_id", "sample_id", "sequence_id", "id") if name in values and values[name][index] is not None),
                        f"{parquet_path}:{index}",
                    )
                    raw = values.get("images", [None] * batch.num_rows)[index]
                    if raw is None:
                        raw = next((values[name][index] for name in ("image", "image_path", "path") if name in values), None)
                    items = tuple(_image_items(raw))
                    if not items:
                        errors.append(f"image evidence missing for row {identity}")
                        continue
                    for item in items:
                        raw_path = _image_path(item)
                        embedded = _embedded_image_bytes(item)
                        resolved = _resolve_existing_path(
                            raw_path,
                            (path if path.is_dir() else path.parent, parquet_path.parent, *image_roots),
                        ) if raw_path else None
                        try:
                            payload = embedded if embedded is not None else (resolved.read_bytes() if resolved else None)
                        except OSError as exc:
                            errors.append(f"unreadable image {raw_path or identity}: {exc}")
                            continue
                        if payload is None:
                            errors.append(f"unreadable image {raw_path or identity}: bytes/path unavailable")
                            continue
                        label = Path(raw_path).name if raw_path else identity
                        try:
                            record = _decode_image_evidence(payload, label)
                            record["mapping_key"] = _mapping_key(label)
                            record["row_identity"] = identity
                            evidence.append(record)
                        except ValueError as exc:
                            errors.append(str(exc))
    elif path.is_file() and path.suffix.lower() == ".json":
        try:
            payload = _json_loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [], [f"forbidden benchmark JSON is unreadable: {path}: {exc}"]
        rows = payload if isinstance(payload, list) else next(
            (payload.get(key) for key in ("data", "samples", "rows") if isinstance(payload, dict) and isinstance(payload.get(key), list)),
            None,
        )
        if not isinstance(rows, list):
            return [], [f"benchmark JSON must contain a list of samples: {path}"]
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"benchmark row {index} is not an object")
                continue
            identity = next((str(row[name]) for name in ("prompt_id", "sample_id", "sequence_id", "id") if row.get(name) is not None), f"{path}:{index}")
            raw = row.get("images")
            if raw is None:
                raw = next((row.get(name) for name in ("image", "image_path", "path") if row.get(name) is not None), None)
            items = tuple(_image_items(raw))
            if not items:
                errors.append(f"image evidence missing for benchmark row {identity}")
                continue
            for item in items:
                raw_path = _image_path(item)
                embedded = _embedded_image_bytes(item)
                resolved = _resolve_existing_path(raw_path, (path.parent, *image_roots)) if raw_path else None
                try:
                    image_bytes = embedded if embedded is not None else (resolved.read_bytes() if resolved else None)
                except OSError as exc:
                    errors.append(f"unreadable image {raw_path or identity}: {exc}")
                    continue
                if image_bytes is None:
                    errors.append(f"unreadable image {raw_path or identity}: bytes/path unavailable")
                    continue
                label = Path(raw_path).name if raw_path else identity
                try:
                    record = _decode_image_evidence(image_bytes, label)
                    record["mapping_key"] = _mapping_key(label)
                    record["row_identity"] = identity
                    evidence.append(record)
                except ValueError as exc:
                    errors.append(str(exc))
    else:
        errors.append(f"image evidence source is not readable parquet/JSON: {path}")
    return evidence, errors


def _near_duplicate_pairs(
    left: list[dict[str, Any]], right: list[dict[str, Any]], threshold: int
) -> tuple[int, list[dict[str, Any]]]:
    count = 0
    examples: list[dict[str, Any]] = []
    for first in left:
        for second in right:
            distance = _phash_distance(first["phash"], second["phash"])
            if distance <= threshold:
                count += 1
                if len(examples) < 20:
                    examples.append({
                        "left": first["row_identity"], "right": second["row_identity"],
                        "left_sha256": first["sha256"], "right_sha256": second["sha256"],
                        "phash": first["phash"], "right_phash": second["phash"],
                        "hamming_distance": distance,
                    })
    return count, examples


def _validate_complete_publication(data_path: Path) -> list[str]:
    root = data_path if data_path.is_dir() else data_path.parent
    manifest_path = root / "selection_manifest.json"
    marker_path = root / "_COMPLETE.json"
    errors: list[str] = []
    if not manifest_path.is_file() or not marker_path.is_file():
        return ["formal frontier_rl requires selection_manifest.json and _COMPLETE.json"]
    try:
        manifest = _json_loads(manifest_path.read_text(encoding="utf-8"))
        marker = _json_loads(marker_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 3 or manifest.get("data_mode") != "frontier_rl" or manifest.get("publication") != "atomic_directory_v1":
            errors.append("frontier manifest publication contract is invalid")
        if manifest.get("classifier_thresholds") != FORMAL_CLASSIFIER_THRESHOLDS:
            errors.append("frontier manifest classifier thresholds are not frozen")
        if manifest.get("outcome_ratio") != .75 or manifest.get("process_ratio") != .25:
            errors.append("frontier manifest outcome/process composition must be 75/25")
        if manifest.get("bucket_ratios") != FORMAL_RATIOS:
            errors.append("frontier manifest selection ratios are not frozen")
        requested = manifest.get("requested_sample_count")
        selection = manifest.get("selection")
        if type(requested) is not int or requested < 1 or not isinstance(selection, dict):
            errors.append("frontier manifest selection count evidence is missing")
        else:
            raw_quotas = [requested * FORMAL_RATIOS[label] for label in BUCKET_LABELS]
            quotas = [math.floor(value) for value in raw_quotas]
            for index in sorted(range(5), key=lambda item: (-(raw_quotas[item] - quotas[item]), item))[: requested - sum(quotas)]:
                quotas[index] += 1
            for label, quota in zip(BUCKET_LABELS, quotas):
                row = selection.get(label)
                category_raw = [quota * .75, quota * .25]
                categories = [math.floor(value) for value in category_raw]
                for index in sorted(range(2), key=lambda item: (-(category_raw[item] - categories[item]), item))[: quota - sum(categories)]:
                    categories[index] += 1
                if not isinstance(row, dict) or row != {
                    "quota": quota, "outcome-frontier": categories[0], "process-hard": categories[1],
                }:
                    errors.append(f"frontier manifest selection/category quota mismatch for {label}")
        mining = manifest.get("mining_generation_contract")
        seeds = manifest.get("seeds")
        expected_mining_keys = {
            "model_checkpoint_path", "model_checkpoint_sha256", "generation_config_by_seed",
            "sampling_config", "seed_difference_allowlist", "source_binding_contract",
            "source_binding_sha256",
        }
        if not isinstance(mining, dict) or set(mining) != expected_mining_keys:
            errors.append("frontier manifest mining generation contract schema mismatch")
        elif not isinstance(seeds, list) or len(seeds) != 2 or mining.get("seed_difference_allowlist") != ["seed"]:
            errors.append("frontier manifest mining seeds/allowlist mismatch")
        else:
            if (
                not isinstance(mining.get("model_checkpoint_path"), str)
                or not Path(mining["model_checkpoint_path"]).is_absolute()
                or not isinstance(mining.get("model_checkpoint_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", mining["model_checkpoint_sha256"]) is None
                or mining.get("source_binding_contract") != "source_row_image_rendered_prompt_v1"
                or not isinstance(mining.get("source_binding_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", mining["source_binding_sha256"]) is None
            ):
                errors.append("frontier manifest mining model/source binding provenance is invalid")
            generation = mining.get("generation_config_by_seed")
            sampling = mining.get("sampling_config")
            normalized = []
            if not isinstance(generation, dict) or set(generation) != {str(seed) for seed in seeds}:
                errors.append("frontier manifest generation configs do not cover both seeds")
            else:
                for seed in seeds:
                    config = generation[str(seed)]
                    if not isinstance(config, dict) or str(config.get("seed")) != str(seed) or set(config) & {"generation_seed", "declared_seed"}:
                        errors.append("frontier manifest generation seed declaration mismatch")
                        break
                    normalized.append({key: value for key, value in config.items() if key != "seed"})
                if len(normalized) == 2 and normalized[0] != normalized[1]:
                    errors.append("frontier manifest generation config differs outside seed")
            if not isinstance(sampling, dict) or set(sampling) & {"seed", "generation_seed", "declared_seed"}:
                errors.append("frontier manifest sampling config must be seed-independent")
        hashes = marker.get("artifact_sha256")
        required = {"frontier_rl.parquet", "selected_ids.json", "selection_manifest.json"}
        ordered = ["frontier_rl.parquet", "selected_ids.json", "selection_manifest.json"]
        entries = list(root.iterdir())
        actual = {item.name for item in entries}
        invalid_entries = [
            item.name for item in entries
            if stat.S_ISLNK(os.lstat(item).st_mode)
            or not stat.S_ISREG(os.lstat(item).st_mode)
        ]
        if manifest.get("published_artifacts") != ordered or marker.get("published_artifacts") != ordered:
            errors.append("publication artifact enumeration is missing or invalid")
        if actual != required | {"_COMPLETE.json"} or invalid_entries:
            errors.append("publication contains missing or unenumerated artifacts")
        if not isinstance(hashes, dict) or set(hashes) != required:
            errors.append("complete marker must hash exactly every published artifact")
        else:
            for name in required:
                artifact_path = root / name
                if not artifact_path.is_file() or _tree_sha256(artifact_path) != hashes[name]:
                    errors.append(f"complete marker hash mismatch: {name}")
        manifest_hashes = manifest.get("artifact_sha256")
        if not isinstance(manifest_hashes, dict) or any(manifest_hashes.get(name) != hashes.get(name) for name in ("frontier_rl.parquet", "selected_ids.json")):
            errors.append("frontier manifest artifact hashes mismatch complete marker")
    except (OSError, json.JSONDecodeError, AttributeError, TypeError) as exc:
        errors.append(f"invalid complete publication evidence: {exc}")
    return errors


def _materialize_training_loader(
    data_path: Path, model_path: Path, *, max_prompt_length: int,
    max_pixels: int, min_pixels: int, filter_num_proc: int,
    format_prompt: Optional[Path], system_prompt: Optional[Path],
) -> dict[str, Any]:
    """Run the exact training dataset/filter and read every retained row."""
    if not model_path.is_dir():
        raise ValueError(f"RLHFDataset loader model path is missing: {model_path}")
    cache_root = Path(os.environ.get("V37_PREFLIGHT_CACHE_DIR", os.environ.get("TMPDIR", "/tmp"))) / "v37-hf-cache"
    datasets_cache = cache_root / "datasets"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from transformers import AutoProcessor, AutoTokenizer
        from verl.utils.dataset import RLHFDataset
    except ImportError as exc:
        raise RuntimeError("formal preflight requires transformers, datasets, torch, and the real RLHFDataset") from exc
    try:
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        loader = RLHFDataset(
            data_path=str(data_path), tokenizer=tokenizer, processor=processor,
            prompt_key="prompt", answer_key="answer", image_key="images",
            max_prompt_length=max_prompt_length, truncation="error",
            format_prompt=str(format_prompt) if format_prompt else None,
            system_prompt_file=str(system_prompt) if system_prompt else None,
            max_pixels=max_pixels, min_pixels=min_pixels,
            filter_overlong_prompts=True, filter_overlong_num_proc=filter_num_proc,
        )
    except Exception as exc:
        raise ValueError(f"real RLHFDataset construction/filter failed: {exc}") from exc

    observed: list[dict[str, Any]] = []
    buckets = {label: 0 for label in BUCKET_LABELS}
    ids: set[tuple[str, str]] = set()
    for index in range(len(loader)):
        try:
            raw = loader.dataset[index]
            sample_id = raw.get("sample_id")
            prompt_id = raw.get("prompt_id")
            if not isinstance(sample_id, str) or not sample_id.strip() or not isinstance(prompt_id, str) or not prompt_id.strip():
                raise ValueError("sample_id/prompt_id must be nonempty strings")
            identity = (sample_id.strip(), prompt_id.strip())
            if identity in ids:
                raise ValueError(f"duplicate retained identity {identity}")
            ids.add(identity)
            answer = _parse_answer_strict(raw.get("answer"))
            label = next((name for (low, high), name in zip(BUCKETS, BUCKET_LABELS) if answer is not None and low <= answer <= high), None)
            if label is None:
                raise ValueError(f"retained row {identity} has invalid answer")
            # __getitem__ is the actual point where image decoding, chat templating,
            # processor tokenization, truncation, and answer extraction occur.
            materialized = loader[index]
            if "input_ids" not in materialized or "ground_truth" not in materialized:
                raise ValueError("training loader output lacks input_ids/ground_truth")
            buckets[label] += 1
            observed.append({"sample_id": identity[0], "prompt_id": identity[1], "bucket": label})
        except Exception as exc:
            raise ValueError(f"training loader cannot read retained row {index}: {exc}") from exc
    observed.sort(key=lambda row: (row["sample_id"], row["prompt_id"]))
    ids_payload = [[row["sample_id"], row["prompt_id"]] for row in observed]
    return {
        "loader_contract": "verl.utils.dataset.RLHFDataset:v1",
        "loader_config": {
            "prompt_key": "prompt", "answer_key": "answer", "image_key": "images",
            "max_prompt_length": max_prompt_length, "truncation": "error",
            "max_pixels": max_pixels, "min_pixels": min_pixels,
            "filter_overlong_prompts": True, "filter_overlong_num_proc": filter_num_proc,
            "format_prompt": str(format_prompt.resolve()) if format_prompt else None,
            "system_prompt": str(system_prompt.resolve()) if system_prompt else None,
        },
        "sample_count": len(observed),
        "answer_buckets": buckets,
        "observed_ids_sha256": hashlib.sha256(
            json.dumps(ids_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "observed_rows_sha256": hashlib.sha256(
            json.dumps(observed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
    }


def _formal_contract(
    data_path: Path, metadata_path: Path, masks_dir: Path, pq: Any, batch_size: int,
    image_roots: tuple[Path, ...], data_mode: str, entrypoint_contract: Optional[str],
    training_entrypoint: Optional[Path], coverage_report: Optional[Path],
    filtered_manifest: Optional[Path], base_report: dict[str, Any],
    val_paths: tuple[Path, ...], forbidden_paths: tuple[Path, ...],
    phash_hamming_threshold: int, entrypoint_contract_path: Optional[Path],
    entrypoint_contract_sha256: Optional[str], expected_training_entrypoint_sha256: Optional[str],
    loader_model_path: Optional[Path], loader_max_prompt_length: int,
    loader_max_pixels: int, loader_min_pixels: int, loader_filter_num_proc: int,
    loader_format_prompt: Optional[Path], loader_system_prompt: Optional[Path],
) -> dict[str, Any]:
    errors: list[str] = []
    files = _parquet_files(data_path)
    dataset_hash = _tree_sha256(data_path)
    forbidden_training = CANDIDATE_ONLY_COLUMNS
    shard_schemas: dict[str, list[str]] = {}
    for path in files:
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        shard_schemas[str(path)] = sorted(columns)
        required_loader_columns = {"sample_id", "prompt_id", "prompt", "answer", "images"}
        if not required_loader_columns <= columns:
            errors.append(f"formal shard {path} is not RLHFDataset-consumable; missing {sorted(required_loader_columns - columns)}")
        if data_mode == "frontier_rl" and columns & forbidden_training:
            errors.append(f"frontier_rl shard {path} forbids response/trajectory training columns: " + ",".join(sorted(columns & forbidden_training)))
        if data_mode == "strict_winner_rft":
            required_winner_columns = {"response", "raw_success", "answer_exact"}
            if not required_winner_columns <= columns:
                errors.append(f"strict_winner_rft shard {path} requires response/raw_success/answer_exact columns")
            else:
                row_number = 0
                for batch in parquet.iter_batches(batch_size=batch_size, columns=["response", "raw_success", "answer_exact"]):
                    values = batch.to_pydict()
                    for response, raw_success, answer_exact in zip(values["response"], values["raw_success"], values["answer_exact"]):
                        if not isinstance(response, str) or not response.strip() or raw_success is not True or answer_exact is not True:
                            errors.append(f"strict_winner_rft row {path}:{row_number} must be an exact raw-success winner with nonempty response")
                        row_number += 1
    if entrypoint_contract != data_mode:
        errors.append(f"formal {data_mode} requires matching --entrypoint-contract")
    if training_entrypoint is None or not training_entrypoint.is_file():
        errors.append("formal training entrypoint must be a real file")
    elif not isinstance(expected_training_entrypoint_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_training_entrypoint_sha256):
        errors.append("formal training entrypoint requires an expected lowercase SHA256")
    elif _tree_sha256(training_entrypoint) != expected_training_entrypoint_sha256:
        errors.append("formal training entrypoint SHA256 mismatch")
    elif data_mode == "frontier_rl" and training_entrypoint.resolve() != (
        Path(__file__).resolve().parents[1] / "examples" / "v32_sparse_0_10_stable_drfix.sh"
    ).resolve():
        errors.append("frontier_rl requires the audited online-RL training entrypoint")
    # The explicit enum above is a supported contract.  If a contract document
    # is supplied, authenticate it; never infer compatibility from a filename.
    if entrypoint_contract_path is not None or entrypoint_contract_sha256 is not None:
        if entrypoint_contract_path is None or not entrypoint_contract_path.is_file():
            errors.append("entrypoint contract document is missing")
        elif not isinstance(entrypoint_contract_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", entrypoint_contract_sha256):
            errors.append("entrypoint contract document requires a SHA256")
        elif _tree_sha256(entrypoint_contract_path) != entrypoint_contract_sha256.lower():
            errors.append("entrypoint contract document SHA256 mismatch")

    if data_mode == "frontier_rl":
        errors.extend(_validate_complete_publication(data_path))
    if not val_paths:
        errors.append("formal requires nonempty --val-data")
    if not forbidden_paths:
        errors.append("formal requires nonempty --forbidden-data benchmark inputs")
    if base_report.get("validation_sample_count", 0) <= 0:
        errors.append("formal validation split must be nonempty")
    if base_report.get("forbidden_sample_count", 0) <= 0:
        errors.append("formal forbidden benchmark inputs must be nonempty")
    if base_report.get("train_validation_overlap_count"):
        errors.append("formal train/validation canonical identity or exact-SHA overlap")
    if base_report.get("train_benchmark_overlap_count"):
        errors.append("formal train/benchmark canonical identity or exact-SHA overlap")
    if base_report.get("validation_benchmark_overlap_count"):
        errors.append("formal validation/benchmark canonical identity or exact-SHA overlap")

    # Materialized real RLHFDataset filtering/row reads are authoritative.
    loader_observation: dict[str, Any] | None = None
    if loader_filter_num_proc != FORMAL_FILTER_OVERLONG_NUM_PROC:
        errors.append(
            f"formal loader_filter_num_proc must equal {FORMAL_FILTER_OVERLONG_NUM_PROC}"
        )
    elif loader_model_path is None:
        errors.append("formal requires --loader-model-path for real RLHFDataset materialization")
    else:
        try:
            loader_observation = _materialize_training_loader(
                data_path, loader_model_path,
                max_prompt_length=loader_max_prompt_length,
                max_pixels=loader_max_pixels, min_pixels=loader_min_pixels,
                filter_num_proc=loader_filter_num_proc,
                format_prompt=loader_format_prompt, system_prompt=loader_system_prompt,
            )
            if loader_observation["sample_count"] != base_report["sample_count"]:
                errors.append("real RLHFDataset overlength filter removed selected formal samples")
        except (ValueError, RuntimeError, OSError) as exc:
            errors.append(str(exc))
    if filtered_manifest is None or not filtered_manifest.is_file():
        errors.append("formal requires --filtered-manifest with expected post-RLHFDataset identities")
    else:
        try:
            filtered = _json_loads(filtered_manifest.read_text(encoding="utf-8"))
            expected_keys = {"schema_version", "dataset_sha256", "loader_contract", "loader_config", "sample_count", "answer_buckets", "observed_ids_sha256", "observed_rows_sha256"}
            if not isinstance(filtered, dict) or set(filtered) != expected_keys:
                errors.append("filtered manifest exact schema mismatch")
            elif filtered.get("schema_version") != 1 or filtered.get("dataset_sha256") != dataset_hash:
                errors.append("filtered manifest dataset/schema mismatch")
            elif loader_observation is not None and any(filtered.get(key) != loader_observation.get(key) for key in expected_keys - {"schema_version", "dataset_sha256"}):
                errors.append("filtered manifest differs from observed real RLHFDataset samples")
        except (OSError, json.JSONDecodeError, AttributeError, ValueError) as exc:
            errors.append(f"invalid filtered manifest: {exc}")

    metadata_hash = _tree_sha256(metadata_path) if metadata_path.is_file() else None
    mask_hash = _tree_sha256(masks_dir) if masks_dir.is_dir() else None
    if coverage_report is None or not coverage_report.is_file():
        errors.append("formal requires --coverage-report")
    else:
        try:
            coverage = _json_loads(coverage_report.read_text(encoding="utf-8"))
            required = {"dataset_sha256": dataset_hash, "metadata_sha256": metadata_hash, "mask_tree_sha256": mask_hash}
            for key, expected in required.items():
                if coverage.get(key) != expected:
                    errors.append(f"coverage report {key} mismatch")
            if coverage.get("coverage") != 1.0:
                errors.append("coverage report must certify coverage=1.0")
            if coverage.get("sample_count") != base_report["sample_count"]:
                errors.append("coverage report sample_count mismatch")
            if coverage.get("mapped_sample_count") != base_report["sample_count"]:
                errors.append("coverage report must prove every sample is mapped")
            for field in ("missing_sample_count", "ambiguous_sample_count", "dimension_mismatch_count", "unreadable_mask_count"):
                if coverage.get(field) != 0:
                    errors.append(f"coverage report must certify {field}=0")
            if coverage.get("mapping_key_normalization") != "nfkc_basename_casefold_v1":
                errors.append("coverage report mapping key normalization evidence missing")
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            errors.append(f"invalid coverage report: {exc}")

    mapping: dict[str, list[str]] = {}
    mask_dimensions: dict[str, tuple[int, int]] = {}
    try:
        for basename, mask in _metadata_entries(_json_loads(metadata_path.read_text(encoding="utf-8"))):
            mapping.setdefault(_mapping_key(basename), []).append(mask)
        ambiguous = sorted(name for name, values in mapping.items() if len(set(values)) != 1 or len(values) != 1)
        if ambiguous:
            errors.append(f"normalized basename ambiguity in metadata: {ambiguous[:10]}")
        for mapping_key, values in mapping.items():
            for raw in values:
                mask_path = (Path(raw) if Path(raw).is_absolute() else masks_dir / raw).resolve()
                masks_root = masks_dir.resolve()
                if mask_path != masks_root and masks_root not in mask_path.parents:
                    errors.append(f"mask path escapes masks_dir for {mapping_key}: {mask_path}")
                    continue
                if not mask_path.is_file():
                    errors.append(f"missing mask for {mapping_key}: {mask_path}")
                    continue
                try:
                    from PIL import Image
                    with Image.open(mask_path) as image:
                        image.load()
                        mask_dimensions[mapping_key] = image.size
                except Exception:
                    errors.append(f"mask is not decodable: {mask_path}")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"invalid mask metadata: {exc}")

    training_images, image_errors = _collect_image_evidence(data_path, pq, batch_size, image_roots)
    errors.extend(f"formal training {error}" for error in image_errors)
    validation_images: list[dict[str, Any]] = []
    benchmark_images: list[dict[str, Any]] = []
    for path in val_paths:
        found, found_errors = _collect_image_evidence(path, pq, batch_size, image_roots)
        validation_images.extend(found)
        errors.extend(f"formal validation {error}" for error in found_errors)
    for path in forbidden_paths:
        found, found_errors = _collect_image_evidence(path, pq, batch_size, image_roots)
        benchmark_images.extend(found)
        errors.extend(f"formal forbidden benchmark {error}" for error in found_errors)

    seen_sha: dict[str, str] = {}
    phash_to_sha: dict[str, set[str]] = {}
    basename_evidence: dict[str, set[str]] = {}
    mapped_by_split = {"training": 0, "validation": 0}
    for split, images in (("training", training_images), ("validation", validation_images)):
      for image in images:
        basename_evidence.setdefault(image["mapping_key"], set()).add(image["sha256"])
        key = image["mapping_key"]
        if key not in mapping:
            errors.append(f"{split} metadata mapping missing for normalized basename: {key}")
        elif key in mask_dimensions and mask_dimensions[key] != (image["width"], image["height"]):
            errors.append(f"{split} mask/image dimension mismatch for {image['label']}: mask={mask_dimensions[key]} image={(image['width'], image['height'])}")
        elif key in mask_dimensions:
            mapped_by_split[split] += 1
    ambiguous_images = sorted(key for key, hashes in basename_evidence.items() if len(hashes) != 1)
    if ambiguous_images:
        errors.append(f"normalized basename ambiguity in train/validation images: {ambiguous_images[:10]}")
    for image in training_images:
        if image["sha256"] in seen_sha:
            errors.append(f"duplicate image SHA256: {image['label']} and {seen_sha[image['sha256']]}")
        seen_sha[image["sha256"]] = image["label"]
        phash_to_sha.setdefault(image["phash"], set()).add(image["sha256"])
    phash_collisions = {value: sorted(hashes) for value, hashes in phash_to_sha.items() if len(hashes) > 1}
    if phash_collisions:
        errors.append(f"exact pHash collision/duplicate with distinct SHA256: {len(phash_collisions)}")

    train_val_near_count, train_val_examples = _near_duplicate_pairs(training_images, validation_images, phash_hamming_threshold)
    train_benchmark_near_count, train_benchmark_examples = _near_duplicate_pairs(training_images, benchmark_images, phash_hamming_threshold)
    val_benchmark_near_count, val_benchmark_examples = _near_duplicate_pairs(validation_images, benchmark_images, phash_hamming_threshold)
    if train_val_near_count:
        errors.append(f"train/validation perceptual near-duplicate overlap detected: {train_val_near_count}")
    if train_benchmark_near_count:
        errors.append(f"train/benchmark perceptual near-duplicate overlap detected: {train_benchmark_near_count}")
    if val_benchmark_near_count:
        errors.append(f"validation/benchmark perceptual near-duplicate overlap detected: {val_benchmark_near_count}")
    evidence_payload = [
        {"row_identity": item["row_identity"], "sha256": item["sha256"], "phash": item["phash"]}
        for item in training_images
    ]
    return {
        "dataset_sha256": dataset_hash,
        "metadata_sha256": metadata_hash,
        "mask_tree_sha256": mask_hash,
        "shard_schemas": shard_schemas,
        "loader_observation": loader_observation,
        "phash_hamming_threshold": phash_hamming_threshold,
        "image_phash_count": len(training_images),
        "validation_phash_count": len(validation_images),
        "benchmark_phash_count": len(benchmark_images),
        "training_mask_mapped_count": mapped_by_split["training"],
        "training_mask_mapping_coverage": mapped_by_split["training"] / len(training_images) if training_images else 0.0,
        "validation_mask_mapped_count": mapped_by_split["validation"],
        "validation_mask_mapping_coverage": mapped_by_split["validation"] / len(validation_images) if validation_images else 0.0,
        "phash_to_sha256_evidence_sha256": hashlib.sha256(json.dumps(evidence_payload, sort_keys=True).encode()).hexdigest(),
        "phash_to_sha256": {value: sorted(hashes) for value, hashes in sorted(phash_to_sha.items())},
        "phash_collision_count": len(phash_collisions),
        "train_validation_near_duplicate_count": train_val_near_count,
        "train_benchmark_near_duplicate_count": train_benchmark_near_count,
        "validation_benchmark_near_duplicate_count": val_benchmark_near_count,
        "near_duplicate_examples": (train_val_examples + train_benchmark_examples + val_benchmark_examples)[:20],
        "errors": errors,
    }


def _validate_eval_rows(
    data_path: Path,
    pq: Any,
    batch_size: int,
    image_roots: tuple[Path, ...],
    *,
    allow_answer_outside_train_range: bool,
    strict_numeric_answers: bool = False,
    require_decodable_images: bool = False,
) -> dict[str, Any]:
    counts = {label: 0 for label in BUCKET_LABELS}
    row_count = 0
    missing_answer_count = 0
    invalid_answer_count = 0
    invalid_image_count = 0
    data_root = data_path if data_path.is_dir() else data_path.parent
    for parquet_path in _parquet_files(data_path):
        parquet_file = pq.ParquetFile(parquet_path)
        available = set(parquet_file.schema_arrow.names)
        columns = [name for name in ("answer", "images", "image", "image_path", "path") if name in available]
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            values = batch.to_pydict()
            for row_index in range(batch.num_rows):
                row_count += 1
                raw_answer = values.get("answer", [None] * batch.num_rows)[row_index]
                if raw_answer is None or (isinstance(raw_answer, str) and not raw_answer.strip()):
                    missing_answer_count += 1
                else:
                    answer = _parse_answer_strict(raw_answer) if strict_numeric_answers else _parse_answer(raw_answer)
                    if answer is None or (
                        not allow_answer_outside_train_range and not 2 <= answer <= 50
                    ):
                        invalid_answer_count += 1
                    elif answer is not None:
                        for (low, high), label in zip(BUCKETS, BUCKET_LABELS):
                            if low <= answer <= high:
                                counts[label] += 1
                                break

                raw_images = values.get("images", [None] * batch.num_rows)[row_index]
                if raw_images is None:
                    raw_images = next((values[name][row_index] for name in ("image", "image_path", "path") if name in values), None)
                items = tuple(_image_items(raw_images))
                row_has_usable_image = False
                for item in items:
                    embedded = _embedded_image_bytes(item)
                    path = _image_path(item)
                    if embedded is not None:
                        row_has_usable_image = row_has_usable_image or _embedded_image_is_decodable(embedded)
                    elif path is not None:
                        resolved = _resolve_existing_path(path, (data_root, parquet_path.parent, *image_roots))
                        row_has_usable_image = row_has_usable_image or (
                            resolved is not None
                            and (not require_decodable_images or _image_file_is_decodable(resolved))
                        )
                if not row_has_usable_image:
                    invalid_image_count += 1
    return {
        "sample_count": row_count,
        "missing_answer_count": missing_answer_count,
        "invalid_answer_count": invalid_answer_count,
        "invalid_image_count": invalid_image_count,
        "answer_buckets": counts,
    }


def validate(
    data_path: Path,
    metadata_path: Path,
    masks_dir: Path,
    image_roots: tuple[Path, ...],
    expected_ratios: Optional[dict[str, float]],
    tolerance: float,
    batch_size: int,
    val_paths: tuple[Path, ...] = (),
    forbidden_paths: tuple[Path, ...] = (),
    allow_val_without_image_path: bool = False,
    allow_val_answer_outside_train_range: bool = False,
    require_val_all_buckets: bool = False,
    strict_numeric_answers: bool = False,
    require_decodable_images: bool = False,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only in incomplete environments
        raise RuntimeError("pyarrow is required; install the repository requirements before running preflight") from exc

    files = _parquet_files(data_path)
    errors: list[str] = []
    if not data_path.exists():
        errors.append(f"training data path does not exist: {data_path}")
    elif not files:
        errors.append(f"no parquet files found under: {data_path}")
    if not metadata_path.is_file():
        errors.append(f"mask metadata file does not exist: {metadata_path}")
    if not masks_dir.is_dir():
        errors.append(f"masks directory does not exist: {masks_dir}")
    for image_root in image_roots:
        if not image_root.is_dir():
            errors.append(f"image root directory does not exist: {image_root}")

    counts = {label: 0 for label in BUCKET_LABELS}
    sample_count = 0
    missing_answer_count = 0
    invalid_answer_count = 0
    rows_without_image_count = 0
    image_path_count = 0
    existing_image_path_count = 0
    embedded_image_count = 0
    invalid_embedded_image_count = 0
    usable_image_count = 0
    missing_image_payload_count = 0
    invalid_examples: list[dict[str, Any]] = []
    missing_image_examples: list[dict[str, Any]] = []
    invalid_embedded_examples: list[dict[str, Any]] = []
    training_identity_keys: set[str] = set()
    training_sample_ids: set[str] = set()
    training_prompt_ids: set[str] = set()
    duplicate_training_sample_ids: set[str] = set()
    duplicate_training_prompt_ids: set[str] = set()

    data_root = data_path if data_path.is_dir() else data_path.parent
    for parquet_path in files:
        parquet_file = pq.ParquetFile(parquet_path)
        available = set(parquet_file.schema_arrow.names)
        columns = [
            name
            for name in ("answer", "images", "image", "image_path", "path", "prompt_id", "id", "sample_id", "sequence_id")
            if name in available
        ]
        if not columns:
            rows = parquet_file.metadata.num_rows
            sample_count += rows
            missing_answer_count += rows
            rows_without_image_count += rows
            continue

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            values = batch.to_pydict()
            for row_index in range(batch.num_rows):
                absolute_index = sample_count
                sample_count += 1

                raw_answer = values["answer"][row_index] if "answer" in values else None
                if raw_answer is None or (isinstance(raw_answer, str) and not raw_answer.strip()):
                    missing_answer_count += 1
                else:
                    answer = _parse_answer_strict(raw_answer) if strict_numeric_answers else _parse_answer(raw_answer)
                    label = next(
                        (bucket_label for (low, high), bucket_label in zip(BUCKETS, BUCKET_LABELS) if answer is not None and low <= answer <= high),
                        None,
                    )
                    if label is None:
                        invalid_answer_count += 1
                        if len(invalid_examples) < 20:
                            invalid_examples.append(
                                {"file": str(parquet_path), "row": absolute_index, "answer": repr(raw_answer)[:200]}
                            )
                    else:
                        counts[label] += 1

                raw_images = values["images"][row_index] if "images" in values else None
                if raw_images is None:
                    raw_images = next((values[name][row_index] for name in ("image", "image_path", "path") if name in values), None)
                items = tuple(_image_items(raw_images))
                if not items:
                    rows_without_image_count += 1
                    if len(missing_image_examples) < 20:
                        missing_image_examples.append(
                            {"file": str(parquet_path), "row": absolute_index, "path": None}
                        )
                    continue
                found_path = False
                for item in items:
                    path = _image_path(item)
                    training_identity_keys.update(
                        _identity_keys(item, (data_root, parquet_path.parent, *image_roots))
                    )
                    if path is None:
                        continue
                    found_path = True
                    image_path_count += 1
                    embedded_payload = _embedded_image_bytes(item)
                    if embedded_payload is not None:
                        # HuggingFace Image parquet rows commonly preserve the
                        # basename for mask lookup while storing the actual image
                        # in bytes.  Such rows are self-contained and must not be
                        # rejected merely because the basename is not materialized.
                        embedded_image_count += 1
                        if _embedded_image_is_decodable(embedded_payload):
                            usable_image_count += 1
                        else:
                            invalid_embedded_image_count += 1
                            if len(invalid_embedded_examples) < 20:
                                invalid_embedded_examples.append(
                                    {"file": str(parquet_path), "row": absolute_index, "path": path}
                                )
                    else:
                        resolved = _resolve_existing_path(path, (data_root, parquet_path.parent, *image_roots))
                    if embedded_payload is None and resolved is not None and (
                        not require_decodable_images or _image_file_is_decodable(resolved)
                    ):
                        existing_image_path_count += 1
                        usable_image_count += 1
                    elif embedded_payload is None:
                        missing_image_payload_count += 1
                        if len(missing_image_examples) < 20:
                            missing_image_examples.append(
                                {"file": str(parquet_path), "row": absolute_index, "path": path}
                            )
                for column in ("prompt_id", "sample_id", "sequence_id", "id"):
                    _add_identifier(training_identity_keys, values.get(column, [None] * batch.num_rows)[row_index])
                for column, seen, duplicates in (
                    ("sample_id", training_sample_ids, duplicate_training_sample_ids),
                    ("prompt_id", training_prompt_ids, duplicate_training_prompt_ids),
                ):
                    identifier = _canonical_id(values.get(column, [None] * batch.num_rows)[row_index])
                    if identifier is not None:
                        if identifier in seen:
                            duplicates.add(identifier)
                        seen.add(identifier)
                if not found_path:
                    rows_without_image_count += 1
                    if len(missing_image_examples) < 20:
                        missing_image_examples.append(
                            {"file": str(parquet_path), "row": absolute_index, "path": None}
                        )

    valid_answer_count = sum(counts.values())
    ratios = {
        label: (counts[label] / valid_answer_count if valid_answer_count else 0.0) for label in BUCKET_LABELS
    }
    ratio_failures: dict[str, dict[str, float]] = {}
    if expected_ratios is not None:
        for label in BUCKET_LABELS:
            delta = abs(ratios[label] - expected_ratios[label])
            if delta > tolerance:
                ratio_failures[label] = {
                    "actual": ratios[label],
                    "expected": expected_ratios[label],
                    "delta": delta,
                }

    if sample_count == 0:
        errors.append("training data contains zero samples")
    if missing_answer_count:
        errors.append(f"{missing_answer_count} samples have a missing answer")
    if invalid_answer_count:
        errors.append(f"{invalid_answer_count} samples have an invalid or out-of-range answer")
    if rows_without_image_count:
        errors.append(f"{rows_without_image_count} samples have no usable image path")
    if missing_image_payload_count:
        errors.append(
            f"{missing_image_payload_count} images have neither embedded bytes nor an existing path"
        )
    if invalid_embedded_image_count:
        errors.append(f"{invalid_embedded_image_count} embedded images are not decodable")
    if duplicate_training_sample_ids:
        errors.append(f"duplicate training sample_id values: {len(duplicate_training_sample_ids)}")
    if duplicate_training_prompt_ids:
        errors.append(f"duplicate training prompt_id values: {len(duplicate_training_prompt_ids)}")
    if ratio_failures:
        errors.append(f"{len(ratio_failures)} answer buckets exceed ratio tolerance {tolerance}")

    val_identity_keys: set[str] = set()
    val_identity_rows = 0
    val_rows_without_identity = 0
    val_rows_without_image_path = 0
    val_parquet_files: list[str] = []
    val_content_reports: list[dict[str, Any]] = []
    val_bucket_counts = {label: 0 for label in BUCKET_LABELS}
    val_missing_answer_count = 0
    val_invalid_answer_count = 0
    val_invalid_image_count = 0
    for val_path in val_paths:
        val_files = _parquet_files(val_path)
        if not val_path.exists() or not val_files:
            errors.append(f"validation data has no readable parquet files: {val_path}")
            continue
        keys, rows, no_identity, no_image_path, path_files = _collect_identity_keys(
            val_path, pq, batch_size, image_roots
        )
        val_identity_keys.update(keys)
        val_identity_rows += rows
        val_rows_without_identity += no_identity
        val_rows_without_image_path += no_image_path
        val_parquet_files.extend(path_files)
        content_report = _validate_eval_rows(
            val_path,
            pq,
            batch_size,
            image_roots,
            allow_answer_outside_train_range=allow_val_answer_outside_train_range,
            strict_numeric_answers=strict_numeric_answers,
            require_decodable_images=require_decodable_images,
        )
        val_content_reports.append({"path": str(val_path), **content_report})
        val_missing_answer_count += content_report["missing_answer_count"]
        val_invalid_answer_count += content_report["invalid_answer_count"]
        val_invalid_image_count += content_report["invalid_image_count"]
        for label in BUCKET_LABELS:
            val_bucket_counts[label] += content_report["answer_buckets"][label]
    if val_paths and val_rows_without_identity:
        errors.append(f"{val_rows_without_identity} validation samples have no stable identity for leakage checks")
    if val_paths and val_rows_without_image_path and not allow_val_without_image_path:
        errors.append(
            f"{val_rows_without_image_path} validation samples have no image path for mask-aware reward"
        )
    if val_missing_answer_count:
        errors.append(f"{val_missing_answer_count} validation samples have a missing answer")
    if val_invalid_answer_count:
        errors.append(f"{val_invalid_answer_count} validation samples have an invalid answer")
    if val_invalid_image_count:
        errors.append(f"{val_invalid_image_count} validation samples have no decodable image payload")
    if require_val_all_buckets:
        missing_buckets = [label for label, count in val_bucket_counts.items() if count == 0]
        if missing_buckets:
            errors.append(
                "validation data must cover every answer bucket; missing=" + ",".join(missing_buckets)
            )
    identity_overlap = sorted(training_identity_keys & val_identity_keys)
    if identity_overlap:
        errors.append(f"train/validation identity overlap detected: {len(identity_overlap)} keys")

    forbidden_identity_keys: set[str] = set()
    forbidden_identity_rows = 0
    forbidden_rows_without_identity = 0
    forbidden_sources: list[str] = []
    if forbidden_paths:
        try:
            (
                forbidden_identity_keys,
                forbidden_identity_rows,
                forbidden_rows_without_identity,
                forbidden_sources,
            ) = _collect_forbidden_identity_keys(
                forbidden_paths, pq, batch_size, image_roots
            )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            errors.append(str(exc))
    if forbidden_paths and forbidden_rows_without_identity:
        errors.append(
            f"{forbidden_rows_without_identity} forbidden benchmark samples have no stable identity"
        )
    if require_decodable_images:
        for forbidden_path in forbidden_paths:
            _, decode_errors = _collect_image_evidence(
                forbidden_path, pq, batch_size, image_roots
            )
            errors.extend(f"non-debug forbidden benchmark {error}" for error in decode_errors)
    forbidden_overlap = sorted(training_identity_keys & forbidden_identity_keys)
    if forbidden_overlap:
        errors.append(f"train/benchmark identity overlap detected: {len(forbidden_overlap)} keys")
    validation_forbidden_overlap = sorted(val_identity_keys & forbidden_identity_keys)
    if validation_forbidden_overlap:
        errors.append(
            f"validation/benchmark identity overlap detected: {len(validation_forbidden_overlap)} keys"
        )

    image_rate_denominator = usable_image_count + missing_image_payload_count
    return {
        "ok": not errors,
        "data_path": str(data_path),
        "parquet_files": [str(path) for path in files],
        "metadata_path": str(metadata_path),
        "masks_dir": str(masks_dir),
        # Existence is checked here; sample-to-metadata/mask matching must be
        # certified by the dataset build report before a V37 launch.
        "metadata_coverage": "not_checked",
        "image_roots": [str(path) for path in image_roots],
        "sample_count": sample_count,
        "answer_buckets": {
            label: {"count": counts[label], "ratio": ratios[label]} for label in BUCKET_LABELS
        },
        "valid_answer_count": valid_answer_count,
        "missing_answer_count": missing_answer_count,
        "invalid_answer_count": invalid_answer_count,
        "image_path_count": image_path_count,
        "existing_image_path_count": existing_image_path_count,
        "embedded_image_count": embedded_image_count,
        "invalid_embedded_image_count": invalid_embedded_image_count,
        "usable_image_count": usable_image_count,
        "missing_image_payload_count": missing_image_payload_count,
        # Compatibility alias for consumers of the first preflight draft.
        "missing_image_path_count": missing_image_payload_count,
        "rows_without_image_count": rows_without_image_count,
        "image_payload_available_rate": (
            usable_image_count / image_rate_denominator if image_rate_denominator else 0.0
        ),
        "image_path_existence_rate": (
            existing_image_path_count / image_path_count if image_path_count else 0.0
        ),
        "expected_ratios": expected_ratios,
        "ratio_tolerance": tolerance,
        "ratio_failures": ratio_failures,
        "invalid_answer_examples": invalid_examples,
        "missing_image_examples": missing_image_examples,
        "invalid_embedded_examples": invalid_embedded_examples,
        "training_identity_key_count": len(training_identity_keys),
        "training_identity_sha256": hashlib.sha256(
            "\n".join(sorted(training_identity_keys)).encode("utf-8")
        ).hexdigest(),
        "validation_paths": [str(path) for path in val_paths],
        "validation_parquet_files": val_parquet_files,
        "validation_sample_count": val_identity_rows,
        "validation_rows_without_identity": val_rows_without_identity,
        "validation_rows_without_image_path": val_rows_without_image_path,
        "validation_image_path_requirement_relaxed": allow_val_without_image_path,
        "validation_answer_range_relaxed": allow_val_answer_outside_train_range,
        "validation_all_buckets_required": require_val_all_buckets,
        "validation_content_reports": val_content_reports,
        "validation_missing_answer_count": val_missing_answer_count,
        "validation_invalid_answer_count": val_invalid_answer_count,
        "validation_invalid_image_count": val_invalid_image_count,
        "validation_answer_buckets": val_bucket_counts,
        "validation_identity_key_count": len(val_identity_keys),
        "validation_identity_sha256": hashlib.sha256(
            "\n".join(sorted(val_identity_keys)).encode("utf-8")
        ).hexdigest(),
        "train_validation_overlap_count": len(identity_overlap),
        "train_validation_overlap_examples": identity_overlap[:20],
        "forbidden_paths": [str(path) for path in forbidden_paths],
        "forbidden_sources": forbidden_sources,
        "forbidden_sample_count": forbidden_identity_rows,
        "forbidden_rows_without_identity": forbidden_rows_without_identity,
        "forbidden_identity_key_count": len(forbidden_identity_keys),
        "forbidden_identity_sha256": hashlib.sha256(
            "\n".join(sorted(forbidden_identity_keys)).encode("utf-8")
        ).hexdigest(),
        "train_benchmark_overlap_count": len(forbidden_overlap),
        "train_benchmark_overlap_examples": forbidden_overlap[:20],
        "validation_benchmark_overlap_count": len(validation_forbidden_overlap),
        "validation_benchmark_overlap_examples": validation_forbidden_overlap[:20],
        "errors": errors,
    }


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    parent_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _print_report(report: dict[str, Any]) -> None:
    print("[V37-preflight] data:", report["data_path"])
    print(
        f"[V37-preflight] samples={report['sample_count']} parquet_files={len(report['parquet_files'])} "
        f"valid_answers={report['valid_answer_count']} missing_answers={report['missing_answer_count']} "
        f"invalid_answers={report['invalid_answer_count']}"
    )
    distribution = " ".join(
        f"{label}={values['count']}({values['ratio']:.2%})" for label, values in report["answer_buckets"].items()
    )
    print(f"[V37-preflight] answer_distribution {distribution}")
    print(
        f"[V37-preflight] image_paths={report['image_path_count']} "
        f"disk={report['existing_image_path_count']} embedded={report['embedded_image_count']} "
        f"invalid_embedded={report['invalid_embedded_image_count']} "
        f"missing_payload={report['missing_image_payload_count']} "
        f"rows_without_path={report['rows_without_image_count']} "
        f"payload_rate={report['image_payload_available_rate']:.2%}"
    )
    print(
        f"[V37-preflight] leakage_check train_keys={report['training_identity_key_count']} "
        f"val_samples={report['validation_sample_count']} val_keys={report['validation_identity_key_count']} "
        f"val_without_image_path={report['validation_rows_without_image_path']} "
        f"overlap={report['train_validation_overlap_count']}"
    )
    print(
        f"[V37-preflight] validation_content missing_answers={report['validation_missing_answer_count']} "
        f"invalid_answers={report['validation_invalid_answer_count']} "
        f"invalid_images={report['validation_invalid_image_count']} "
        f"buckets={report['validation_answer_buckets']}"
    )
    print(
        f"[V37-preflight] benchmark_leakage forbidden_samples={report['forbidden_sample_count']} "
        f"forbidden_keys={report['forbidden_identity_key_count']} "
        f"rows_without_identity={report['forbidden_rows_without_identity']} "
        f"overlap={report['train_benchmark_overlap_count']}"
    )
    print(
        "[V37-preflight] metadata_coverage=not_checked "
        "(requires an external dataset-build coverage report)"
    )
    if report["errors"]:
        for error in report["errors"]:
            print(f"[V37-preflight][ERROR] {error}", file=sys.stderr)
    else:
        print("[V37-preflight] PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="parquet file or directory containing parquet shards")
    parser.add_argument("--metadata", type=Path, required=True, help="mask metadata JSON file")
    parser.add_argument("--masks-dir", type=Path, required=True, help="directory containing mask files")
    parser.add_argument(
        "--image-root",
        type=Path,
        action="append",
        default=[],
        help="additional root for relative image paths; may be specified more than once",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        action="append",
        default=[],
        help="held-out parquet file/directory checked for sample/image overlap; may be repeated",
    )
    parser.add_argument(
        "--allow-val-without-image-path",
        action="store_true",
        help="debug-only: allow answer-only validation rows that cannot use mask-aware reward",
    )
    parser.add_argument(
        "--allow-val-answer-outside-train-range",
        action="store_true",
        help="debug-only: allow parseable validation answers outside the V37 2-50 range",
    )
    parser.add_argument(
        "--require-val-all-buckets",
        action="store_true",
        help="require validation data to cover all five V37 answer buckets",
    )
    parser.add_argument(
        "--forbidden-data",
        type=Path,
        action="append",
        default=[],
        help="benchmark parquet/JSON checked for train leakage; may be repeated",
    )
    parser.add_argument(
        "--expected-ratios",
        help=(
            "five ratios for buckets 2-10,11-20,21-30,31-40,41-50; accepts comma-separated values, "
            "a JSON list/object, or bucket=value pairs"
        ),
    )
    parser.add_argument("--tolerance", type=float, default=0.05, help="maximum absolute per-bucket ratio delta")
    parser.add_argument("--batch-size", type=int, default=4096, help="streaming parquet batch size")
    parser.add_argument("--json-out", type=Path, help="write the complete report as JSON")
    parser.add_argument(
        "--data-mode", choices=("frontier_rl", "strict_winner_rft"),
        default=os.environ.get("V37_DATA_MODE"),
        help="V37_DATA_MODE contract; omitted preserves legacy dry-run behavior",
    )
    parser.add_argument(
        "--run-class", choices=("debug", "canary", "formal"),
        default=os.environ.get("V37_RUN_CLASS"),
        help="formal enables all fail-closed identity/mask/filtering contracts",
    )
    parser.add_argument("--entrypoint-contract", choices=("frontier_rl", "strict_winner_rft"))
    parser.add_argument("--training-entrypoint", type=Path)
    parser.add_argument("--entrypoint-contract-path", type=Path)
    parser.add_argument("--entrypoint-contract-sha256")
    parser.add_argument("--expected-training-entrypoint-sha256")
    parser.add_argument("--coverage-report", type=Path)
    parser.add_argument("--filtered-manifest", type=Path)
    parser.add_argument("--loader-model-path", type=Path)
    parser.add_argument("--loader-max-prompt-length", type=int, default=12000)
    parser.add_argument("--loader-max-pixels", type=int, default=12845056)
    parser.add_argument("--loader-min-pixels", type=int, default=262144)
    parser.add_argument("--loader-filter-num-proc", type=int, default=1)
    parser.add_argument("--loader-format-prompt", type=Path)
    parser.add_argument("--loader-system-prompt", type=Path)
    parser.add_argument("--phash-hamming-threshold", type=int, default=4)
    parser.add_argument("--expected-data-sha256")
    parser.add_argument("--expected-metadata-sha256")
    parser.add_argument("--expected-masks-sha256")
    parser.add_argument("--expected-val-sha256", action="append")
    parser.add_argument("--expected-forbidden-sha256", action="append")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.tolerance) or args.tolerance < 0 or args.tolerance > 1:
        parser.error("--tolerance must be between 0 and 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0 <= args.phash_hamming_threshold <= 64:
        parser.error("--phash-hamming-threshold must be between 0 and 64")
    if min(args.loader_max_prompt_length, args.loader_max_pixels, args.loader_min_pixels, args.loader_filter_num_proc) < 1:
        parser.error("loader length/pixel/process settings must be positive")
    if args.loader_min_pixels > args.loader_max_pixels:
        parser.error("--loader-min-pixels cannot exceed --loader-max-pixels")
    try:
        expected_ratios = _parse_expected_ratios(args.expected_ratios)
        formal_snapshots: dict[str, Any] | None = None
        formal_external_image_snapshots: dict[str, Any] | None = None
        formal_external_image_errors: list[str] = []
        if args.run_class == "formal":
            if args.tolerance != FORMAL_RATIO_TOLERANCE:
                raise ValueError(f"formal ratio tolerance is locked to {FORMAL_RATIO_TOLERANCE}")
            if args.phash_hamming_threshold != FORMAL_PHASH_HAMMING_THRESHOLD:
                raise ValueError(f"formal pHash Hamming threshold is locked to {FORMAL_PHASH_HAMMING_THRESHOLD}")
            if args.loader_filter_num_proc != FORMAL_FILTER_OVERLONG_NUM_PROC:
                raise ValueError(
                    "formal --loader-filter-num-proc is locked to "
                    f"{FORMAL_FILTER_OVERLONG_NUM_PROC}"
                )
            expected_hash_fields = {
                "data": args.expected_data_sha256,
                "metadata": args.expected_metadata_sha256,
                "masks": args.expected_masks_sha256,
            }
            if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) or set(value) == {"0"} for value in expected_hash_fields.values()):
                raise ValueError("formal requires expected SHA256 for data, metadata, and masks")
            if not args.val_data or not isinstance(args.expected_val_sha256, list) or len(args.expected_val_sha256) != len(args.val_data) or any(not re.fullmatch(r"[0-9a-f]{64}", value) or set(value) == {"0"} for value in args.expected_val_sha256):
                raise ValueError("formal requires one expected SHA256 per validation input")
            if not args.forbidden_data or not isinstance(args.expected_forbidden_sha256, list) or len(args.expected_forbidden_sha256) != len(args.forbidden_data) or any(not re.fullmatch(r"[0-9a-f]{64}", value) or set(value) == {"0"} for value in args.expected_forbidden_sha256):
                raise ValueError("formal requires one expected SHA256 per forbidden benchmark input")
            formal_snapshots = {
                "data": _tree_sha256(args.data),
                "metadata": _tree_sha256(args.metadata),
                "masks": _tree_sha256(args.masks_dir),
                "validation": [_tree_sha256(path) for path in args.val_data],
                "forbidden": [_tree_sha256(path) for path in args.forbidden_data],
            }
            expected_snapshot = {
                "data": args.expected_data_sha256.lower(),
                "metadata": args.expected_metadata_sha256.lower(),
                "masks": args.expected_masks_sha256.lower(),
                "validation": [value.lower() for value in args.expected_val_sha256],
                "forbidden": [value.lower() for value in args.expected_forbidden_sha256],
            }
            if formal_snapshots != expected_snapshot:
                raise ValueError("formal input SHA256 mismatch")
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError(
                    "pyarrow is required; install the repository requirements before running preflight"
                ) from exc
            formal_external_image_snapshots, formal_external_image_errors = (
                _collect_external_image_snapshots(
                    args.data, tuple(args.val_data), tuple(args.forbidden_data), pq,
                    args.batch_size, tuple(args.image_root),
                )
            )
        report = validate(
            data_path=args.data,
            metadata_path=args.metadata,
            masks_dir=args.masks_dir,
            image_roots=tuple(args.image_root),
            expected_ratios=expected_ratios,
            tolerance=args.tolerance,
            batch_size=args.batch_size,
            val_paths=tuple(args.val_data),
            forbidden_paths=tuple(args.forbidden_data),
            allow_val_without_image_path=args.allow_val_without_image_path,
            allow_val_answer_outside_train_range=args.allow_val_answer_outside_train_range,
            require_val_all_buckets=args.require_val_all_buckets,
            strict_numeric_answers=args.run_class in {"canary", "formal"},
            require_decodable_images=args.run_class in {"canary", "formal"},
        )
        report["data_mode"] = args.data_mode or "legacy"
        report["run_class"] = args.run_class or "legacy"
        if args.data_mode is not None and args.run_class != "formal":
            import pyarrow.parquet as pq
            for parquet_path in _parquet_files(args.data):
                parquet = pq.ParquetFile(parquet_path)
                mode_columns = set(parquet.schema_arrow.names)
                if args.data_mode == "frontier_rl" and mode_columns & CANDIDATE_ONLY_COLUMNS:
                    report["errors"].append(f"frontier_rl shard {parquet_path} forbids candidate response/audit columns: " + ",".join(sorted(mode_columns & CANDIDATE_ONLY_COLUMNS)))
                if args.data_mode == "strict_winner_rft":
                    required_winner_columns = {"response", "raw_success", "answer_exact"}
                    if not required_winner_columns <= mode_columns:
                        report["errors"].append(f"strict_winner_rft shard {parquet_path} requires response/raw_success/answer_exact columns")
                    else:
                        row_number = 0
                        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=["response", "raw_success", "answer_exact"]):
                            values = batch.to_pydict()
                            for response, raw_success, answer_exact in zip(values["response"], values["raw_success"], values["answer_exact"]):
                                if not isinstance(response, str) or not response.strip() or raw_success is not True or answer_exact is not True:
                                    report["errors"].append(f"strict_winner_rft row {parquet_path}:{row_number} must be an exact raw-success winner with nonempty response")
                                row_number += 1
            if args.data_mode == "strict_winner_rft" and args.entrypoint_contract != "strict_winner_rft":
                report["errors"].append("strict_winner_rft requires an explicit supported entrypoint contract")
            report["ok"] = not report["errors"]
        if args.run_class == "formal":
            if expected_ratios != FORMAL_RATIOS:
                report["errors"].append("formal expected ratios are locked to 0.40,0.10,0.20,0.20,0.10")
            if not args.require_val_all_buckets:
                report["errors"].append("formal requires --require-val-all-buckets")
            if args.allow_val_without_image_path or args.allow_val_answer_outside_train_range:
                report["errors"].append("formal forbids validation relaxation flags")
            if args.data_mode is None:
                report["errors"].append("formal requires --data-mode or V37_DATA_MODE")
            else:
                import pyarrow.parquet as pq
                formal = _formal_contract(
                    args.data, args.metadata, args.masks_dir, pq, args.batch_size,
                    tuple(args.image_root), args.data_mode, args.entrypoint_contract,
                    args.training_entrypoint, args.coverage_report, args.filtered_manifest, report,
                    tuple(args.val_data), tuple(args.forbidden_data),
                    args.phash_hamming_threshold, args.entrypoint_contract_path,
                    args.entrypoint_contract_sha256, args.expected_training_entrypoint_sha256,
                    args.loader_model_path, args.loader_max_prompt_length,
                    args.loader_max_pixels, args.loader_min_pixels, args.loader_filter_num_proc,
                    args.loader_format_prompt, args.loader_system_prompt,
                )
                assert formal_external_image_snapshots is not None
                formal["external_image_snapshots"] = formal_external_image_snapshots
                formal["errors"].extend(
                    f"formal external image snapshot {error}"
                    for error in formal_external_image_errors
                )
                after_external_snapshots, after_external_errors = (
                    _collect_external_image_snapshots(
                        args.data, tuple(args.val_data), tuple(args.forbidden_data), pq,
                        args.batch_size, tuple(args.image_root),
                    )
                )
                formal["errors"].extend(
                    f"formal external image snapshot {error}"
                    for error in after_external_errors
                    if error not in formal_external_image_errors
                )
                if after_external_snapshots != formal_external_image_snapshots:
                    formal["errors"].append(
                        "formal external image inputs mutated while preflight was reading them"
                    )
                report["formal_contract"] = formal
                report["errors"].extend(formal["errors"])
            assert formal_snapshots is not None
            after_snapshots = {
                "data": _tree_sha256(args.data),
                "metadata": _tree_sha256(args.metadata),
                "masks": _tree_sha256(args.masks_dir),
                "validation": [_tree_sha256(path) for path in args.val_data],
                "forbidden": [_tree_sha256(path) for path in args.forbidden_data],
            }
            if after_snapshots != formal_snapshots:
                report["errors"].append("formal input mutated while preflight was reading it")
            report["formal_input_snapshots"] = formal_snapshots
            report["ok"] = not report["errors"]
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"[V37-preflight][ERROR] {exc}", file=sys.stderr)
        return 2

    if args.json_out is not None:
        _write_json(args.json_out, report)
    _print_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
