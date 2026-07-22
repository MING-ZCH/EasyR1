#!/usr/bin/env python3
"""Build the 500-row, non-benchmark V37 diagnostic heldout split.

Source parquet is scanned in bounded record batches. Selection keeps only a
bounded heap per answer bucket, then streams selected rows back without
materializing a potentially multi-gigabyte row group.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import secrets
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


BUCKETS = ((2, 10), (11, 20), (21, 30), (31, 40), (41, 50))
LABELS = tuple(f"{low}-{high}" for low, high in BUCKETS)
QUOTAS = dict(zip(LABELS, (200, 50, 100, 100, 50)))
SELECTION_SALT = "v37-diagnostic-heldout:500:40-10-20-20-10:v2"
PARQUET_BATCH_SIZE = 8
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_SOURCE_ROWS = 1_000_000
MAX_BENCHMARK_ROWS = 100_000
MAX_EXACT_KEYS = 500_000
MAX_FOCUSED_ROWS = 100_000
DEFAULT_ROOT = Path(os.environ.get(
    "STEPCOUNT_DATA_ROOT",
    "/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset",
))


class BuildError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    source_index: int
    source_label: str
    parquet_path: Path
    source_file: str
    row_group: int
    row_group_local_idx: int
    local_idx: int
    answer: int
    bucket: str
    rank: int
    image_keys: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise BuildError(f"cannot hash empty input: {path}")
    entries = [{"path": item.relative_to(path).as_posix(), "size": item.stat().st_size,
                "sha256": sha256_file(item)} for item in files]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(6)}")
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parquet_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path.resolve()]
    if not path.is_dir():
        raise BuildError(f"source is not a parquet file/directory: {path}")
    files = sorted((path / "data").glob("*.parquet"))
    if not files:
        files = sorted(path.glob("*.parquet"))
    if not files:
        raise BuildError(f"no parquet files found under: {path}")
    return [item.resolve() for item in files]


def parse_answer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, dict):
        for name in ("answer", "count_number", "final_count", "N", "total_count"):
            if name in value:
                parsed = parse_answer(value[name])
                if parsed is not None:
                    return parsed
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    match = re.search(r"-?\d+", str(value))
    return int(match.group()) if match else None


def bucket_for(answer: int) -> str | None:
    return next((label for (low, high), label in zip(BUCKETS, LABELS) if low <= answer <= high), None)


def image_items(value: Any) -> Iterator[Any]:
    if value is None:
        return
    if isinstance(value, (str, os.PathLike, bytes, bytearray, memoryview, dict)):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from image_items(item)


def image_value(row: dict[str, Any]) -> Any:
    for name in ("images", "image", "image_path", "path", "file_name"):
        if row.get(name) is not None:
            return row[name]
    return None


def resolve_image(path_text: str, bases: Sequence[Path]) -> Path | None:
    requested = Path(os.path.expanduser(path_text))
    candidates = [requested] if requested.is_absolute() else [base / requested for base in bases]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def image_keys(value: Any, bases: Sequence[Path], *, require_exact: bool) -> tuple[str, ...]:
    keys: set[str] = set()
    for item in image_items(value):
        path_value: Any = None
        payload: Any = None
        exact_evidence = False
        if isinstance(item, (str, os.PathLike)):
            path_value = os.fspath(item)
        elif isinstance(item, (bytes, bytearray, memoryview)):
            payload = bytes(item)
        elif isinstance(item, dict):
            path_value = next((item.get(key) for key in ("path", "image_path", "file_name") if item.get(key)), None)
            payload = item.get("bytes")
        if isinstance(payload, (bytes, bytearray, memoryview)) and payload:
            keys.add("sha256:" + hashlib.sha256(bytes(payload)).hexdigest())
            exact_evidence = True
        if isinstance(path_value, (str, os.PathLike)) and os.fspath(path_value).strip():
            text = os.path.normpath(os.fspath(path_value).strip())
            keys.add("declared_path:" + text)
            resolved = resolve_image(text, bases)
            if resolved is not None:
                keys.add("resolved_path:" + str(resolved))
                keys.add("sha256:" + sha256_file(resolved))
                exact_evidence = True
        if require_exact and not exact_evidence:
            raise BuildError(f"image has neither embedded bytes nor a resolvable file: {path_value!r}")
    if not keys:
        raise BuildError("row has no usable image path/bytes")
    return tuple(sorted(keys))


def strict_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BuildError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    if path.stat().st_size > MAX_JSON_BYTES:
        raise BuildError(f"JSON exceeds bounded parser limit ({MAX_JSON_BYTES} bytes): {path}")
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(BuildError(f"non-finite JSON in {path}: {value}")),
    )


def focused_exclusions(
    path: Path,
    source_path: Path,
    source_files: Sequence[Path],
) -> dict[tuple[str, int], str]:
    payload = strict_json(path)
    rows = payload.get("selected_indices") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise BuildError("focused selection manifest requires nonempty selected_indices")
    if len(rows) > MAX_FOCUSED_ROWS:
        raise BuildError(f"focused selection exceeds bounded row limit: {len(rows)}")
    declared_source = payload.get("source_dir") if isinstance(payload, dict) else None
    if not isinstance(declared_source, str) or Path(declared_source).resolve() != source_path.resolve():
        raise BuildError("focused selection manifest source_dir does not bind the 11-50 source")
    source_row_count = sum(pq.ParquetFile(item).metadata.num_rows for item in source_files)
    if payload.get("source_rows") != source_row_count:
        raise BuildError("focused selection manifest source_rows does not match parquet metadata")
    source_names = {item.name for item in source_files}
    if len(source_names) != len(source_files):
        raise BuildError("focused source parquet basenames must be unique")
    result: dict[tuple[str, int], str] = {}
    sequence_ids: set[str] = set()
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("source_file"), str)
            or Path(row["source_file"]).name != row["source_file"]
            or row["source_file"] not in source_names
            or type(row.get("local_idx")) is not int
            or not isinstance(row.get("sequence_id"), str)
            or re.fullmatch(r"[0-9a-f]{64}", row["sequence_id"]) is None
        ):
            raise BuildError(f"invalid focused selected_indices[{index}]")
        if row["local_idx"] < 0:
            raise BuildError(f"negative focused local_idx at index {index}")
        key = (row["source_file"], row["local_idx"])
        if key in result:
            raise BuildError(f"duplicate focused source row binding: {key}")
        if row["sequence_id"] in sequence_ids:
            raise BuildError(f"duplicate focused image identity: {row['sequence_id']}")
        result[key] = row["sequence_id"]
        sequence_ids.add(row["sequence_id"])
    return result


def focused_image_identity(value: Any) -> tuple[set[str], set[str]]:
    content_sha256: set[str] = set()
    declared_stems: set[str] = set()
    for item in image_items(value):
        if isinstance(item, (bytes, bytearray, memoryview)):
            payload, path_value = bytes(item), None
        elif isinstance(item, dict):
            payload = item.get("bytes")
            path_value = next(
                (item.get(key) for key in ("path", "image_path", "file_name") if item.get(key)),
                None,
            )
        else:
            payload, path_value = None, item if isinstance(item, (str, os.PathLike)) else None
        if isinstance(payload, (bytes, bytearray, memoryview)) and payload:
            content_sha256.add(hashlib.sha256(bytes(payload)).hexdigest())
        if isinstance(path_value, (str, os.PathLike)) and os.fspath(path_value).strip():
            declared_stems.add(Path(os.fspath(path_value).strip()).stem)
    return content_sha256, declared_stems


def bind_focused_content(
    focused: dict[tuple[str, int], str],
    source_files: Sequence[Path],
) -> dict[tuple[str, int], str]:
    """Bind focused source coordinates/path IDs to authoritative embedded bytes."""
    result: dict[tuple[str, int], str] = {}
    for parquet_path in source_files:
        parquet = pq.ParquetFile(parquet_path)
        names = set(parquet.schema_arrow.names)
        columns = [name for name in ("images", "image", "image_path", "path", "file_name") if name in names]
        if not columns:
            raise BuildError(f"focused source parquet has no image column: {parquet_path}")
        local_offset = 0
        for row_group in range(parquet.metadata.num_row_groups):
            group_index = 0
            for batch in parquet.iter_batches(
                batch_size=PARQUET_BATCH_SIZE,
                row_groups=[row_group],
                columns=columns,
            ):
                for batch_index, row in enumerate(batch.to_pylist()):
                    local_idx = local_offset + group_index + batch_index
                    key = (parquet_path.name, local_idx)
                    sequence_id = focused.get(key)
                    if sequence_id is None:
                        continue
                    content_sha256, declared_stems = focused_image_identity(image_value(row))
                    if declared_stems != {sequence_id} or len(content_sha256) != 1:
                        raise BuildError(
                            "focused row sequence_id/path or embedded image binding differs: "
                            f"{parquet_path}:{local_idx}"
                        )
                    result[key] = next(iter(content_sha256))
                group_index += batch.num_rows
            local_offset += parquet.metadata.row_group(row_group).num_rows
    missing = sorted(set(focused) - set(result))
    if missing:
        raise BuildError(f"focused source coordinates were not resolved: {missing[:5]}")
    return result


def benchmark_rows(path: Path) -> Iterable[tuple[Any, Sequence[Path]]]:
    bases = (path.parent.resolve(), path.resolve() if path.is_dir() else path.parent.resolve())
    if path.suffix.lower() == ".json":
        payload = strict_json(path)
        if isinstance(payload, dict):
            payload = next((payload.get(key) for key in ("data", "samples", "items") if isinstance(payload.get(key), list)), None)
        if not isinstance(payload, list):
            raise BuildError(f"benchmark JSON must contain a row list: {path}")
        for row in payload:
            if not isinstance(row, dict):
                raise BuildError(f"benchmark JSON row is not an object: {path}")
            yield image_value(row), bases
        return
    for parquet_path in parquet_files(path):
        parquet = pq.ParquetFile(parquet_path)
        names = set(parquet.schema_arrow.names)
        columns = [name for name in ("images", "image", "image_path", "path", "file_name") if name in names]
        if not columns:
            raise BuildError(f"benchmark parquet has no image column: {parquet_path}")
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=columns):
            for row in batch.to_pylist():
                yield image_value(row), (parquet_path.parent, *bases)


def collect_benchmark_keys(paths: Sequence[Path], image_roots: Sequence[Path]) -> tuple[set[str], int]:
    keys: set[str] = set()
    rows = 0
    for path in paths:
        if not path.exists():
            raise BuildError(f"benchmark input is missing: {path}")
        for value, bases in benchmark_rows(path):
            rows += 1
            if rows > MAX_BENCHMARK_ROWS:
                raise BuildError(f"benchmark evidence exceeds bounded row limit: {rows}")
            keys.update(image_keys(value, (*image_roots, *bases), require_exact=True))
            if len(keys) > MAX_EXACT_KEYS:
                raise BuildError(f"benchmark exact evidence exceeds bounded key limit: {len(keys)}")
    if rows == 0 or not keys:
        raise BuildError("benchmark inputs contain no exact image evidence")
    return keys, rows


def rank_for(source_label: str, source_file: str, local_idx: int, keys: Sequence[str]) -> int:
    payload = f"{SELECTION_SALT}\0{source_label}\0{source_file}\0{local_idx}\0{'|'.join(keys)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def scan_sources(
    source_files: Sequence[tuple[str, list[Path]]], focused: dict[tuple[str, int], str],
    focused_content: dict[tuple[str, int], str], benchmark_keys: set[str], image_roots: Sequence[Path],
) -> tuple[list[Candidate], dict[str, int]]:
    heaps: dict[str, list[tuple[int, int, int, int, Candidate]]] = {label: [] for label in LABELS}
    seen_images: set[tuple[str, ...]] = set()
    focused_sha_keys = {f"sha256:{value}" for value in focused_content.values()}
    stats: Counter[str] = Counter()
    for source_index, (source_label, files) in enumerate(source_files):
        for parquet_path in files:
            parquet = pq.ParquetFile(parquet_path)
            names = set(parquet.schema_arrow.names)
            if "answer" not in names:
                raise BuildError(f"source parquet has no answer column: {parquet_path}")
            columns = [name for name in ("answer", "images", "image", "image_path", "path", "file_name") if name in names]
            local_offset = 0
            for row_group in range(parquet.metadata.num_row_groups):
                group_index = 0
                for batch in parquet.iter_batches(
                    batch_size=PARQUET_BATCH_SIZE,
                    row_groups=[row_group],
                    columns=columns,
                ):
                    for batch_index, row in enumerate(batch.to_pylist()):
                        row_group_local_idx = group_index + batch_index
                        stats["source_rows"] += 1
                        if stats["source_rows"] > MAX_SOURCE_ROWS:
                            raise BuildError(f"source rows exceed bounded scan limit: {stats['source_rows']}")
                        local_idx = local_offset + row_group_local_idx
                        focused_sequence_id = (
                            focused.get((parquet_path.name, local_idx))
                            if source_label == "11-50"
                            else None
                        )
                        if focused_sequence_id is not None:
                            if (parquet_path.name, local_idx) not in focused_content:
                                raise BuildError(f"focused content binding is missing: {parquet_path}:{local_idx}")
                            stats["excluded_focused"] += 1
                            continue
                        answer = parse_answer(row.get("answer"))
                        if answer is not None and (
                            (source_label == "0-10" and not 0 <= answer <= 10)
                            or (source_label == "11-50" and not 11 <= answer <= 50)
                        ):
                            raise BuildError(
                                f"{source_label} source contains answer {answer} outside its declared range: "
                                f"{parquet_path}:{local_idx}"
                            )
                        label = bucket_for(answer) if answer is not None else None
                        if label is None:
                            stats["excluded_outside_buckets"] += 1
                            continue
                        keys = image_keys(
                            image_value(row), (parquet_path.parent, *image_roots), require_exact=True,
                        )
                        if focused_sha_keys.intersection(keys):
                            stats["excluded_focused_content"] += 1
                            continue
                        if benchmark_keys.intersection(keys):
                            stats["excluded_benchmark_exact"] += 1
                            continue
                        dedupe_key = tuple(
                            key for key in keys if key.startswith(("sha256:", "resolved_path:"))
                        ) or keys
                        if dedupe_key in seen_images:
                            stats["excluded_source_duplicate"] += 1
                            continue
                        seen_images.add(dedupe_key)
                        rank = rank_for(source_label, parquet_path.name, local_idx, keys)
                        candidate = Candidate(
                            source_index, source_label, parquet_path, parquet_path.name,
                            row_group, row_group_local_idx, local_idx, answer, label, rank, keys,
                        )
                        heap = heaps[label]
                        entry = (-rank, -source_index, -local_idx, -row_group_local_idx, candidate)
                        if len(heap) < QUOTAS[label]:
                            heapq.heappush(heap, entry)
                        elif rank < -heap[0][0]:
                            heapq.heapreplace(heap, entry)
                    group_index += batch.num_rows
                expected_group_rows = parquet.metadata.row_group(row_group).num_rows
                if group_index != expected_group_rows:
                    raise BuildError(
                        f"streamed row count mismatch for {parquet_path} row_group={row_group}: "
                        f"{group_index} != {expected_group_rows}"
                    )
                local_offset += expected_group_rows
    missing = {label: QUOTAS[label] - len(heap) for label, heap in heaps.items() if len(heap) != QUOTAS[label]}
    if missing:
        raise BuildError(f"insufficient eligible rows for fixed quotas: {missing}")
    selected = [entry[-1] for label in LABELS for entry in heaps[label]]
    selected.sort(key=lambda item: (item.source_index, str(item.parquet_path), item.local_idx))
    stats["selected_rows"] = len(selected)
    return selected, dict(stats)


def write_selected(selected: Sequence[Candidate], path: Path) -> None:
    by_group: dict[tuple[Path, int], list[Candidate]] = {}
    for candidate in selected:
        by_group.setdefault((candidate.parquet_path, candidate.row_group), []).append(candidate)
    writer: pq.ParquetWriter | None = None
    expected_schema: pa.Schema | None = None
    try:
        for (parquet_path, row_group), candidates in sorted(
            by_group.items(), key=lambda item: (str(item[0][0]), item[0][1])
        ):
            parquet = pq.ParquetFile(parquet_path)
            wanted = {item.row_group_local_idx for item in candidates}
            emitted: set[int] = set()
            group_offset = 0
            for batch in parquet.iter_batches(
                batch_size=PARQUET_BATCH_SIZE,
                row_groups=[row_group],
            ):
                offsets = [
                    index - group_offset
                    for index in sorted(wanted)
                    if group_offset <= index < group_offset + batch.num_rows
                ]
                if offsets:
                    table = (
                        pa.Table.from_batches([batch])
                        .take(pa.array(offsets, type=pa.int64()))
                        .replace_schema_metadata(None)
                    )
                    if expected_schema is None:
                        expected_schema = table.schema
                        writer = pq.ParquetWriter(path, expected_schema, compression="zstd")
                    elif table.schema != expected_schema:
                        raise BuildError(
                            f"source schemas differ after metadata normalization: {parquet_path}"
                        )
                    assert writer is not None
                    writer.write_table(table)
                    emitted.update(group_offset + offset for offset in offsets)
                group_offset += batch.num_rows
            if emitted != wanted:
                raise BuildError(
                    f"failed to stream selected rows from {parquet_path} row_group={row_group}: "
                    f"missing={sorted(wanted - emitted)}"
                )
    finally:
        if writer is not None:
            writer.close()
    if expected_schema is None:
        raise BuildError("selection unexpectedly produced no rows")


def build(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise BuildError(f"refusing to overwrite output: {output}")
    source_paths = [args.source_0_10.resolve(), args.source_11_50.resolve()]
    source_files = [("0-10", parquet_files(source_paths[0])), ("11-50", parquet_files(source_paths[1]))]
    focused_path = args.focused_selection_manifest.resolve()
    focused = focused_exclusions(focused_path, source_paths[1], source_files[1][1])
    focused_content = bind_focused_content(focused, source_files[1][1])
    benchmark_paths = [path.resolve() for path in args.benchmark]
    source_hashes_before = {
        str(path): sha256_file(path) for _, files in source_files for path in files
    }
    focused_hash_before = sha256_file(focused_path)
    benchmark_hashes_before = {str(path): sha256_path(path) for path in benchmark_paths}
    image_roots = [path.resolve() for path in args.image_root]
    if any(not path.is_dir() for path in image_roots):
        raise BuildError("every --image-root must be an existing directory")
    benchmark_keys, benchmark_rows_count = collect_benchmark_keys(benchmark_paths, image_roots)
    selected, stats = scan_sources(
        source_files, focused, focused_content, benchmark_keys, image_roots,
    )
    if stats.get("excluded_focused", 0) != len(focused):
        raise BuildError(
            "focused selection manifest does not bind exactly to the 11-50 source: "
            f"matched={stats.get('excluded_focused', 0)} expected={len(focused)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        parquet_path = staging / "diagnostic_heldout.parquet"
        write_selected(selected, parquet_path)
        with parquet_path.open("rb") as handle:
            os.fsync(handle.fileno())
        if (
            source_hashes_before != {
                str(path): sha256_file(path) for _, files in source_files for path in files
            }
            or focused_hash_before != sha256_file(focused_path)
            or benchmark_hashes_before != {
                str(path): sha256_path(path) for path in benchmark_paths
            }
        ):
            raise BuildError("an input mutated while the heldout build was running")
        distribution = Counter(item.bucket for item in selected)
        manifest = {
            "schema_version": 2,
            "dataset_kind": "v37_nonbenchmark_diagnostic_heldout",
            "publication": "atomic_directory_v1",
            "promotable": False,
            "selection_salt": SELECTION_SALT,
            "parquet_batch_size": PARQUET_BATCH_SIZE,
            "source_paths": [str(path) for path in source_paths],
            "source_parquet_sha256": source_hashes_before,
            "focused_selection_manifest": str(focused_path),
            "focused_selection_manifest_sha256": focused_hash_before,
            "focused_sequence_key_count": len(set(focused.values())),
            "focused_sequence_keys_sha256": hashlib.sha256(
                json.dumps(sorted(set(focused.values())), separators=(",", ":")).encode()
            ).hexdigest(),
            "focused_exact_key_count": len(set(focused_content.values())),
            "focused_exact_duplicate_count": len(focused_content) - len(set(focused_content.values())),
            "focused_exact_keys_sha256": hashlib.sha256(
                json.dumps(sorted(set(focused_content.values())), separators=(",", ":")).encode()
            ).hexdigest(),
            "benchmark_paths": [str(path) for path in benchmark_paths],
            "benchmark_input_sha256": benchmark_hashes_before,
            "benchmark_evidence_rows": benchmark_rows_count,
            "benchmark_exact_key_count": len(benchmark_keys),
            "benchmark_exact_keys_sha256": hashlib.sha256(
                json.dumps(sorted(benchmark_keys), separators=(",", ":")).encode()
            ).hexdigest(),
            "benchmark_overlap_selected": 0,
            "quotas": QUOTAS,
            "selected_distribution": {label: distribution[label] for label in LABELS},
            "selected_rows": len(selected),
            "scan_stats": stats,
            "artifact_sha256": {"diagnostic_heldout.parquet": sha256_file(parquet_path)},
            "selected_indices": [
                {
                    "source": item.source_label, "source_file": item.source_file,
                    "local_idx": item.local_idx, "answer": item.answer,
                    "bucket": item.bucket, "selection_rank_sha256": f"{item.rank:064x}",
                    "image_exact_keys": list(item.image_keys),
                }
                for item in selected
            ],
        }
        atomic_json(staging / "selection_manifest.json", manifest)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output.exists() or output.is_symlink():
            raise BuildError(f"output appeared during build: {output}")
        os.replace(staging, output)
        return output
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-0-10", type=Path, default=DEFAULT_ROOT / "StepCountQA-RL-Traj_0_10")
    result.add_argument("--source-11-50", type=Path, default=DEFAULT_ROOT / "StepCountQA-RL-Traj_11_50_Combined_maskcomplete_38332")
    result.add_argument(
        "--focused-selection-manifest", type=Path,
        default=DEFAULT_ROOT / "StepCountQA-RL-Traj_11_30_Focused10k_quality_maskcomplete_v36_20260713/selection_manifest.json",
    )
    result.add_argument("--benchmark", type=Path, action="append", required=True, help="repeat for every forbidden benchmark")
    result.add_argument("--image-root", type=Path, action="append", default=[])
    result.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "StepCountQA-RL-Diagnostic-Heldout-v37")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        output = build(args)
    except (BuildError, OSError, json.JSONDecodeError) as exc:
        print(f"[V37-diagnostic-builder][ERROR] {exc}", file=os.sys.stderr)
        return 2
    print(f"[V37-diagnostic-builder] published={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
