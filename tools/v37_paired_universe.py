#!/usr/bin/env python3
"""Build the preregistered V37 held-out sample universe from parquet data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

try:
    from tools import v37_gate as gate
except ModuleNotFoundError:  # Direct execution from tools/.
    import v37_gate as gate


class UniverseError(ValueError):
    """The held-out dataset cannot produce an unambiguous formal universe."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise UniverseError("sample universe contains non-canonical JSON") from exc


def _absolute_nonsymlink(path: os.PathLike[str] | str, label: str) -> Path:
    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise UniverseError(f"{label} does not exist: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise UniverseError(f"{label} contains a symlink component: {current}")
    return absolute


def _identifier(value: Any, label: str) -> str:
    try:
        return gate._paired_source_identifier(value, label)
    except gate.GateError as exc:
        raise UniverseError(str(exc)) from exc


def _answer(value: Any, label: str) -> int:
    try:
        return gate._paired_source_answer(value, label)
    except gate.GateError as exc:
        raise UniverseError(str(exc)) from exc


def build(
    *,
    data: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
) -> dict[str, Any]:
    output_path = Path(os.path.abspath(os.path.expanduser(os.fspath(output))))
    if output_path.exists() or output_path.is_symlink():
        raise UniverseError(f"output already exists: {output_path}")
    data_path = _absolute_nonsymlink(data, "held-out dataset")
    parent = _absolute_nonsymlink(output_path.parent, "output parent")
    if not parent.is_dir():
        raise UniverseError("output parent must be a directory")
    for root, directories, filenames in os.walk(data_path, followlinks=False):
        for name in directories + filenames:
            if (Path(root) / name).is_symlink():
                raise UniverseError("held-out dataset tree contains a symlink")
    data_snapshot_before = gate.content_snapshot(data_path, parquet_only=True)
    try:
        rows = gate._recompute_paired_source_rows(data_path)
    except gate.GateError as exc:
        raise UniverseError(str(exc)) from exc
    bucket_counts = {
        bucket: sum(row["bucket"] == bucket for row in rows) for bucket in gate.BUCKETS
    }
    if any(value == 0 for value in bucket_counts.values()):
        raise UniverseError("formal held-out universe must cover all five answer buckets")
    data_snapshot_after = gate.content_snapshot(data_path, parquet_only=True)
    if gate.canonical(data_snapshot_before) != gate.canonical(data_snapshot_after):
        raise UniverseError("held-out dataset mutated while building the sample universe")
    data_hash = data_snapshot_before["sha256"]
    payload = {
        "schema_version": gate.PAIRED_UNIVERSE_SCHEMA_VERSION,
        "artifact_type": gate.PAIRED_UNIVERSE_ARTIFACT_TYPE,
        "data_path": str(data_path),
        "data_sha256": data_hash,
        "sample_count": len(rows),
        "bucket_counts": bucket_counts,
        "ordered_rows_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
        "rows": rows,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publication is atomic and fails if another writer creates
        # the preregistration path after the initial existence check.
        os.link(temporary, output_path)
        os.unlink(temporary)
        parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = build(
            data=args.data,
            output=args.output,
        )
    except (OSError, UniverseError, gate.GateError) as exc:
        print(f"[V37-paired-universe][ERROR] {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "sample_count": payload["sample_count"],
        "ordered_rows_sha256": payload["ordered_rows_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
