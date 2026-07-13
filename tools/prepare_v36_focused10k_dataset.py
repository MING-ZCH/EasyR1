#!/usr/bin/env python3
"""Build a deterministic 10k V36 dense-training subset focused on answers 11-30."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_LOCAL_ROOT = Path(os.environ.get("LOCAL_ROOT", "/mnt/shared-storage-user/zhangchenhao"))
DEFAULT_DATA_ROOT = Path(
    os.environ.get("STEPCOUNT_DATA_ROOT", DEFAULT_LOCAL_ROOT / "work/StepcountModel/dataset")
)
DEFAULT_SOURCE = str(DEFAULT_DATA_ROOT / "StepCountQA-RL-Traj_11_50_Combined_maskcomplete_38332")
DEFAULT_OUTPUT = str(DEFAULT_DATA_ROOT / "StepCountQA-RL-Traj_11_30_Focused10k_maskcomplete_v36_20260713")
HASH_SALT = "v36-focused10k:maskcomplete:20260713"


def parse_answer(value: object) -> int:
    match = re.search(r"-?\d+", str(value))
    if not match:
        raise ValueError(f"cannot parse answer from {value!r}")
    return int(match.group())


def answer_quota(answer: int) -> int:
    if 11 <= answer <= 20:
        return 400
    if 21 <= answer <= 30:
        return 450
    if 31 <= answer <= 40:
        return 100
    if 41 <= answer <= 50:
        return 50
    raise ValueError(f"answer {answer} outside 11-50")


def bin_label(answer: int) -> str:
    for low, high in ((11, 20), (21, 30), (31, 40), (41, 50)):
        if low <= answer <= high:
            return f"{low}-{high}"
    raise ValueError(f"answer {answer} outside 11-50")


def stable_key(row: dict[str, object]) -> str:
    payload = (
        f"{HASH_SALT}:file={row['source_file']}:local={row['local_idx']}:"
        f"answer={row['answer']}:problem={row['problem']}"
    ).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def discover_rows(files: Iterable[Path]) -> tuple[list[dict[str, object]], Counter[int], Counter[str]]:
    rows: list[dict[str, object]] = []
    answer_counts: Counter[int] = Counter()
    bin_counts: Counter[str] = Counter()
    global_idx = 0
    for file_idx, path in enumerate(files):
        table = pq.read_table(path, columns=["problem", "answer"])
        for local_idx in range(table.num_rows):
            answer = parse_answer(table["answer"][local_idx].as_py())
            problem = table["problem"][local_idx].as_py()
            row = {
                "global_idx": global_idx,
                "file_idx": file_idx,
                "source_file": path.name,
                "local_idx": local_idx,
                "answer": answer,
                "bin": bin_label(answer),
                "problem": problem,
            }
            rows.append(row)
            answer_counts[answer] += 1
            bin_counts[row["bin"]] += 1
            global_idx += 1
    return rows, answer_counts, bin_counts


def select_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_answer: Dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_answer[int(row["answer"])].append(row)

    selected: list[dict[str, object]] = []
    for answer in range(11, 51):
        candidates = sorted(by_answer[answer], key=stable_key)
        quota = answer_quota(answer)
        if len(candidates) < quota:
            raise RuntimeError(f"answer={answer} has {len(candidates)} rows, quota={quota}")
        selected.extend(candidates[:quota])
    return sorted(selected, key=lambda r: int(r["global_idx"]))


def write_selected_dataset(files: list[Path], selected: list[dict[str, object]], output_dir: Path, rows_per_shard: int) -> None:
    output_data = output_dir / "data"
    output_data.mkdir(parents=True, exist_ok=True)
    for stale in output_data.glob("train-*.parquet"):
        stale.unlink()

    selected_by_file: Dict[int, list[int]] = defaultdict(list)
    for row in selected:
        selected_by_file[int(row["file_idx"])].append(int(row["local_idx"]))

    writer: pq.ParquetWriter | None = None
    shard_idx = 0
    rows_in_shard = 0
    shard_paths: list[Path] = []

    def open_writer(schema: pa.Schema) -> pq.ParquetWriter:
        nonlocal shard_idx
        shard_path = output_data / f"train-{shard_idx:05d}-of-00000.parquet"
        shard_paths.append(shard_path)
        shard_idx += 1
        return pq.ParquetWriter(shard_path, schema=schema, compression="zstd")

    for file_idx, path in enumerate(files):
        local_indices = sorted(selected_by_file.get(file_idx, []))
        if not local_indices:
            continue
        parquet_file = pq.ParquetFile(path)
        row_group_start = 0
        cursor = 0
        for row_group_idx in range(parquet_file.metadata.num_row_groups):
            row_group_rows = parquet_file.metadata.row_group(row_group_idx).num_rows
            row_group_end = row_group_start + row_group_rows
            group_indices: list[int] = []
            while cursor < len(local_indices) and local_indices[cursor] < row_group_end:
                if local_indices[cursor] >= row_group_start:
                    group_indices.append(local_indices[cursor] - row_group_start)
                cursor += 1
            row_group_start = row_group_end
            if not group_indices:
                continue
            table = parquet_file.read_row_group(row_group_idx)
            selected_table = table.take(pa.array(group_indices, type=pa.int64()))
            offset = 0
            while offset < selected_table.num_rows:
                if writer is None:
                    writer = open_writer(selected_table.schema)
                    rows_in_shard = 0
                take = min(rows_per_shard - rows_in_shard, selected_table.num_rows - offset)
                writer.write_table(selected_table.slice(offset, take))
                rows_in_shard += take
                offset += take
                if rows_in_shard >= rows_per_shard:
                    writer.close()
                    writer = None
        if cursor != len(local_indices):
            raise RuntimeError(f"not all selected rows were written for {path}: {cursor}/{len(local_indices)}")
    if writer is not None:
        writer.close()

    total_shards = len(shard_paths)
    for idx, old_path in enumerate(shard_paths):
        new_path = output_data / f"train-{idx:05d}-of-{total_shards:05d}.parquet"
        old_path.rename(new_path)


def counter_to_sorted_dict(counter: Counter[int] | Counter[str]) -> dict[str, int]:
    return {str(k): int(counter[k]) for k in sorted(counter, key=lambda item: int(item) if str(item).isdigit() else str(item))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--rows-per-shard", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    files = sorted((source_dir / "data").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {source_dir / 'data'}")

    rows, source_answer_counts, source_bin_counts = discover_rows(files)
    selected = select_rows(rows)
    selected_answer_counts = Counter(int(row["answer"]) for row in selected)
    selected_bin_counts = Counter(str(row["bin"]) for row in selected)

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "algorithm": "deterministic answer-stratified hash sample from mask-complete rows",
        "hash_salt": HASH_SALT,
        "target_rows": 10000,
        "selected_rows": len(selected),
        "source_rows": len(rows),
        "quality_gate": (
            "source dataset is already mask-complete: GT trajectory turns exactly 0..answer-1, "
            "count equals answer, no duplicate/extra turns, and mask files exist"
        ),
        "quota_by_range": {
            "11-20": 4000,
            "21-30": 4500,
            "31-40": 1000,
            "41-50": 500,
        },
        "quota_by_answer": {str(answer): answer_quota(answer) for answer in range(11, 51)},
        "source_answer_distribution": counter_to_sorted_dict(source_answer_counts),
        "source_bin_distribution": counter_to_sorted_dict(source_bin_counts),
        "selected_answer_distribution": counter_to_sorted_dict(selected_answer_counts),
        "selected_bin_distribution": counter_to_sorted_dict(selected_bin_counts),
        "selected_indices": [
            {
                "global_idx": int(row["global_idx"]),
                "source_file": str(row["source_file"]),
                "local_idx": int(row["local_idx"]),
                "answer": int(row["answer"]),
            }
            for row in selected
        ],
    }

    print(json.dumps({k: v for k, v in manifest.items() if k != "selected_indices"}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_selected_dataset(files, selected, output_dir, args.rows_per_shard)
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# StepCountQA-RL-Traj 11-30 Focused10k Maskcomplete V36\n\n"
        "Deterministic 10k training subset from `StepCountQA-RL-Traj_11_50_Combined_maskcomplete_38332`.\n\n"
        "- Selected rows: 10,000\n"
        "- Focus: 8,500 rows from answer 11-30\n"
        "- Retained long-count coverage: 1,500 rows from answer 31-50\n"
        "- Quotas: 11-20 = 4,000; 21-30 = 4,500; 31-40 = 1,000; 41-50 = 500\n"
        "- Selection: deterministic SHA256 ranking per answer\n"
        "- Quality gate: inherited from mask-complete source manifest\n"
        f"- Hash salt: `{HASH_SALT}`\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
