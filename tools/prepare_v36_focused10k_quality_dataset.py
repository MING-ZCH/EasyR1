#!/usr/bin/env python3
"""Build a quality-prioritized 10k V36 dense-training subset.

The output keeps the focused10k answer quota exactly unchanged while replacing
low-quality mask rows with better rows from the same mask-complete source set.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_LOCAL_ROOT = Path(os.environ.get("LOCAL_ROOT", "/mnt/shared-storage-user/zhangchenhao"))
DEFAULT_DATA_ROOT = Path(
    os.environ.get("STEPCOUNT_DATA_ROOT", DEFAULT_LOCAL_ROOT / "work/StepcountModel/dataset")
)
DEFAULT_SOURCE = str(DEFAULT_DATA_ROOT / "StepCountQA-RL-Traj_11_50_Combined_maskcomplete_38332")
DEFAULT_CURRENT = str(DEFAULT_DATA_ROOT / "StepCountQA-RL-Traj_11_30_Focused10k_maskcomplete_v36_20260713")
DEFAULT_OUTPUT = str(
    DEFAULT_DATA_ROOT / "StepCountQA-RL-Traj_11_30_Focused10k_quality_maskcomplete_v36_20260713"
)
DEFAULT_METADATA = str(
    DEFAULT_DATA_ROOT
    / "sam_mask/StepCount-RL_masks_output_merged/"
    "final_rl_mask_manifest_v36_grouped_missing_remaining12h8_20260705_070420_"
    "grouped_progressive_issue_filtered_pointin_quality_upgraded/"
    "masks_metadata.easy_r1_compatible.grouped_progressive_issue_filtered.json"
)
HASH_SALT = "v36-focused10k-quality:maskcomplete:20260713"

SUFFIX_RE = re.compile(r"^(.+?)_(\d+)$")
K_RE = re.compile(r"^(.+?)_k\d+$")

OK_TIERS = {"curated_high_quality", "base_ideal_ok", "recall_hard_ok"}
RISK_TIERS = {
    "base_large_area_risk",
    "recall_large_area_risk",
    "recall_neighbor_risk",
    "base_point_false_risk",
    "recall_invalid_placeholder",
}
WEAK_TIERS = {
    "base_low_quality",
    "recall_low_quality",
    "actual_pointin_fallback",
    "recall_point_local_fallback",
    *RISK_TIERS,
}
TIER_SCORE = {
    "curated_high_quality": 100.0,
    "base_ideal_ok": 90.0,
    "recall_hard_ok": 85.0,
    # Some curated recovery rows were emitted without a tier; they still passed
    # the component-clean/issue-filter gate, so rank them above unverified base.
    "None": 78.0,
    "base_unverified_metadata": 55.0,
    "actual_pointin_fallback": 42.0,
    "recall_point_local_fallback": 40.0,
    "base_low_quality": 25.0,
    "recall_low_quality": 24.0,
    "base_large_area_risk": 8.0,
    "recall_large_area_risk": 8.0,
    "recall_neighbor_risk": 5.0,
    "base_point_false_risk": 3.0,
    "recall_invalid_placeholder": 0.0,
}


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
    if 11 <= answer <= 20:
        return "11-20"
    if 21 <= answer <= 30:
        return "21-30"
    if 31 <= answer <= 40:
        return "31-40"
    if 41 <= answer <= 50:
        return "41-50"
    raise ValueError(f"answer {answer} outside 11-50")


def extract_sequence_id(image_path: object) -> str:
    basename = os.path.basename(str(image_path or ""))
    basename = re.sub(r"\s+", "", basename)
    name, _ = os.path.splitext(basename)
    match = SUFFIX_RE.match(name)
    return match.group(1) if match else name


def extract_turn_number(image_path: object) -> int:
    basename = os.path.basename(str(image_path or ""))
    basename = re.sub(r"\s+", "", basename)
    name, _ = os.path.splitext(basename)
    match = SUFFIX_RE.match(name)
    return int(match.group(2)) if match else 0


def base_sequence_id(sequence_id: str) -> str:
    match = K_RE.match(sequence_id or "")
    return match.group(1) if match else sequence_id


def row_identity(problem: object, answer: int, image_path: object) -> str:
    payload = f"{image_path}\t{answer}\t{problem}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def stable_tiebreak(row: dict[str, object]) -> str:
    payload = (
        f"{HASH_SALT}:file={row['source_file']}:local={row['local_idx']}:"
        f"answer={row['answer']}:seq={row['sequence_id']}"
    ).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def discover_rows(dataset_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    files = sorted((dataset_dir / "data").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {dataset_dir / 'data'}")

    for file_idx, path in enumerate(files):
        parquet_file = pq.ParquetFile(path)
        local_offset = 0
        for row_group_idx in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(
                row_group_idx,
                columns=["images.list.element.path", "problem", "answer"],
            )
            for row_group_local_idx, row in enumerate(table.to_pylist()):
                answer = parse_answer(row["answer"])
                images = row.get("images") or []
                image_path = images[0].get("path") if images and isinstance(images[0], dict) else ""
                problem = row.get("problem") or ""
                sequence_id = base_sequence_id(extract_sequence_id(image_path))
                rows.append(
                    {
                        "global_idx": len(rows),
                        "file_idx": file_idx,
                        "source_file": path.name,
                        "local_idx": local_offset + row_group_local_idx,
                        "answer": answer,
                        "bin": bin_label(answer),
                        "sequence_id": sequence_id,
                        "image_path": image_path,
                        "problem": problem,
                        "row_identity": row_identity(problem, answer, image_path),
                    }
                )
            local_offset += table.num_rows
    return rows


def stream_metadata_for_rows(metadata_path: Path, rows: list[dict[str, object]]) -> dict[tuple[str, int], tuple[str, str, str]]:
    expected: dict[tuple[str, int], int] = {}
    for row_idx, row in enumerate(rows):
        sequence_id = str(row["sequence_id"])
        for turn in range(int(row["answer"])):
            expected[(sequence_id, turn)] = row_idx

    result: dict[tuple[str, int], tuple[str, str, str]] = {}
    decoder = json.JSONDecoder()
    buffer = ""
    index = 0
    started = False
    eof = False

    with metadata_path.open("r", encoding="utf-8") as handle:
        while True:
            if not eof and len(buffer) - index < 1024 * 1024:
                chunk = handle.read(8 * 1024 * 1024)
                if chunk:
                    buffer = (buffer[index:] if index else buffer) + chunk
                    index = 0
                else:
                    eof = True

            n_chars = len(buffer)
            while index < n_chars and buffer[index] in " \t\r\n,":
                index += 1

            if not started:
                if index < n_chars and buffer[index] == "[":
                    index += 1
                    started = True
                    continue

            if index < n_chars and buffer[index] == "]":
                break

            if index >= n_chars:
                if eof:
                    break
                continue

            try:
                obj, next_index = decoder.raw_decode(buffer, index)
            except json.JSONDecodeError:
                if eof:
                    raise
                buffer = buffer[index:]
                index = 0
                continue

            index = next_index
            key = (
                base_sequence_id(extract_sequence_id(obj.get("image_path") or "")),
                extract_turn_number(obj.get("image_path") or ""),
            )
            if key in expected:
                result[key] = (
                    str(obj.get("quality_tier") or "None"),
                    str(obj.get("quality_reason") or "None"),
                    str(obj.get("source") or "None"),
                )

            if index > 16 * 1024 * 1024:
                buffer = buffer[index:]
                index = 0

    return result


def score_rows(rows: list[dict[str, object]], metadata: dict[tuple[str, int], tuple[str, str, str]]) -> None:
    for row in rows:
        tier_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        total_score = 0.0
        missing_turns = 0
        for turn in range(int(row["answer"])):
            meta = metadata.get((str(row["sequence_id"]), turn))
            if not meta:
                missing_turns += 1
                total_score -= 100.0
                continue
            tier, reason, _source = meta
            tier_counts[tier] += 1
            reason_counts[reason] += 1
            total_score += TIER_SCORE.get(tier, 50.0)

        answer = int(row["answer"])
        row["missing_turns"] = missing_turns
        row["tier_counts"] = dict(tier_counts)
        row["reason_counts"] = dict(reason_counts)
        row["risk_turns"] = sum(tier_counts[tier] for tier in RISK_TIERS)
        row["weak_turns"] = sum(tier_counts[tier] for tier in WEAK_TIERS)
        row["ok_turns"] = sum(tier_counts[tier] for tier in OK_TIERS)
        row["ok_ratio"] = round(float(row["ok_turns"]) / max(answer, 1), 6)
        row["mask_score"] = round(total_score / max(answer, 1), 6)


def choose_quality_rows(rows: list[dict[str, object]], current_identities: set[str]) -> list[dict[str, object]]:
    by_answer: Dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_answer[int(row["answer"])].append(row)

    selected: list[dict[str, object]] = []
    for answer in range(11, 51):
        candidates = by_answer[answer]
        quota = answer_quota(answer)
        if len(candidates) < quota:
            raise RuntimeError(f"answer={answer} has {len(candidates)} rows, quota={quota}")
        candidates = sorted(
            candidates,
            key=lambda row: (
                int(row["missing_turns"]) == 0,
                -int(row["risk_turns"]),
                -int(row["weak_turns"]),
                float(row["mask_score"]),
                float(row["ok_ratio"]),
                str(row["row_identity"]) in current_identities,
                stable_tiebreak(row),
            ),
            reverse=True,
        )
        selected.extend(candidates[:quota])
    return sorted(selected, key=lambda row: int(row["global_idx"]))


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


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"rows": 0}
    return {
        "rows": len(rows),
        "avg_score": round(sum(float(row["mask_score"]) for row in rows) / len(rows), 6),
        "avg_ok_ratio": round(sum(float(row["ok_ratio"]) for row in rows) / len(rows), 6),
        "missing_rows": sum(int(row["missing_turns"]) > 0 for row in rows),
        "risk_rows": sum(int(row["risk_turns"]) > 0 for row in rows),
        "weak_rows": sum(int(row["weak_turns"]) > 0 for row in rows),
        "all_ok_rows": sum(int(row["ok_turns"]) == int(row["answer"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE)
    parser.add_argument("--current-dir", default=DEFAULT_CURRENT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--rows-per-shard", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start = time.time()
    source_dir = Path(args.source_dir)
    current_dir = Path(args.current_dir)
    output_dir = Path(args.output_dir)
    metadata_path = Path(args.metadata)

    rows = discover_rows(source_dir)
    current_rows = discover_rows(current_dir) if current_dir.exists() else []
    current_identities = {str(row["row_identity"]) for row in current_rows}
    metadata = stream_metadata_for_rows(metadata_path, rows)
    score_rows(rows, metadata)
    selected = choose_quality_rows(rows, current_identities)

    selected_identities = {str(row["row_identity"]) for row in selected}
    current_source_rows = [row for row in rows if str(row["row_identity"]) in current_identities]
    removed = [row for row in current_source_rows if str(row["row_identity"]) not in selected_identities]
    added = [row for row in selected if str(row["row_identity"]) not in current_identities]

    manifest = {
        "source_dir": str(source_dir),
        "current_dir": str(current_dir),
        "output_dir": str(output_dir),
        "metadata": str(metadata_path),
        "algorithm": "answer-stratified quality-prioritized selection from mask-complete rows",
        "hash_salt": HASH_SALT,
        "target_rows": 10000,
        "selected_rows": len(selected),
        "source_rows": len(rows),
        "metadata_turns_matched_source": len(metadata),
        "quota_by_range": {"11-20": 4000, "21-30": 4500, "31-40": 1000, "41-50": 500},
        "quota_by_answer": {str(answer): answer_quota(answer) for answer in range(11, 51)},
        "selected_answer_distribution": dict(sorted(Counter(int(row["answer"]) for row in selected).items())),
        "selected_bin_distribution": dict(sorted(Counter(str(row["bin"]) for row in selected).items())),
        "current_summary": summarize(current_source_rows),
        "selected_summary": summarize(selected),
        "current_by_bin": {label: summarize([row for row in current_source_rows if row["bin"] == label]) for label in ("11-20", "21-30", "31-40", "41-50")},
        "selected_by_bin": {label: summarize([row for row in selected if row["bin"] == label]) for label in ("11-20", "21-30", "31-40", "41-50")},
        "replacement_count": {"added": len(added), "removed": len(removed)},
        "removed_worst_examples": [
            {
                "source_file": row["source_file"],
                "local_idx": int(row["local_idx"]),
                "answer": int(row["answer"]),
                "sequence_id": row["sequence_id"],
                "mask_score": row["mask_score"],
                "ok_ratio": row["ok_ratio"],
                "risk_turns": row["risk_turns"],
                "weak_turns": row["weak_turns"],
                "tier_counts": row["tier_counts"],
            }
            for row in sorted(removed, key=lambda item: (float(item["mask_score"]), -int(item["risk_turns"]), -int(item["weak_turns"])))[:100]
        ],
        "selected_indices": [
            {
                "global_idx": int(row["global_idx"]),
                "source_file": str(row["source_file"]),
                "local_idx": int(row["local_idx"]),
                "answer": int(row["answer"]),
                "sequence_id": str(row["sequence_id"]),
                "mask_score": row["mask_score"],
                "ok_ratio": row["ok_ratio"],
                "risk_turns": row["risk_turns"],
                "weak_turns": row["weak_turns"],
            }
            for row in selected
        ],
        "elapsed_sec": round(time.time() - start, 3),
    }

    print(json.dumps({key: value for key, value in manifest.items() if key != "selected_indices"}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((source_dir / "data").glob("*.parquet"))
    write_selected_dataset(files, selected, output_dir, args.rows_per_shard)
    (output_dir / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# StepCountQA-RL-Traj 11-30 Focused10k Quality Maskcomplete V36\n\n"
        "Quality-prioritized 10k training subset from `StepCountQA-RL-Traj_11_50_Combined_maskcomplete_38332`.\n\n"
        "- Selected rows: 10,000\n"
        "- Focus: 8,500 rows from answer 11-30\n"
        "- Retained long-count coverage: 1,500 rows from answer 31-50\n"
        "- Quotas: 11-20 = 4,000; 21-30 = 4,500; 31-40 = 1,000; 41-50 = 500\n"
        "- Selection: per-answer quality ranking using the latest EasyR1-compatible mask metadata\n"
        f"- Metadata: `{metadata_path}`\n"
        f"- Hash salt: `{HASH_SALT}`\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
