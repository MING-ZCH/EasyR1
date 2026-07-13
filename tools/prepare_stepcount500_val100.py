#!/usr/bin/env python3
"""Build a deterministic v36 100-sample validation subset from Stepcount-500.

The source benchmark is balanced across answers 11-50 with 500 total rows.
This script preserves the 10-count difficulty bins and spreads the extra
per-answer samples symmetrically, so mean/median/std match the full benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_LOCAL_ROOT = Path(os.environ.get("LOCAL_ROOT", "/mnt/shared-storage-user/zhangchenhao"))
DEFAULT_DATA_ROOT = Path(
    os.environ.get("STEPCOUNT_DATA_ROOT", DEFAULT_LOCAL_ROOT / "work/StepcountModel/dataset")
)
DEFAULT_SOURCE = str(DEFAULT_DATA_ROOT / "stepcount-500/data/train-00000-of-00001.parquet")
DEFAULT_OUTPUT = str(DEFAULT_DATA_ROOT / "stepcount-500-v36-val-100")
DEFAULT_BINS = [(11, 20), (21, 30), (31, 40), (41, 50)]
HASH_SALT = "stepcount-500:v36-val-100"


def parse_answer(value: object) -> int:
    match = re.search(r"-?\d+", str(value))
    if not match:
        raise ValueError(f"cannot parse numeric answer from {value!r}")
    return int(match.group())


def bin_label(answer: int, bins: Iterable[Tuple[int, int]]) -> str:
    for low, high in bins:
        if low <= answer <= high:
            return f"{low}-{high}"
    raise ValueError(f"answer {answer} is outside configured bins")


def exact_answer_quotas() -> Dict[int, int]:
    """Quota recommended by reviewer consensus.

    Each answer gets at least two samples. The 11-20 and 21-30 bins each get
    six extra samples; 31-40 and 41-50 each get four extra samples.
    """
    quotas = {answer: 2 for answer in range(11, 51)}
    for start in (11, 21):
        for offset in (0, 2, 4, 5, 7, 9):
            quotas[start + offset] += 1
    for start in (31, 41):
        for offset in (1, 3, 6, 8):
            quotas[start + offset] += 1
    return quotas


def stable_hash_key(answer: int, idx: int) -> str:
    payload = f"{HASH_SALT}:answer={answer}:idx={idx}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_pick(indices: Sequence[int], answer: int, k: int) -> List[int]:
    ranked = sorted(indices, key=lambda idx: stable_hash_key(answer, idx))
    return sorted(ranked[:k])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output_dir)
    if not source.exists():
        raise FileNotFoundError(source)

    table = pq.read_table(source)
    answer_col = table.column("answer")
    problem_col = table.column("problem") if "problem" in table.column_names else None

    rows = []
    bin_counts: Counter[str] = Counter()
    answer_counts: Counter[int] = Counter()
    answer_to_indices: Dict[Tuple[str, int], List[int]] = defaultdict(list)

    for idx in range(table.num_rows):
        answer = parse_answer(answer_col[idx].as_py())
        label = bin_label(answer, DEFAULT_BINS)
        problem = problem_col[idx].as_py() if problem_col is not None else ""
        rows.append({"idx": idx, "answer": answer, "bin": label, "problem": problem})
        bin_counts[label] += 1
        answer_counts[answer] += 1
        answer_to_indices[(label, answer)].append(idx)

    quotas = exact_answer_quotas()
    selected: List[int] = []
    answer_quotas: Dict[str, Dict[str, int]] = {}
    for low, high in DEFAULT_BINS:
        label = f"{low}-{high}"
        answer_order = list(range(low, high + 1))
        answer_quotas[label] = {str(answer): quotas[answer] for answer in answer_order}
        for answer in answer_order:
            candidates = answer_to_indices[(label, answer)]
            if len(candidates) < quotas[answer]:
                raise RuntimeError(f"answer={answer} has only {len(candidates)} rows for quota={quotas[answer]}")
            selected.extend(hash_pick(candidates, answer, quotas[answer]))

    selected = sorted(selected)
    if len(selected) != 100:
        raise RuntimeError(f"selected {len(selected)} rows, expected 100")

    selected_table = table.take(pa.array(selected, type=pa.int64()))
    selected_answers = [rows[idx]["answer"] for idx in selected]
    selected_bins = [rows[idx]["bin"] for idx in selected]

    manifest = {
        "source": str(source),
        "output_dir": str(output_dir),
        "algorithm": "per-answer quota with stable sha256 hash ordering",
        "hash_salt": HASH_SALT,
        "target_size": 100,
        "source_rows": table.num_rows,
        "selected_rows": len(selected),
        "selected_indices": selected,
        "source_answer_distribution": dict(sorted(answer_counts.items())),
        "source_bin_distribution": dict(sorted(bin_counts.items())),
        "selected_answer_distribution": dict(sorted(Counter(selected_answers).items())),
        "selected_bin_distribution": dict(sorted(Counter(selected_bins).items())),
        "bin_quotas": {label: sum(int(v) for v in answer_quotas[label].values()) for label in answer_quotas},
        "answer_quotas_by_bin": answer_quotas,
        "note": (
            "images.path is preserved from source; Stepcount-500 stores image bytes and has null image paths. "
            "Use this as answer-level high-count validation, not mask-sequence validation."
        ),
    }

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(selected_table, output_dir / "train.parquet")
    (output_dir / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Stepcount-500 Val100\n\n"
        "Deterministic 100-sample validation subset from local Stepcount-500.\n\n"
        "- Source: `stepcount-500/data/train-00000-of-00001.parquet`\n"
        "- Count range: 11-50\n"
        "- Bin quotas: 11-20 = 26, 21-30 = 26, 31-40 = 24, 41-50 = 24\n"
        f"- Stable hash salt: `{HASH_SALT}`\n"
        "- Image paths: null in source and preserved here; image bytes are present.\n"
        "- Intended use: answer-level high-count validation alongside `pixmo-test`, not mask-sequence validation.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
