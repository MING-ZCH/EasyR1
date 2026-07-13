#!/usr/bin/env python3
"""V34 Phase2: Prepare a dense count-bucket subset (+ optional 0-10 replay) dataset.

Realises the "11-20 curriculum with 0-10 replay" WITHOUT any training-code change:
filters rows by the integer `answer` (count), optionally mixes in a sampled fraction
of the 0-10 dataset, and writes NEW parquet shards with the SAME schema. The existing
datasets are never modified.

STREAMING + CHUNK-SAFE: these parquet files embed images as large binary, so a whole
`pq.read_table(...).take(...)` or `pa.concat_tables(...)` overflows pyarrow's int32
offset limit (2GB/chunk) -> `ArrowInvalid: offset overflow`. We therefore iterate
row-batches and write filtered batches directly via a sharded ParquetWriter, never
materialising a giant in-memory table.

Schema expected: columns = ['images', 'problem', 'answer'].

Usage:
  python3 tools/v34_prepare_dense_subset.py \
    --combined /apdcephfs_hldy2/.../StepCountQA-RL-Traj_11_50_Combined/data \
    --replay   /apdcephfs_hldy2/.../StepCountQA-RL-Traj_0_10/data \
    --count_min 11 --count_max 20 --replay_ratio 0.25 \
    --out_dir  /apdcephfs_hldy2/.../StepCountQA-RL-Traj_11_20_plus_replay/data
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


def _to_int(x) -> Optional[int]:
    if x is None:
        return None
    m = re.search(r"-?\d+", str(x))
    return int(m.group()) if m else None


def _list_parquet(d: str) -> List[str]:
    if os.path.isfile(d) and d.endswith(".parquet"):
        return [d]
    fs = sorted(glob.glob(os.path.join(d, "*.parquet")))
    if not fs:
        fs = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
    return fs


class ShardWriter:
    """Accumulate filtered batches and flush fixed-size parquet shards."""

    def __init__(self, out_dir: str, shard_rows: int, schema: pa.Schema):
        self.out_dir = out_dir
        self.shard_rows = shard_rows
        self.schema = schema
        self.buf: List[pa.RecordBatch] = []
        self.buf_rows = 0
        self.shard_idx = 0
        self.total_rows = 0
        os.makedirs(out_dir, exist_ok=True)

    def add(self, batch: pa.RecordBatch) -> None:
        if batch.num_rows == 0:
            return
        # align column order to schema
        batch = batch.select(self.schema.names)
        self.buf.append(batch)
        self.buf_rows += batch.num_rows
        self.total_rows += batch.num_rows
        if self.buf_rows >= self.shard_rows:
            self._flush()

    def _flush(self) -> None:
        if not self.buf:
            return
        out = os.path.join(self.out_dir, f"train-{self.shard_idx:05d}.parquet")
        writer = pq.ParquetWriter(out, self.schema)
        for b in self.buf:
            writer.write_batch(b)
        writer.close()
        print(f"[prep] wrote shard {out} ({self.buf_rows} rows)")
        self.shard_idx += 1
        self.buf = []
        self.buf_rows = 0

    def close(self) -> None:
        self._flush()
        # rename shards to include final count of shards
        n = self.shard_idx
        for i in range(n):
            src = os.path.join(self.out_dir, f"train-{i:05d}.parquet")
            dst = os.path.join(self.out_dir, f"train-{i:05d}-of-{n:05d}.parquet")
            if os.path.exists(src):
                os.rename(src, dst)


def _filter_batch(batch: pa.RecordBatch, cmin: int, cmax: int) -> pa.RecordBatch:
    ans = batch.column(batch.schema.get_field_index("answer")).to_pylist()
    idx = [i for i, a in enumerate(ans) if (_to_int(a) is not None and cmin <= _to_int(a) <= cmax)]
    if len(idx) == batch.num_rows:
        return batch
    if not idx:
        return batch.slice(0, 0)
    return batch.take(pa.array(idx))


def _count_rows(files: List[str], cmin: int, cmax: int, batch_size: int) -> int:
    n = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=batch_size, columns=["answer"]):
            ans = b.column(0).to_pylist()
            n += sum(1 for a in ans if (_to_int(a) is not None and cmin <= _to_int(a) <= cmax))
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", required=True, help="dense combined data dir (parquet)")
    ap.add_argument("--replay", default="", help="0-10 data dir for replay mixing (optional)")
    ap.add_argument("--count_min", type=int, default=11)
    ap.add_argument("--count_max", type=int, default=20)
    ap.add_argument("--replay_ratio", type=float, default=0.25, help="fraction of final set from replay")
    ap.add_argument("--replay_count_min", type=int, default=0)
    ap.add_argument("--replay_count_max", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard_rows", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=256, help="row-group batch size (chunk-safe)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    dense_files = _list_parquet(args.combined)
    if not dense_files:
        raise SystemExit(f"No parquet under --combined {args.combined}")
    schema = pq.read_schema(dense_files[0])
    print(f"[prep] schema: {schema.names}")

    # Pass 1: count dense rows (cheap, answer column only) to size the replay draw.
    n_dense = _count_rows(dense_files, args.count_min, args.count_max, args.batch_size)
    print(f"[prep] dense [{args.count_min},{args.count_max}]: {n_dense} rows")

    n_replay_target = 0
    replay_p = 0.0
    replay_files: List[str] = []
    if args.replay and args.replay_ratio > 0:
        replay_files = _list_parquet(args.replay)
        n_replay_total = _count_rows(replay_files, args.replay_count_min, args.replay_count_max, args.batch_size)
        n_replay_target = int(round(args.replay_ratio / (1 - args.replay_ratio) * n_dense))
        n_replay_target = min(n_replay_target, n_replay_total)
        replay_p = (n_replay_target / n_replay_total) if n_replay_total > 0 else 0.0
        print(f"[prep] replay [{args.replay_count_min},{args.replay_count_max}]: "
              f"total={n_replay_total} target={n_replay_target} p={replay_p:.4f}")

    n_final = n_dense + n_replay_target
    print(f"[prep] FINAL: {n_final} rows -> {args.out_dir} "
          f"(replay ratio={n_replay_target/max(n_final,1):.3f})")
    if args.dry_run:
        print("[prep] dry_run: not writing")
        return

    writer = ShardWriter(args.out_dir, args.shard_rows, schema)

    # Stream dense (all matching rows).
    for f in dense_files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=args.batch_size):
            fb = _filter_batch(b, args.count_min, args.count_max)
            writer.add(fb)

    # Stream replay (Bernoulli-sample per row at probability replay_p).
    if replay_p > 0:
        for f in replay_files:
            pf = pq.ParquetFile(f)
            for b in pf.iter_batches(batch_size=args.batch_size):
                fb = _filter_batch(b, args.replay_count_min, args.replay_count_max)
                if fb.num_rows == 0:
                    continue
                keep = [i for i in range(fb.num_rows) if rng.random() < replay_p]
                if not keep:
                    continue
                writer.add(fb.take(pa.array(keep)))

    writer.close()
    print(f"[prep] DONE: ~{writer.total_rows} rows in {writer.shard_idx} shard(s). "
          f"Set STEPCOUNT_TRAIN_DATA to the parent dir of {args.out_dir}.")


if __name__ == "__main__":
    main()
