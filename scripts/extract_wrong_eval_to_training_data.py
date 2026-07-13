#!/usr/bin/env python3
"""Extract wrong predictions from eval JSONs and convert to StepCountQA-RL-Traj parquet format."""

import argparse
import hashlib
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def load_eval_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected list, got {type(data)} in {json_path}")


def extract_wrong_predictions(eval_data, count_range=None):
    wrong = []
    for sample in eval_data:
        correct = str(sample.get("correct_answer", "")).strip()
        predicted = str(sample.get("predicted_answer", "")).strip()
        if correct != predicted:
            if count_range is not None:
                try:
                    gt_val = int(correct)
                    if gt_val < count_range[0] or gt_val > count_range[1]:
                        continue
                except ValueError:
                    continue
            wrong.append(sample)
    return wrong


def find_original_image(sample, eval_base_dir):
    image_paths = sample.get("image_paths", [])
    if not image_paths:
        return None
    first_img = image_paths[0]
    full_path = os.path.join(eval_base_dir, first_img)
    if os.path.exists(full_path):
        return full_path
    if os.path.exists(first_img):
        return first_img
    return None


def sample_to_training_row(sample, eval_base_dir):
    img_path = find_original_image(sample, eval_base_dir)
    if img_path is None:
        return None
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    img_filename = os.path.basename(img_path)
    return {
        "images": [{"bytes": img_bytes, "path": img_filename}],
        "problem": f"<image>\n{sample['question']}",
        "answer": str(sample["correct_answer"]),
    }


def build_parquet_table(rows):
    image_struct = pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ])
    schema = pa.schema([
        pa.field("images", pa.list_(image_struct)),
        pa.field("problem", pa.string()),
        pa.field("answer", pa.string()),
    ])
    images_col = [r["images"] for r in rows]
    problems = [r["problem"] for r in rows]
    answers = [r["answer"] for r in rows]
    return pa.table({"images": images_col, "problem": problems, "answer": answers}, schema=schema)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_files", nargs="+", required=True)
    parser.add_argument("--eval_base_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--count_range", nargs=2, type=int, default=None)
    parser.add_argument("--samples_per_shard", type=int, default=500)
    parser.add_argument("--dedup", action="store_true")
    args = parser.parse_args()

    if len(args.eval_base_dirs) == 1:
        args.eval_base_dirs = args.eval_base_dirs * len(args.eval_files)
    if len(args.eval_base_dirs) != len(args.eval_files):
        print("ERROR: --eval_base_dirs count mismatch")
        sys.exit(1)

    count_range = tuple(args.count_range) if args.count_range else None
    all_rows = []
    seen_keys = set()
    stats = {"total_eval": 0, "total_wrong": 0, "converted": 0, "skipped": 0, "dedup": 0}

    for eval_file, base_dir in zip(args.eval_files, args.eval_base_dirs):
        print(f"\nProcessing: {os.path.basename(eval_file)}")
        eval_data = load_eval_json(eval_file)
        wrong = extract_wrong_predictions(eval_data, count_range)
        stats["total_eval"] += len(eval_data)
        stats["total_wrong"] += len(wrong)
        print(f"  Total: {len(eval_data)}, Wrong: {len(wrong)} ({100*len(wrong)/max(len(eval_data),1):.1f}%)")

        for sample in wrong:
            row = sample_to_training_row(sample, base_dir)
            if row is None:
                stats["skipped"] += 1
                print(f"  WARNING: No image for {sample.get('id', '?')}")
                continue
            if args.dedup:
                img_hash = hashlib.md5(row["images"][0]["bytes"]).hexdigest()
                key = (row["problem"], img_hash)
                if key in seen_keys:
                    stats["dedup"] += 1
                    continue
                seen_keys.add(key)
            all_rows.append(row)
            stats["converted"] += 1

    if not all_rows:
        print("\nNo samples to write!")
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)
    num_shards = max(1, (len(all_rows) + args.samples_per_shard - 1) // args.samples_per_shard)
    shard_size = (len(all_rows) + num_shards - 1) // num_shards

    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(all_rows))
        shard_rows = all_rows[start:end]
        table = build_parquet_table(shard_rows)
        shard_name = f"train-{shard_idx:05d}-of-{num_shards:05d}.parquet"
        out_path = os.path.join(args.output_dir, shard_name)
        pq.write_table(table, out_path)
        print(f"  Shard {shard_idx}: {len(shard_rows)} samples -> {out_path}")

    print(f"\n=== Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  Output: {args.output_dir} ({num_shards} shard(s))")


if __name__ == "__main__":
    main()
