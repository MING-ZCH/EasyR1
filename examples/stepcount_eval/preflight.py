#!/usr/bin/env python3
"""不加载模型权重的 V36 bundle 预检；正式运行会逐张验证图片。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bundle_common import (
    SUITES,
    ids_sha256,
    parse_image_remaps,
    parse_key_value,
    remap_image_path,
    sha256_file,
    validate_bf16_safetensors_model,
    validate_dataset_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[], required=True, metavar="SUITE=JSON")
    parser.add_argument("--image-root", action="append", default=[], metavar="SUITE=DIR")
    parser.add_argument("--image-remap", action="append", default=[], metavar="FROM=TO")
    parser.add_argument("--model", action="append", default=[], required=True, metavar="LABEL=DIR")
    parser.add_argument("--expected-gpus", type=int)
    parser.add_argument("--workers-per-gpu", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_model(model_path: Path) -> dict:
    # HF cache snapshots commonly expose immutable blobs through symlinks.
    # Eval may read those targets; release staging applies the stricter default.
    model = validate_bf16_safetensors_model(model_path, allow_external_shard_symlinks=True)
    return {
        "path": str(model_path),
        "tensors": model["tensor_count"],
        "shards": len(model["shards"]),
        "dtypes": model["dtypes"],
        "weight_bytes": model["weight_bytes"],
    }


def validate_dataset(suite: str, dataset_path: Path, image_root: str | None, remaps) -> dict:
    with dataset_path.open("r", encoding="utf-8") as handle:
        rows, ids, answers = validate_dataset_rows(json.load(handle), suite)
    spec = SUITES[suite]
    if len(rows) != spec.expected_count:
        raise ValueError(f"{suite}: expected {spec.expected_count} samples, got {len(rows)}")
    if (min(answers), max(answers)) != (spec.expected_min_gt, spec.expected_max_gt):
        raise ValueError(
            f"{suite}: GT range expected {spec.expected_min_gt}..{spec.expected_max_gt}, "
            f"got {min(answers)}..{max(answers)}"
        )
    missing: list[tuple[str, str, str]] = []
    for row in rows:
        resolved = remap_image_path(row["image_path"], dataset_path=dataset_path, image_root=image_root, remaps=remaps)
        if not resolved.is_file():
            missing.append((str(row["id"]), str(row["image_path"]), str(resolved)))
    if missing:
        preview = [
            {"id": sample_id, "json_path": raw, "resolved_path": resolved} for sample_id, raw, resolved in missing[:10]
        ]
        raise FileNotFoundError(f"{suite}: remap 后有 {len(missing)} 张图片不存在；前 10 项: {preview}")
    return {
        "dataset": str(dataset_path),
        "dataset_id": suite,
        "samples": len(rows),
        "dataset_sha256": sha256_file(dataset_path),
        "sample_ids_sha256": ids_sha256(ids),
        "gt_range": [min(answers), max(answers)],
        "effective_turn_range": [
            min(min(spec.task_cap, answer + 3) for answer in answers),
            max(min(spec.task_cap, answer + 3) for answer in answers),
        ],
        "all_images_exist": True,
    }


def validate_gpus(expected: int, workers: int, model_reports: dict) -> dict:
    if expected <= 0:
        raise ValueError("--expected-gpus 必须为正整数")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"期望 {expected} 张 CUDA GPU，但 CUDA 不可用")
    actual = torch.cuda.device_count()
    if actual != expected:
        raise RuntimeError(f"期望恰好 {expected} 张可见 GPU，实际 {actual}")
    unsupported_bf16 = []
    for device in range(actual):
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                unsupported_bf16.append(device)
    if unsupported_bf16:
        raise RuntimeError(f"以下可见 CUDA 设备不支持 BF16: {unsupported_bf16}")
    max_weight_bytes = max((item["weight_bytes"] for item in model_reports.values()), default=0)
    estimated = workers * (max_weight_bytes + 24 * 1024**3)
    insufficient = []
    for device in range(actual):
        total = torch.cuda.get_device_properties(device).total_memory
        if total < int(estimated * 1.10):
            insufficient.append({"device": device, "total_bytes": total, "estimated_bytes": estimated})
    if insufficient:
        raise RuntimeError(f"worker 并发显存保守估计不足: {insufficient}")
    return {"expected": expected, "actual": actual, "bf16_supported": True}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    datasets = parse_key_value(args.dataset, what="--dataset")
    roots = parse_key_value(args.image_root, what="--image-root")
    models = parse_key_value(args.model, what="--model")
    if set(datasets) != set(SUITES):
        raise ValueError(f"--dataset 必须恰好覆盖 {list(SUITES)}")
    if set(roots).difference(SUITES):
        raise ValueError(f"--image-root 含未知 suite: {sorted(set(roots).difference(SUITES))}")
    if set(models) != {"step77", "step60"}:
        raise ValueError("--model 必须恰好包含 step77 和 step60")
    remaps = parse_image_remaps(args.image_remap)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "datasets": datasets,
                    "models": models,
                    "image_roots": roots,
                    "image_remaps": args.image_remap,
                    "note": "未访问文件、模型或 GPU",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    dataset_reports = {
        suite: validate_dataset(suite, Path(path).expanduser().resolve(), roots.get(suite), remaps)
        for suite, path in datasets.items()
    }
    model_reports = {label: validate_model(Path(path).expanduser().resolve()) for label, path in models.items()}
    gpu_report = (
        validate_gpus(args.expected_gpus, args.workers_per_gpu, model_reports)
        if args.expected_gpus is not None
        else None
    )
    print(
        json.dumps(
            {
                "protocol": "strict_oracle_gt_plus_v3",
                "datasets": dataset_reports,
                "models": model_reports,
                "gpus": gpu_report,
                "workers_per_gpu": args.workers_per_gpu,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
