#!/usr/bin/env python3
"""按 step77 -> step60、四 suite 固定顺序执行 V36 正式评测。"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from bundle_common import MODEL_ORDER, PROTOCOL_TAG, PROTOCOL_VERSION, SUITE_ORDER, parse_key_value


DATASET_ENVS = {
    "pixmo-test": "V36_PIXMO_JSON",
    "stepcount-500": "V36_STEPCOUNT500_JSON",
    "countqa": "V36_COUNTQA_JSON",
    "bias": "V36_BIAS_JSON",
}
IMAGE_ROOT_ENVS = {
    "pixmo-test": "V36_PIXMO_IMAGE_ROOT",
    "stepcount-500": "V36_STEPCOUNT500_IMAGE_ROOT",
    "countqa": "V36_COUNTQA_IMAGE_ROOT",
    "bias": "V36_BIAS_IMAGE_ROOT",
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step77-model")
    parser.add_argument("--step60-model")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="SUITE=JSON",
        help="可重复；覆盖对应 V36_*_JSON 环境变量",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        metavar="SUITE=DIR",
        help="相对 image_path 的基准目录；可重复",
    )
    parser.add_argument(
        "--image-remap",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="绝对 image_path 的安全前缀映射；可重复",
    )
    parser.add_argument("--output-root")
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--workers-per-gpu", type=int, choices=(1, 2, 3))
    parser.add_argument("--max-pixels", type=int, default=12845056)
    parser.add_argument("--python", dest="python_bin")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    cli_datasets = parse_key_value(args.dataset, what="--dataset")
    cli_roots = parse_key_value(args.image_root, what="--image-root")
    unknown = (set(cli_datasets) | set(cli_roots)).difference(SUITE_ORDER)
    if unknown:
        raise ValueError(f"未知 suite: {sorted(unknown)}")

    models = {
        "77": args.step77_model or os.environ.get("V36_STEP77_MODEL"),
        "60": args.step60_model or os.environ.get("V36_STEP60_MODEL"),
    }
    datasets = {suite: cli_datasets.get(suite) or os.environ.get(DATASET_ENVS[suite]) for suite in SUITE_ORDER}
    global_image_root = os.environ.get("V36_IMAGE_ROOT")
    image_roots = {
        suite: cli_roots.get(suite) or os.environ.get(IMAGE_ROOT_ENVS[suite]) or global_image_root
        for suite in SUITE_ORDER
    }
    env_remaps = [item for item in os.environ.get("V36_IMAGE_REMAPS", "").split(";;") if item]
    output_root = args.output_root or os.environ.get("V36_EVAL_OUTPUT_ROOT")
    num_gpus = args.num_gpus if args.num_gpus is not None else _env_int("V36_EVAL_NUM_GPUS", 8)
    workers = args.workers_per_gpu if args.workers_per_gpu is not None else _env_int("V36_EVAL_WORKERS_PER_GPU", 1)
    python_bin = args.python_bin or os.environ.get("PYTHON_BIN") or sys.executable

    missing = [
        *(f"V36_STEP{step}_MODEL/--step{step}-model" for step, value in models.items() if not value),
        *(f"{DATASET_ENVS[suite]}/--dataset {suite}=..." for suite, value in datasets.items() if not value),
    ]
    if not output_root:
        output_root = str(Path.cwd() / "v36_eval_results")
    if missing and not args.dry_run:
        raise ValueError("缺少正式运行参数: " + ", ".join(missing))
    if args.dry_run:
        models = {step: value or f"/REQUIRED/model/step{step}" for step, value in models.items()}
        datasets = {suite: value or f"/REQUIRED/data/{suite}.json" for suite, value in datasets.items()}
    if num_gpus <= 0:
        raise ValueError("--num-gpus 必须为正整数")
    if workers not in (1, 2, 3):
        raise ValueError("workers-per-gpu 只允许 1、2、3")

    return {
        "models": models,
        "datasets": datasets,
        "image_roots": image_roots,
        "image_remaps": [*env_remaps, *args.image_remap],
        "output_root": str(Path(output_root).expanduser()),
        "num_gpus": num_gpus,
        "workers_per_gpu": workers,
        "max_pixels": args.max_pixels,
        "python": python_bin,
        "resume": args.resume,
    }


def build_commands(config: dict, bundle_dir: Path) -> list[tuple[str, list[str]]]:
    preflight = [
        config["python"],
        str(bundle_dir / "preflight.py"),
        "--expected-gpus",
        str(config["num_gpus"]),
        "--workers-per-gpu",
        str(config["workers_per_gpu"]),
    ]
    for step in MODEL_ORDER:
        preflight.extend(("--model", f"step{step}={config['models'][step]}"))
    for suite in SUITE_ORDER:
        preflight.extend(("--dataset", f"{suite}={config['datasets'][suite]}"))
        if config["image_roots"][suite]:
            preflight.extend(("--image-root", f"{suite}={config['image_roots'][suite]}"))
    for remap in config["image_remaps"]:
        preflight.extend(("--image-remap", remap))

    commands: list[tuple[str, list[str]]] = [("preflight", preflight)]
    for step in MODEL_ORDER:
        for suite in SUITE_ORDER:
            output_dir = Path(config["output_root"]) / f"step{step}" / suite
            command = [
                config["python"],
                str(bundle_dir / "eval_stepcount.py"),
                "--model",
                config["models"][step],
                "--model-label",
                f"step{step}",
                "--task",
                suite,
                "--dataset",
                config["datasets"][suite],
                "--output-dir",
                str(output_dir),
                "--workers-per-gpu",
                str(config["workers_per_gpu"]),
                "--max-pixels",
                str(config["max_pixels"]),
            ]
            if config["image_roots"][suite]:
                command.extend(("--image-root", config["image_roots"][suite]))
            for remap in config["image_remaps"]:
                command.extend(("--image-remap", remap))
            if config["resume"]:
                command.append("--resume")
            commands.append((f"step{step}/{suite}", command))

    analyze = [
        config["python"],
        str(bundle_dir / "analyze_results.py"),
        "--results-root",
        config["output_root"],
    ]
    for suite in SUITE_ORDER:
        analyze.extend(("--dataset", f"{suite}={config['datasets'][suite]}"))
    commands.append(("analyze", analyze))
    return commands


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(args)
    bundle_dir = Path(__file__).resolve().parent
    commands = build_commands(config, bundle_dir)
    summary = {
        "protocol": PROTOCOL_VERSION,
        "protocol_tag": PROTOCOL_TAG,
        "model_order": [f"step{step}" for step in MODEL_ORDER],
        "suite_order": list(SUITE_ORDER),
        "num_gpus": config["num_gpus"],
        "workers_per_gpu": config["workers_per_gpu"],
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    for label, command in commands:
        print(f"[{label}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    if args.dry_run:
        print("DRY-RUN：未检查文件、未加载模型/GPU、未执行评测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
