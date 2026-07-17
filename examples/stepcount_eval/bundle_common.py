#!/usr/bin/env python3
"""共享的 V36 严格协议、路径映射与可验证元数据工具。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROTOCOL_VERSION = "strict_oracle_gt_plus_v3"
PROTOCOL_TAG = "bf16-oracle-gtplus3-strict-v3"
MODEL_ORDER = ("77", "60")
SUITE_ORDER = ("pixmo-test", "stepcount-500", "countqa", "bias")


@dataclass(frozen=True)
class SuiteSpec:
    expected_count: int
    expected_min_gt: int
    expected_max_gt: int
    task_cap: int
    dot_radius: int


SUITES: Mapping[str, SuiteSpec] = {
    "pixmo-test": SuiteSpec(529, 2, 10, 13, 20),
    "stepcount-500": SuiteSpec(500, 11, 50, 53, 10),
    "countqa": SuiteSpec(491, 2, 10, 13, 20),
    "bias": SuiteSpec(1992, 2, 51, 54, 20),
}


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bf16_safetensors_model(
    root: os.PathLike[str] | str, *, allow_external_shard_symlinks: bool = False
) -> dict:
    """Validate an indexed BF16 model, including exact key-to-shard placement."""
    root = Path(root)
    required = ("config.json", "preprocessor_config.json", "model.safetensors.index.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{root}: 缺少模型文件 {missing}")

    index_path = root / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{root}: safetensors index 的 weight_map 为空")
    if any(not isinstance(key, str) or not key for key in weight_map):
        raise ValueError(f"{root}: safetensors index 含无效 tensor key")
    if any(not isinstance(name, str) or not name for name in weight_map.values()):
        raise ValueError(f"{root}: safetensors index 含无效 shard 名")

    shards = sorted(set(weight_map.values()))
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in shards):
        raise ValueError(f"{root}: safetensors index 含不安全 shard 路径")
    root_real = root.resolve(strict=True)
    shard_paths: dict[str, Path] = {}
    for name in shards:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"{root}: 缺少 shard {name!r}")
        if path.is_symlink() and not allow_external_shard_symlinks:
            raise ValueError(f"{root}: shard 不允许是 symlink: {name!r}")
        path_real = path.resolve(strict=True)
        if not allow_external_shard_symlinks and not _is_relative_to(path_real, root_real):
            raise ValueError(f"{root}: shard 逃逸模型目录: {name!r}")
        shard_paths[name] = path

    indexed_files = set(shards)
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.safetensors")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != indexed_files:
        raise ValueError(
            f"{root}: safetensors 文件集与 index 不一致；"
            f"missing={sorted(indexed_files - actual_files)} "
            f"unexpected={sorted(actual_files - indexed_files)}"
        )

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("模型 dtype/index 校验需要安装 safetensors") from exc

    expected_by_shard = {
        shard: {key for key, mapped_shard in weight_map.items() if mapped_shard == shard} for shard in shards
    }
    observed_dtypes: set[str] = set()
    observed_keys: set[str] = set()
    placement_errors: list[str] = []
    for shard, path in shard_paths.items():
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            observed_keys.update(actual_keys)
            missing_keys = expected_by_shard[shard] - actual_keys
            unexpected_keys = actual_keys - expected_by_shard[shard]
            if missing_keys or unexpected_keys:
                placement_errors.append(
                    f"{shard}: missing={sorted(missing_keys)[:10]} unexpected={sorted(unexpected_keys)[:10]}"
                )
            for key in actual_keys:
                observed_dtypes.add(str(handle.get_slice(key).get_dtype()))
    if placement_errors:
        raise ValueError(f"{root}: index 的 key-to-shard 映射不一致；" + "; ".join(placement_errors[:10]))
    if observed_keys != set(weight_map):
        raise ValueError(
            f"{root}: index/tensor key 不一致；"
            f"missing={sorted(set(weight_map) - observed_keys)[:10]} "
            f"unexpected={sorted(observed_keys - set(weight_map))[:10]}"
        )
    if observed_dtypes != {"BF16"}:
        raise ValueError(f"{root}: 正式协议要求全部 tensor 为 BF16，实际 {sorted(observed_dtypes)}")
    return {
        "index_sha256": sha256_file(index_path),
        "tensor_count": len(weight_map),
        "shards": shards,
        "dtypes": sorted(observed_dtypes),
        "weight_bytes": sum(path.stat().st_size for path in shard_paths.values()),
    }


def ids_sha256(ids: Iterable[object]) -> str:
    encoded = json.dumps([str(value) for value in ids], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_key_value(items: Sequence[str], *, what: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{what} 必须使用 KEY=VALUE: {item!r}")
        key, value = item.split("=", 1)
        if not key or not value:
            raise ValueError(f"{what} 的两侧都不能为空: {item!r}")
        if key in parsed and parsed[key] != value:
            raise ValueError(f"{what} 重复且冲突: {key!r}")
        parsed[key] = value
    return parsed


def parse_image_remaps(items: Sequence[str]) -> tuple[tuple[Path, Path], ...]:
    """解析 FROM=TO；两侧必须绝对，拒绝重复冲突和根目录 FROM。"""
    parsed: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"image remap 必须使用 FROM=TO: {item!r}")
        source_text, target_text = item.split("=", 1)
        if not source_text or not target_text:
            raise ValueError(f"image remap 的两侧都不能为空: {item!r}")
        source = Path(source_text).expanduser()
        target = Path(target_text).expanduser()
        if not source.is_absolute():
            raise ValueError(f"image remap FROM 必须是绝对路径: {source_text!r}")
        if not target.is_absolute():
            raise ValueError(f"image remap TO 必须是绝对路径: {target_text!r}")
        normalized_source = os.path.normpath(str(source))
        if Path(normalized_source) == Path(Path(normalized_source).anchor):
            raise ValueError("image remap 禁止把整个文件系统根目录作为 FROM")
        normalized_target = Path(os.path.normpath(str(target)))
        previous = parsed.get(normalized_source)
        if previous is not None and previous != normalized_target:
            raise ValueError(f"image remap 的 FROM 重复且目标冲突: {source_text!r}")
        parsed[normalized_source] = normalized_target
    return tuple(
        sorted(
            ((Path(source), target) for source, target in parsed.items()),
            key=lambda pair: (-len(pair[0].parts), str(pair[0])),
        )
    )


def _is_relative_to(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
        return True
    except ValueError:
        return False


def remap_image_path(
    raw_path: os.PathLike[str] | str,
    *,
    dataset_path: os.PathLike[str] | str,
    image_root: os.PathLike[str] | str | None = None,
    remaps: Sequence[tuple[Path, Path]] = (),
) -> Path:
    """确定性解析图片路径；绝不使用 basename 或目录搜索猜测。"""
    raw = Path(str(raw_path)).expanduser()
    if not raw.is_absolute():
        base = Path(image_root).expanduser() if image_root else Path(dataset_path).expanduser().parent
        normalized_base = Path(os.path.realpath(os.path.abspath(base)))
        candidate = Path(os.path.realpath(os.path.abspath(normalized_base / raw)))
        if not _is_relative_to(candidate, normalized_base):
            raise ValueError(f"relative image path escapes its configured root: {raw_path!r}")
        return candidate

    normalized = Path(os.path.normpath(str(raw)))
    matches = [(source, target) for source, target in remaps if _is_relative_to(normalized, source)]
    if not matches:
        return normalized
    longest = len(matches[0][0].parts)
    best = [pair for pair in matches if len(pair[0].parts) == longest]
    destinations = {target / normalized.relative_to(source) for source, target in best}
    if len(destinations) != 1:
        raise ValueError(f"image path 存在歧义映射: {raw_path!r} -> {sorted(map(str, destinations))}")
    return next(iter(destinations))


def validate_dataset_rows(rows: object, suite: str) -> tuple[list[dict], list[str], list[int]]:
    if suite not in SUITES:
        raise ValueError(f"未知 suite: {suite}")
    if not isinstance(rows, list):
        raise TypeError("dataset JSON 顶层必须是 list")
    normalized: list[dict] = []
    ids: list[str] = []
    answers: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"dataset row {index} 不是 object")
        missing = {"id", "question", "answer", "image_path"}.difference(row)
        if missing:
            raise ValueError(f"dataset row {index} 缺少字段: {sorted(missing)}")
        if isinstance(row["answer"], bool):
            raise ValueError(f"dataset row {index} 的 answer 不是整数")
        try:
            answer = int(str(row["answer"]).strip())
        except ValueError as exc:
            raise ValueError(f"dataset row {index} 的 answer 不是整数: {row['answer']!r}") from exc
        if str(answer) != str(row["answer"]).strip() and not isinstance(row["answer"], int):
            raise ValueError(f"dataset row {index} 的 answer 不是规范整数: {row['answer']!r}")
        ids.append(str(row["id"]))
        answers.append(answer)
        normalized.append(row)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{suite}: sample ID 不唯一")
    return normalized, ids, answers


def protocol_manifest(
    *,
    suite: str,
    dataset_path: Path,
    model_label: str,
    model_identity: str,
    output_path: Path,
    workers_per_gpu: int,
    sample_ids: Sequence[str],
) -> dict:
    spec = SUITES[suite]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_tag": PROTOCOL_TAG,
        "suite": suite,
        "dataset_id": suite,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "sample_ids_sha256": ids_sha256(sample_ids),
        "expected_samples": spec.expected_count,
        "model_label": model_label,
        "model_identity": model_identity,
        "output_path": str(output_path),
        "dtype": "bfloat16",
        "do_sample": False,
        "num_beams": 1,
        "num_beam_groups": 1,
        "num_return_sequences": 1,
        "history_mode": 0,
        "adaptive_max_rounds": True,
        "adaptive_max_rounds_extra": 3,
        "task_cap": spec.task_cap,
        "require_explicit_answer": True,
        "allow_point_count_fallback": False,
        "stop_after_first_complete_tag": True,
        "stop_on_no_progress": False,
        "keep_eval_images": False,
        "workers_per_gpu": workers_per_gpu,
    }
