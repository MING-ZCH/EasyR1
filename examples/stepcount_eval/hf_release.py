#!/usr/bin/env python3
"""Stage, validate, upload, and verify one V36 model release."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
from pathlib import Path

from bundle_common import sha256_file, validate_bf16_safetensors_model


DEFAULT_REPOS = {
    "60": "SI-Lab/StepCount-7B-v36-focused10k-step60",
    "77": "SI-Lab/StepCount-7B-v36-focused10k-step77",
}
LARGE_FILE_BYTES = 10 * 1024 * 1024


def model_card_path(step: str) -> Path:
    return Path(__file__).resolve().parent / "model_cards" / f"step{step}_README.md"


def iter_release_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"release tree contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in {".git", ".cache"}:
            continue
        yield relative, path


def validate_bf16_model(root: Path) -> dict:
    model = validate_bf16_safetensors_model(root)
    return {key: model[key] for key in ("index_sha256", "tensor_count", "shards", "dtypes")}


def resolve_release_roots(source: Path, staging: Path) -> tuple[Path, Path]:
    """Reject a symlink at either CLI root before canonicalizing containment."""
    source_raw = Path(os.path.abspath(source.expanduser()))
    staging_raw = Path(os.path.abspath(staging.expanduser()))
    if source_raw.is_symlink():
        raise ValueError(f"source-dir 不允许是 symlink: {source_raw}")
    if staging_raw.is_symlink():
        raise ValueError(f"staging-dir 不允许是 symlink: {staging_raw}")
    source_real = source_raw.resolve(strict=True)
    staging_real = staging_raw.resolve()
    if source_real == staging_real or source_real in staging_real.parents or staging_real in source_real.parents:
        raise ValueError("source-dir 与 staging-dir 必须是互不包含的独立目录")
    return source_real, staging_real


def build_file_manifest(root: Path) -> dict[str, dict]:
    return {
        relative.as_posix(): {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for relative, path in iter_release_files(root)
        if relative.as_posix() != "release_manifest.json"
    }


def sanitize_json_value(value, repo_id: str):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "_name_or_path" and isinstance(item, str) and item.startswith("/"):
                result[key] = repo_id
            elif key in {"dtype", "torch_dtype"} and item in {"float", "float32"}:
                result[key] = "bfloat16"
            else:
                result[key] = sanitize_json_value(item, repo_id)
        return result
    if isinstance(value, list):
        return [sanitize_json_value(item, repo_id) for item in value]
    return value


def prepare_staging(source: Path, staging: Path, step: str, repo_id: str) -> None:
    source, staging = resolve_release_roots(source, staging)
    if staging.exists() and any(staging.iterdir()):
        raise FileExistsError(f"staging-dir 必须不存在或为空目录: {staging}")
    staging.mkdir(parents=True, exist_ok=True)

    for relative, source_file in iter_release_files(source):
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Only immutable weight shards may be hardlinked. JSON metadata and
        # README can be rewritten below, so hardlinking them would mutate source.
        if source_file.suffix == ".safetensors" and source_file.stat().st_size >= LARGE_FILE_BYTES:
            try:
                os.link(source_file, target)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                shutil.copy2(source_file, target)
        else:
            shutil.copy2(source_file, target)

    for json_path in staging.rglob("*.json"):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        sanitized = sanitize_json_value(payload, repo_id)
        if sanitized != payload:
            json_path.write_text(
                json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    shutil.copyfile(model_card_path(step), staging / "README.md")

    for path in staging.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".txt", ".jinja"}:
            continue
        text = path.read_text(encoding="utf-8")
        if any(prefix in text for prefix in ("/mnt/", "/apdcephfs/", "/root/")):
            raise ValueError(f"release text leaks an internal path: {path}")


def write_release_manifest(staging: Path, step: str, repo_id: str, model: dict) -> dict:
    files = build_file_manifest(staging)
    payload = {
        "schema_version": 1,
        "step": step,
        "repo_id": repo_id,
        "model": model,
        "files": files,
        "note": "release_manifest.json is excluded from its self-referential file list",
    }
    (staging / "release_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def remote_manifest(api, repo_id: str, revision: str) -> dict[str, dict]:
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    remote = {}
    for sibling in info.siblings:
        size = getattr(sibling, "size", None)
        lfs = getattr(sibling, "lfs", None)
        sha256 = None
        if isinstance(lfs, dict):
            size = lfs.get("size", size)
            sha256 = lfs.get("sha256") or lfs.get("oid")
        elif lfs is not None:
            size = getattr(lfs, "size", size)
            sha256 = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
        if isinstance(sha256, str) and sha256.startswith("sha256:"):
            sha256 = sha256.split(":", 1)[1]
        remote[sibling.rfilename] = {"size": size, "sha256": sha256}
    return remote


def verify_remote(local: dict[str, dict], remote: dict[str, dict]) -> dict:
    allowed_extra = {".gitattributes"}
    missing = sorted(set(local) - set(remote))
    unexpected = sorted(set(remote) - set(local) - allowed_extra)
    size_mismatch = []
    hash_mismatch = []
    large_without_hash = []
    for name, item in local.items():
        other = remote.get(name)
        if other is None:
            continue
        if other.get("size") != item["size"]:
            size_mismatch.append(name)
        remote_hash = other.get("sha256")
        if remote_hash and remote_hash != item["sha256"]:
            hash_mismatch.append(name)
        if item["size"] >= LARGE_FILE_BYTES and not remote_hash:
            large_without_hash.append(name)
    if missing or unexpected or size_mismatch or hash_mismatch or large_without_hash:
        raise RuntimeError(
            "远端 manifest 校验失败: "
            f"missing={missing}, unexpected={unexpected}, size={size_mismatch}, "
            f"hash={hash_mismatch}, large_without_hash={large_without_hash}"
        )
    return {
        "verified_files": len(local),
        "remote_extra_files": sorted(set(remote) - set(local)),
        "large_lfs_sha256_verified": sum(item["size"] >= LARGE_FILE_BYTES for item in local.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, choices=("60", "77"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--enable-xet",
        action="store_true",
        help="opt in to Xet transfer; default disables Xet for bounded publisher RAM",
    )
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def configure_hf_transfer(enable_xet: bool) -> None:
    """Make the explicit CLI transfer mode override inherited process state."""
    if enable_xet:
        os.environ.pop("HF_HUB_DISABLE_XET", None)
    else:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    from huggingface_hub import constants

    constants.HF_HUB_DISABLE_XET = not enable_xet


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_workers <= 0:
        raise ValueError("--num-workers 必须为正整数")
    repo_id = args.repo_id or DEFAULT_REPOS[args.step]
    plan = {
        "step": args.step,
        "repo_id": repo_id,
        "source_dir": str(args.source_dir.expanduser()),
        "staging_dir": str(args.staging_dir.expanduser()),
        "private": not args.public,
        "model_card": str(model_card_path(args.step)),
        "token_source": "HF_TOKEN environment only",
        "operation": "huggingface_hub.HfApi.upload_large_folder",
        "transfer_mode": "xet" if args.enable_xet else "low-memory-lfs",
        "num_workers": args.num_workers,
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    **plan,
                    "dry_run": True,
                    "note": "未访问目录、网络或 HF_TOKEN，未写文件",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("正式上传要求环境变量 HF_TOKEN")

    source, staging = resolve_release_roots(args.source_dir, args.staging_dir)
    if not source.is_dir():
        raise NotADirectoryError(source)
    source_report = validate_bf16_model(source)
    prepare_staging(source, staging, args.step, repo_id)
    staged_report = validate_bf16_model(staging)
    if source_report != staged_report:
        raise RuntimeError("staging 后模型 index/tensor 验证结果变化")
    write_release_manifest(staging, args.step, repo_id, staged_report)
    local = {
        relative.as_posix(): {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for relative, path in iter_release_files(staging)
    }

    configure_hf_transfer(args.enable_xet)
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )
    info = api.model_info(repo_id)
    if bool(info.private) != (not args.public):
        raise RuntimeError(f"repo visibility 与请求不一致: {repo_id}")
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(staging),
        private=not args.public,
        ignore_patterns=[".cache/**"],
        num_workers=args.num_workers,
    )
    info = api.model_info(repo_id, files_metadata=True)
    if bool(info.private) != (not args.public):
        raise RuntimeError(f"上传后 repo visibility 变化: {repo_id}")
    remote = remote_manifest(api, repo_id, info.sha)
    remote_report = verify_remote(local, remote)
    receipt = {
        **plan,
        "dry_run": False,
        "revision": info.sha,
        "source_model": source_report,
        "remote": remote_report,
    }
    (staging / "upload_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
