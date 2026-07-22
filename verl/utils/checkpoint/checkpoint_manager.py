# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from filelock import FileLock
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedTokenizer, ProcessorMixin


CHECKPOINT_TRACKER = "checkpoint_tracker.json"
CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
CHECKPOINT_MANIFEST_VERSION = 1


def _validate_relative_path(relative_path: Any) -> str:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise RuntimeError(f"Invalid checkpoint-relative path {relative_path!r}.")
    normalized = os.path.normpath(relative_path)
    if (
        os.path.isabs(relative_path)
        or normalized in (".", "..")
        or normalized.startswith(f"..{os.path.sep}")
    ):
        raise RuntimeError(f"Checkpoint path escapes its root: {relative_path!r}.")
    return normalized


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(os.path.dirname(path), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _checkpoint_artifacts(path: str) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for root, directories, filenames in os.walk(path):
        directories.sort()
        filenames.sort()
        for directory in directories:
            directory_path = os.path.join(root, directory)
            if os.path.islink(directory_path):
                raise RuntimeError(f"Checkpoint directories must not be symlinks: {directory_path}.")
        for filename in filenames:
            absolute_path = os.path.join(root, filename)
            relative_path = os.path.relpath(absolute_path, path).replace(os.path.sep, "/")
            if relative_path == CHECKPOINT_MANIFEST:
                continue
            if os.path.islink(absolute_path) or not os.path.isfile(absolute_path):
                raise RuntimeError(f"Checkpoint artifacts must be regular files: {absolute_path}.")
            artifacts[relative_path] = {
                "size": os.path.getsize(absolute_path),
                "sha256": _sha256_file(absolute_path),
            }
    return artifacts


def _fsync_checkpoint_tree(path: str) -> None:
    for root, directories, filenames in os.walk(path, topdown=False):
        for filename in filenames:
            artifact_fd = os.open(os.path.join(root, filename), os.O_RDONLY)
            try:
                os.fsync(artifact_fd)
            finally:
                os.close(artifact_fd)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def write_checkpoint_manifest(path: str, global_step: int, required_paths: list[str]) -> dict[str, Any]:
    """Seal a fully-written staging directory with hashes for every artifact."""
    normalized_required = sorted({_validate_relative_path(item.strip("/")) for item in required_paths})
    for relative_path in normalized_required:
        absolute_required = os.path.join(path, relative_path)
        if not relative_path or not os.path.exists(absolute_required):
            raise RuntimeError(f"Checkpoint is missing required artifact {relative_path!r}.")
    artifacts = _checkpoint_artifacts(path)
    if not artifacts:
        raise RuntimeError(f"Checkpoint staging directory contains no artifacts: {path}.")
    for relative_path in normalized_required:
        if os.path.isdir(os.path.join(path, relative_path)) and not any(
            artifact == relative_path or artifact.startswith(f"{relative_path}/") for artifact in artifacts
        ):
            raise RuntimeError(f"Required checkpoint directory is empty: {relative_path!r}.")
    manifest = {
        "version": CHECKPOINT_MANIFEST_VERSION,
        "global_step": int(global_step),
        "required_paths": normalized_required,
        "artifacts": artifacts,
    }
    _atomic_write_json(os.path.join(path, CHECKPOINT_MANIFEST), manifest)
    _fsync_checkpoint_tree(path)
    return manifest


def validate_checkpoint_manifest(path: str, expected_step: Optional[int] = None) -> dict[str, Any]:
    """Fail closed unless a checkpoint is complete and byte-for-byte intact."""
    manifest_path = os.path.join(path, CHECKPOINT_MANIFEST)
    if not os.path.isfile(manifest_path):
        raise RuntimeError(f"Checkpoint is missing {manifest_path}.")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("version") != CHECKPOINT_MANIFEST_VERSION:
        raise RuntimeError(f"Unsupported checkpoint manifest contract in {manifest_path}.")
    step = manifest.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or (expected_step is not None and step != expected_step):
        raise RuntimeError(f"Checkpoint manifest global_step mismatch in {manifest_path}.")
    required_paths = manifest.get("required_paths")
    artifacts = manifest.get("artifacts")
    if not isinstance(required_paths, list) or not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError(f"Invalid checkpoint manifest payload in {manifest_path}.")
    for relative_path in required_paths:
        relative_path = _validate_relative_path(relative_path)
        if not os.path.exists(os.path.join(path, relative_path)):
            raise RuntimeError(f"Checkpoint is missing required artifact {relative_path!r}.")
    for relative_path, expected in artifacts.items():
        _validate_relative_path(relative_path)
        if (
            not isinstance(expected, dict)
            or set(expected) != {"size", "sha256"}
            or isinstance(expected.get("size"), bool)
            or not isinstance(expected.get("size"), int)
            or expected["size"] < 0
            or not isinstance(expected.get("sha256"), str)
        ):
            raise RuntimeError(f"Invalid manifest entry for {relative_path!r}.")
    observed = _checkpoint_artifacts(path)
    if set(observed) != set(artifacts):
        missing = sorted(set(artifacts) - set(observed))
        unexpected = sorted(set(observed) - set(artifacts))
        raise RuntimeError(f"Checkpoint artifact set mismatch; missing={missing}, unexpected={unexpected}.")
    for relative_path in required_paths:
        if os.path.isdir(os.path.join(path, relative_path)) and not any(
            artifact == relative_path or artifact.startswith(f"{relative_path}/") for artifact in artifacts
        ):
            raise RuntimeError(f"Required checkpoint directory is empty: {relative_path!r}.")
    for relative_path, expected in artifacts.items():
        if expected != observed[relative_path]:
            artifact_path = os.path.join(path, *relative_path.split("/"))
            raise RuntimeError(f"Checkpoint artifact hash mismatch: {artifact_path}.")
    return manifest


def checkpoint_manifest_hash(path: str) -> str:
    return _sha256_file(os.path.join(path, CHECKPOINT_MANIFEST))


def publish_staged_checkpoint(staging_path: str, final_path: str, global_step: int) -> dict[str, Any]:
    """Atomically rename a validated staging tree into its public name."""
    manifest = validate_checkpoint_manifest(staging_path, expected_step=global_step)
    if os.path.exists(final_path):
        # A complete same-step save is idempotent. Never replace an invalid or
        # partially-published directory in place.
        return validate_checkpoint_manifest(final_path, expected_step=global_step)
    os.replace(staging_path, final_path)
    directory_fd = os.open(os.path.dirname(final_path), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return manifest


def reference_content_identity(path_or_id: Optional[str]) -> dict[str, str]:
    """Return a stable content identity for a local reference tree or model id."""
    value = "" if path_or_id is None else str(path_or_id)
    digest = hashlib.sha256()
    if os.path.isfile(value):
        digest.update(b"file\0")
        digest.update(_sha256_file(value).encode("ascii"))
        return {"kind": "file-content", "sha256": digest.hexdigest()}
    if os.path.isdir(value):
        digest.update(b"directory\0")
        found_file = False
        for root, directories, filenames in os.walk(value):
            directories.sort()
            filenames.sort()
            for directory in directories:
                directory_path = os.path.join(root, directory)
                if os.path.islink(directory_path):
                    raise RuntimeError(f"Reference model identity refuses symlink directory {directory_path}.")
            for filename in filenames:
                absolute_path = os.path.join(root, filename)
                if not os.path.isfile(absolute_path):
                    raise RuntimeError(f"Reference model identity refuses non-regular file {absolute_path}.")
                found_file = True
                relative_path = os.path.relpath(absolute_path, value).replace(os.path.sep, "/")
                digest.update(relative_path.encode("utf-8"))
                digest.update(b"\0")
                digest.update(_sha256_file(absolute_path).encode("ascii"))
                digest.update(b"\0")
        if not found_file:
            raise RuntimeError(f"Reference model directory is empty: {value}.")
        return {"kind": "directory-content", "sha256": digest.hexdigest()}
    try:
        from huggingface_hub import snapshot_download

        cached_path = snapshot_download(repo_id=value, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot compute an exact content hash for reference model {value!r}; "
            "use a local path or populate the Hugging Face snapshot cache."
        ) from exc
    identity = reference_content_identity(cached_path)
    return {"kind": "cached-directory-content", "sha256": identity["sha256"]}


class BaseCheckpointManager(ABC):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin],
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class

        assert isinstance(self.model, FSDP)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    @abstractmethod
    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def local_mkdir(path: str) -> str:
        if not os.path.isabs(path):
            working_dir = os.getcwd()
            path = os.path.join(working_dir, path)

        lock_id = hashlib.sha256(os.path.realpath(path).encode("utf-8")).hexdigest()[:24]
        lock_filename = f"ckpt_{lock_id}.lock"
        lock_path = os.path.join(tempfile.gettempdir(), lock_filename)

        try:
            with FileLock(lock_path, timeout=60):
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            if os.environ.get("V37_TRAINING_EVIDENCE_REQUIRED") == "1":
                raise RuntimeError(f"V37 failed to acquire checkpoint directory lock for {path}") from e
            print(f"Warning: Failed to acquire lock for {path}: {e}")
            os.makedirs(path, exist_ok=True)  # even if the lock is not acquired, try to create the directory

        return path

    @staticmethod
    def get_rng_state() -> dict[str, Any]:
        rng_state = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        return rng_state

    @staticmethod
    def load_rng_state(rng_state: dict[str, Any]):
        torch.set_rng_state(rng_state["cpu"])
        torch.cuda.set_rng_state(rng_state["cuda"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])


def get_checkpoint_tracker_filename(root_path: str) -> str:
    """
    Tracker file rescords the latest chckpoint during training to restart from.
    """
    return os.path.join(root_path, CHECKPOINT_TRACKER)


def find_latest_ckpt(
    path: str, directory_format: str = "global_step_{}"
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """
    Find the latest checkpoint in the save path.
    """
    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_file):
        return None, None

    with open(tracker_file, "rb") as f:
        checkpointer_tracker_info = json.load(f)

    if isinstance(checkpointer_tracker_info, int):
        checkpointer_tracker_info = {"last_global_step": checkpointer_tracker_info}
    if not isinstance(checkpointer_tracker_info, dict) or not isinstance(
        checkpointer_tracker_info.get("last_global_step"), int
    ):
        raise ValueError(f"Invalid checkpoint tracker contract in {tracker_file}.")

    ckpt_path = os.path.join(path, directory_format.format(checkpointer_tracker_info["last_global_step"]))
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint does not exist: {ckpt_path}")
        return None, None

    manifest_path = os.path.join(ckpt_path, CHECKPOINT_MANIFEST)
    tracker_manifest_hash = checkpointer_tracker_info.get("manifest_sha256")
    if tracker_manifest_hash is not None or os.path.exists(manifest_path):
        validate_checkpoint_manifest(ckpt_path, expected_step=checkpointer_tracker_info["last_global_step"])
        if tracker_manifest_hash is not None and tracker_manifest_hash != checkpoint_manifest_hash(ckpt_path):
            raise RuntimeError(f"Checkpoint tracker manifest hash mismatch for {ckpt_path}.")

    print(f"Found latest checkpoint: {ckpt_path}, will resume from it. Turn off `find_last_checkpoint` to disable it.")
    return ckpt_path, checkpointer_tracker_info


def remove_obsolete_ckpt(
    path: str, global_step: int, best_global_step: int, save_limit: int = -1, directory_format: str = "global_step_{}"
):
    """
    Remove the obsolete checkpoints that exceed the save limit.
    """
    if save_limit <= 0 or not os.path.exists(path):
        return

    num_ckpt_to_keep = save_limit - 1  # exclude the current ckpt
    pattern = re.escape(directory_format).replace(r"\{\}", r"(\d+)")
    ckpt_global_steps = []
    for folder in os.listdir(path):
        if match := re.fullmatch(pattern, folder):
            step = int(match.group(1))
            if step < global_step:
                ckpt_global_steps.append(step)

    ckpt_global_steps.sort(reverse=True)
    if best_global_step in ckpt_global_steps:  # do not remove the best ckpt
        ckpt_global_steps.remove(best_global_step)
        num_ckpt_to_keep = max(num_ckpt_to_keep - 1, 0)

    failures = []
    for step in ckpt_global_steps[num_ckpt_to_keep:]:
        folder_path = os.path.join(path, directory_format.format(step))
        try:
            shutil.rmtree(folder_path)
            if os.path.exists(folder_path):
                raise RuntimeError("checkpoint directory still exists after recursive deletion")
            print(f"Removed obsolete checkpoint: {folder_path}")
        except Exception as e:
            failures.append((folder_path, e))

    if failures:
        details = "; ".join(f"{folder}: {error}" for folder, error in failures)
        raise RuntimeError(
            "Failed to remove one or more obsolete checkpoints after the current checkpoint "
            f"was published; the authoritative checkpoint was not targeted. {details}"
        )
