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


import json
import os
import random
import shutil
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.core_algos import AdaptiveKLController
from verl.utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from verl.utils.checkpoint import checkpoint_manager as checkpoint_manager_module
from verl.utils.checkpoint.checkpoint_manager import (
    checkpoint_manifest_hash,
    publish_staged_checkpoint,
    validate_checkpoint_manifest,
    write_checkpoint_manifest,
)
from verl.trainer.ray_trainer import (
    TRAINER_RUNTIME_STATE_FILE,
    RayPPOTrainer,
)


@pytest.fixture
def save_checkpoint_path():
    ckpt_dir = os.path.join("checkpoints", str(uuid.uuid4()))
    os.makedirs(ckpt_dir, exist_ok=True)
    yield ckpt_dir
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_find_latest_ckpt(save_checkpoint_path):
    with open(os.path.join(save_checkpoint_path, CHECKPOINT_TRACKER), "w") as f:
        json.dump({"last_global_step": 10}, f, ensure_ascii=False, indent=2)

    assert find_latest_ckpt(save_checkpoint_path)[0] is None
    os.makedirs(os.path.join(save_checkpoint_path, "global_step_10"), exist_ok=True)
    assert find_latest_ckpt(save_checkpoint_path)[0] == os.path.join(save_checkpoint_path, "global_step_10")


def test_find_latest_ckpt_accepts_legacy_integer_tracker(save_checkpoint_path):
    with open(os.path.join(save_checkpoint_path, CHECKPOINT_TRACKER), "w") as f:
        f.write("12")
    os.makedirs(os.path.join(save_checkpoint_path, "global_step_12"), exist_ok=True)

    path, info = find_latest_ckpt(save_checkpoint_path)

    assert path == os.path.join(save_checkpoint_path, "global_step_12")
    assert info == {"last_global_step": 12}


def test_find_latest_ckpt_rejects_invalid_json_contract(save_checkpoint_path):
    with open(os.path.join(save_checkpoint_path, CHECKPOINT_TRACKER), "w") as f:
        json.dump({"step": 3}, f)

    with pytest.raises(ValueError, match="checkpoint tracker contract"):
        find_latest_ckpt(save_checkpoint_path)


def test_remove_obsolete_ckpt(save_checkpoint_path):
    for step in range(5, 30, 5):
        os.makedirs(os.path.join(save_checkpoint_path, f"global_step_{step}"), exist_ok=True)

    remove_obsolete_ckpt(save_checkpoint_path, global_step=30, best_global_step=10, save_limit=3)
    for step in range(5, 30, 5):
        is_exist = step in [10, 25]
        assert os.path.exists(os.path.join(save_checkpoint_path, f"global_step_{step}")) == is_exist


def test_retention_failure_is_explicit_and_never_targets_authoritative_checkpoint(
    save_checkpoint_path, monkeypatch
):
    for step in (5, 10, 30):
        os.makedirs(os.path.join(save_checkpoint_path, f"global_step_{step}"))
    original_rmtree = checkpoint_manager_module.shutil.rmtree

    def fail_oldest(path):
        if path.endswith("global_step_5"):
            raise PermissionError("injected retention failure")
        original_rmtree(path)

    monkeypatch.setattr(checkpoint_manager_module.shutil, "rmtree", fail_oldest)
    with pytest.raises(RuntimeError, match="injected retention failure"):
        remove_obsolete_ckpt(
            save_checkpoint_path, global_step=30, best_global_step=-1, save_limit=2
        )

    assert os.path.isdir(os.path.join(save_checkpoint_path, "global_step_30"))
    assert os.path.isdir(os.path.join(save_checkpoint_path, "global_step_5"))


def _sealed_checkpoint(root, step):
    staging = os.path.join(root, f".stage-{step}")
    final = os.path.join(root, f"global_step_{step}")
    os.makedirs(os.path.join(staging, "actor"))
    torch.save({"cursor": step}, os.path.join(staging, "dataloader.pt"))
    with open(os.path.join(staging, "actor", "rank_0.pt"), "wb") as handle:
        handle.write(f"actor-{step}".encode())
    write_checkpoint_manifest(staging, step, ["actor", "dataloader.pt"])
    publish_staged_checkpoint(staging, final, step)
    return final


def test_manifest_detects_missing_and_hash_corruption(save_checkpoint_path):
    checkpoint = _sealed_checkpoint(save_checkpoint_path, 4)
    tracker = {
        "last_global_step": 4,
        "manifest_sha256": checkpoint_manifest_hash(checkpoint),
    }
    with open(os.path.join(save_checkpoint_path, CHECKPOINT_TRACKER), "w") as handle:
        json.dump(tracker, handle)

    assert find_latest_ckpt(save_checkpoint_path)[0] == checkpoint
    with open(os.path.join(checkpoint, "actor", "rank_0.pt"), "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        find_latest_ckpt(save_checkpoint_path)

    os.remove(os.path.join(checkpoint, "actor", "rank_0.pt"))
    with pytest.raises(RuntimeError, match="artifact set mismatch"):
        validate_checkpoint_manifest(checkpoint, expected_step=4)


def test_transaction_failure_preserves_previous_checkpoint_and_tracker(save_checkpoint_path):
    previous = _sealed_checkpoint(save_checkpoint_path, 1)
    previous_tracker = {
        "last_global_step": 1,
        "manifest_sha256": checkpoint_manifest_hash(previous),
    }
    tracker_path = os.path.join(save_checkpoint_path, CHECKPOINT_TRACKER)
    with open(tracker_path, "w") as handle:
        json.dump(previous_tracker, handle)

    class FailingWorkerGroup:
        def save_checkpoint(self, path):
            os.makedirs(path)
            with open(os.path.join(path, "partial.pt"), "wb") as handle:
                handle.write(b"partial")
            raise RuntimeError("injected distributed save failure")

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(save_checkpoint_path=save_checkpoint_path, save_limit=1)
    )
    trainer.global_step = 2
    trainer.adaptive_actor_kl = False
    trainer._current_step_complete = True
    trainer.use_critic = False
    trainer.actor_rollout_wg = FailingWorkerGroup()
    trainer.train_dataloader = SimpleNamespace(state_dict=lambda: {"cursor": 2})

    with pytest.raises(RuntimeError, match="injected"):
        trainer._save_checkpoint()

    with open(tracker_path) as handle:
        assert json.load(handle) == previous_tracker
    validate_checkpoint_manifest(previous, expected_step=1)
    assert not os.path.exists(os.path.join(save_checkpoint_path, "global_step_2"))
    assert not any("staging" in name for name in os.listdir(save_checkpoint_path))


def test_adaptive_incomplete_checkpoint_reports_skipped_without_writing(save_checkpoint_path):
    previous = _sealed_checkpoint(save_checkpoint_path, 7)
    previous_tracker = {
        "last_global_step": 7,
        "manifest_sha256": checkpoint_manifest_hash(previous),
    }
    tracker_path = os.path.join(save_checkpoint_path, CHECKPOINT_TRACKER)
    with open(tracker_path, "w") as handle:
        json.dump(previous_tracker, handle)

    class MustNotRun:
        def save_checkpoint(self, path):
            raise AssertionError("incomplete adaptive step attempted a distributed save")

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(save_checkpoint_path=save_checkpoint_path, save_limit=2)
    )
    trainer.global_step = 8
    trainer.adaptive_actor_kl = True
    trainer._current_step_complete = False
    trainer.use_critic = False
    trainer.actor_rollout_wg = MustNotRun()

    assert trainer._save_checkpoint() is False
    assert not os.path.exists(os.path.join(save_checkpoint_path, "global_step_8"))
    with open(tracker_path) as handle:
        assert json.load(handle) == previous_tracker
    validate_checkpoint_manifest(previous, expected_step=7)


def test_complete_same_step_save_is_idempotent(save_checkpoint_path):
    checkpoint = _sealed_checkpoint(save_checkpoint_path, 7)

    class MustNotRun:
        def save_checkpoint(self, path):
            raise AssertionError("idempotent save rewrote distributed shards")

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(save_checkpoint_path=save_checkpoint_path, save_limit=2)
    )
    trainer.global_step = 7
    trainer.adaptive_actor_kl = False
    trainer._current_step_complete = True
    trainer.use_critic = False
    trainer.actor_rollout_wg = MustNotRun()
    assert trainer._save_checkpoint() is True

    with open(os.path.join(save_checkpoint_path, CHECKPOINT_TRACKER)) as handle:
        tracker = json.load(handle)
    assert tracker["last_global_step"] == 7
    assert tracker["manifest_sha256"] == checkpoint_manifest_hash(checkpoint)


class _RuntimeWorkerGroup:
    world_size = 1

    def save_checkpoint(self, path):
        os.makedirs(path)
        torch.save({"worker": "state"}, os.path.join(path, "rank_0.pt"))

    def load_checkpoint(self, path):
        assert os.path.isfile(os.path.join(path, "rank_0.pt"))
        # Loading is deliberately allowed to consume controller RNG.  The
        # trainer runtime snapshot must be restored after this call.
        random.random()
        np.random.random()
        torch.rand(1)


class _RuntimeDataloader:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"cursor": 3}

    def load_state_dict(self, state):
        self.loaded = state
        random.random()
        np.random.random()
        torch.rand(1)


def _adaptive_checkpoint_trainer(root, load_path=None):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            save_checkpoint_path=root,
            load_checkpoint_path=load_path,
            save_limit=-1,
        )
    )
    trainer.global_step = 3
    trainer.adaptive_actor_kl = True
    trainer._current_step_complete = True
    trainer.use_critic = False
    trainer.actor_rollout_wg = _RuntimeWorkerGroup()
    trainer.train_dataloader = _RuntimeDataloader()
    trainer.kl_ctrl = AdaptiveKLController(0.08, 0.15, 1000, strict_nonnegative_kl=True)
    trainer.effective_actor_updates = 7
    trainer._adaptive_ref_identity = {"kind": "test", "sha256": "reference"}
    return trainer


def test_adaptive_trainer_runtime_rng_roundtrip_is_sealed_and_restored(save_checkpoint_path):
    random.seed(1234)
    np.random.seed(5678)
    torch.manual_seed(9012)
    saver = _adaptive_checkpoint_trainer(save_checkpoint_path)
    saver._save_checkpoint()
    checkpoint = os.path.join(save_checkpoint_path, "global_step_3")

    manifest = validate_checkpoint_manifest(checkpoint, expected_step=3)
    assert TRAINER_RUNTIME_STATE_FILE in manifest["required_paths"]
    assert TRAINER_RUNTIME_STATE_FILE in manifest["artifacts"]
    expected = (random.random(), np.random.random(), torch.rand(4))

    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    loader = _adaptive_checkpoint_trainer(save_checkpoint_path, checkpoint)
    assert loader._load_checkpoint() is True
    actual = (random.random(), np.random.random(), torch.rand(4))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert loader.effective_actor_updates == 7
    assert loader.train_dataloader.loaded == {"cursor": 3}


@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_adaptive_trainer_runtime_missing_or_corrupt_fails_closed(
    save_checkpoint_path, mutation
):
    saver = _adaptive_checkpoint_trainer(save_checkpoint_path)
    saver._save_checkpoint()
    checkpoint = os.path.join(save_checkpoint_path, "global_step_3")
    runtime_path = os.path.join(checkpoint, TRAINER_RUNTIME_STATE_FILE)
    if mutation == "missing":
        os.remove(runtime_path)
    else:
        with open(runtime_path, "ab") as handle:
            handle.write(b"corrupt")

    loader = _adaptive_checkpoint_trainer(save_checkpoint_path, checkpoint)
    with pytest.raises(RuntimeError, match="artifact set mismatch|hash mismatch"):
        loader._load_checkpoint()


def test_adaptive_trainer_runtime_topology_mismatch_fails_closed(save_checkpoint_path):
    saver = _adaptive_checkpoint_trainer(save_checkpoint_path)
    saver._save_checkpoint()
    checkpoint = os.path.join(save_checkpoint_path, "global_step_3")
    manifest = validate_checkpoint_manifest(checkpoint, expected_step=3)
    runtime_path = os.path.join(checkpoint, TRAINER_RUNTIME_STATE_FILE)
    runtime = torch.load(runtime_path, weights_only=False)
    runtime["worker_world_size"] = 2
    torch.save(runtime, runtime_path)
    write_checkpoint_manifest(checkpoint, 3, manifest["required_paths"])

    loader = _adaptive_checkpoint_trainer(save_checkpoint_path, checkpoint)
    with pytest.raises(RuntimeError, match="worker topology"):
        loader._load_checkpoint()


def test_adaptive_controller_cuda_rng_topology_mismatch_fails_closed(save_checkpoint_path):
    saver = _adaptive_checkpoint_trainer(save_checkpoint_path)
    saver._save_checkpoint()
    checkpoint = os.path.join(save_checkpoint_path, "global_step_3")
    manifest = validate_checkpoint_manifest(checkpoint, expected_step=3)
    runtime_path = os.path.join(checkpoint, TRAINER_RUNTIME_STATE_FILE)
    runtime = torch.load(runtime_path, weights_only=False)
    runtime["controller_rng"]["cuda_device_count"] += 1
    torch.save(runtime, runtime_path)
    write_checkpoint_manifest(checkpoint, 3, manifest["required_paths"])

    loader = _adaptive_checkpoint_trainer(save_checkpoint_path, checkpoint)
    with pytest.raises(RuntimeError, match="CUDA topology"):
        loader._load_checkpoint()


def test_terminal_only_validation_is_after_terminal_checkpoint():
    events = []
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.global_step = 9
    trainer.val_reward_fn = object()
    trainer.adaptive_actor_kl = True
    trainer._save_checkpoint = lambda: events.append("save") or True
    trainer._validate = lambda: events.append("validate") or {"score": 1.0}
    trainer.logger = SimpleNamespace(log=lambda **kwargs: events.append("log"))

    trainer._finalize_training(None, last_validation_step=None, last_checkpoint_step=None)

    assert events == ["save", "validate", "log"]


def test_v36_terminal_validation_order_is_unchanged():
    events = []
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.global_step = 9
    trainer.val_reward_fn = object()
    trainer.adaptive_actor_kl = False
    trainer._save_checkpoint = lambda: events.append("save") or True
    trainer._validate = lambda: events.append("validate") or {"score": 1.0}
    trainer.logger = SimpleNamespace(log=lambda **kwargs: events.append("log"))

    trainer._finalize_training(None, last_validation_step=None, last_checkpoint_step=None)

    assert events == ["validate", "log", "save"]


def test_periodic_validation_keeps_validation_then_checkpoint_order():
    events = ["periodic_validate"]
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.global_step = 9
    trainer.val_reward_fn = object()
    trainer._save_checkpoint = lambda: events.append("save") or True
    trainer._validate = lambda: (_ for _ in ()).throw(AssertionError("duplicate validation"))
    trainer.logger = SimpleNamespace(log=lambda **kwargs: None)

    trainer._finalize_training(
        {"score": 1.0}, last_validation_step=9, last_checkpoint_step=None
    )

    assert events == ["periodic_validate", "save"]
