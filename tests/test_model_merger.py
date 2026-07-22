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

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


class _FakeDevice:
    type = "cpu"

    def __str__(self):
        return "cpu"


class _FakeTensor:
    def __init__(self, shape, dtype="torch.bfloat16", pointer=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = _FakeDevice()
        self._pointer = id(self) if pointer is None else pointer

    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def data_ptr(self):
        return self._pointer

    def storage_offset(self):
        return 0

    def stride(self):
        if not self.shape:
            return ()
        strides = []
        stride = 1
        for dim in reversed(self.shape):
            strides.append(stride)
            stride *= dim
        return tuple(reversed(strides))


def _dependency_stubs():
    torch = ModuleType("torch")
    torch.Tensor = _FakeTensor
    torch.bfloat16 = "torch.bfloat16"

    distributed = ModuleType("torch.distributed")
    distributed_tensor = ModuleType("torch.distributed._tensor")
    distributed_tensor.DTensor = type("DTensor", (), {})
    distributed_tensor.Placement = type("Placement", (), {})
    distributed_tensor.Shard = type("Shard", (), {})
    distributed._tensor = distributed_tensor
    torch.distributed = distributed

    peft = ModuleType("peft")
    peft.LoraConfig = type("LoraConfig", (), {})
    peft.get_peft_model = lambda *_args, **_kwargs: None

    safetensors = ModuleType("safetensors")
    safetensors.safe_open = lambda *_args, **_kwargs: None

    transformers = ModuleType("transformers")
    for name in (
        "AutoConfig",
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForTokenClassification",
        "AutoModelForVision2Seq",
        "PretrainedConfig",
        "PreTrainedModel",
    ):
        setattr(transformers, name, type(name, (), {}))

    return {
        "torch": torch,
        "torch.distributed": distributed,
        "torch.distributed._tensor": distributed_tensor,
        "peft": peft,
        "safetensors": safetensors,
        "transformers": transformers,
    }


@pytest.fixture(scope="module")
def merger_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "model_merger.py"
    spec = importlib.util.spec_from_file_location("model_merger_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, _dependency_stubs()):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


class _FakeSlice:
    def __init__(self, shape, dtype):
        self._shape = shape
        self._dtype = dtype

    def get_shape(self):
        return self._shape

    def get_dtype(self):
        return self._dtype

    def get_tensor(self):
        raise AssertionError("metadata validation materialized a tensor")


class _FakeSafeOpen:
    def __init__(self, tensors):
        self._tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def keys(self):
        return list(self._tensors)

    def get_slice(self, key):
        return _FakeSlice(*self._tensors[key])

    def get_tensor(self, _key):
        raise AssertionError("metadata validation materialized a tensor")


def _install_safe_open(monkeypatch, merger_module, tensor_files):
    def fake_safe_open(path, *, framework, device):
        assert framework == "pt"
        assert device == "cpu"
        return _FakeSafeOpen(tensor_files[Path(path).name])

    monkeypatch.setattr(merger_module, "safe_open", fake_safe_open)


def test_validate_shard_key_sets_is_symmetric_and_non_mutating(merger_module):
    rank0 = {"weight": object(), "bias": object()}
    rank1 = {"bias": object(), "weight": object()}

    assert merger_module.validate_shard_key_sets([rank0, rank1], ranks=[0, 7]) == {
        "weight",
        "bias",
    }
    assert set(rank0) == {"weight", "bias"}
    assert set(rank1) == {"weight", "bias"}


def test_validate_shard_key_sets_reports_missing_and_extra_rank_keys(merger_module):
    with pytest.raises(RuntimeError) as exc_info:
        merger_module.validate_shard_key_sets(
            [{"weight": object(), "bias": object()}, {"weight": object(), "extra": object()}],
            ranks=[0, 3],
        )

    message = str(exc_info.value)
    assert "rank 3" in message
    assert "bias" in message
    assert "extra" in message
    assert "missing=1" in message
    assert "unexpected=1" in message


def test_validate_shard_key_sets_rejects_empty_reference(merger_module):
    with pytest.raises(RuntimeError, match="rank 0.*empty"):
        merger_module.validate_shard_key_sets([{}], ranks=[0])


def test_hsdp_replica_key_scan_uses_mmap(monkeypatch, merger_module):
    calls = []

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return {"weight": _FakeTensor((2, 2)), "bias": _FakeTensor((2,))}

    monkeypatch.setattr(merger_module.torch, "load", fake_load, raising=False)

    assert merger_module.load_shard_keys_mmap("rank7.pt") == {"weight", "bias"}
    assert calls == [
        (
            "rank7.pt",
            {"map_location": "cpu", "weights_only": False, "mmap": True},
        )
    ]


def test_device_mesh_rank_order_rejects_silent_shard_reordering(merger_module):
    merger_module.validate_device_mesh_rank_order(
        merger_module.np.array([[0, 1], [2, 3]]), world_size=4
    )

    with pytest.raises(RuntimeError, match="Non-canonical DeviceMesh rank order"):
        merger_module.validate_device_mesh_rank_order(
            merger_module.np.array([[0, 2], [1, 3]]), world_size=4
        )


def test_saved_manifest_checks_key_shape_and_dtype_without_loading_tensors(
    tmp_path, monkeypatch, merger_module
):
    (tmp_path / "model.safetensors").write_bytes(b"header-only-test-double")
    _install_safe_open(
        monkeypatch,
        merger_module,
        {
            "model.safetensors": {
                "model.weight": ((2, 3), "BF16"),
                "model.counter": ((), "I64"),
            }
        },
    )
    expected = merger_module.StateDictManifest(
        tensors={
            "model.weight": merger_module.TensorSpec(shape=(2, 3), dtype="BF16"),
            "model.counter": merger_module.TensorSpec(shape=(), dtype="I64"),
        }
    )

    assert merger_module.verify_saved_bfloat16_model(
        str(tmp_path), expected_manifest=expected
    ) == ["model.safetensors"]


def test_saved_manifest_reports_missing_extra_and_shape_mismatch(
    tmp_path, monkeypatch, merger_module
):
    (tmp_path / "model.safetensors").write_bytes(b"header-only-test-double")
    _install_safe_open(
        monkeypatch,
        merger_module,
        {
            "model.safetensors": {
                "model.weight": ((3, 2), "BF16"),
                "model.extra": ((1,), "BF16"),
            }
        },
    )
    expected = merger_module.StateDictManifest(
        tensors={
            "model.weight": merger_module.TensorSpec(shape=(2, 3), dtype="BF16"),
            "model.bias": merger_module.TensorSpec(shape=(2,), dtype="BF16"),
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        merger_module.verify_saved_bfloat16_model(str(tmp_path), expected_manifest=expected)

    message = str(exc_info.value)
    assert "missing=1" in message
    assert "unexpected=1" in message
    assert "shape_mismatches=1" in message


def test_saved_manifest_allows_omitted_exact_tied_alias(tmp_path, monkeypatch, merger_module):
    (tmp_path / "model.safetensors").write_bytes(b"header-only-test-double")
    _install_safe_open(
        monkeypatch,
        merger_module,
        {"model.safetensors": {"model.embed.weight": ((8, 4), "BF16")}},
    )
    spec = merger_module.TensorSpec(shape=(8, 4), dtype="BF16")
    expected = merger_module.StateDictManifest(
        tensors={"model.embed.weight": spec, "lm_head.weight": spec},
        alias_groups=(("lm_head.weight", "model.embed.weight"),),
    )

    assert merger_module.verify_saved_bfloat16_model(
        str(tmp_path), expected_manifest=expected
    ) == ["model.safetensors"]


def test_saved_manifest_validates_exact_index_mapping(tmp_path, monkeypatch, merger_module):
    shard1 = "model-00001-of-00002.safetensors"
    shard2 = "model-00002-of-00002.safetensors"
    (tmp_path / shard1).write_bytes(b"header-only-test-double")
    (tmp_path / shard2).write_bytes(b"header-only-test-double")
    with open(tmp_path / "model.safetensors.index.json", "w", encoding="utf-8") as handle:
        json.dump({"weight_map": {"a": shard2, "b": shard1}}, handle)
    _install_safe_open(
        monkeypatch,
        merger_module,
        {
            shard1: {"a": ((1,), "BF16")},
            shard2: {"b": ((1,), "BF16")},
        },
    )
    expected = merger_module.StateDictManifest(
        tensors={
            "a": merger_module.TensorSpec(shape=(1,), dtype="BF16"),
            "b": merger_module.TensorSpec(shape=(1,), dtype="BF16"),
        }
    )

    with pytest.raises(RuntimeError, match="key-to-shard mapping"):
        merger_module.verify_saved_bfloat16_model(str(tmp_path), expected_manifest=expected)


def test_multiple_saved_shards_require_an_index(tmp_path, merger_module):
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"x")

    with pytest.raises(RuntimeError, match="require model.safetensors.index.json"):
        merger_module.verify_saved_bfloat16_model(str(tmp_path))


def test_remove_stale_weights_preserves_non_model_assets(tmp_path, merger_module):
    stale_weights = {
        "model.safetensors",
        "model-00001-of-00002.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model.bin.index.json",
    }
    preserved = {"training_args.bin", "adapter_model.safetensors", "tokenizer.json"}
    for name in stale_weights | preserved:
        (tmp_path / name).write_bytes(b"x")

    merger_module.remove_stale_weight_artifacts(str(tmp_path))

    assert not stale_weights & {path.name for path in tmp_path.iterdir()}
    assert preserved <= {path.name for path in tmp_path.iterdir()}


def test_host_memory_preflight_is_opt_in_and_enforces_threshold(
    tmp_path, monkeypatch, merger_module, capsys
):
    shard1 = tmp_path / "rank0.pt"
    shard2 = tmp_path / "rank1.pt"
    shard1.write_bytes(b"a" * 10)
    shard2.write_bytes(b"b" * 20)

    monkeypatch.setattr(
        merger_module,
        "available_host_memory_bytes",
        lambda: (_ for _ in ()).throw(AssertionError("off policy read host memory")),
    )
    assert merger_module.maybe_preflight_host_memory([str(shard1), str(shard2)]) is None

    estimate = merger_module.preflight_host_memory(
        [str(shard1), str(shard2)], safety_factor=2.0, available_bytes=60
    )
    assert estimate.shard_bytes == 30
    assert estimate.required_bytes == 60

    with pytest.raises(MemoryError, match="Host-memory preflight failed"):
        merger_module.maybe_preflight_host_memory(
            [str(shard1), str(shard2)],
            policy="error",
            safety_factor=2.0,
            available_bytes=59,
        )

    assert (
        merger_module.maybe_preflight_host_memory(
            [str(shard1), str(shard2)],
            policy="warn",
            safety_factor=2.0,
            available_bytes=59,
        )
        is None
    )
    assert "WARNING: Host-memory preflight failed" in capsys.readouterr().out


def test_host_memory_preflight_cli_defaults_off(merger_module):
    args = merger_module.build_argument_parser().parse_args(["--local_dir", "/tmp/checkpoint"])

    assert args.host_memory_preflight == "off"
    assert args.host_memory_safety_factor == 2.0
    assert args.hf_upload_path is None

    compatible = merger_module.build_argument_parser().parse_args([
        "--local_dir", "/tmp/checkpoint", "--hf_upload_path", "HF-SI-Lab/model",
    ])
    assert compatible.hf_upload_path == "HF-SI-Lab/model"


def test_manifest_tracks_exact_aliases_without_tensor_copies(merger_module):
    state_dict = {
        "model.embed.weight": _FakeTensor((8, 4), pointer=1234),
        "lm_head.weight": _FakeTensor((8, 4), pointer=1234),
        "model.bias": _FakeTensor((4,)),
    }

    manifest = merger_module.build_state_dict_manifest(state_dict, label="test state_dict")

    assert manifest.tensors["model.embed.weight"].shape == (8, 4)
    assert manifest.tensors["model.embed.weight"].dtype == "BF16"
    assert manifest.alias_groups == (("lm_head.weight", "model.embed.weight"),)


def test_lora_manifest_comes_from_dense_merged_model_and_loading_is_offline(
    tmp_path, monkeypatch, merger_module
):
    local_dir = tmp_path / "checkpoint"
    adapter_dir = local_dir / "lora_adapter"
    hf_path = local_dir / "huggingface"
    adapter_dir.mkdir(parents=True)
    hf_path.mkdir()
    with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as handle:
        json.dump({"base_model_name_or_path": "/local/base-model"}, handle)
    (hf_path / "model.safetensors").write_bytes(b"stale")

    class FakeLoraConfig:
        @staticmethod
        def from_json_file(_path):
            return FakeLoraConfig()

    dense_state_dict = {"model.weight": _FakeTensor((2, 2))}
    save_calls = []

    class FakeMerged:
        config = SimpleNamespace(dtype="torch.float32")

        def state_dict(self):
            return dense_state_dict

        def save_pretrained(self, path, **kwargs):
            save_calls.append((path, kwargs))

    class FakePeftModel:
        def __init__(self):
            self.loaded_state_dict = None

        def load_state_dict(self, state_dict):
            self.loaded_state_dict = state_dict

        def merge_and_unload(self):
            return FakeMerged()

    peft_model = FakePeftModel()
    from_pretrained_calls = []

    class FakeAutoClass:
        @staticmethod
        def from_pretrained(path, **kwargs):
            from_pretrained_calls.append((path, kwargs))
            return object()

    monkeypatch.setattr(merger_module, "LoraConfig", FakeLoraConfig)
    monkeypatch.setattr(merger_module, "get_peft_model", lambda *_args: peft_model)
    adapter_state_dict = {"lora_A.weight": _FakeTensor((1, 2))}

    manifest = merger_module.merge_lora_into_base_and_save(
        str(local_dir), adapter_state_dict, str(hf_path), FakeAutoClass
    )

    assert set(manifest.tensors) == {"model.weight"}
    assert peft_model.loaded_state_dict is adapter_state_dict
    assert from_pretrained_calls[0][0] == "/local/base-model"
    assert from_pretrained_calls[0][1]["local_files_only"] is True
    assert save_calls[0][1]["state_dict"] is dense_state_dict
    assert not (hf_path / "model.safetensors").exists()
