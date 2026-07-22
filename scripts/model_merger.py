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

import argparse
import dataclasses
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
)

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    shape: Tuple[int, ...]
    dtype: str


@dataclasses.dataclass(frozen=True)
class StateDictManifest:
    tensors: Dict[str, TensorSpec]
    alias_groups: Tuple[Tuple[str, ...], ...] = ()


@dataclasses.dataclass(frozen=True)
class HostMemoryEstimate:
    shard_bytes: int
    required_bytes: int
    available_bytes: int
    safety_factor: float


_SAFETENSORS_DTYPE_NAMES = {
    "bool": "BOOL",
    "torch.bool": "BOOL",
    "uint8": "U8",
    "torch.uint8": "U8",
    "int8": "I8",
    "torch.int8": "I8",
    "int16": "I16",
    "torch.int16": "I16",
    "int32": "I32",
    "torch.int32": "I32",
    "int64": "I64",
    "torch.int64": "I64",
    "float16": "F16",
    "torch.float16": "F16",
    "half": "F16",
    "bfloat16": "BF16",
    "torch.bfloat16": "BF16",
    "float32": "F32",
    "torch.float32": "F32",
    "float": "F32",
    "float64": "F64",
    "torch.float64": "F64",
    "double": "F64",
    "complex64": "C64",
    "torch.complex64": "C64",
    "complex128": "C128",
    "torch.complex128": "C128",
    "float8_e4m3fn": "F8_E4M3",
    "torch.float8_e4m3fn": "F8_E4M3",
    "float8_e5m2": "F8_E5M2",
    "torch.float8_e5m2": "F8_E5M2",
}
_SAFETENSORS_WEIGHT_RE = re.compile(r"^model(?:-\d+-of-\d+)?\.safetensors$")
_BIN_WEIGHT_RE = re.compile(r"^pytorch_model(?:-\d+-of-\d+)?\.bin$")


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def _canonical_dtype_name(dtype: object) -> str:
    raw = str(dtype)
    return _SAFETENSORS_DTYPE_NAMES.get(raw.lower(), raw.upper())


def _tensor_alias_identity(tensor: object) -> Optional[Tuple[object, ...]]:
    """Return an identity only for exact, materialized tensor aliases."""
    try:
        if tensor.numel() == 0 or getattr(tensor.device, "type", None) == "meta":
            return None
        return (
            str(tensor.device),
            int(tensor.data_ptr()),
            int(tensor.storage_offset()),
            tuple(int(dim) for dim in tensor.shape),
            tuple(int(dim) for dim in tensor.stride()),
            _canonical_dtype_name(tensor.dtype),
        )
    except (AttributeError, RuntimeError, TypeError):
        return None


def build_state_dict_manifest(state_dict: Mapping[str, object], *, label: str) -> StateDictManifest:
    """Snapshot tensor metadata without copying or materializing tensor data."""
    if not state_dict:
        raise RuntimeError(f"{label} is empty")

    tensors: Dict[str, TensorSpec] = {}
    aliases: Dict[Tuple[object, ...], List[str]] = {}
    for key in sorted(state_dict):
        tensor = state_dict[key]
        try:
            shape = tuple(int(dim) for dim in tensor.shape)
            dtype = _canonical_dtype_name(tensor.dtype)
        except (AttributeError, TypeError) as exc:
            raise TypeError(f"{label} entry {key!r} is not tensor-like") from exc
        tensors[key] = TensorSpec(shape=shape, dtype=dtype)
        alias_identity = _tensor_alias_identity(tensor)
        if alias_identity is not None:
            aliases.setdefault(alias_identity, []).append(key)

    alias_groups = tuple(
        sorted(tuple(sorted(keys)) for keys in aliases.values() if len(keys) > 1)
    )
    return StateDictManifest(tensors=tensors, alias_groups=alias_groups)


def _sample(values: Sequence[object], limit: int = 8) -> str:
    rendered = [str(value) for value in values[:limit]]
    if len(values) > limit:
        rendered.append(f"... (+{len(values) - limit} more)")
    return "[" + ", ".join(rendered) + "]"


def validate_shard_key_sets(
    shards: Sequence[Iterable[str]],
    *,
    ranks: Optional[Sequence[int]] = None,
) -> Set[str]:
    """Validate all shard key sets symmetrically before any shard is mutated."""
    if not shards:
        raise RuntimeError("No FSDP model shards were loaded")
    if ranks is None:
        ranks = tuple(range(len(shards)))
    if len(ranks) != len(shards):
        raise ValueError("ranks and shards must have the same length")

    reference_keys = set(shards[0])
    if not reference_keys:
        raise RuntimeError(f"FSDP model shard rank {ranks[0]} has an empty state_dict")

    mismatches = []
    for rank, shard in zip(ranks[1:], shards[1:]):
        shard_keys = set(shard)
        missing = sorted(reference_keys - shard_keys)
        unexpected = sorted(shard_keys - reference_keys)
        if missing or unexpected:
            mismatches.append(
                f"rank {rank}: missing={len(missing)} {_sample(missing)}, "
                f"unexpected={len(unexpected)} {_sample(unexpected)}"
            )
    if mismatches:
        raise RuntimeError(
            f"FSDP shard key-set mismatch against rank {ranks[0]}: " + "; ".join(mismatches)
        )
    return reference_keys


def load_shard_keys_mmap(model_path: str) -> Set[str]:
    """Inspect a PyTorch ZIP checkpoint without making tensor payloads resident."""
    try:
        state_dict = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except (OSError, RuntimeError, TypeError) as exc:
        raise RuntimeError(
            f"Cannot inspect HSDP replica shard keys with mmap without materializing tensors: {model_path}"
        ) from exc
    keys = set(state_dict)
    del state_dict
    return keys


def validate_device_mesh_rank_order(mesh: object, world_size: int) -> None:
    """Fail instead of concatenating shards in the wrong order for a custom mesh."""
    try:
        mesh_ranks = [int(rank) for rank in mesh.reshape(-1).tolist()]
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Cannot inspect DeviceMesh rank order: {mesh}") from exc
    expected_ranks = list(range(world_size))
    if mesh_ranks != expected_ranks:
        raise RuntimeError(
            "Non-canonical DeviceMesh rank order is not supported by this merger; "
            f"expected={expected_ranks}, actual={mesh_ranks}"
        )


def validate_state_dict_manifest(
    actual: StateDictManifest,
    expected: StateDictManifest,
    *,
    context: str,
    check_dtype: bool,
    allow_missing_aliases: bool = False,
) -> None:
    actual_keys = set(actual.tensors)
    expected_keys = set(expected.tensors)
    missing = expected_keys - actual_keys
    if allow_missing_aliases and missing:
        for alias_group in expected.alias_groups:
            aliases = set(alias_group)
            if aliases & actual_keys:
                missing -= aliases
    unexpected = actual_keys - expected_keys

    common_keys = actual_keys & expected_keys
    shape_mismatches = sorted(
        (key, expected.tensors[key].shape, actual.tensors[key].shape)
        for key in common_keys
        if actual.tensors[key].shape != expected.tensors[key].shape
    )
    dtype_mismatches = []
    if check_dtype:
        dtype_mismatches = sorted(
            (key, expected.tensors[key].dtype, actual.tensors[key].dtype)
            for key in common_keys
            if actual.tensors[key].dtype != expected.tensors[key].dtype
        )

    if missing or unexpected or shape_mismatches or dtype_mismatches:
        details = [
            f"missing={len(missing)} {_sample(sorted(missing))}",
            f"unexpected={len(unexpected)} {_sample(sorted(unexpected))}",
            f"shape_mismatches={len(shape_mismatches)} {_sample(shape_mismatches)}",
        ]
        if check_dtype:
            details.append(f"dtype_mismatches={len(dtype_mismatches)} {_sample(dtype_mismatches)}")
        raise RuntimeError(f"{context} manifest mismatch: " + "; ".join(details))


def mark_config_bfloat16(config: PretrainedConfig) -> None:
    """Keep saved HF config dtype aligned with the merged BF16 tensors."""
    for candidate in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "vision_config", None),
    ):
        if candidate is None:
            continue
        if hasattr(candidate, "dtype"):
            candidate.dtype = torch.bfloat16
        elif hasattr(candidate, "torch_dtype"):
            candidate.torch_dtype = torch.bfloat16


def remove_stale_weight_artifacts(hf_path: str) -> None:
    """Prevent old shards or indexes from surviving a new merge."""
    for name in os.listdir(hf_path):
        is_weight = bool(_SAFETENSORS_WEIGHT_RE.fullmatch(name) or _BIN_WEIGHT_RE.fullmatch(name))
        is_weight_index = name in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
        if is_weight or is_weight_index:
            os.remove(os.path.join(hf_path, name))


def verify_saved_bfloat16_model(
    hf_path: str,
    expected_manifest: Optional[StateDictManifest] = None,
) -> List[str]:
    """Verify saved weights from safetensors headers without loading tensor payloads."""
    weight_files = sorted(name for name in os.listdir(hf_path) if _SAFETENSORS_WEIGHT_RE.fullmatch(name))
    bin_files = sorted(name for name in os.listdir(hf_path) if _BIN_WEIGHT_RE.fullmatch(name))
    if not weight_files or bin_files:
        raise RuntimeError(
            f"Expected safetensors-only merged weights; safetensors={weight_files}, bin={bin_files}"
        )

    index_path = os.path.join(hf_path, "model.safetensors.index.json")
    weight_map = None
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or not all(isinstance(key, str) and isinstance(name, str) for key, name in weight_map.items())
        ):
            raise RuntimeError("Saved safetensors index has no non-empty weight_map")
        indexed_files = set(weight_map.values())
        if indexed_files != set(weight_files):
            raise RuntimeError(
                "Saved safetensors index files do not match the weight shards: "
                f"index={sorted(indexed_files)}, files={weight_files}"
            )
    elif weight_files != ["model.safetensors"]:
        raise RuntimeError(f"Multiple safetensors shards require model.safetensors.index.json: {weight_files}")

    observed_tensors: Dict[str, TensorSpec] = {}
    observed_weight_map: Dict[str, str] = {}
    for name in weight_files:
        with safe_open(os.path.join(hf_path, name), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in observed_tensors:
                    raise RuntimeError(f"Duplicate tensor key across saved shards: {key}")
                tensor_slice = handle.get_slice(key)
                observed_tensors[key] = TensorSpec(
                    shape=tuple(int(dim) for dim in tensor_slice.get_shape()),
                    dtype=_canonical_dtype_name(tensor_slice.get_dtype()),
                )
                observed_weight_map[key] = name
    if not observed_tensors:
        raise RuntimeError("Merged safetensors output contains no tensors")
    if weight_map is not None and weight_map != observed_weight_map:
        raise RuntimeError("Saved safetensors index key-to-shard mapping does not match tensor headers")

    observed_manifest = StateDictManifest(tensors=observed_tensors)
    if expected_manifest is None:
        observed_dtypes = {spec.dtype for spec in observed_tensors.values()}
        if observed_dtypes != {"BF16"}:
            raise RuntimeError(
                f"Merged tensor verification failed: tensors={len(observed_tensors)}, "
                f"dtypes={sorted(observed_dtypes)}"
            )
    else:
        validate_state_dict_manifest(
            observed_manifest,
            expected_manifest,
            context="Saved safetensors",
            check_dtype=True,
            allow_missing_aliases=True,
        )
    return weight_files


def available_host_memory_bytes() -> Optional[int]:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass

    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def preflight_host_memory(
    shard_paths: Sequence[str],
    *,
    safety_factor: float = 2.0,
    available_bytes: Optional[int] = None,
) -> HostMemoryEstimate:
    """Apply a checkpoint-size heuristic; this is not an OOM guarantee."""
    if safety_factor < 1.0:
        raise ValueError("host-memory safety_factor must be at least 1.0")
    shard_bytes = sum(os.path.getsize(path) for path in shard_paths)
    required_bytes = int(shard_bytes * safety_factor)
    if available_bytes is None:
        available_bytes = available_host_memory_bytes()
    if available_bytes is None:
        raise RuntimeError("Cannot determine available host memory for merger preflight")

    estimate = HostMemoryEstimate(
        shard_bytes=shard_bytes,
        required_bytes=required_bytes,
        available_bytes=available_bytes,
        safety_factor=safety_factor,
    )
    if available_bytes < required_bytes:
        gib = 1024**3
        raise MemoryError(
            "Host-memory preflight failed: "
            f"checkpoint_files={shard_bytes / gib:.2f} GiB, "
            f"estimated_required={required_bytes / gib:.2f} GiB "
            f"({safety_factor:.2f}x), MemAvailable={available_bytes / gib:.2f} GiB. "
            "The estimate is heuristic; adjust the factor or use policy=off only when capacity is managed externally."
        )
    return estimate


def maybe_preflight_host_memory(
    shard_paths: Sequence[str],
    *,
    policy: str = "off",
    safety_factor: float = 2.0,
    available_bytes: Optional[int] = None,
) -> Optional[HostMemoryEstimate]:
    if policy not in {"off", "warn", "error"}:
        raise ValueError(f"Unknown host-memory preflight policy: {policy}")
    if policy == "off":
        return None
    try:
        return preflight_host_memory(
            shard_paths,
            safety_factor=safety_factor,
            available_bytes=available_bytes,
        )
    except (MemoryError, RuntimeError) as exc:
        if policy == "warn":
            print(f"WARNING: {exc}")
            return None
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str, help="The path for your saved model")
    parser.add_argument(
        "--hf_upload_path",
        default=None,
        type=str,
        help="Deprecated compatibility option: upload the verified local HF output to this repo",
    )
    parser.add_argument(
        "--host-memory-preflight",
        choices=("off", "warn", "error"),
        default="off",
        help="Optional checkpoint-size host-memory check before loading remaining shards (default: off)",
    )
    parser.add_argument(
        "--host-memory-safety-factor",
        default=2.0,
        type=float,
        help="Multiplier applied to checkpoint shard bytes by the optional host-memory preflight",
    )
    return parser


def upload_model_to_huggingface(local_path: str, remote_path: str) -> None:
    """Upload only after local structural verification has completed."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=remote_path, private=False, exist_ok=True)
    api.upload_folder(repo_id=remote_path, folder_path=local_path, repo_type="model")


def merge_lora_into_base_and_save(
    local_dir: str,
    state_dict: dict[str, torch.Tensor],
    hf_path: str,
    auto_class: type[PreTrainedModel],
) -> Optional[StateDictManifest]:
    """Merge LoRA weights into the base model before saving a dense HF checkpoint for vLLM."""
    adapter_cfg = os.path.join(local_dir, "lora_adapter", "adapter_config.json")
    if not os.path.isfile(adapter_cfg):
        return None

    with open(adapter_cfg, encoding="utf-8") as f:
        raw_cfg = json.load(f)
    base_path = raw_cfg.get("base_model_name_or_path")
    if not base_path:
        raise ValueError(f"adapter_config.json has no base_model_name_or_path: {adapter_cfg}")

    peft_config = LoraConfig.from_json_file(adapter_cfg)
    if isinstance(peft_config, dict):
        field_names = {f.name for f in dataclasses.fields(LoraConfig)}
        kwargs = {k: v for k, v in peft_config.items() if k in field_names}
        peft_config = LoraConfig(**kwargs)

    print(f"Loading base model from {base_path}...")
    base_model = auto_class.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    peft_model = get_peft_model(base_model, peft_config)
    peft_model.load_state_dict(state_dict)

    print("Merging LoRA weights into base model...")
    merged = peft_model.merge_and_unload()
    mark_config_bfloat16(merged.config)
    merged_state_dict = merged.state_dict()
    output_manifest = build_state_dict_manifest(merged_state_dict, label="Merged LoRA state_dict")
    remove_stale_weight_artifacts(hf_path)
    print(f"Saving merged model to {hf_path}...")
    merged.save_pretrained(
        hf_path,
        state_dict=merged_state_dict,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    return output_manifest


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    local_dir: str = os.path.abspath(args.local_dir)

    assert not local_dir.endswith("huggingface"), "The local_dir should not end with huggingface."

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break

    assert world_size, "No model file with the proper format."

    world_size = int(world_size)
    expected_shards = [
        os.path.join(local_dir, f"model_world_size_{world_size}_rank_{shard_rank}.pt")
        for shard_rank in range(world_size)
    ]
    missing_shards = [path for path in expected_shards if not os.path.isfile(path) or os.path.getsize(path) == 0]
    if missing_shards:
        raise FileNotFoundError(f"Missing or empty model shards: {missing_shards}")

    rank0_weight_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    state_dict = torch.load(rank0_weight_path, map_location="cpu", weights_only=False)
    validate_shard_key_sets([state_dict], ranks=[rank])
    dtensor_weight = next((tensor for tensor in state_dict.values() if isinstance(tensor, DTensor)), None)
    if dtensor_weight is not None:
        # get sharding info
        device_mesh = dtensor_weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        # for non-DTensor
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    if mesh_dim_names not in (("fsdp",), ("ddp", "fsdp")):
        raise RuntimeError(f"Unsupported mesh_dim_names {mesh_dim_names}.")
    if dtensor_weight is not None:
        validate_device_mesh_rank_order(mesh, world_size)

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    total_shards = int(total_shards)
    if total_shards < 1 or total_shards > world_size:
        raise RuntimeError(
            f"Invalid participant shard count {total_shards} for checkpoint world_size {world_size}"
        )

    memory_estimate = maybe_preflight_host_memory(
        expected_shards[:total_shards],
        policy=args.host_memory_preflight,
        safety_factor=args.host_memory_safety_factor,
    )
    if memory_estimate is not None:
        gib = 1024**3
        print(
            "Host-memory preflight completed for participant shards: "
            f"checkpoint_files={memory_estimate.shard_bytes / gib:.2f} GiB, "
            f"estimated_required={memory_estimate.required_bytes / gib:.2f} GiB, "
            f"MemAvailable={memory_estimate.available_bytes / gib:.2f} GiB."
        )

    print(f"Processing {total_shards} model shards in total.")

    # HSDP only merges the first FSDP replica group. Validate the other rank files
    # one at a time so integrity coverage does not make all replicas resident at once.
    for shard_rank in range(total_shards, world_size):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{shard_rank}.pt")
        replica_keys = load_shard_keys_mmap(model_path)
        validate_shard_key_sets([state_dict, replica_keys], ranks=[rank, shard_rank])

    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank, model_state_dict_lst):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
        futures = [
            executor.submit(process_one_shard, shard_rank, model_state_dict_lst)
            for shard_rank in range(1, total_shards)
        ]
        for future in futures:
            future.result()

    keys = validate_shard_key_sets(model_state_dict_lst, ranks=tuple(range(total_shards)))
    print(f"Validated identical key sets across {world_size} model rank shards ({len(keys)} keys).")

    state_dict: Dict[str, List[torch.Tensor]] = {}
    param_placements: Dict[str, Tuple[Placement, ...]] = {}
    for key in sorted(keys):
        state_dict[key] = []
        for shard_rank, model_state_dict in enumerate(model_state_dict_lst):
            try:
                tensor = model_state_dict.pop(key)
            except KeyError as exc:
                raise KeyError(f"Cannot find key {key} in model shard rank {shard_rank}") from exc

            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at ddp dimension can be discarded
                if mesh_dim_names[0] == "ddp":
                    placements = placements[1:]

                if key not in param_placements:
                    param_placements[key] = placements
                elif param_placements[key] != placements:
                    raise RuntimeError(
                        f"DTensor placement mismatch for {key} in rank {shard_rank}: "
                        f"expected={param_placements[key]}, actual={placements}"
                    )
            else:
                state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue

        if key in param_placements:
            # merge shards
            placements: Tuple[Shard] = param_placements[key]
            if len(mesh_shape) == 1:
                # 1-D list, FSDP without TP
                if len(placements) != 1:
                    raise RuntimeError(f"Expected one FSDP placement for {key}, got {placements}")
                shards = state_dict[key]
                state_dict[key] = merge_by_placement(shards, placements[0])
            else:
                # 2-D list, FSDP + TP
                raise NotImplementedError("FSDP + TP is not supported yet.")
        else:
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    print("Merge completed.")
    hf_path = os.path.join(local_dir, "huggingface")

    if not os.path.isdir(hf_path):
        raise ValueError(
            f"Directory {hf_path} does not exist. Please create it and copy the base model's "
            "config.json (and tokenizer files) into it."
        )

    config: PretrainedConfig = AutoConfig.from_pretrained(hf_path, local_files_only=True)
    mark_config_bfloat16(config)
    architectures: List[str] = getattr(config, "architectures", ["Unknown"])

    # Fix for TypeError if architectures is None
    if architectures is None:
        architectures = ["Unknown"]

    if "ForTokenClassification" in architectures[0]:
        AutoClass = AutoModelForTokenClassification
    elif "ForCausalLM" in architectures[0]:
        AutoClass = AutoModelForCausalLM
    elif "ForConditionalGeneration" in architectures[0]:
        AutoClass = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f"Unknown architecture {architectures}.")

    output_manifest = merge_lora_into_base_and_save(local_dir, state_dict, hf_path, AutoClass)
    if output_manifest is not None:
        del state_dict
    else:
        with torch.device("meta"):
            model: PreTrainedModel = AutoClass.from_config(config, torch_dtype=torch.bfloat16)

        assert isinstance(model, PreTrainedModel)
        output_manifest = build_state_dict_manifest(state_dict, label="Merged FSDP state_dict")
        model_manifest = build_state_dict_manifest(model.state_dict(), label="HF model schema")
        validate_state_dict_manifest(
            output_manifest,
            model_manifest,
            context="Merged FSDP state_dict versus HF model schema",
            check_dtype=False,
        )
        model.to_empty(device="cpu")

        remove_stale_weight_artifacts(hf_path)
        print(f"Saving model to {hf_path}...")
        model.save_pretrained(
            hf_path,
            state_dict=state_dict,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        del state_dict, model

    saved_weight_files = verify_saved_bfloat16_model(hf_path, expected_manifest=output_manifest)
    print(
        f"Verified {len(output_manifest.tensors)} state_dict entries across "
        f"{len(saved_weight_files)} HF weight files in {hf_path}: {saved_weight_files}"
    )
    if args.hf_upload_path:
        upload_model_to_huggingface(hf_path, args.hf_upload_path)
