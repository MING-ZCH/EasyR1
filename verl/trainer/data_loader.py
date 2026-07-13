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

from typing import Dict, List, Optional, Tuple, Union

import os
import json
import re

import torch
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


def _extract_sequence_id(image_path: str) -> str:
    """Extract sequence ID from image path (mirrors StepCount_mask_reward.py logic)."""
    basename = os.path.basename(image_path)
    basename = re.sub(r"\s+", "", basename)
    name, _ext = os.path.splitext(basename)
    match = re.match(r'^(.+?)_(\d+)$', name)
    if match:
        return match.group(1)
    return name


def _get_sample_seq_id(row, image_key: str) -> Optional[str]:
    """Extract sequence ID from a dataset row's first image."""
    images = row.get(image_key)
    if not images:
        return None
    img = images[0] if isinstance(images, list) else images
    img_path = None
    if isinstance(img, dict):
        img_path = img.get("path", "")
    elif isinstance(img, str):
        img_path = img
    if not img_path:
        return None
    return _extract_sequence_id(img_path)


def _load_mask_seq_ids(metadata_path: str) -> set:
    """Load unique sequence IDs from masks_metadata.json (list-of-dicts format)."""
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    ids = set()
    for item in metadata:
        ip = item.get("image_path", "") if isinstance(item, dict) else ""
        if ip:
            ids.add(_extract_sequence_id(ip))
    del metadata
    return ids


def _load_hard_seq_ids(hard_ref_path: str) -> set:
    """Load unique sequence IDs from a reference hard dataset (parquet dir or file)."""
    import pyarrow.parquet as pq
    import glob
    ids = set()
    if os.path.isdir(hard_ref_path):
        parquet_files = sorted(glob.glob(os.path.join(hard_ref_path, "**/*.parquet"), recursive=True))
    elif os.path.isfile(hard_ref_path):
        parquet_files = [hard_ref_path]
    else:
        return ids
    for pf_path in parquet_files:
        pf = pq.ParquetFile(pf_path)
        for batch in pf.iter_batches(batch_size=500, columns=["images"]):
            for i in range(batch.num_rows):
                imgs = batch.column("images")[i].as_py()
                if imgs and isinstance(imgs, list) and len(imgs) > 0:
                    img = imgs[0]
                    path = img.get("path", "") if isinstance(img, dict) else ""
                    if path:
                        ids.add(_extract_sequence_id(path))
    return ids


def _build_sample_weights(dataset, image_key: str) -> Optional[list]:
    """Build per-sample weights for weighted sampling.

    Supports two independent weight multipliers (multiplicative):
      1. TRAIN_OVERSAMPLE_NO_MASK_FACTOR: oversample samples without mask metadata
      2. TRAIN_HARD_OVERSAMPLE_FACTOR: oversample hard samples

    Env vars:
      TRAIN_OVERSAMPLE_NO_MASK_FACTOR: float >= 1 (default 1 = off)
      TRAIN_HARD_OVERSAMPLE_FACTOR: float >= 1 (default 1 = off)
      TRAIN_HARD_REFERENCE_PATH: path to hard-only dataset (parquet dir) for identification
      STEPCOUNT_MASKS_METADATA: path to masks_metadata.json

    Returns list of weights or None if all factors are 1.
    """
    no_mask_factor = float(os.environ.get("TRAIN_OVERSAMPLE_NO_MASK_FACTOR", "1"))
    hard_factor = float(os.environ.get("TRAIN_HARD_OVERSAMPLE_FACTOR", "1"))

    if no_mask_factor <= 1.0 and hard_factor <= 1.0:
        return None

    # Load reference ID sets as needed
    mask_seq_ids = None
    if no_mask_factor > 1.0:
        metadata_path = os.environ.get("STEPCOUNT_MASKS_METADATA", "")
        if metadata_path and os.path.exists(metadata_path):
            print(f"[Oversample] Loading mask metadata from {metadata_path} ...")
            mask_seq_ids = _load_mask_seq_ids(metadata_path)
            print(f"[Oversample] Loaded {len(mask_seq_ids)} mask sequence IDs")
        else:
            print(f"[Oversample] no_mask_factor={no_mask_factor} but STEPCOUNT_MASKS_METADATA not found, skipping no-mask oversample")
            no_mask_factor = 1.0

    hard_seq_ids = None
    if hard_factor > 1.0:
        hard_ref = os.environ.get("TRAIN_HARD_REFERENCE_PATH", "")
        if hard_ref and os.path.exists(hard_ref):
            print(f"[Oversample] Loading hard reference IDs from {hard_ref} ...")
            hard_seq_ids = _load_hard_seq_ids(hard_ref)
            print(f"[Oversample] Loaded {len(hard_seq_ids)} hard sequence IDs")
        else:
            print(f"[Oversample] hard_factor={hard_factor} but TRAIN_HARD_REFERENCE_PATH not found, skipping hard oversample")
            hard_factor = 1.0

    if no_mask_factor <= 1.0 and hard_factor <= 1.0:
        return None

    # Scan dataset and compute per-sample weights
    n_total = len(dataset)
    weights = [1.0] * n_total
    n_no_mask = 0
    n_hard = 0
    n_both = 0

    for i in range(n_total):
        row = dataset.dataset[i]
        seq_id = _get_sample_seq_id(row, image_key)
        if seq_id is None:
            continue

        w = 1.0
        is_no_mask = (mask_seq_ids is not None and seq_id not in mask_seq_ids)
        is_hard = (hard_seq_ids is not None and seq_id in hard_seq_ids)

        if is_no_mask:
            w *= no_mask_factor
            n_no_mask += 1
        if is_hard:
            w *= hard_factor
            n_hard += 1
        if is_no_mask and is_hard:
            n_both += 1

        weights[i] = w

    # Report statistics
    n_easy = n_total - n_hard
    w_sum = sum(weights)
    print(f"[Oversample] === Weight Distribution ===")
    print(f"[Oversample] Total samples: {n_total} (easy={n_easy}, hard={n_hard})")
    if no_mask_factor > 1.0:
        print(f"[Oversample] No-mask: {n_no_mask} ({n_no_mask/n_total:.1%}), factor={no_mask_factor}")
    if hard_factor > 1.0:
        print(f"[Oversample] Hard: {n_hard} ({n_hard/n_total:.1%}), factor={hard_factor}")
    if n_both > 0:
        print(f"[Oversample] Both (no-mask & hard): {n_both}")
    print(f"[Oversample] Effective sampling rates: "
          f"easy={n_easy/w_sum:.1%}, hard={sum(weights[i] for i in range(n_total) if weights[i] > 1.0)/w_sum:.1%}")

    return weights


def _safe_val_name(path: str) -> str:
    base = str(path).split("@", 1)[0].rstrip("/")
    name = os.path.basename(base) or base.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "val"


def _parse_val_file_specs(val_files: Union[str, List[str], Tuple[str, ...]]) -> List[Tuple[str, str]]:
    if isinstance(val_files, (list, tuple)):
        raw_specs = [str(x).strip() for x in val_files if str(x).strip()]
    else:
        raw = str(val_files or "").strip()
        raw_specs = [part.strip() for part in raw.split(",") if part.strip()]

    specs: List[Tuple[str, str]] = []
    used_names = set()
    for spec in raw_specs:
        if "::" in spec:
            name, path = spec.split("::", 1)
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_") or _safe_val_name(path)
            path = path.strip()
        else:
            path = spec
            name = _safe_val_name(path)

        base_name = name
        suffix = 2
        while name in used_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(name)
        specs.append((name, path))
    return specs


def _create_val_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin], data_path: str) -> StatefulDataLoader:
    val_dataset = RLHFDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        system_prompt_file=config.system_prompt_file,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_num_proc=config.filter_overlong_num_proc,
    )
    return StatefulDataLoader(
        dataset=val_dataset,
        batch_size=len(val_dataset) if config.val_batch_size == -1 else config.val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]):
    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        system_prompt_file=config.system_prompt_file,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_num_proc=config.filter_overlong_num_proc,
    )
    # use sampler for better ckpt resume
    oversample_weights = _build_sample_weights(train_dataset, config.image_key)
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        if oversample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=oversample_weights,
                num_samples=len(train_dataset),
                replacement=True,
                generator=train_dataloader_generator,
            )
        else:
            sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.rollout_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_specs = _parse_val_file_specs(config.val_files)
    if len(val_specs) == 1:
        val_dataloader = _create_val_dataloader(config, tokenizer, processor, val_specs[0][1])
    else:
        val_dataloader: Dict[str, StatefulDataLoader] = {}
        for name, path in val_specs:
            val_dataloader[name] = _create_val_dataloader(config, tokenizer, processor, path)

    assert len(train_dataloader) >= 1
    if isinstance(val_dataloader, dict):
        assert len(val_dataloader) >= 1
        for name, loader in val_dataloader.items():
            assert len(loader) >= 1
            print(f"Size of val dataloader[{name}]: {len(loader)}")
    else:
        assert len(val_dataloader) >= 1
        print(f"Size of val dataloader: {len(val_dataloader)}")
    print(f"Size of train dataloader: {len(train_dataloader)}")
    return train_dataloader, val_dataloader
