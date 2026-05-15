#!/usr/bin/env python3
"""CPU runnable preflight checks for interleaved trajectory training.

Goals:
- Fail fast on missing/empty prompt template files.
- Preview rendered prompts for first/process turns.
- Sanity-check DataProto repeat/union alignment for n>1 with non-tensor fields.

This script does NOT run vLLM or require GPU.
"""

from __future__ import annotations

import argparse
import copy
import sys
from typing import Any, Dict, List, Optional

try:
    from jinja2 import Template
except Exception as e:  # pragma: no cover
    Template = None  # type: ignore


def _format_turn_prompt(template: Optional[str], question_text: str) -> str:
    if not template:
        return ""
    if "{{" in template and "}}" in template and Template is not None:
        try:
            return Template(template).render(content=question_text, question=question_text)
        except Exception:
            pass
    try:
        return template.format(question=question_text, content=question_text)
    except Exception:
        return template


def _load_template_from_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None

    candidate_paths = [path]
    if not path.startswith("/"):
        # scripts/ is under EasyR1/, so project root is one parent.
        import os

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidate_paths.append(os.path.join(project_root, path))

    resolved = None
    import os

    for p in candidate_paths:
        if os.path.exists(p):
            resolved = p
            break
    if resolved is None:
        return None

    with open(resolved, encoding="utf-8") as f:
        content = f.read().strip()
    return content if content else None


def _die(msg: str) -> None:
    print(f"[PreflightError] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load_required(path: str, name: str) -> str:
    content = _load_template_from_path(path)
    if not content:
        _die(f"{name} not found or empty: {path}")
    return content


def _preview(text: str, n: int) -> str:
    if text is None:
        return "<None>"
    t = str(text)
    return t if len(t) <= n else (t[:n] + "...")


def _simulate_alignment(n: int) -> None:
    """Simulate the trainer+rollout non-tensor behavior without vLLM."""
    try:
        import numpy as np
        import torch
        from tensordict import TensorDict

        from verl.protocol import DataProto
    except Exception as e:  # pragma: no cover
        print(f"[Preflight] Skip DataProto alignment (missing deps): {e}")
        return

    batch_size = 2
    prompt_len = 4

    tensors = {
        "input_ids": torch.arange(batch_size * prompt_len, dtype=torch.int64).view(batch_size, prompt_len),
        "attention_mask": torch.ones((batch_size, prompt_len), dtype=torch.int64),
        "position_ids": torch.arange(prompt_len, dtype=torch.int64).view(1, -1).expand(batch_size, -1),
    }

    non_tensors: Dict[str, Any] = {
        "raw_prompt_ids": np.array([[1, 2, 3], [4, 5]], dtype=object),
        "multi_modal_data": np.array(
            [{"image": [b"fakeimg0"]}, {"image": [b"fakeimg1"]}],
            dtype=object,
        ),
        "prompt": np.array(["Q: count objects", "Q: count cars"], dtype=object),
        "ground_truth": np.array(["1", "2"], dtype=object),
    }

    batch = DataProto(batch=TensorDict(source=tensors, batch_size=(batch_size,)), non_tensor_batch=non_tensors)

    # Trainer pop (mirrors ray_trainer.py after our patch)
    gen_batch = batch.pop(
        batch_keys=["input_ids", "attention_mask", "position_ids"],
        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "prompt"],
    )

    if len(gen_batch) != batch_size:
        _die(f"gen_batch length mismatch: {len(gen_batch)} != {batch_size}")

    # Simulate FSDP worker caching raw multi_modal_data before rollout preprocess
    cached_mm = copy.deepcopy(gen_batch.non_tensor_batch.get("multi_modal_data"))

    # Simulate vllm_rollout_spmd.generate_sequences non-tensor popping behavior
    non_tensor_for_rollout = dict(gen_batch.non_tensor_batch)

    # Extract question texts (same key order as rollout)
    question_texts: List[str] = []
    for i in range(batch_size):
        v = non_tensor_for_rollout.get("prompt")[i]
        question_texts.append(str(v) if isinstance(v, str) else "")

    # Pop question keys + raw_prompt_ids + multi_modal_data
    for k in ("problem", "prompt", "question", "query", "instruction"):
        non_tensor_for_rollout.pop(k, None)
    non_tensor_for_rollout.pop("raw_prompt_ids", None)
    non_tensor_for_rollout.pop("multi_modal_data", None)

    # Rollout output non-tensors should be empty at this point
    if len(non_tensor_for_rollout) != 0:
        _die(f"Unexpected leftover non_tensor keys from rollout: {list(non_tensor_for_rollout.keys())}")

    # Simulate fsdp_workers restoring multi_modal_data and repeating for n
    restored_mm = cached_mm
    if restored_mm is None:
        _die("cached multi_modal_data missing")
    if n > 1:
        restored_mm = np.repeat(restored_mm, repeats=n, axis=0)

    # Simulate rollout output batch (only shape matters for union)
    out_bs = batch_size * n
    rollout_out = DataProto(
        batch=TensorDict(
            source={
                "prompts": torch.zeros((out_bs, prompt_len), dtype=torch.int64),
                "responses": torch.zeros((out_bs, 3), dtype=torch.int64),
                "attention_mask": torch.ones((out_bs, prompt_len + 3), dtype=torch.int64),
                "response_mask": torch.ones((out_bs, 3), dtype=torch.int64),
                "position_ids": torch.zeros((out_bs, prompt_len + 3), dtype=torch.int64),
                "input_ids": torch.zeros((out_bs, prompt_len + 3), dtype=torch.int64),
            },
            batch_size=(out_bs,),
        ),
        non_tensor_batch={"multi_modal_data": restored_mm},
        meta_info={},
    )

    # Trainer repeats batch to align with rollout output
    batch.non_tensor_batch["uid"] = np.array(["u0", "u1"], dtype=object)
    batch_rep = batch.repeat(repeat_times=n, interleave=True)

    # Union should succeed and lengths must match
    merged = batch_rep.union(rollout_out)
    if len(merged) != out_bs:
        _die(f"Merged length mismatch: {len(merged)} != {out_bs}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interleaved-process-prompt-file", default=None)
    ap.add_argument("--interleaved-first-turn-prompt-file", default=None)
    ap.add_argument("--data-format-prompt", default=None, help="Path to data.format_prompt (jinja) used by dataset")
    ap.add_argument("--question", default="How many objects are there?")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--preview-chars", type=int, default=200)
    ap.add_argument(
        "--skip-alignment-check",
        action="store_true",
        help="Skip DataProto alignment simulation (useful in minimal envs without torch/numpy)",
    )
    args = ap.parse_args()

    print("[Preflight] interleaved trajectory checks")
    print(f"[Preflight] n={args.n}")

    if Template is None:
        print("[Preflight] Warning: jinja2 not available; will only use str.format rendering")

    if args.interleaved_process_prompt_file:
        process_tpl = _load_required(args.interleaved_process_prompt_file, "interleaved_process_prompt_file")
        print(f"[Preflight] Loaded process prompt template: {args.interleaved_process_prompt_file}")
        rendered = _format_turn_prompt(process_tpl, args.question)
        print(f"[Preflight] Rendered process prompt preview: {_preview(rendered, args.preview_chars)}")

    if args.interleaved_first_turn_prompt_file:
        first_tpl = _load_required(args.interleaved_first_turn_prompt_file, "interleaved_first_turn_prompt_file")
        print(f"[Preflight] Loaded first-turn prompt template: {args.interleaved_first_turn_prompt_file}")
        rendered = _format_turn_prompt(first_tpl, args.question)
        print(f"[Preflight] Rendered first-turn prompt preview: {_preview(rendered, args.preview_chars)}")

    if args.data_format_prompt:
        data_tpl = _load_required(args.data_format_prompt, "data.format_prompt")
        # dataset uses Template(...).render(content=prompt)
        # Here we just show the raw jinja content length so users catch empty files.
        print(f"[Preflight] Loaded data.format_prompt: {args.data_format_prompt} (chars={len(data_tpl)})")

    if not args.skip_alignment_check:
        _simulate_alignment(n=max(int(args.n), 1))
        print("[Preflight] DataProto alignment OK")


if __name__ == "__main__":
    main()
