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

import os
import re
import json
from io import BytesIO
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams
from PIL import Image as PILImage
from PIL import ImageDraw
from PIL import ImageFont
from jinja2 import Template

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.tokenizer import get_processor
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(model_path: str, trust_remote_code: bool) -> Optional[Dict[int, float]]:
    processor = get_processor(model_path, trust_remote_code=trust_remote_code)
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _resolve_stop_token_id(tokenizer: PreTrainedTokenizer, stop_text: str) -> Optional[int]:
    token_ids = tokenizer.encode(stop_text, add_special_tokens=False)
    if len(token_ids) == 1:
        return int(token_ids[0])

    token_id = tokenizer.convert_tokens_to_ids(stop_text)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if isinstance(token_id, int) and token_id >= 0 and token_id != unk_id:
        return token_id

    return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any],
    min_pixels: int,
    max_pixels: int,
    video_fps: float,
    return_video_metadata: bool = False,
) -> dict[str, Any]:
    """Convert paths / raw multimodal payloads into vLLM-ready dict.

    Supports both upstream-style keys (`images` / `videos`) and this fork's dataset keys
    (`image` / `video`).
    """
    images, videos = [], []
    image_iterable = multi_modal_data.get("images")
    if image_iterable is None and "image" in multi_modal_data:
        image_iterable = multi_modal_data["image"]
    if image_iterable is not None:
        if not isinstance(image_iterable, list):
            image_iterable = [image_iterable]
        for image in image_iterable:
            images.append(process_image(image, min_pixels, max_pixels))

    video_iterable = multi_modal_data.get("videos")
    if video_iterable is None and "video" in multi_modal_data:
        video_iterable = multi_modal_data["video"]
    if video_iterable is not None:
        if not isinstance(video_iterable, list):
            video_iterable = [video_iterable]
        for video in video_iterable:
            videos.append(
                process_video(
                    video,
                    min_pixels,
                    max_pixels,
                    video_fps,
                    return_metadata=return_video_metadata,
                )
            )

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    # No recognizable multimodal fields; keep original (already vLLM-ready or empty).
    return multi_modal_data


class _LLMProxy:
    """A thin proxy so we can hot-swap the underlying vLLM engine.

    FSDPVLLMShardingManager holds a reference to `inference_engine`. If we ever need to
    rebuild the vLLM LLM (e.g., enable chunked prefill after OOM), we can swap the
    underlying engine without changing the reference held by the sharding manager.
    """

    def __init__(self, engine: LLM):
        self._engine = engine

    def swap(self, engine: LLM) -> None:
        self._engine = engine

    def __getattr__(self, name: str):
        return getattr(self._engine, name)


def _to_pil_image(image_obj: Any) -> Optional[PILImage.Image]:
    if isinstance(image_obj, PILImage.Image):
        return image_obj.convert("RGB") if image_obj.mode != "RGB" else image_obj
    if isinstance(image_obj, str) and os.path.exists(image_obj):
        with PILImage.open(image_obj) as im:
            return im.convert("RGB")
    if isinstance(image_obj, dict) and "bytes" in image_obj:
        with PILImage.open(BytesIO(image_obj["bytes"])) as im:
            return im.convert("RGB")
    if isinstance(image_obj, (bytes, bytearray)):
        with PILImage.open(BytesIO(image_obj)) as im:
            return im.convert("RGB")
    return None


def _parse_points_from_text(text: str) -> List[Tuple[float, float, Optional[int]]]:
    parsed_points: List[Tuple[float, float, Optional[int]]] = []
    point_matches = re.findall(r"<point>(.*?)</point>", text, re.DOTALL)
    if not point_matches:
        return parsed_points

    for payload_raw in point_matches:
        payload = payload_raw.strip()
        parsed: Optional[Tuple[float, float, Optional[int]]] = None

        try:
            data = json.loads(payload)
            if isinstance(data, dict) and "point_2d" in data:
                p = data["point_2d"]
                count_number = data.get("count_number")
                parsed_count = None
                try:
                    if count_number is not None:
                        parsed_count = int(count_number)
                except Exception:
                    parsed_count = None
                parsed = (float(p[0]), float(p[1]), parsed_count)
        except Exception:
            parsed = None

        if parsed is None:
            coord_match = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", payload)
            if coord_match:
                parsed = (float(coord_match.group(1)), float(coord_match.group(2)), None)

        if parsed is not None:
            parsed_points.append(parsed)

    return parsed_points


def _clean_question_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"<\s*image\s*>", " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _draw_point_on_image(image_obj: Any, point: Tuple[float, float], point_number: int) -> Any:
    image = _to_pil_image(image_obj)
    if image is None:
        return image_obj

    BASE_DOT_RADIUS = 10
    BASE_FONT_SIZE = 20

    x, y = point
    width, height = image.size
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x *= width
        y *= height

    scale = max(0.6, min(width, height) / 1024.0)
    radius = max(4, int(round(BASE_DOT_RADIUS * scale)))
    font_size = max(10, int(round(BASE_FONT_SIZE * scale)))

    x = max(0, min(int(round(x)), width - 1))
    y = max(0, min(int(round(y)), height - 1))

    draw = ImageDraw.Draw(image)

    # Solid red dot with white border — matches eval drawing style.
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 fill=(255, 0, 0), outline=(255, 255, 255), width=2)

    try:
        if os.name == "nt":
            font = ImageFont.truetype("arial.ttf", font_size)
        else:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    text = str(point_number)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    # 文本默认放在点外侧，优先尝试四个象限，尽量减少对目标遮挡。
    offset = radius + max(6, int(4 * scale))
    candidates = [
        (x + offset, y - offset - text_h),
        (x + offset, y + offset),
        (x - offset - text_w, y - offset - text_h),
        (x - offset - text_w, y + offset),
    ]

    chosen_x, chosen_y = candidates[0]
    for cx, cy in candidates:
        if 0 <= cx <= max(0, width - text_w - 1) and 0 <= cy <= max(0, height - text_h - 1):
            chosen_x, chosen_y = cx, cy
            break

    bg_pad = 2
    draw.rectangle(
        [
            chosen_x - bg_pad,
            chosen_y - bg_pad,
            chosen_x + text_w + bg_pad,
            chosen_y + text_h + bg_pad,
        ],
        fill=(255, 255, 255),
    )
    draw.text((chosen_x, chosen_y), text, font=font, fill=(0, 0, 0))

    return image


def _extract_question_text(non_tensor_batch: Dict[str, Any], index: int) -> str:
    for key in ("problem", "prompt", "question", "query", "instruction"):
        if key in non_tensor_batch:
            try:
                value = non_tensor_batch[key][index]
            except Exception:
                continue
            if isinstance(value, str):
                return _clean_question_text(value)
    return ""


def _format_turn_prompt(template: Optional[str], question_text: str) -> str:
    if not template:
        return ""
    if "{{" in template and "}}" in template:
        try:
            return Template(template).render(content=question_text, question=question_text)
        except Exception:
            pass
    # Use simple string replace to avoid conflicts with JSON braces in templates
    result = template.replace("{question}", question_text).replace("{content}", question_text)
    return result


def _encode_followup_chat_tokens(tokenizer: PreTrainedTokenizer, text: str) -> List[int]:
    if not text:
        return []

    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    im_start_id = tokenizer.convert_tokens_to_ids(im_start)
    im_end_id = tokenizer.convert_tokens_to_ids(im_end)
    unk_id = getattr(tokenizer, "unk_token_id", None)

    has_chat_tokens = (
        isinstance(im_start_id, int)
        and isinstance(im_end_id, int)
        and im_start_id >= 0
        and im_end_id >= 0
        and im_start_id != unk_id
        and im_end_id != unk_id
    )

    if not has_chat_tokens:
        # Fallback for non-chat templates
        return tokenizer.encode("\n" + text + "\n", add_special_tokens=False)

    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    user_ids = tokenizer.encode("user", add_special_tokens=False)
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
    tokens: List[int] = []
    tokens.append(int(im_start_id))
    tokens.extend(user_ids)
    tokens.extend(newline_ids)
    tokens.extend(text_ids)
    tokens.append(int(im_end_id))
    tokens.extend(newline_ids)
    tokens.append(int(im_start_id))
    tokens.extend(assistant_ids)
    tokens.extend(newline_ids)
    return tokens


def _encode_assistant_generation_prompt(tokenizer: PreTrainedTokenizer) -> List[int]:
    """Best-effort token ids for starting an assistant generation turn.

    For Qwen-style chat templates this corresponds to: "\n<|im_start|>assistant\n".
    If the tokenizer doesn't have chat tokens, returns a newline.
    """

    im_start = "<|im_start|>"
    im_start_id = tokenizer.convert_tokens_to_ids(im_start)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if not (isinstance(im_start_id, int) and im_start_id >= 0 and im_start_id != unk_id):
        return tokenizer.encode("\n", add_special_tokens=False)

    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    # \n<|im_start|>assistant\n
    return list(newline_ids) + [int(im_start_id)] + list(assistant_ids) + list(newline_ids)


def _load_template_from_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None

    candidate_paths = [path]
    if not os.path.isabs(path):
        # Best-effort fallback: resolve relative to EasyR1 project root.
        # __file__: .../EasyR1/verl/workers/rollout/vllm_rollout_spmd.py
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        candidate_paths.append(os.path.join(project_root, path))

    resolved_path = None
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            resolved_path = candidate
            break

    if resolved_path is None:
        return None

    with open(resolved_path, encoding="utf-8") as f:
        content = f.read().strip()
    return content if content else None


def _require_template_from_path(path: Optional[str], field_name: str) -> Optional[str]:
    if not path:
        return None

    content = _load_template_from_path(path)
    if content:
        return content

    candidate_paths = [path]
    if not os.path.isabs(path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        candidate_paths.append(os.path.join(project_root, path))

    raise FileNotFoundError(
        f"Failed to load {field_name} from '{path}'. Tried: {candidate_paths}. "
        "If you want the built-in default, do not set the *_prompt_file field."
    )


def _dtype_nbytes(dtype_str: str) -> int:
    ds = (dtype_str or "").lower()
    if ds in {"bf16", "bfloat16", "float16", "fp16", "half"}:
        return 2
    if ds in {"float32", "fp32"}:
        return 4
    if ds in {"float64", "fp64"}:
        return 8
    return 2


def _get_cuda_total_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.get_device_properties(0).total_memory)


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg or "failed to allocate" in msg


def _get_prompt_lens(attention_mask: torch.Tensor) -> torch.Tensor:
    # attention_mask is left-padded; sum gives effective length
    if attention_mask.dtype != torch.int64 and attention_mask.dtype != torch.int32:
        return attention_mask.long().sum(dim=-1)
    return attention_mask.sum(dim=-1)


def _has_low_diversity_tail(token_ids: List[int], tail_k: int = 128, max_unique: int = 2) -> bool:
    if not token_ids:
        return False
    tail = token_ids[-tail_k:] if len(token_ids) > tail_k else token_ids
    if len(tail) < max(16, tail_k // 4):
        return False
    return len(set(int(x) for x in tail)) <= max_unique


def _extract_attn_meta(model: Any) -> Optional[Tuple[int, int, int]]:
    """Return (num_layers, num_kv_heads, head_dim) if available."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None

    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    num_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    num_kv_heads = (
        getattr(cfg, "num_key_value_heads", None)
        or getattr(cfg, "num_kv_heads", None)
        or getattr(cfg, "num_kv_head", None)
        or num_heads
    )
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
        if hidden is not None and num_heads is not None and num_heads > 0:
            head_dim = hidden // num_heads

    if not (num_layers and num_kv_heads and head_dim):
        return None
    return int(num_layers), int(num_kv_heads), int(head_dim)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        _proc = get_processor(model_path, trust_remote_code=config.trust_remote_code)
        self.return_video_metadata = (
            _proc is not None and "Qwen3VLProcessor" in _proc.__class__.__name__
        )
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        required_tokens = config.prompt_length + config.response_length
        if config.max_num_batched_tokens < required_tokens:
            # Auto-adjust max_num_batched_tokens with a buffer
            adjusted_value = required_tokens + 2048  # Add 2k buffer for safety
            if self.rank == 0:
                print(
                    f"Warning: max_num_batched_tokens ({config.max_num_batched_tokens}) is less than "
                    f"prompt_length + response_length ({config.prompt_length} + {config.response_length} = {required_tokens}). "
                    f"Auto-adjusting to {adjusted_value}."
                )
            config.max_num_batched_tokens = adjusted_value

        self._engine_kwargs = {}
        if config.limit_images:
            self._engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self._engine_init_args = dict(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            enable_sleep_mode=True,
        )

        engine = self._create_engine(enable_chunked_prefill=config.enable_chunked_prefill)
        self.inference_engine = _LLMProxy(engine)
        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        self._attn_meta: Optional[Tuple[int, int, int]] = None

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(model_path, trust_remote_code=config.trust_remote_code),
        }
        default_sampling_params = SamplingParams()
        # FIX(seed-leak): Exclude 'seed' from config→SamplingParams copy.
        # RolloutConfig.seed is intended for the vLLM *engine* constructor
        # (reproducible init), NOT for SamplingParams (per-request determinism).
        # Leaking seed=1 into SamplingParams causes ALL n copies of the same
        # prompt to produce IDENTICAL outputs, defeating RL exploration.
        _SAMPLING_PARAMS_SKIP_KEYS = {"seed"}
        for key in config.to_dict().keys():
            if key in _SAMPLING_PARAMS_SKIP_KEYS:
                continue
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        # vLLM requires detokenize=True when using stop strings.
        # Auto-convert stop strings to stop_token_ids when possible to keep detokenize=False fast path.
        stop_strings = sampling_kwargs.get("stop")
        detokenize = bool(sampling_kwargs.get("detokenize", False))
        if stop_strings and not detokenize:
            if isinstance(stop_strings, str):
                stop_strings = [stop_strings]
            converted_ids: List[int] = []
            unresolved_stops: List[str] = []
            for stop_text in stop_strings:
                resolved_id = _resolve_stop_token_id(tokenizer, stop_text)
                if resolved_id is None:
                    unresolved_stops.append(stop_text)
                else:
                    converted_ids.append(resolved_id)

            if len(unresolved_stops) == 0 and len(converted_ids) > 0:
                sampling_kwargs.pop("stop", None)
                merged_ids = list(sampling_kwargs.get("stop_token_ids", [])) + converted_ids
                sampling_kwargs["stop_token_ids"] = sorted(set(int(x) for x in merged_ids))
                if self.rank == 0:
                    print(
                        f"[Rollout] Converted stop strings to stop_token_ids: {sampling_kwargs['stop_token_ids']}"
                    )
            else:
                sampling_kwargs["detokenize"] = True
                sampling_kwargs["stop"] = stop_strings
                if self.rank == 0:
                    print(
                        "[Rollout] Warning: cannot safely convert some stop strings to single token ids "
                        f"({unresolved_stops}); fallback to detokenize=True."
                    )

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

        # For stop-string fallback warnings in update_sampling_params.
        self._stop_string_detokenize_warned = False

        self._interleaved_first_turn_prompt_template = (
            _require_template_from_path(
                self.config.interleaved_first_turn_prompt_file, "interleaved_first_turn_prompt_file"
            )
            or self.config.interleaved_first_turn_prompt_template
        )
        self._interleaved_process_prompt_template = (
            _require_template_from_path(
                self.config.interleaved_process_prompt_file, "interleaved_process_prompt_file"
            )
            or self.config.interleaved_process_prompt_template
            or (
                "{question}\n"
                "Continue your reasoning process inside <think> and </think>. \n"
                "If needed, you can continue to count on the observation image, by outputting <point> and </point> as before. \n"
                "If the final answer is confirmed, put your final answer inside <answer> and </answer>."
            )
        )

        if self.rank == 0 and self.config.interleaved_point_to_count:
            first_src = (
                f"file:{self.config.interleaved_first_turn_prompt_file}"
                if self.config.interleaved_first_turn_prompt_file
                else ("inline" if self.config.interleaved_first_turn_prompt_template else "none")
            )
            process_src = (
                f"file:{self.config.interleaved_process_prompt_file}"
                if self.config.interleaved_process_prompt_file
                else (
                    "inline"
                    if self.config.interleaved_process_prompt_template
                    else "builtin-default"
                )
            )
            print(
                "[InterleavedDebug] template_sources "
                f"first={first_src} process={process_src} "
                f"first_len={len(self._interleaved_first_turn_prompt_template or '')} "
                f"process_len={len(self._interleaved_process_prompt_template or '')}"
            )

    def _create_engine(self, enable_chunked_prefill: bool) -> LLM:
        return LLM(
            **self._engine_init_args,
            enable_chunked_prefill=enable_chunked_prefill,
            **self._engine_kwargs,
        )

    def _maybe_init_attn_meta(self) -> None:
        if self._attn_meta is not None:
            return
        try:
            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            self._attn_meta = _extract_attn_meta(model)
        except Exception:
            self._attn_meta = None

    def _estimate_kv_cache_bytes(self, batch_size: int, n: int, max_total_tokens: int) -> Optional[int]:
        self._maybe_init_attn_meta()
        if self._attn_meta is None:
            return None
        num_layers, num_kv_heads, head_dim = self._attn_meta
        bytes_per_elem = _dtype_nbytes(self.config.dtype)
        # per token, per sequence: K+V
        per_token_per_seq = 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
        return int(per_token_per_seq * max_total_tokens * batch_size * n)

    def _maybe_enable_chunked_prefill(self, reason: str) -> None:
        if self.config.enable_chunked_prefill:
            return
        if not self.config.auto_protect_enable_chunked_prefill_on_oom:
            return
        self.config.enable_chunked_prefill = True
        if self.rank == 0:
            print(f"[Rollout] Auto-protect: enabling chunked prefill ({reason}).")
        # Rebuild engine and hot-swap via proxy so sharding manager keeps working.
        torch.cuda.empty_cache()
        new_engine = self._create_engine(enable_chunked_prefill=True)
        self.inference_engine.swap(new_engine)
        self.inference_engine.sleep(level=1)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        # NOTE(vLLM): stop strings are only supported when detokenize=True.
        # We keep the fast path (detokenize=False) by converting stop strings to
        # stop_token_ids when each stop can be represented by a single token id.
        old_sampling_params_args: Dict[str, Any] = {}

        if kwargs and "stop" in kwargs and kwargs.get("stop") is not None:
            stop_value = kwargs.get("stop")
            stop_list = [stop_value] if isinstance(stop_value, str) else list(stop_value)
            effective_detokenize = bool(kwargs.get("detokenize", getattr(self.sampling_params, "detokenize", True)))

            if stop_list and not effective_detokenize:
                converted_ids: List[int] = []
                unresolved: List[str] = []
                for stop_text in stop_list:
                    resolved_id = _resolve_stop_token_id(self.tokenizer, stop_text)
                    if resolved_id is None:
                        unresolved.append(str(stop_text))
                    else:
                        converted_ids.append(int(resolved_id))

                if len(unresolved) == 0 and converted_ids:
                    existing_ids: List[int] = []
                    if "stop_token_ids" in kwargs and kwargs.get("stop_token_ids") is not None:
                        sti = kwargs.get("stop_token_ids")
                        existing_ids = [int(sti)] if isinstance(sti, int) else [int(x) for x in list(sti)]
                    else:
                        current_ids = getattr(self.sampling_params, "stop_token_ids", None)
                        if current_ids:
                            existing_ids = [int(x) for x in list(current_ids)]

                    merged_ids = sorted(set(existing_ids + converted_ids))
                    kwargs["stop"] = None
                    kwargs["stop_token_ids"] = merged_ids
                else:
                    # Fallback: enable detokenize so vLLM accepts stop strings.
                    kwargs["detokenize"] = True
                    kwargs["stop"] = stop_list

                    if self.rank == 0 and unresolved and not getattr(self, "_stop_string_detokenize_warned", False):
                        print(
                            "[Rollout] Warning: stop strings cannot be converted to single token ids "
                            f"({unresolved}); using detokenize=True for this generate call."
                        )
                        self._stop_string_detokenize_warned = True

        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def _generate_interleaved_responses(
        self,
        vllm_inputs: List[Dict[str, Any]],
        desired_n: int,
        question_texts: Optional[List[str]] = None,
        first_turn_prompt_template: Optional[str] = None,
        process_prompt_template: Optional[str] = None,
        step_info: Any = "?",
    ) -> torch.Tensor:
        active_prompt_ids: List[List[int]] = []
        active_mm_data: List[Optional[Dict[str, Any]]] = []
        active_question_texts: List[str] = []
        repeat_times = max(desired_n, 1)

        for sample_idx, item in enumerate(vllm_inputs):
            sample_question = ""
            if question_texts is not None and sample_idx < len(question_texts):
                sample_question = question_texts[sample_idx] or ""

            base_mm_data = item.get("multi_modal_data")
            for _ in range(repeat_times):
                active_prompt_ids.append(list(item["prompt_token_ids"]))
                if isinstance(base_mm_data, dict):
                    mm_data = dict(base_mm_data)
                    images = mm_data.get("image")
                    if isinstance(images, list):
                        mm_data["image"] = list(images)
                    active_mm_data.append(mm_data)
                else:
                    active_mm_data.append(base_mm_data)
                active_question_texts.append(sample_question)

        response_token_ids: List[List[int]] = [[] for _ in active_prompt_ids]
        active_flags: List[bool] = [True for _ in active_prompt_ids]
        point_counts: List[int] = [0 for _ in active_prompt_ids]
        answer_mode_flags: List[bool] = [False for _ in active_prompt_ids]
        point_parse_buffers: List[str] = ["" for _ in active_prompt_ids]
        control_tag_buffers: List[str] = ["" for _ in active_prompt_ids]
        no_point_streaks: List[int] = [0 for _ in active_prompt_ids]

        point_reminder_text = (
            "\nReminder: if counting is not complete, output exactly one point now in this format: "
            '<point>{"point_2d": [x, y], "label": "object", "count_number": "k"}</point>. '
            "Do not output final answer yet unless counting is complete."
        )
        point_reminder_token_ids = _encode_followup_chat_tokens(self.tokenizer, point_reminder_text)

        first_turn_inject_mode = "none"
        if first_turn_prompt_template:
            for idx in range(len(active_prompt_ids)):
                first_text = _format_turn_prompt(first_turn_prompt_template, active_question_texts[idx])
                if first_text:
                    # IMPORTANT(interleaved): for chat models (e.g., Qwen2/2.5-VL), the model expects
                    # a proper user->assistant turn boundary. Appending raw text to the prompt tail can
                    # cause the model to emit a user-like segment (often echoing the question and
                    # closing with <|im_end|>), which systematically drops the first <point>.
                    #
                    # We therefore append the first-turn template as a *new user turn* followed by an
                    # assistant generation prompt, consistent with process prompts and reminders.
                    #
                    # If the template itself already contains chat tokens (<|im_start|>/<|im_end|>),
                    # avoid double-wrapping and just append the raw ids.
                    if "<|im_start|>" in first_text or "<|im_end|>" in first_text:
                        first_prompt_token_ids = self.tokenizer.encode(first_text, add_special_tokens=False)
                        if "<|im_start|>assistant" not in first_text:
                            # Some templates include only the user segment and end with <|im_end|>.
                            # Ensure we always end at an assistant generation boundary.
                            first_turn_inject_mode = "raw_chat_tokens+assistant_prompt"
                            first_prompt_token_ids = first_prompt_token_ids + _encode_assistant_generation_prompt(
                                self.tokenizer
                            )
                        else:
                            first_turn_inject_mode = "raw_chat_tokens"
                    else:
                        first_turn_inject_mode = "user_chat_turn"
                        first_prompt_token_ids = _encode_followup_chat_tokens(self.tokenizer, first_text)
                    if first_prompt_token_ids:
                        prompt_token_ids = active_prompt_ids[idx]
                        suffix_len = len(first_prompt_token_ids)
                        already_has_first_prompt = (
                            len(prompt_token_ids) >= suffix_len
                            and prompt_token_ids[-suffix_len:] == first_prompt_token_ids
                        )
                        if not already_has_first_prompt:
                            active_prompt_ids[idx].extend(first_prompt_token_ids)

        history_mode = int(getattr(self.config, "interleaved_history_mode", -1))
        keep_full_history = (history_mode == -1)  # -1=full, 0=none, N>0=last N turns
        keep_partial_history = (history_mode > 0)  # Keep last N turns
        partial_history_n = max(1, history_mode) if keep_partial_history else 0
        base_prompt_ids: List[List[int]] = [list(x) for x in active_prompt_ids]
        # Track where each turn's content starts in active_prompt_ids (for partial history)
        # turn_starts[i] = list of positions; turn_starts[i][k] = index in active_prompt_ids
        # where turn k's generated content begins. Used to trim history to last N turns.
        turn_starts: List[List[int]] = [[len(ids)] for ids in active_prompt_ids]  # turn 0 starts after base

        # Pre-compute turn2 image prefix for history_mode=0 prompt reconstruction.
        # When set, turn 2+ builds [system + user(image + process_text)] instead
        # of [base_prompt + user(process_text)], matching eval inference structure.
        turn2_image_prefixes: Optional[List[List[int]]] = None
        turn2_suffix_ids: List[int] = []
        if not keep_full_history and not keep_partial_history and not first_turn_prompt_template:
            _vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            _im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            _im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
            _unk_id = getattr(self.tokenizer, "unk_token_id", None)
            _has_vision_tokens = (
                isinstance(_vision_end_id, int) and _vision_end_id >= 0
                and _vision_end_id != _unk_id
                and isinstance(_im_end_id, int) and _im_end_id >= 0
            )
            if _has_vision_tokens:
                _prefixes: List[List[int]] = []
                for ids in base_prompt_ids:
                    # Find last <|vision_end|> position
                    _ve_pos = None
                    for _i in range(len(ids) - 1, -1, -1):
                        if ids[_i] == _vision_end_id:
                            _ve_pos = _i
                            break
                    if _ve_pos is not None:
                        _prefixes.append(ids[:_ve_pos + 1])
                    else:
                        _prefixes = None
                        break
                if _prefixes is not None:
                    turn2_image_prefixes = _prefixes
                    _newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
                    _assistant_ids = self.tokenizer.encode("assistant", add_special_tokens=False)
                    turn2_suffix_ids = [_im_end_id] + _newline_ids + [_im_start_id] + _assistant_ids + _newline_ids

        stop_tag = self.config.interleaved_stop_tag or "</answer>"
        per_turn_max_tokens = max(1, int(self.config.interleaved_per_turn_max_tokens))
        answer_turn_max_tokens = max(per_turn_max_tokens, int(self.config.interleaved_answer_turn_max_tokens))
        max_turns = max(1, int(self.config.interleaved_max_turns))
        debug_enabled = bool(self.config.interleaved_debug)
        debug_chars = max(50, int(self.config.interleaved_debug_print_chars))

        if debug_enabled and self.rank == 0:
            preview_question = active_question_texts[0] if active_question_texts else ""
            preview_first = _format_turn_prompt(first_turn_prompt_template, preview_question)
            preview_process = _format_turn_prompt(process_prompt_template, preview_question)
            first_truncated = len(preview_first) > debug_chars
            process_truncated = len(preview_process) > debug_chars
            try:
                prompt_tail_ids = active_prompt_ids[0][-200:] if active_prompt_ids else []
                prompt_tail_text = self.tokenizer.decode(prompt_tail_ids, skip_special_tokens=False)
            except Exception:
                prompt_tail_text = ""
            print(
                "[InterleavedDebug] history_mode "
                f"value={history_mode} keep_full_history={keep_full_history} "
                f"keep_partial={keep_partial_history} partial_n={partial_history_n} "
                "(-1=full,0=none,N>0=last_N_turns)"
            )
            print(
                "[InterleavedDebug] first_turn_prompt_injected "
                f"step={step_info} mode={first_turn_inject_mode} "
                f"sample0_prompt_tail='{prompt_tail_text[:debug_chars]}'"
            )
            print(
                "[InterleavedDebug] prompt_preview "
                f"first='{preview_first[:debug_chars]}'{('...(truncated)' if first_truncated else '')} "
                f"| process='{preview_process[:debug_chars]}'{('...(truncated)' if process_truncated else '')} "
                f"(preview_chars={debug_chars}, first_len={len(preview_first)}, process_len={len(preview_process)})"
            )

        with self.update_sampling_params(n=1):
            total_points_all_turns = 0
            total_point_tag_seen_all_turns = 0
            stall_warning_emitted = False
            for turn_idx in range(max_turns):
                turn_indices = [idx for idx, flag in enumerate(active_flags) if flag]
                if not turn_indices:
                    break

                normal_indices = [idx for idx in turn_indices if not answer_mode_flags[idx]]
                answer_indices = [idx for idx in turn_indices if answer_mode_flags[idx]]

                completion_pairs: List[Tuple[int, RequestOutput]] = []

                def _run_generate(sub_indices: List[int], max_tokens_for_group: int, stop_for_group: Optional[List[str]] = None):
                    if not sub_indices:
                        return
                    turn_prompts: List[Dict[str, Any]] = []
                    for idx in sub_indices:
                        item: Dict[str, Any] = {"prompt_token_ids": active_prompt_ids[idx]}
                        if active_mm_data[idx] is not None:
                            item["multi_modal_data"] = active_mm_data[idx]
                        turn_prompts.append(item)
                    with self.update_sampling_params(max_tokens=max_tokens_for_group, stop=stop_for_group):
                        completions = self.inference_engine.generate(
                            prompts=turn_prompts, sampling_params=self.sampling_params, use_tqdm=False
                        )
                    for local_idx, output in enumerate(completions):
                        completion_pairs.append((sub_indices[local_idx], output))

                # Point turns: stop as soon as </point> is produced, so the turn stays short and
                # doesn't spill into explanations that can break parsing/reward.
                # On the LAST turn, give normal-mode samples the full answer budget
                # so they can fit <think>...</think><answer>N</answer> without truncation.
                normal_max_tokens = answer_turn_max_tokens if turn_idx == max_turns - 1 else per_turn_max_tokens
                _run_generate(normal_indices, normal_max_tokens, stop_for_group=["</point>", "</answer>"])
                # Answer/long turns: stop at </answer>.
                _run_generate(answer_indices, answer_turn_max_tokens, stop_for_group=["</answer>"])

                if debug_enabled and self.rank == 0:
                    print(
                        "[InterleavedDebug] "
                        f"step={step_info} turn={turn_idx + 1}/{max_turns} active={len(turn_indices)} "
                        f"normal={len(normal_indices)} answer_mode={len(answer_indices)}"
                    )
                    turn_indices_head = turn_indices[:8]
                    print(
                        "[InterleavedDebug] "
                        f"step={step_info} turn={turn_idx + 1} active_indices_head={turn_indices_head} "
                        "(sample0_idx below is the first completed active index of this turn, not a fixed sample id)"
                    )

                turn_points = 0
                turn_answer_open = 0
                turn_answer_closed = 0
                turn_point_tag_seen = 0
                turn_point_parsed = 0
                turn_empty = 0
                turn_partial_point_open = 0
                turn_stall_candidates = 0

                for global_idx, output in completion_pairs:
                    token_ids = output.outputs[0].token_ids if output.outputs else []
                    if not token_ids:
                        active_flags[global_idx] = False
                        turn_empty += 1
                        continue

                    remaining_budget = self.config.response_length - len(response_token_ids[global_idx])
                    if remaining_budget <= 0:
                        active_flags[global_idx] = False
                        continue
                    token_ids = token_ids[:remaining_budget]

                    response_token_ids[global_idx].extend(token_ids)
                    active_prompt_ids[global_idx].extend(token_ids)
                    decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                    if "<point" in decoded_text:
                        turn_point_tag_seen += 1

                    control_combined = (control_tag_buffers[global_idx] + decoded_text)[-2048:]
                    control_tag_buffers[global_idx] = control_combined

                    has_answer_open = "<answer>" in control_combined
                    has_answer_close = "</answer>" in control_combined
                    
                    last_point_open = control_combined.rfind("<point")
                    last_point_close = control_combined.rfind("</point>")
                    is_inside_point = last_point_open > last_point_close
                    
                    last_think_open = control_combined.rfind("<think>")
                    last_think_close = control_combined.rfind("</think>")
                    is_inside_think = last_think_open > last_think_close
                    
                    last_answer_open = control_combined.rfind("<answer>")
                    last_answer_close = control_combined.rfind("</answer>")
                    is_inside_answer = last_answer_open > last_answer_close

                    if has_answer_open:
                        turn_answer_open += 1
                    if has_answer_close:
                        turn_answer_closed += 1

                    if stop_tag and (stop_tag in decoded_text or stop_tag in control_combined):
                        active_flags[global_idx] = False
                        answer_mode_flags[global_idx] = False
                        continue

                    if self.config.interleaved_append_process_prompt:
                        # If answer or think has started but not closed, avoid appending process prompt
                        # and switch to answer-mode long generation next turn.
                        # Also treat an unclosed <point> tag as long-generation mode, otherwise
                        # the next turn's process prompt can split the tag and break parsing/reward.
                        if is_inside_think or is_inside_answer or is_inside_point:
                            answer_mode_flags[global_idx] = True
                        else:
                            answer_mode_flags[global_idx] = False

                        if keep_partial_history:
                            # Keep base + last N turns of generated content
                            ts = turn_starts[global_idx]
                            if len(ts) > partial_history_n:
                                # Keep last partial_history_n turns
                                keep_from = ts[-partial_history_n]
                                base_len = len(base_prompt_ids[global_idx])
                                offset = base_len - keep_from
                                active_prompt_ids[global_idx] = list(base_prompt_ids[global_idx]) + active_prompt_ids[global_idx][keep_from:]
                                # Adjust remaining turn_starts for the new trimmed prompt
                                turn_starts[global_idx] = [pos + offset for pos in ts[-partial_history_n:]]
                            # else: not enough turns yet, keep all
                        elif not keep_full_history:
                            if turn2_image_prefixes is not None and not answer_mode_flags[global_idx]:
                                # Fresh prompt: [system + user(image + <process_text>)]
                                active_prompt_ids[global_idx] = list(turn2_image_prefixes[global_idx])
                            else:
                                active_prompt_ids[global_idx] = list(base_prompt_ids[global_idx])

                        process_prompt = _format_turn_prompt(
                            process_prompt_template,
                            active_question_texts[global_idx],
                        )
                        if process_prompt and not answer_mode_flags[global_idx]:
                            if turn2_image_prefixes is not None:
                                # Append process text into existing user message + close user + start assistant
                                _process_text_ids = self.tokenizer.encode(process_prompt, add_special_tokens=False)
                                active_prompt_ids[global_idx].extend(_process_text_ids)
                                active_prompt_ids[global_idx].extend(turn2_suffix_ids)
                            else:
                                active_prompt_ids[global_idx].extend(
                                    _encode_followup_chat_tokens(self.tokenizer, process_prompt)
                                )
                            # Mark the start of the NEXT turn (after process prompt)
                            if keep_partial_history:
                                turn_starts[global_idx].append(len(active_prompt_ids[global_idx]))

                    point_combined = point_parse_buffers[global_idx] + decoded_text
                    parsed_points = _parse_points_from_text(point_combined)
                    turn_point_parsed += len(parsed_points)
                    if "<point" in point_combined and "</point>" not in point_combined:
                        turn_partial_point_open += 1
                    last_close = point_combined.rfind("</point>")
                    if last_close >= 0:
                        point_parse_buffers[global_idx] = point_combined[last_close + len("</point>"):]
                    else:
                        point_parse_buffers[global_idx] = point_combined[-512:]

                    if not parsed_points:
                        if is_inside_point:
                            # We are in the middle of a <point> tag; do not count as a stall and
                            # do not inject reminders. We'll give it more tokens next turn.
                            continue
                        if (not answer_mode_flags[global_idx]) and (not has_answer_open):
                            no_point_streaks[global_idx] += 1
                            turn_stall_candidates += 1
                            if no_point_streaks[global_idx] >= 2 and point_reminder_token_ids:
                                active_prompt_ids[global_idx].extend(point_reminder_token_ids)
                        continue
                    mm_data = active_mm_data[global_idx]
                    if not isinstance(mm_data, dict) or "image" not in mm_data:
                        continue

                    images = mm_data.get("image")
                    if not isinstance(images, list) or len(images) == 0:
                        continue

                    first_image = images[0]
                    if isinstance(first_image, PILImage.Image):
                        first_image = first_image.copy()
                    else:
                        converted = _to_pil_image(first_image)
                        if converted is None:
                            continue
                        first_image = converted

                    no_point_streaks[global_idx] = 0
                    for parsed_point in parsed_points:
                        pred_point = (parsed_point[0], parsed_point[1])
                        parsed_count = parsed_point[2]
                        if parsed_count is not None and parsed_count > 0:
                            point_counts[global_idx] = parsed_count
                        else:
                            point_counts[global_idx] += 1
                        first_image = _draw_point_on_image(
                            first_image, pred_point, point_number=point_counts[global_idx]
                        )
                        turn_points += 1

                    updated_images = list(images)
                    updated_images[0] = first_image
                    mm_data["image"] = updated_images

                total_points_all_turns += turn_points
                total_point_tag_seen_all_turns += turn_point_tag_seen

                if debug_enabled and self.rank == 0:
                    print(
                        "[InterleavedDebug] "
                        f"step={step_info} turn={turn_idx + 1} points={turn_points} "
                        f"answer_open={turn_answer_open} answer_closed={turn_answer_closed} "
                        f"point_tag_seen={turn_point_tag_seen} point_parsed={turn_point_parsed} "
                        f"partial_point_open={turn_partial_point_open} empty={turn_empty} "
                        f"stall_candidates={turn_stall_candidates}"
                    )
                    if completion_pairs:
                        sample_idx, sample_out = completion_pairs[0]
                        sample_token_ids = sample_out.outputs[0].token_ids if sample_out.outputs else []
                        sample_text = self.tokenizer.decode(sample_token_ids, skip_special_tokens=False)
                        try:
                            base_sample_idx = int(sample_idx) // int(repeat_times)
                            rollout_k = int(sample_idx) % int(repeat_times)
                            idx_hint = f"base_sample={base_sample_idx} rollout_k={rollout_k} repeat_times={repeat_times}"
                        except Exception:
                            idx_hint = ""
                        print(
                            "[InterleavedDebug] "
                            f"step={step_info} turn={turn_idx + 1} sample0_idx={sample_idx} "
                            f"{idx_hint} sample0_text='{sample_text[:debug_chars]}'"
                        )

                if (
                    debug_enabled
                    and self.rank == 0
                    and (not stall_warning_emitted)
                    and turn_idx >= 2
                    and len(turn_indices) > 0
                    and turn_points == 0
                    and turn_answer_open == 0
                    and turn_answer_closed == 0
                ):
                    avg_no_point_streak = float(sum(no_point_streaks[idx] for idx in turn_indices)) / max(len(turn_indices), 1)
                    sample_idx = turn_indices[0]
                    prompt_tail_ids = active_prompt_ids[sample_idx][-200:]
                    prompt_tail_text = self.tokenizer.decode(prompt_tail_ids, skip_special_tokens=False)
                    print(
                        "[InterleavedWarning] "
                        f"step={step_info} turn={turn_idx + 1} stalled_no_points "
                        f"active={len(turn_indices)} avg_no_point_streak={avg_no_point_streak:.2f} "
                        f"point_tag_seen={turn_point_tag_seen} point_parsed={turn_point_parsed} "
                        f"partial_point_open={turn_partial_point_open}"
                    )
                    print(
                        "[InterleavedWarning] "
                        f"step={step_info} turn={turn_idx + 1} sample0_prompt_tail='{prompt_tail_text[:debug_chars]}'"
                    )
                    stall_warning_emitted = True

            if debug_enabled and self.rank == 0:
                print(
                    "[InterleavedSummary] "
                    f"step={step_info} total_points={total_points_all_turns} "
                    f"total_point_tag_seen={total_point_tag_seen_all_turns}"
                )
                if total_points_all_turns == 0:
                    print(
                        "[InterleavedWarning] "
                        f"step={step_info} no_points_across_all_turns; check sample0_text/point_tag_seen/point_parsed logs above."
                    )

                # Dump exactly one full trajectory for manual verification.
                # This prints the complete model output (points + final answer) for sample0.
                try:
                    dumped = bool(getattr(self, "_interleaved_dumped_once", False))
                except Exception:
                    dumped = False
                if not dumped and response_token_ids:
                    try:
                        sample_text = self.tokenizer.decode(response_token_ids[0], skip_special_tokens=False)
                    except Exception:
                        sample_text = ""
                    print("[InterleavedTrajectory] BEGIN sample0")
                    print(sample_text)
                    print("[InterleavedTrajectory] END sample0")
                    try:
                        setattr(self, "_interleaved_dumped_once", True)
                    except Exception:
                        pass

        return VF.pad_2d_list_to_length(
            response_token_ids, self.pad_token_id, max_length=self.config.response_length
        )

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # Get step info from meta_info if available
        step_info = prompts.meta_info.get("global_step", "?")
        if self.rank == 0:
            print(f"[Rollout] Step {step_info}: Start generating sequences.")

        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        # For interleaved follow-up turns we want the original question text.
        # However, when n>1 we must not return any non-tensor fields that still have the pre-repeat batch size.
        question_texts = [_extract_question_text(non_tensor_batch, i) for i in range(batch_size)]
        for key in ("problem", "prompt", "question", "query", "instruction"):
            if key in non_tensor_batch:
                non_tensor_batch.pop(key, None)

        has_multi_modal = "multi_modal_data" in non_tensor_batch
        if has_multi_modal:
            vllm_inputs = []
            _vfps = float(prompts.meta_info.get("video_fps", 2.0))
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            _vfps,
                            return_video_metadata=self.return_video_metadata,
                        ),
                    }
                )
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        prompt_lens = _get_prompt_lens(attention_mask)
        max_prompt_len = int(prompt_lens.max().item()) if prompt_lens.numel() else 0

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            desired_n = int(getattr(self.sampling_params, "n", 1) or 1)
            free_bytes = int(torch.cuda.mem_get_info()[0]) if torch.cuda.is_available() else 0
            free_gb = free_bytes / (1024**3) if free_bytes else 0.0

            # If we are extremely tight on headroom with very long prompts, proactively switch on
            # chunked prefill once (kept on afterwards) to avoid prefill-side spikes.
            if (
                self.config.auto_protect
                and (not self.config.enable_chunked_prefill)
                and free_gb > 0
                and free_gb < (float(self.config.auto_protect_free_gb_threshold) / 2.0)
                and max_prompt_len >= int(self.config.auto_protect_enable_chunked_prefill_prompt_threshold)
            ):
                self._maybe_enable_chunked_prefill(reason="predicted low headroom for long prompts")

            # Heuristic risk check (no overhead on normal path): if headroom is low and prompts are very long,
            # we may reduce `n` temporarily; if we still OOM, we will enable chunked prefill and retry.
            actual_n = desired_n
            if (
                self.config.auto_protect
                and desired_n > self.config.auto_protect_min_n
                and free_gb > 0
                and free_gb < float(self.config.auto_protect_free_gb_threshold)
                and max_prompt_len >= int(self.config.auto_protect_enable_chunked_prefill_prompt_threshold)
            ):
                actual_n = max(self.config.auto_protect_min_n, desired_n // 2)
                if self.rank == 0:
                    print(
                        f"[Rollout] Auto-protect: low free mem ({free_gb:.2f} GiB) with long prompt ({max_prompt_len}); "
                        f"temporarily reducing n {desired_n} -> {actual_n}."
                    )

            # IMPORTANT: downstream (trainer + fsdp worker wrapper) assumes meta_info['n'] matches
            # the actual number of sequences produced. Auto-protect may reduce n, so we must
            # propagate the effective n here to keep tensor/non-tensor/multi-modal fields aligned.
            prompts.meta_info["n"] = int(actual_n)

            if self.config.interleaved_debug and self.rank == 0:
                non_empty_questions = sum(1 for x in question_texts if isinstance(x, str) and x.strip())
                print(
                    "[InterleavedDebug] mode_decision "
                    f"step={step_info} interleaved_enable={self.config.interleaved_point_to_count} "
                    f"has_multi_modal={has_multi_modal} batch_size={batch_size} "
                    f"question_non_empty={non_empty_questions}/{len(question_texts)} "
                    f"data_format_prompt={prompts.meta_info.get('data_format_prompt')}"
                )

            if self.config.interleaved_point_to_count and has_multi_modal:
                data_format_prompt_path = prompts.meta_info.get("data_format_prompt")
                # Interleaved first turn should be aligned with dataset format prompt when available
                # (e.g. StepCount_format.jinja), so trajectory start condition matches normal training.
                first_turn_prompt_template = _load_template_from_path(data_format_prompt_path)
                first_turn_source = "data.format_prompt"
                if not first_turn_prompt_template:
                    first_turn_prompt_template = self._interleaved_first_turn_prompt_template
                    first_turn_source = "interleaved_first_turn_prompt"

                process_prompt_template = self._interleaved_process_prompt_template
                process_source = "interleaved_process_prompt"
                if not process_prompt_template:
                    process_prompt_template = _load_template_from_path(data_format_prompt_path)
                    process_source = "data.format_prompt"

                if self.config.interleaved_debug and self.rank == 0:
                    print(
                        "[InterleavedDebug] prompt_effective_source "
                        f"step={step_info} first={first_turn_source} process={process_source} "
                        f"first_exists={bool(first_turn_prompt_template)} process_exists={bool(process_prompt_template)}"
                    )

                response_ids = self._generate_interleaved_responses(
                    vllm_inputs=vllm_inputs,
                    desired_n=actual_n,
                    question_texts=question_texts,
                    first_turn_prompt_template=first_turn_prompt_template,
                    process_prompt_template=process_prompt_template,
                    step_info=step_info,
                ).to(
                    input_ids.device
                )
            else:
                completions: List[RequestOutput] = []
                retries = 0
                last_exc: Optional[BaseException] = None
                while retries <= int(self.config.auto_protect_max_retries):
                    try:
                        with (self.update_sampling_params(n=actual_n) if actual_n != desired_n else nullcontext()):
                            # One-shot generate
                            completions = self.inference_engine.generate(
                                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=False
                            )
                        last_exc = None
                        break
                    except Exception as exc:  # vLLM may wrap OOM into RuntimeError
                        last_exc = exc
                        if not self.config.auto_protect or not _is_oom_error(exc):
                            raise

                        if self.rank == 0:
                            print(f"[Rollout] Auto-protect: caught OOM during generate (retry {retries}).")

                        torch.cuda.empty_cache()

                        # First: reduce n aggressively
                        if actual_n > 1:
                            new_n = max(1, actual_n // 2)
                            if self.rank == 0:
                                print(f"[Rollout] Auto-protect: reducing n {actual_n} -> {new_n} and retry.")
                            actual_n = new_n
                            retries += 1
                            continue

                        # Second: enable chunked prefill and retry once
                        if not self.config.enable_chunked_prefill:
                            self._maybe_enable_chunked_prefill(reason="OOM during rollout")
                            retries += 1
                            continue

                        retries += 1

                if last_exc is not None:
                    raise last_exc

                # Non-interleaved safety guard:
                # keep default one-shot behavior, but if generation collapses to
                # full-length low-diversity repetitions, do a single rescue retry.
                if (not self.config.interleaved_point_to_count) and completions:
                    try:
                        seqs: List[List[int]] = [
                            list(out.token_ids)
                            for comp in completions
                            for out in comp.outputs
                            if out is not None and hasattr(out, "token_ids")
                        ]
                        total_seq = len(seqs)
                        if total_seq > 0:
                            full_len = int(self.config.response_length)
                            stalled = sum(
                                1
                                for ids in seqs
                                if len(ids) >= full_len and _has_low_diversity_tail(ids, tail_k=128, max_unique=2)
                            )
                            stall_ratio = float(stalled) / float(total_seq)
                            if stall_ratio >= 0.70:
                                rescue_temp = max(0.1, float(getattr(self.sampling_params, "temperature", 1.0)) * 0.5)
                                if self.rank == 0:
                                    print(
                                        "[Rollout] Non-interleaved stall guard: "
                                        f"stalled={stalled}/{total_seq} ({stall_ratio:.2%}); "
                                        f"retry once with stop=['</answer>'], temperature={rescue_temp:.2f}."
                                    )
                                with self.update_sampling_params(stop=["</answer>"], temperature=rescue_temp):
                                    completions = self.inference_engine.generate(
                                        prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=False
                                    )
                    except Exception as guard_exc:
                        if self.rank == 0:
                            print(f"[Rollout] Non-interleaved stall guard skipped due to: {guard_exc}")

                response_ids = [output.token_ids for completion in completions for output in completion.outputs]
                response_ids = VF.pad_2d_list_to_length(
                    response_ids, self.pad_token_id, max_length=self.config.response_length
                ).to(input_ids.device)

            if actual_n > 1:
                batch_size = batch_size * actual_n
                input_ids = _repeat_interleave(input_ids, actual_n)
                attention_mask = _repeat_interleave(attention_mask, actual_n)
                position_ids = _repeat_interleave(position_ids, actual_n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # NOTE (interleaved point->count): for Qwen2/2.5-VL tokenizers, `eos_token_id` is often
        # `<|im_end|>` (see logs: eos_token_id=151645). Interleaved trajectories may legitimately
        # contain `<|im_end|>` very early (e.g., right after the echoed question), so an EOS-based
        # mask would truncate the response and break reward parsing (points/answer become invisible).
        # In this mode, we instead treat non-pad tokens as valid response tokens.
        use_pad_based_response_mask = bool(self.config.interleaved_point_to_count and has_multi_modal)
        if use_pad_based_response_mask:
            response_mask = response_ids.ne(self.pad_token_id).to(attention_mask.dtype)
            if self.config.interleaved_debug and self.rank == 0:
                try:
                    eos_tok = self.tokenizer.convert_ids_to_tokens(int(eos_token_id))
                except Exception:
                    eos_tok = str(eos_token_id)
                print(
                    "[InterleavedDebug] response_mask_mode=pad_based "
                    f"pad_token_id={self.pad_token_id} eos_token_id={eos_token_id} eos_token={eos_tok}"
                )
        else:
            response_mask = VF.get_response_mask(
                response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
            )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.rank == 0:
            print(f"[Rollout] Step {step_info}: Finish generating sequences.")

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=prompts.meta_info.copy(),
        )
