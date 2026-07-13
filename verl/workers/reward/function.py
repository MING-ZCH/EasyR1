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
import math
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[str, str], RewardScore]

BatchRewardFunction = Callable[[List[str], List[str]], List[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer
        self._reward_debug_calls = 0
        self._reward_debug_every = max(1, int(os.environ.get("EASYR1_REWARD_DEBUG_EVERY", "50")))
        self._reward_sample_debug = os.environ.get("EASYR1_REWARD_SAMPLE_DEBUG", "1") == "1"
        self._reward_sample_debug_max = max(1, int(os.environ.get("EASYR1_REWARD_SAMPLE_DEBUG_MAX", "3")))
        self._reward_health_debug = os.environ.get("EASYR1_REWARD_HEALTH_DEBUG", "1") == "1"
        self._reward_health_every = max(1, int(os.environ.get("EASYR1_REWARD_HEALTH_DEBUG_EVERY", str(self._reward_debug_every))))

    @staticmethod
    def _safe_mean(values: List[float]) -> Optional[float]:
        if not values:
            return None
        try:
            nums = [float(v) for v in values]
            nums = [v for v in nums if not math.isnan(v)]
            if not nums:
                return None
            return float(sum(nums) / len(nums))
        except Exception:
            return None

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    @staticmethod
    def _extract_first_image_path(multi_modal_data) -> Optional[str]:
        if not isinstance(multi_modal_data, dict) or "image" not in multi_modal_data:
            return None

        images = multi_modal_data["image"]
        if not isinstance(images, list) or len(images) == 0:
            return None

        first_image = images[0]
        if isinstance(first_image, str):
            return first_image
        # HuggingFace datasets often store images as dicts: {"bytes": ..., "path": "xxx.jpg"}
        if isinstance(first_image, dict):
            path = first_image.get("path")
            if isinstance(path, str) and path:
                return path
        return None

    @staticmethod
    def _extract_problem_text(non_tensor_batch: Dict[str, List], index: int) -> Optional[str]:
        for key in ("problem", "prompt", "question", "query", "instruction"):
            if key in non_tensor_batch:
                value = non_tensor_batch[key][index]
                if isinstance(value, str):
                    return value
        return None

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        self._reward_debug_calls += 1
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        with_images_count = 0
        with_problem_count = 0
        sample_debug_rows = []

        # --- Parallel reward computation ---
        # Pre-decode all responses and prepare kwargs (thread-safe prep)
        num_workers = int(os.environ.get("REWARD_NUM_WORKERS", "4"))
        batch_size = len(data)

        def _compute_single_reward(i):
            """Compute reward for a single sample. Thread-safe."""
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            ground_truth = data.non_tensor_batch["ground_truth"][i]

            kwargs = {}
            _has_images = False
            _has_problem = False
            if "multi_modal_data" in data.non_tensor_batch:
                multi_modal_data = data.non_tensor_batch["multi_modal_data"][i]
                if isinstance(multi_modal_data, dict) and "image" in multi_modal_data:
                    kwargs["images"] = multi_modal_data["image"]
                    _has_images = True
                image_path = self._extract_first_image_path(multi_modal_data)
                if image_path is not None:
                    kwargs["image_path"] = image_path

            problem_text = self._extract_problem_text(data.non_tensor_batch, i)
            if problem_text is not None:
                kwargs["problem"] = problem_text
                _has_problem = True

            if "id" in data.non_tensor_batch:
                kwargs["sample_id"] = data.non_tensor_batch["id"][i]

            kwargs["sample_index"] = i
            kwargs["is_eval"] = bool(getattr(data, "meta_info", {}).get("is_validation", False))

            try:
                score = self.reward_fn(response_str, ground_truth, **kwargs)
            except Exception as exc:
                try:
                    response_tail = response_str[-260:].replace("\n", " ")
                except Exception:
                    response_tail = ""
                print(
                    "[RewardError] "
                    f"reward_fn_exception idx={i} sample_id={kwargs.get('sample_id')} "
                    f"has_image={('images' in kwargs)} has_problem={('problem' in kwargs)} "
                    f"image_path={kwargs.get('image_path')} gt={str(ground_truth)} "
                    f"exc={type(exc).__name__}: {exc} tail='{response_tail}'"
                )
                score = {"overall": 0.0, "format": 0.0, "content": 0.0, "answer": 0.0, "point": 0.0, "format_fail": 1.0, "stop_violation": 1.0}

            if not isinstance(score, dict) or "overall" not in score:
                score = {"overall": 0.0, "format": 0.0, "content": 0.0, "answer": 0.0, "point": 0.0, "format_fail": 1.0, "stop_violation": 1.0}

            return i, score, response_str, _has_images, _has_problem

        # Execute in parallel if num_workers > 1, otherwise sequential
        if num_workers > 1 and batch_size > 16:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(_compute_single_reward, range(batch_size)))
        else:
            results = [_compute_single_reward(i) for i in range(batch_size)]

        # Gather results (sequential - fast)
        for i, score, response_str, _has_images, _has_problem in results:
            if _has_images:
                with_images_count += 1
            if _has_problem:
                with_problem_count += 1

            # --- Per-turn process reward placement ---
            # When enabled, distribute per-step point rewards at turn boundary
            # token positions (</point> tags), with the answer+format portion
            # at the last token. This enables step-level GRPO (GSPO).
            _step_token_positions = []
            _step_values_at_positions = []  # parallel _point_step_value aligned to positions
            _step_value_list = score.get("_point_step_value", None)  # optional (progress/stoptiming arms)
            _process_reward_mode = os.environ.get("PROCESS_REWARD_ENABLE", "0") in ("1", "true", "yes")
            # The explicit per-turn VALUE channel is active ONLY for the new value-based
            # arms (progress/stoptiming). For legacy `pointhit` (or unset) we must NOT emit
            # `_point_step_value` at all, otherwise the trainer would build a point_step_value
            # tensor (from fallback=placed rewards) and trigger AUX re-centering — silently
            # changing legacy behaviour. Gating on the env keeps the batch list aligned.
            _value_channel_active = os.environ.get("BOK_STEP_SIGNAL", "pointhit").strip().lower() in ("progress", "stoptiming")
            if _process_reward_mode and "_step_rewards" in score:
                step_rewards = score["_step_rewards"]  # list of floats (per-turn point scores)
                # Find </point> token positions in the response
                _point_end_tag = "</point>"
                _tag_char_positions = []
                _search_start = 0
                for _ in range(50):  # safety bound
                    _pos = response_str.find(_point_end_tag, _search_start)
                    if _pos == -1:
                        break
                    _tag_char_positions.append(_pos + len(_point_end_tag))
                    _search_start = _pos + len(_point_end_tag)

                if _tag_char_positions and step_rewards:
                    # Map character positions to token positions
                    # Encode prefix strings to get token offsets
                    _n_placed = 0
                    for _step_idx, _char_pos in enumerate(_tag_char_positions):
                        if _step_idx >= len(step_rewards):
                            break
                        # Encode the prefix up to this character position
                        _prefix_text = response_str[:_char_pos]
                        _prefix_tokens = self.tokenizer.encode(
                            _prefix_text, add_special_tokens=False
                        )
                        _token_pos = min(len(_prefix_tokens) - 1, response_length[i].item() - 1)
                        if _token_pos >= 0 and _token_pos < reward_tensor.shape[1]:
                            reward_tensor[i, _token_pos] = float(step_rewards[_step_idx])
                            _step_token_positions.append(int(_token_pos))
                            # Record the parallel step VALUE (progress/stoptiming) at the
                            # SAME position. When the value channel is active but this row
                            # lacks an explicit value (or has fewer than positions), emit
                            # 0.0 (NOT the placed pointhit reward) to avoid silently mixing
                            # pointhit semantics into a progress/stoptiming arm.
                            if _step_value_list is not None and _step_idx < len(_step_value_list):
                                _step_values_at_positions.append(float(_step_value_list[_step_idx]))
                            else:
                                _step_values_at_positions.append(0.0)
                            _n_placed += 1

                    # Place answer portion at the last token
                    # answer_reward = overall - sum(step_rewards placed)
                    _placed_sum = sum(step_rewards[:_n_placed]) if _n_placed > 0 else 0.0
                    _answer_portion = float(score["overall"]) - _placed_sum
                    reward_tensor[i, response_length[i] - 1] = _answer_portion
                else:
                    # Fallback: place entire reward at last token
                    reward_tensor[i, response_length[i] - 1] = score["overall"]
            else:
                reward_tensor[i, response_length[i] - 1] = score["overall"]

            if _process_reward_mode:
                reward_metrics["_point_step_token_positions"].append(_step_token_positions)
                if _value_channel_active:
                    reward_metrics["_point_step_value"].append(_step_values_at_positions)

            for key, value in score.items():
                if not key.startswith("_"):  # skip internal keys
                    reward_metrics[key].append(value)

            if self._reward_sample_debug and len(sample_debug_rows) < self._reward_sample_debug_max:
                try:
                    response_tail = response_str[-220:].replace("\n", " ")
                except Exception:
                    response_tail = ""
                sample_debug_rows.append(
                    {
                        "idx": i,
                        "gt": str(data.non_tensor_batch["ground_truth"][i]),
                        "overall": float(score.get("overall", 0.0)),
                        "format": float(score.get("format", 0.0)),
                        "point": float(score.get("point", 0.0)),
                        "answer": float(score.get("answer", 0.0)),
                        "stop_violation": float(score.get("stop_violation", 0.0)),
                        "format_fail": float(score.get("format_fail", 0.0)),
                        "tail": response_tail,
                    }
                )

        if self._reward_debug_calls % self._reward_debug_every == 0:
            metric_keys = sorted([key for key in reward_metrics.keys() if not str(key).startswith("_")])
            print(
                "[RewardDebug] "
                f"calls={self._reward_debug_calls} batch_size={len(data)} "
                f"with_images={with_images_count}/{len(data)} "
                f"with_problem={with_problem_count}/{len(data)} "
                f"metrics={metric_keys}"
            )
            if self._reward_sample_debug:
                for row in sample_debug_rows:
                    print(
                        "[RewardDebug][sample] "
                        f"idx={row['idx']} gt={row['gt']} "
                        f"overall={row['overall']:.4f} format={row['format']:.4f} "
                        f"point={row['point']:.4f} answer={row['answer']:.4f} "
                        f"stop_violation={row['stop_violation']:.1f} format_fail={row['format_fail']:.1f} "
                        f"tail='{row['tail']}'"
                    )

        if self._reward_health_debug and (self._reward_debug_calls % self._reward_health_every == 0):
            overall_mean = self._safe_mean(reward_metrics.get("overall", []))
            answer_mean = self._safe_mean(reward_metrics.get("answer", []))
            point_mean = self._safe_mean(reward_metrics.get("point", []))
            format_fail_mean = self._safe_mean(reward_metrics.get("format_fail", []))
            stop_violation_mean = self._safe_mean(reward_metrics.get("stop_violation", []))
            stopped_by_answer_mean = self._safe_mean(reward_metrics.get("stopped_by_answer", []))
            turns_exceeded_mean = self._safe_mean(reward_metrics.get("turns_exceeded", []))
            no_point_pred_mean = self._safe_mean(reward_metrics.get("no_point_pred", []))
            point_key_typo_mean = self._safe_mean(reward_metrics.get("point_key_typo", []))
            point_key_typo_rate = self._safe_mean(reward_metrics.get("point_key_typo_rate", []))

            def _fmt(x: Optional[float]) -> str:
                return "na" if x is None else f"{x:.4f}"

            print(
                "[RewardHealth] "
                f"calls={self._reward_debug_calls} "
                f"overall_mean={_fmt(overall_mean)} answer_mean={_fmt(answer_mean)} point_mean={_fmt(point_mean)} "
                f"format_fail_rate={_fmt(format_fail_mean)} stop_violation_rate={_fmt(stop_violation_mean)} "
                f"stopped_by_answer_rate={_fmt(stopped_by_answer_mean)} turns_exceeded_rate={_fmt(turns_exceeded_mean)} "
                f"no_point_pred_rate={_fmt(no_point_pred_mean)} "
                f"point_key_typo_mean={_fmt(point_key_typo_mean)} point_key_typo_rate={_fmt(point_key_typo_rate)}"
            )

            if (
                overall_mean is not None
                and stop_violation_mean is not None
                and format_fail_mean is not None
                and stopped_by_answer_mean is not None
                and overall_mean <= 0.01
                and (stop_violation_mean >= 0.80 or format_fail_mean >= 0.80)
            ):
                dominant_failure = "stop_violation" if stop_violation_mean >= format_fail_mean else "format_fail"
                print(
                    "[RewardHealth][ALERT] "
                    f"reward collapse detected: dominant={dominant_failure} "
                    f"overall_mean={overall_mean:.4f} stop_violation_rate={stop_violation_mean:.4f} "
                    f"format_fail_rate={format_fail_mean:.4f} stopped_by_answer_rate={stopped_by_answer_mean:.4f}. "
                    "Check interleaved prompt composition/termination and trajectory reward reasons."
                )

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        response_str, ground_truth = [], []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str.append(
                self.tokenizer.decode(valid_response_ids, skip_special_tokens=self.config.skip_special_tokens)
            )
            ground_truth.append(data.non_tensor_batch["ground_truth"][i])

        scores = self.reward_fn(response_str, ground_truth)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
