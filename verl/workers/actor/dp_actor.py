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
"""
Implement Actor
"""

import os
import time
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
try:
    from transformers.modeling_flash_attention_utils import index_first_axis, pad_input, unpad_input
except ImportError:
    from ...utils.padding_utils import index_first_axis, pad_input, unpad_input

from ...protocol import DataProto
from ...trainer import core_algos
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


__all__ = ["DataParallelPPOActor"]


def _corrected_torch_fallback_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return VF.log_probs_from_logits(
        logits,
        labels,
        correct_torch_fallback_logprob_sign=True,
    )


def resolve_logprob_function(fallback_mode: str):
    """Resolve a feature-gated log-prob kernel without changing V36 defaults."""
    mode = str(fallback_mode).strip().lower()
    if mode not in ("legacy", "correct", "error"):
        raise ValueError(
            "torch_logprob_fallback_mode must be one of legacy, correct, or error, "
            f"got {fallback_mode!r}."
        )
    if mode == "error" and not VF.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        raise RuntimeError(
            "torch_logprob_fallback_mode=error requires the FlashAttention "
            "cross-entropy kernel; refusing the historical positive-CE fallback."
        )
    if mode == "correct":
        return _corrected_torch_fallback_log_probs
    return VF.log_probs_from_logits


def _is_grad_spike_against_ema(
    grad_norm: float, ema: Optional[float], threshold: float, *, ignore_zero_ema: bool = False
) -> bool:
    """Apply legacy zero-EMA behavior unless the V37 safety mode opts out."""
    if ema is None:
        return False
    if ignore_zero_ema and ema <= 0.0:
        return False
    return grad_norm > ema * threshold


def _validate_adaptive_optimizer_rpc_counts(attempted: int, executed: int) -> None:
    """Keep one scheduler/controller transition per adaptive actor RPC."""
    if attempted != 1 or executed not in (0, 1) or executed > attempted:
        raise RuntimeError(
            "adaptive_actor_kl requires exactly one optimizer attempt per update_policy RPC; "
            f"got attempted={attempted}, executed={executed}."
        )


def coherent_actual_loss_metrics(
    policy_sum: float,
    entropy_sum: float,
    entropy_bonus_sum: float,
    loss_token_count: float,
    kl_sum: float,
    kl_token_count: float,
    kl_coef: float,
) -> dict[str, float]:
    """Compose actual-loss metrics from compatible token-weighted aggregates."""
    values = (
        policy_sum,
        entropy_sum,
        entropy_bonus_sum,
        loss_token_count,
        kl_sum,
        kl_token_count,
        kl_coef,
    )
    if not all(torch.isfinite(torch.tensor(float(value))).item() for value in values):
        raise RuntimeError("Non-finite actual-loss aggregate.")
    if loss_token_count <= 0 or kl_token_count <= 0 or kl_coef < 0:
        raise RuntimeError("Invalid actual-loss token counts or KL coefficient.")
    policy_mean = policy_sum / loss_token_count
    entropy_mean = entropy_sum / loss_token_count
    entropy_bonus_mean = entropy_bonus_sum / loss_token_count
    kl_mean = kl_sum / kl_token_count
    return {
        "policy_loss": policy_mean,
        "entropy_loss": entropy_mean,
        "entropy_bonus": entropy_bonus_mean,
        "kl_loss": kl_mean,
        "kl_penalty": kl_coef * kl_mean,
        "total_loss": policy_mean + kl_coef * kl_mean - entropy_bonus_mean,
    }


def scale_full_shard_token_sum(
    local_token_sum: torch.Tensor,
    global_token_count: torch.Tensor,
    gradient_world_size: int,
    gradient_accumulation: int = 1,
) -> torch.Tensor:
    """Scale a local token sum before the caller's accumulation division.

    FSDP averages gradients over ``gradient_world_size`` ranks and the caller
    later divides every microbatch loss by ``gradient_accumulation``.  This
    factor makes the resulting averaged gradient exactly the global valid-token
    mean when every microbatch contributes its local token sum.
    """
    if (
        gradient_world_size <= 0
        or gradient_accumulation <= 0
        or global_token_count.numel() != 1
        or global_token_count.item() <= 0
    ):
        raise RuntimeError("Invalid full-shard token-mean scaling inputs.")
    denominator = global_token_count.to(device=local_token_sum.device, dtype=local_token_sum.dtype)
    return local_token_sum * (gradient_world_size * gradient_accumulation) / denominator


def masked_mean_to_token_sum(
    masked_mean: torch.Tensor, valid_token_count: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Invert ``VF.masked_mean`` exactly for a scalar mean and detached count."""
    if valid_token_count.numel() != 1 or valid_token_count.item() < 0:
        raise RuntimeError("Invalid valid-token count for masked-mean reconstruction.")
    denominator = valid_token_count.to(device=masked_mean.device, dtype=masked_mean.dtype) + eps
    return masked_mean * denominator


def validate_adaptive_actor_kl_topology(ulysses_size: int, fsdp_size: int, world_size: int) -> None:
    if ulysses_size > 1:
        raise RuntimeError("adaptive_actor_kl currently fails closed for Ulysses sequence parallelism.")
    if 0 < fsdp_size < world_size:
        raise RuntimeError("adaptive_actor_kl currently fails closed for hybrid-sharded FSDP.")


def _handle_multimodal_logprob_mismatch(
    message: str,
    batch_size: int,
    response_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Fail closed unless the legacy zero-logprob fallback is explicitly enabled."""
    if os.environ.get("EASYR1_ALLOW_ZERO_MM_LOGPROB", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        print(
            "WARNING: EASYR1_ALLOW_ZERO_MM_LOGPROB=1; returning zero log-probabilities "
            f"for a mismatched multimodal batch: {message}"
        )
        return torch.zeros(batch_size, response_length, device=device, dtype=torch.float32)
    raise RuntimeError(
        f"{message} Refusing to continue with zero log-probabilities because that silently "
        "corrupts PPO/GRPO ratios. Fix the multimodal/SP alignment, or explicitly set "
        "EASYR1_ALLOW_ZERO_MM_LOGPROB=1 only for legacy debugging."
    )


def _validate_image_features_and_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multi_modal_inputs: Dict[str, torch.Tensor],
    processor=None,
) -> tuple[bool, Optional[str]]:
    """
    校验 image features 和 image tokens 是否匹配
    
    Args:
        input_ids: 输入token ids
        attention_mask: attention mask
        multi_modal_inputs: 多模态输入，包含 pixel_values 和 image_grid_thw
        processor: 处理器，用于获取 image token id
        
    Returns:
        (is_valid, error_msg): 是否有效，错误信息（如果无效）
    """
    if not multi_modal_inputs or "image_grid_thw" not in multi_modal_inputs:
        return True, None  # 没有图像数据，跳过校验
    
    try:
        # 计算 image features 数量（通过 image_grid_thw）
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is None:
            return True, None  # 没有 image_grid_thw，无法校验
        
        # image_grid_thw shape: (num_images, 3) -> (t, h, w)
        # 每个图像的 token 数量 = t * h * w
        import numpy as np
        if isinstance(image_grid_thw, torch.Tensor):
            num_image_features = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum().item()
        elif isinstance(image_grid_thw, np.ndarray):
            num_image_features = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum()
        else:
            return True, None  # 无法校验，假设有效
        
        # 计算 input_ids 中的 image tokens 数量
        # 对于 Qwen2.5-VL，image token 是 <|image_pad|>
        image_token_id = None
        if processor is not None:
            try:
                image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            except:
                pass
        
        if image_token_id is not None:
            # 只统计有效位置的 tokens (attention_mask == 1)
            if attention_mask is not None:
                valid_input_ids = input_ids[attention_mask == 1]
            else:
                valid_input_ids = input_ids.flatten()
            num_image_tokens = (valid_input_ids == image_token_id).sum().item()
        else:
            # 如果无法获取 image_token_id，无法准确校验，假设匹配
            return True, None
        
        # 允许一定的误差范围（5%或至少10个token），因为可能有舍入误差
        tolerance = max(10, int(num_image_features * 0.05))
        if abs(num_image_tokens - num_image_features) > tolerance:
            error_msg = (
                f"Image features and image tokens mismatch: "
                f"tokens={num_image_tokens}, features={num_image_features} "
                f"(diff={abs(num_image_tokens - num_image_features)}). "
                f"The batch cannot produce trustworthy log-probabilities."
            )
            return False, error_msg
        
        return True, None
    except Exception as e:
        # 如果校验过程出错，记录警告但允许继续（避免因为校验逻辑问题导致训练失败）
        return True, None


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        # === Grad Norm Spike Detection & Adaptive LR Protection ===
        self._spike_enabled = os.getenv("GRAD_SPIKE_PROTECT", "1") == "1"
        self._spike_threshold = float(os.getenv("GRAD_SPIKE_THRESHOLD", "5.0"))   # spike if grad_norm > EMA * threshold
        self._spike_cooldown_steps = int(os.getenv("GRAD_SPIKE_COOLDOWN", "3"))   # reduce LR for N micro-steps after spike
        self._spike_lr_factor = float(os.getenv("GRAD_SPIKE_LR_FACTOR", "0.1"))  # multiply LR by this during cooldown
        self._grad_norm_ema = None       # EMA of grad_norm (initialized on first finite grad)
        self._grad_norm_ema_alpha = 0.1  # EMA smoothing factor
        self._spike_cooldown_remaining = 0  # remaining cooldown micro-steps
        self._spike_count = 0            # total spike count for logging
        self._original_lrs = None        # saved original LRs before reduction
        # Emergency brake: permanently halve LR if too many spikes in a window
        self._spike_brake_window = int(os.getenv("GRAD_SPIKE_BRAKE_WINDOW", "30"))   # window in optimizer steps
        self._spike_brake_max = int(os.getenv("GRAD_SPIKE_BRAKE_MAX", "6"))          # max spikes before brake
        self._spike_brake_activated = False
        self._spike_history = []         # list of optimizer step numbers when spikes occurred
        self._optimizer_step_count = 0   # total optimizer steps for brake window tracking
        self._last_optimizer_step_executed = False
        self._last_optimizer_skip_reason = ""
        # Non-finite grad recovery (NaN/Inf): backoff + cooldown + emergency brake
        self._nonfinite_count = 0
        self._nonfinite_history = []
        self._nonfinite_cooldown_steps = int(
            os.getenv("GRAD_NONFINITE_COOLDOWN", str(self._spike_cooldown_steps))
        )
        self._nonfinite_lr_factor = float(
            os.getenv("GRAD_NONFINITE_LR_FACTOR", str(self._spike_lr_factor))
        )
        self._nonfinite_brake_window = int(
            os.getenv("GRAD_NONFINITE_BRAKE_WINDOW", str(self._spike_brake_window))
        )
        self._nonfinite_brake_max = int(os.getenv("GRAD_NONFINITE_BRAKE_MAX", "3"))
        # Absolute cap on grad_norm (independent of EMA)
        self._grad_spike_absolute_cap = float(os.getenv("GRAD_SPIKE_ABSOLUTE_CAP", "0"))
        if self._spike_enabled and self.rank == 0:
            print(f"[GradSpikeProtect] ENABLED: threshold={self._spike_threshold}x, "
                  f"cooldown={self._spike_cooldown_steps} steps, lr_factor={self._spike_lr_factor}, "
                  f"absolute_cap={'OFF' if self._grad_spike_absolute_cap <= 0 else self._grad_spike_absolute_cap}")

        # === FP16 Gradient Underflow Monitor ===
        self._fp16_monitor_enabled = os.getenv('FP16_GRAD_UNDERFLOW_MONITOR', '0') == '1'
        self._fp16_monitor_every = int(os.getenv('FP16_GRAD_UNDERFLOW_MONITOR_EVERY', '10'))
        self._fp16_monitor_step = 0
        if self._fp16_monitor_enabled and self.rank == 0:
            print(f'[FP16Monitor] ENABLED: check every {self._fp16_monitor_every} optimizer steps')

        self._grad_protection_config_identity = {
            "version": 1,
            "adaptive_actor_kl": bool(getattr(config, "adaptive_actor_kl", False)),
            "spike_enabled": self._spike_enabled,
            "spike_threshold": self._spike_threshold,
            "spike_cooldown_steps": self._spike_cooldown_steps,
            "spike_lr_factor": self._spike_lr_factor,
            "ema_alpha": self._grad_norm_ema_alpha,
            "brake_window": self._spike_brake_window,
            "brake_max": self._spike_brake_max,
            "nonfinite_cooldown_steps": self._nonfinite_cooldown_steps,
            "nonfinite_lr_factor": self._nonfinite_lr_factor,
            "nonfinite_brake_window": self._nonfinite_brake_window,
            "nonfinite_brake_max": self._nonfinite_brake_max,
            "absolute_cap": self._grad_spike_absolute_cap,
            "max_grad_norm": float(config.max_grad_norm),
            "optimizer_param_groups": len(actor_optimizer.param_groups) if actor_optimizer is not None else 0,
            "configured_lr": float(config.optim.lr) if actor_optimizer is not None else None,
        }
        # FSDPCheckpointManager is intentionally actor-agnostic.  Attach narrow
        # adaptive-only callbacks to the optimizer so its existing per-rank
        # extra-state file can persist exact actor runtime state without
        # changing legacy/V36 checkpoint and LR-override behavior.
        if actor_optimizer is not None and bool(getattr(config, "adaptive_actor_kl", False)):
            actor_optimizer._easy_r1_actor_state_getter = self.runtime_state_dict
            actor_optimizer._easy_r1_actor_state_loader = self.load_runtime_state_dict

        logprob_fn = resolve_logprob_function(
            getattr(config, "torch_logprob_fallback_mode", "legacy")
        )
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(logprob_fn, dynamic=True)
        else:
            self.log_probs_from_logits = logprob_fn

    def runtime_state_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "config_identity": dict(self._grad_protection_config_identity),
            "grad_norm_ema": self._grad_norm_ema,
            "spike_cooldown_remaining": self._spike_cooldown_remaining,
            "original_lrs": None if self._original_lrs is None else list(self._original_lrs),
            "current_lrs": [float(group["lr"]) for group in self.actor_optimizer.param_groups],
            "spike_count": self._spike_count,
            "spike_history": list(self._spike_history),
            "spike_brake_activated": self._spike_brake_activated,
            "nonfinite_count": self._nonfinite_count,
            "nonfinite_history": list(self._nonfinite_history),
            "optimizer_attempt_count": self._optimizer_step_count,
            "last_optimizer_step_executed": self._last_optimizer_step_executed,
            "last_optimizer_skip_reason": self._last_optimizer_skip_reason,
            "fp16_monitor_step": self._fp16_monitor_step,
        }

    def load_runtime_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if state is None:
            if bool(getattr(self.config, "adaptive_actor_kl", False)):
                raise RuntimeError("adaptive_actor_kl checkpoint is missing per-rank actor runtime state.")
            return  # adaptive-off legacy checkpoint
        if not isinstance(state, dict) or state.get("version") != 1:
            raise RuntimeError("Unsupported actor runtime checkpoint state.")
        if state.get("config_identity") != self._grad_protection_config_identity:
            raise RuntimeError("Actor runtime checkpoint configuration identity differs from this run.")

        def _counter(name: str) -> int:
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"Invalid actor runtime counter {name}={value!r}.")
            return value

        attempt_count = _counter("optimizer_attempt_count")
        spike_history = state.get("spike_history")
        nonfinite_history = state.get("nonfinite_history")
        for name, history in (("spike_history", spike_history), ("nonfinite_history", nonfinite_history)):
            if not isinstance(history, list) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > attempt_count
                for item in history
            ):
                raise RuntimeError(f"Invalid actor runtime {name}.")
        current_lrs = state.get("current_lrs")
        original_lrs = state.get("original_lrs")
        group_count = len(self.actor_optimizer.param_groups)
        if not isinstance(current_lrs, list) or len(current_lrs) != group_count:
            raise RuntimeError("Actor runtime current_lrs do not match optimizer param groups.")
        if original_lrs is not None and (not isinstance(original_lrs, list) or len(original_lrs) != group_count):
            raise RuntimeError("Actor runtime original_lrs do not match optimizer param groups.")
        for name, values in (("current_lrs", current_lrs), ("original_lrs", original_lrs or [])):
            if any(not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))) or value < 0 for value in values):
                raise RuntimeError(f"Invalid actor runtime {name}.")
        ema = state.get("grad_norm_ema")
        if ema is not None and (not isinstance(ema, (int, float)) or not torch.isfinite(torch.tensor(float(ema))) or ema < 0):
            raise RuntimeError("Invalid actor runtime grad_norm_ema.")
        for name in ("last_optimizer_step_executed", "spike_brake_activated"):
            if type(state.get(name)) is not bool:
                raise RuntimeError(f"Invalid actor runtime boolean {name}.")
        skip_reason = state.get("last_optimizer_skip_reason")
        if not isinstance(skip_reason, str):
            raise RuntimeError("Invalid actor runtime last_optimizer_skip_reason.")

        self._grad_norm_ema = None if ema is None else float(ema)
        self._spike_cooldown_remaining = _counter("spike_cooldown_remaining")
        self._original_lrs = None if original_lrs is None else [float(value) for value in original_lrs]
        self._spike_count = _counter("spike_count")
        self._spike_history = list(spike_history)
        self._spike_brake_activated = state["spike_brake_activated"]
        self._nonfinite_count = _counter("nonfinite_count")
        self._nonfinite_history = list(nonfinite_history)
        self._optimizer_step_count = attempt_count
        self._last_optimizer_step_executed = state["last_optimizer_step_executed"]
        self._last_optimizer_skip_reason = skip_reason
        self._fp16_monitor_step = _counter("fp16_monitor_step")
        for group, lr in zip(self.actor_optimizer.param_groups, current_lrs):
            group["lr"] = float(lr)
        # The worker's legacy post-load LR override compares against config.lr.
        # During cooldown the exact current LR is intentionally below the base;
        # mirror it here so that override cannot destroy restored runtime state.
        self.config.optim.lr = float(current_lrs[0])

    @staticmethod
    def _distributed_agreed_int(name: str, value: int, device: torch.device) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            value_tensor = torch.tensor(int(value), dtype=torch.long, device=device)
            minimum = value_tensor.clone()
            maximum = value_tensor.clone()
            torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
            if int(minimum.item()) != int(maximum.item()):
                raise RuntimeError(
                    f"Distributed {name} disagreement: min={int(minimum.item())}, max={int(maximum.item())}."
                )
            return int(minimum.item())
        return int(value)

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            # Collect all tensors for each key, ensuring they're on the same device
            device = input_ids.device
            for key in micro_batch["multi_modal_inputs"][0].keys():
                tensors_to_cat = []
                for inputs in micro_batch["multi_modal_inputs"]:
                    tensor = inputs[key]
                    # Convert to tensor if needed and move to the correct device
                    if not isinstance(tensor, torch.Tensor):
                        tensor = torch.as_tensor(tensor)
                    # Ensure tensor is on the same device as input_ids
                    if tensor.device != device:
                        tensor = tensor.to(device)
                    tensors_to_cat.append(tensor)
                
                if len(tensors_to_cat) > 0:
                    # Concatenate along the first dimension (batch dimension)
                    # For pixel_values: (num_images_per_sample, ...) -> (total_images, ...)
                    # For image_grid_thw: (num_images_per_sample, 3) -> (total_images, 3)
                    multi_modal_inputs[key] = torch.cat(tensors_to_cat, dim=0)

        # 校验 image features 和 tokens 是否匹配（在调用模型之前）
        if multi_modal_inputs:
            # 尝试获取 processor（如果可用）
            processor = getattr(self.actor_module, 'processor', None)
            if processor is None and hasattr(self.actor_module, 'module'):
                processor = getattr(self.actor_module.module, 'processor', None)
            
            is_valid, error_msg = _validate_image_features_and_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                multi_modal_inputs=multi_modal_inputs,
                processor=processor,
            )
            
            if not is_valid:
                if self.rank == 0:
                    print(f"WARNING: {error_msg}")
                return _handle_multimodal_logprob_mismatch(
                    error_msg or "Image features and image tokens mismatch.",
                    batch_size,
                    response_length,
                    input_ids.device,
                )

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_sequence_parallel_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            try:
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature)
                # ((total_nnz / sp) + pad)
                log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
            except ValueError as e:
                # 捕获 "Image features and image tokens do not match" 错误
                if "Image features and image tokens do not match" in str(e):
                    if self.rank == 0:
                        print(f"WARNING: {str(e)}")
                    return _handle_multimodal_logprob_mismatch(
                        str(e), batch_size, response_length, input_ids.device
                    )
                else:
                    raise  # 重新抛出其他 ValueError

            # gather log_prob if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            # 对于非 padding_free 模式，也需要校验
            if multi_modal_inputs:
                processor = getattr(self.actor_module, 'processor', None)
                if processor is None and hasattr(self.actor_module, 'module'):
                    processor = getattr(self.actor_module.module, 'processor', None)
                
                is_valid, error_msg = _validate_image_features_and_tokens(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    multi_modal_inputs=multi_modal_inputs,
                    processor=processor,
                )
                
                if not is_valid:
                    if self.rank == 0:
                        print(f"WARNING: {error_msg}")
                    return _handle_multimodal_logprob_mismatch(
                        error_msg or "Image features and image tokens mismatch.",
                        batch_size,
                        response_length,
                        input_ids.device,
                    )
            
            try:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits: torch.Tensor = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)
            except ValueError as e:
                # 捕获 "Image features and image tokens do not match" 错误
                if "Image features and image tokens do not match" in str(e):
                    if self.rank == 0:
                        print(f"WARNING: {str(e)}")
                    return _handle_multimodal_logprob_mismatch(
                        str(e), batch_size, response_length, input_ids.device
                    )
                else:
                    raise  # 重新抛出其他 ValueError

        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        self._last_optimizer_step_executed = False
        self._last_optimizer_skip_reason = ""
        self._optimizer_step_count += 1

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        # Decide before any rank mutates optimizer parameters.  A disagreement
        # must fail before a partial distributed update can occur.
        decision = 0  # execute
        if not torch.isfinite(grad_norm):
            decision = 1  # nonfinite
        elif self._spike_enabled:
            grad_norm_for_decision = float(grad_norm.detach().item())
            if self._grad_norm_ema is None and grad_norm_for_decision > 0.0:
                self._grad_norm_ema = grad_norm_for_decision
            if self._grad_spike_absolute_cap > 0 and grad_norm_for_decision > self._grad_spike_absolute_cap:
                decision = 2  # absolute cap
            elif _is_grad_spike_against_ema(
                grad_norm_for_decision,
                self._grad_norm_ema,
                self._spike_threshold,
                ignore_zero_ema=bool(getattr(self.config, "adaptive_actor_kl", False)),
            ):
                decision = 3  # EMA spike
        decision = self._distributed_agreed_int("optimizer decision", decision, grad_norm.device)

        if decision == 1:
            self._last_optimizer_skip_reason = "nonfinite_grad"
            self._nonfinite_count += 1
            current_lr = self.actor_optimizer.param_groups[0]["lr"]
            print(
                f"Gradient norm is not finite. Skip update. "
                f"opt_step={self._optimizer_step_count} nonfinite_count={self._nonfinite_count} lr={current_lr:.2e}"
            )
            self.actor_optimizer.zero_grad()

            if self._spike_enabled:
                # Treat non-finite as the highest-severity spike: immediate LR backoff + cooldown
                self._nonfinite_history.append(self._optimizer_step_count)
                old_factor = self._spike_lr_factor
                self._spike_lr_factor = self._nonfinite_lr_factor
                self._reduce_lr()
                self._spike_lr_factor = old_factor
                self._spike_cooldown_remaining = max(
                    self._spike_cooldown_remaining, self._nonfinite_cooldown_steps
                )

                recent_nonfinite = [
                    s for s in self._nonfinite_history
                    if s > self._optimizer_step_count - self._nonfinite_brake_window
                ]
                if len(recent_nonfinite) >= self._nonfinite_brake_max and not self._spike_brake_activated:
                    self._spike_brake_activated = True
                    # Permanently reduce both current LR and baseline LR by half.
                    for pg in self.actor_optimizer.param_groups:
                        pg["lr"] = pg["lr"] * 0.5
                    if self._original_lrs is not None:
                        self._original_lrs = [lr * 0.5 for lr in self._original_lrs]
                    if self.rank == 0:
                        print(
                            f"[GradSpikeProtect] NONFINITE EMERGENCY BRAKE: "
                            f"{len(recent_nonfinite)} non-finite events in last {self._nonfinite_brake_window} steps. "
                            f"Base LR permanently halved."
                        )
            return grad_norm

        grad_norm_val = grad_norm.detach().item()

        # === FP16 Gradient Underflow Monitor ===
        if self._fp16_monitor_enabled:
            self._fp16_monitor_step += 1
            if self._fp16_monitor_step % self._fp16_monitor_every == 1 and self.rank == 0:
                total_params = 0
                zero_params = 0
                subnormal_params = 0
                max_grad = 0.0
                min_nonzero_grad = float('inf')
                for name, p in self.actor_module.named_parameters():
                    if p.grad is not None:
                        g = p.grad
                        numel = g.numel()
                        total_params += numel
                        abs_g = g.abs()
                        zeros = (abs_g == 0).sum().item()
                        zero_params += zeros
                        nonzero_mask = abs_g > 0
                        if nonzero_mask.any():
                            nonzero_abs = abs_g[nonzero_mask]
                            subnormals = (nonzero_abs < 6.1e-5).sum().item()
                            subnormal_params += subnormals
                            gmax = nonzero_abs.max().item()
                            gmin = nonzero_abs.min().item()
                            if gmax > max_grad:
                                max_grad = gmax
                            if gmin < min_nonzero_grad:
                                min_nonzero_grad = gmin
                if total_params > 0:
                    zero_pct = 100.0 * zero_params / total_params
                    subnormal_pct = 100.0 * subnormal_params / total_params
                    print(f'[FP16Monitor] step={self._fp16_monitor_step}: '
                          f'zero_grad={zero_pct:.2f}% subnormal={subnormal_pct:.2f}% '
                          f'max_grad={max_grad:.4e} min_nonzero={min_nonzero_grad:.4e} '
                          f'grad_norm={grad_norm_val:.4f}')
                    if zero_pct > 30:
                        print(f'[FP16Monitor] WARNING: >30% zero gradients — potential fp16 underflow!')

        # === Grad Norm Spike Detection & Adaptive LR Protection ===
        if self._spike_enabled:
            # Initialize EMA on first finite grad
            if self._grad_norm_ema is None and grad_norm_val > 0.0:
                self._grad_norm_ema = grad_norm_val

            # === Absolute cap check (independent of EMA) ===
            if decision == 2:
                self._last_optimizer_skip_reason = "absolute_grad_cap"
                if self.rank == 0:
                    print(f"[GradSpikeProtect] ABSOLUTE CAP triggered: "
                          f"grad_norm={grad_norm_val:.3f} > cap={self._grad_spike_absolute_cap:.1f}, skipping step")
                self.actor_optimizer.zero_grad()
                return grad_norm

            is_spike = decision == 3

            if is_spike:
                self._last_optimizer_skip_reason = "grad_spike"
                self._spike_count += 1
                self._spike_history.append(self._optimizer_step_count)
                if self.rank == 0:
                    print(f"[GradSpikeProtect] SPIKE DETECTED #{self._spike_count}: "
                          f"grad_norm={grad_norm_val:.4f}, EMA={self._grad_norm_ema:.4f}, "
                          f"ratio={grad_norm_val / max(self._grad_norm_ema, 1e-8):.2f}x. "
                          f"Skipping update & entering cooldown for {self._spike_cooldown_steps} steps.")
                # Skip this update entirely — do NOT apply the spike gradient
                self.actor_optimizer.zero_grad()
                # Enter cooldown: reduce LR for subsequent steps
                self._spike_cooldown_remaining = self._spike_cooldown_steps
                self._reduce_lr()

                # === Emergency Brake: check if too many spikes in recent window ===
                recent_spikes = [s for s in self._spike_history
                                 if s > self._optimizer_step_count - self._spike_brake_window]
                if len(recent_spikes) >= self._spike_brake_max and not self._spike_brake_activated:
                    self._spike_brake_activated = True
                    # Permanently halve the base LR
                    if self._original_lrs is not None:
                        self._original_lrs = [lr * 0.5 for lr in self._original_lrs]
                    if self.rank == 0:
                        print(f"[GradSpikeProtect] EMERGENCY BRAKE: {len(recent_spikes)} spikes "
                              f"in last {self._spike_brake_window} steps. Base LR permanently halved.")

                return grad_norm

            # Handle cooldown: if we're in cooldown, LR is already reduced
            if self._spike_cooldown_remaining > 0:
                self._spike_cooldown_remaining -= 1
                if self.rank == 0:
                    current_lrs = [pg["lr"] for pg in self.actor_optimizer.param_groups]
                    print(f"[GradSpikeProtect] Cooldown step (remaining={self._spike_cooldown_remaining}), "
                          f"grad_norm={grad_norm_val:.4f}, LR={current_lrs[0]:.2e}")
                # Do the step with reduced LR
                self.actor_optimizer.step()
                self._last_optimizer_step_executed = True
                # Restore LR when cooldown ends
                if self._spike_cooldown_remaining == 0:
                    self._restore_lr()
                    if self.rank == 0:
                        restored_lrs = [pg["lr"] for pg in self.actor_optimizer.param_groups]
                        print(f"[GradSpikeProtect] Cooldown ended. LR restored to {restored_lrs[0]:.2e}")
            else:
                # Normal step
                self.actor_optimizer.step()
                self._last_optimizer_step_executed = True

            # FIX: Update EMA only with non-spike AND non-cooldown values
            # During cooldown, grad_norms are elevated and would contaminate the EMA,
            # causing the spike threshold to ratchet up (positive feedback loop).
            if not is_spike and self._spike_cooldown_remaining == 0 and grad_norm_val > 0.0:
                if self._grad_norm_ema is None:
                    self._grad_norm_ema = grad_norm_val
                else:
                    self._grad_norm_ema = (
                        (1 - self._grad_norm_ema_alpha) * self._grad_norm_ema
                        + self._grad_norm_ema_alpha * grad_norm_val
                    )
        else:
            # Spike protection disabled — original behavior
            self.actor_optimizer.step()
            self._last_optimizer_step_executed = True

        # === Structured per-step gradient health log ===
        if self.rank == 0 and self._spike_enabled:
            _ema = self._grad_norm_ema if self._grad_norm_ema is not None else 0.0
            _ratio = grad_norm_val / max(_ema, 1e-8) if self._grad_norm_ema is not None else 0.0
            _lr = self.actor_optimizer.param_groups[0]["lr"]
            print(
                f"[TrainHealth] opt_step={self._optimizer_step_count} "
                f"grad_norm={grad_norm_val:.4f} ema={_ema:.4f} ratio={_ratio:.2f}x "
                f"spike_count={self._spike_count} cooldown={self._spike_cooldown_remaining} "
                f"brake={'ON' if self._spike_brake_activated else 'off'} "
                f"lr={_lr:.2e}"
            )

        self.actor_optimizer.zero_grad()
        return grad_norm

    def _reduce_lr(self):
        """Temporarily reduce LR by spike_lr_factor. Save original LRs."""
        if self._original_lrs is None:
            # First spike: save current LRs and reduce
            self._original_lrs = [pg["lr"] for pg in self.actor_optimizer.param_groups]
            for pg in self.actor_optimizer.param_groups:
                pg["lr"] = pg["lr"] * self._spike_lr_factor
        # If already in cooldown (consecutive spike), just reset counter — LR already reduced

    def _restore_lr(self):
        """Restore LRs to values before reduction."""
        if self._original_lrs is not None:
            for pg, orig_lr in zip(self.actor_optimizer.param_groups, self._original_lrs):
                pg["lr"] = orig_lr
            self._original_lrs = None

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        skipped_batches = 0
        total_batches = 0
        # Disable tqdm to avoid log spam - just use the iterator directly
        # if self.rank == 0:
        #     micro_batches = tqdm(micro_batches, desc="Compute log probs", position=2, disable=True)

        for micro_batch in micro_batches:
            total_batches += 1
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            
            # 检测是否跳过了这个 batch（全零的 log_probs 可能表示跳过）
            # 但要注意，正常的 log_probs 也可能接近零，所以我们需要更精确的检测
            # 这里我们假设如果 log_probs 全为零且 batch 有 multi_modal_inputs，可能是跳过了
            if "multi_modal_inputs" in model_inputs and torch.all(log_probs == 0):
                skipped_batches += 1
                if self.rank == 0 and skipped_batches <= 5:  # 只打印前5个，避免日志过多
                    print(f"INFO: Skipped batch {total_batches} due to image features/tokens mismatch")
            
            log_probs_lst.append(log_probs)

        if skipped_batches > 0 and self.rank == 0:
            print(f"INFO: Skipped {skipped_batches}/{total_batches} batches due to image features/tokens mismatch")

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        adaptive_actor_kl = bool(data.meta_info.get("adaptive_actor_kl", False))
        if adaptive_actor_kl != bool(self.config.adaptive_actor_kl):
            raise ValueError("adaptive_actor_kl driver/worker configuration mismatch.")
        if adaptive_actor_kl:
            distributed_world_size = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 1
            )
            fsdp_size = int(self.config.fsdp.fsdp_size)
            validate_adaptive_actor_kl_topology(
                int(self.config.ulysses_sequence_parallel_size), fsdp_size, distributed_world_size
            )
            if self.config.kl_penalty != "low_var_kl":
                raise ValueError("adaptive_actor_kl only supports sampled-token low_var_kl.")
            if os.environ.get("EASYR1_ALLOW_ZERO_MM_LOGPROB", "0").lower() in ("1", "true", "yes", "on"):
                raise RuntimeError("adaptive_actor_kl forbids the zero multimodal log-probability fallback.")
            adaptive_kl_coef = float(data.meta_info["adaptive_actor_kl_coef"])
            if not torch.isfinite(torch.tensor(adaptive_kl_coef)) or adaptive_kl_coef < 0:
                raise ValueError(f"Invalid adaptive actor KL coefficient: {adaptive_kl_coef}.")
        else:
            adaptive_kl_coef = 0.0
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if (self.config.use_kl_loss or adaptive_actor_kl) and not self.config.disable_kl:
            select_keys.append("ref_log_probs")

        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        optimizer_steps_attempted = 0
        optimizer_steps_executed = 0
        adaptive_kl_sum = 0.0
        adaptive_kl_count = 0.0
        adaptive_policy_sum = 0.0
        adaptive_entropy_sum = 0.0
        adaptive_entropy_bonus_sum = 0.0
        adaptive_loss_token_count = 0.0
        for _ in range(self.config.ppo_epochs):
            # Disable tqdm to avoid log spam
            # if self.rank == 0:
            #     mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=2)

            for mini_batch in mini_batches:
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if adaptive_actor_kl:
                    mini_response_length = mini_batch.batch["responses"].size(1)
                    local_valid_count = mini_batch.batch["attention_mask"][:, -mini_response_length:].sum().double()
                    global_valid_count = local_valid_count.detach().clone()
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.all_reduce(global_valid_count, op=torch.distributed.ReduceOp.SUM)
                        gradient_world_size = torch.distributed.get_world_size()
                    else:
                        gradient_world_size = 1
                    if global_valid_count.item() <= 0:
                        raise ValueError("adaptive_actor_kl received a mini-batch with zero valid response tokens.")
                # Disable tqdm to avoid log spam
                # if self.rank == 0:
                #     micro_batches = tqdm(micro_batches, desc="Update policy", position=3)

                for micro_batch_idx, micro_batch in enumerate(micro_batches):
                    # Periodic empty_cache before each micro_batch: releases PyTorch cached-but-free
                    # memory back to CUDA driver. Avoids false 'reserved>100%' from vLLM+actor
                    # sharing the same CUDA context. Only warn on actual allocation pressure.
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()  # release cached blocks before each micro_batch
                            free_mem, total_mem = torch.cuda.mem_get_info()
                            used_mem = total_mem - free_mem
                            usage_ratio = used_mem / total_mem if total_mem > 0 else 0.0
                            if usage_ratio >= 0.95:  # only actual allocated memory is an OOM risk
                                if self.rank == 0:
                                    reserved_ratio = torch.cuda.memory_reserved() / total_mem if total_mem > 0 else 0.0
                                    print(f"Warning: GPU allocated high (allocated={usage_ratio*100:.1f}%, "
                                          f"reserved={reserved_ratio*100:.1f}%). May OOM soon.")
                        except Exception as e:
                            if self.rank == 0:
                                print(f"Warning: Could not check GPU memory: {e}")
                    
                    try:
                        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                        responses = model_inputs["responses"]
                        response_length = responses.size(1)
                        attention_mask = model_inputs["attention_mask"]
                        response_mask = attention_mask[:, -response_length:]
                        old_log_probs = model_inputs["old_log_probs"]
                        advantages = model_inputs["advantages"]

                        # all return: (bsz, response_length)
                        log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                        
                        # 检查是否跳过了这个 batch（全零 log_probs 且有多模态输入）
                        # 注意：FSDP 需要所有 rank 同步，不能跳过，所以使用零值但继续执行
                        if "multi_modal_inputs" in model_inputs and torch.all(log_probs == 0):
                            if self.rank == 0:
                                print(f"Warning: Image features/tokens mismatch detected. Using zero loss to maintain FSDP sync.")
                            # 不跳过，继续使用 log_probs（已经是零值）进行计算，以保持 FSDP 同步
                        
                        entropy_loss = -VF.masked_mean(log_probs, response_mask)  # estimator of entropy loss

                        policy_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl = core_algos.compute_policy_loss(
                            old_log_probs=old_log_probs,
                            log_probs=log_probs,
                            advantages=advantages,
                            response_mask=response_mask,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            clip_ratio_dual=self.config.clip_ratio_dual,
                        )
                        if adaptive_actor_kl:
                            local_valid_count = response_mask.sum().detach()
                            policy_token_sum = masked_mean_to_token_sum(policy_loss, local_valid_count)
                            entropy_token_sum = masked_mean_to_token_sum(entropy_loss, local_valid_count)
                            ref_log_probs = model_inputs["ref_log_probs"]
                            kld = core_algos.compute_stable_low_var_kl(log_probs, ref_log_probs)
                            local_kl_sum = (kld * response_mask).sum()
                            entropy_bonus_coeff = float(getattr(self.config, "entropy_bonus_coeff", 0.0))
                            local_objective_sum = policy_token_sum + adaptive_kl_coef * local_kl_sum
                            if entropy_bonus_coeff > 0:
                                local_objective_sum = local_objective_sum - entropy_bonus_coeff * entropy_token_sum

                            # The caller below divides by gradient_accumulation;
                            # the helper includes its exact inverse as well as
                            # the inverse of FSDP/DDP rank averaging.
                            total_loss = scale_full_shard_token_sum(
                                local_objective_sum,
                                global_valid_count,
                                gradient_world_size,
                                gradient_accumulation,
                            )

                            metric_token_count = float(local_valid_count.double().item())
                            adaptive_loss_token_count += metric_token_count
                            adaptive_policy_sum += float(policy_token_sum.detach().double().item())
                            adaptive_entropy_sum += float(entropy_token_sum.detach().double().item())
                            adaptive_kl_sum += float(local_kl_sum.detach().double().item())
                            adaptive_kl_count += metric_token_count
                            adaptive_entropy_bonus_sum += float(
                                (entropy_bonus_coeff * entropy_token_sum).detach().double().item()
                            )
                        elif "ref_log_probs" in model_inputs:
                            total_loss = policy_loss
                            ref_log_probs = model_inputs["ref_log_probs"]
                            # compute kl loss
                            kld = core_algos.compute_kl(
                                log_probs=log_probs,
                                ref_log_probs=ref_log_probs,
                                kl_penalty=self.config.kl_penalty,
                            )
                            kl_loss = VF.masked_mean(kld, response_mask)
                            total_loss = total_loss + kl_loss * self.config.kl_coef
                            metrics["actor/kl_loss"] = kl_loss.detach().item()
                            metrics["actor/kl_coef"] = self.config.kl_coef
                        else:
                            total_loss = policy_loss


                        # DiVA-GRPO v3: Entropy preservation bonus
                        # Subtracting entropy_loss (which is -mean(log_probs)) from pg_loss
                        # encourages the policy to maintain higher entropy, preventing mode collapse.
                        if hasattr(self.config, 'entropy_bonus_coeff') and self.config.entropy_bonus_coeff > 0:
                            entropy_bonus = self.config.entropy_bonus_coeff * entropy_loss
                            if not adaptive_actor_kl:
                                total_loss = total_loss - entropy_bonus
                                metrics["actor/entropy_bonus"] = entropy_bonus.detach().item()
                        loss = total_loss / gradient_accumulation
                        
                        # Cache already cleared above; backward can proceed safely.
                        
                        loss.backward()
                        
                        # Clear cache after backward to free memory (更激进的清理)
                        if torch.cuda.is_available():
                            try:
                                # 清理缓存
                                torch.cuda.empty_cache()
                                # 如果内存使用率仍然很高，尝试同步并再次清理
                                free_mem, total_mem = torch.cuda.mem_get_info()
                                usage_ratio = (total_mem - free_mem) / total_mem if total_mem > 0 else 0.0
                                if usage_ratio > 0.98:  # 如果使用率超过 98%
                                    torch.cuda.synchronize()  # 同步所有 CUDA 操作
                                    torch.cuda.empty_cache()  # 再次清理
                            except:
                                pass
                                
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                            if self.rank == 0:
                                print(f"CRITICAL: OOM error during Update policy. Error: {e}")
                            
                            # CRITICAL FIX: 遇到 OOM 必须抛出异常。
                            # 尝试使用 zero_loss.backward() 是无效的，会导致死锁。
                            # 抛出异常后，Ray 或 K8s 会重启 Worker，这是处理分布式 OOM 的唯一正确方式。
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            raise e 
                        else:
                            raise
                    except ValueError as e:
                        # CRITICAL FIX: 数据错误也必须抛出，不能吞掉。
                        if self.rank == 0:
                            print(f"CRITICAL: Data error during Update policy. Error: {e}")
                        raise e

                    batch_metrics = {
                        # Preserve V36's dashboard meaning: actor/pg_loss was
                        # the optimized objective including legacy KL/entropy.
                        "actor/pg_loss": (
                            policy_loss.detach().item()
                            if adaptive_actor_kl
                            else total_loss.detach().item()
                        ),
                        "actor/policy_loss": policy_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_clipfrac_higher.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, batch_metrics)

                optimizer_steps_attempted += 1
                grad_norm = self._optimizer_step()
                if getattr(self, "_last_optimizer_step_executed", False):
                    optimizer_steps_executed += 1
                spike_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                if adaptive_actor_kl:
                    # V37 evidence needs the cumulative counter even when the
                    # optional spike heuristic itself is disabled.
                    spike_metrics["actor/nonfinite_grad_count"] = float(self._nonfinite_count)
                if self._spike_enabled:
                    spike_metrics["actor/grad_norm_ema"] = self._grad_norm_ema if self._grad_norm_ema is not None else 0.0
                    spike_metrics["actor/spike_count"] = float(self._spike_count)
                    spike_metrics["actor/spike_cooldown"] = float(self._spike_cooldown_remaining)
                    spike_metrics["actor/nonfinite_grad_count"] = float(self._nonfinite_count)
                append_to_dict(metrics, spike_metrics)

        optimizer_steps_skipped = optimizer_steps_attempted - optimizer_steps_executed
        if adaptive_actor_kl:
            _validate_adaptive_optimizer_rpc_counts(
                optimizer_steps_attempted, optimizer_steps_executed
            )
            metric_device = data.batch["responses"].device
            optimizer_steps_attempted = self._distributed_agreed_int(
                "optimizer_steps_attempted", optimizer_steps_attempted, metric_device
            )
            optimizer_steps_executed = self._distributed_agreed_int(
                "optimizer_steps_executed", optimizer_steps_executed, metric_device
            )
            optimizer_steps_skipped = self._distributed_agreed_int(
                "optimizer_steps_skipped", optimizer_steps_skipped, metric_device
            )
            nonfinite_count = self._distributed_agreed_int(
                "nonfinite_grad_count", self._nonfinite_count, metric_device
            )
            # Replace per-microstep local observations with one rank-agreed
            # cumulative value. Formal evidence must not infer agreement from
            # a reduced mean that could hide divergent worker counters.
            metrics["actor/nonfinite_grad_count"] = [float(nonfinite_count)]
            metrics["actor/nonfinite_counter_agreement"] = [1.0]
        metrics["actor/optimizer_steps_attempted"] = [float(optimizer_steps_attempted)]
        metrics["actor/optimizer_steps_executed"] = [float(optimizer_steps_executed)]
        metrics["actor/optimizer_steps_skipped"] = [float(optimizer_steps_skipped)]
        metrics["actor/optimizer_counter_agreement"] = [1.0 if adaptive_actor_kl else 0.0]
        if adaptive_actor_kl:
            loss_aggregates = torch.tensor(
                [
                    adaptive_policy_sum,
                    adaptive_entropy_sum,
                    adaptive_entropy_bonus_sum,
                    adaptive_kl_sum,
                    adaptive_loss_token_count,
                    adaptive_kl_count,
                ],
                dtype=torch.float64,
                device=data.batch["responses"].device,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(loss_aggregates, op=torch.distributed.ReduceOp.SUM)
            global_loss_count = float(loss_aggregates[4].item())
            global_kl_count = float(loss_aggregates[5].item())
            if global_loss_count <= 0 or not torch.isfinite(loss_aggregates).all():
                raise RuntimeError("adaptive_actor_kl produced invalid global actual-loss aggregates.")
            if global_kl_count != global_loss_count:
                raise RuntimeError(
                    "adaptive_actor_kl loss components were aggregated over different valid-token counts."
                )
            actual = coherent_actual_loss_metrics(
                policy_sum=float(loss_aggregates[0].item()),
                entropy_sum=float(loss_aggregates[1].item()),
                entropy_bonus_sum=float(loss_aggregates[2].item()),
                loss_token_count=global_loss_count,
                kl_sum=float(loss_aggregates[3].item()),
                kl_token_count=global_kl_count,
                kl_coef=adaptive_kl_coef,
            )
            metrics["actor/pg_loss"] = [actual["policy_loss"]]
            metrics["actor/policy_loss"] = [actual["policy_loss"]]
            metrics["actor/entropy_loss"] = [actual["entropy_loss"]]
            metrics["actor/loss_token_count"] = [global_loss_count]
            if actual["entropy_bonus"] != 0.0:
                metrics["actor/entropy_bonus"] = [actual["entropy_bonus"]]
            metrics["actor/kl_token_sum"] = [float(loss_aggregates[3].item())]
            metrics["actor/kl_token_count"] = [global_kl_count]
            metrics["actor/kl_loss"] = [actual["kl_loss"]]
            metrics["actor/kl_coef"] = [adaptive_kl_coef]
            metrics["actor/kl_penalty"] = [actual["kl_penalty"]]
            metrics["actor/total_loss"] = [actual["total_loss"]]

        return metrics
