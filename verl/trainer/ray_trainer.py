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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import json
import random
import re
import shutil
import subprocess
import sys
import uuid
import time
import logging
import traceback
import tempfile
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Tuple

import numpy as np
import ray
import torch
from ray.exceptions import ActorDiedError, RayTaskError
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.action_ledger import ACTION_TYPE_IDS, validate_action_event_row
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.checkpoint.checkpoint_manager import (
    checkpoint_manifest_hash,
    publish_staged_checkpoint,
    reference_content_identity,
    validate_checkpoint_manifest,
    write_checkpoint_manifest,
)
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..utils.v37_training_evidence import (
    TrainingEvidenceError,
    TrainingEvidenceRecorder,
    require_clean_optimizer_step,
    reward_counts as v37_reward_counts,
    tree_sha256 as v37_tree_sha256,
    validate_resume_checkpoint_binding,
)
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics


TRAINER_RUNTIME_STATE_FILE = "trainer_runtime.pt"
TRAINER_RUNTIME_STATE_VERSION = 2
ADAPTIVE_ACTOR_KL_HORIZON_UNIT = "executed_optimizer_updates"
logger = logging.getLogger(__name__)


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        directory_fd = os.open(os.path.dirname(path), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def capture_controller_rng_state() -> dict[str, Any]:
    """Capture controller-process RNG state without lossy serialization."""
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_states = torch.cuda.get_rng_state_all() if cuda_available else []
    if len(cuda_states) != cuda_device_count:
        raise RuntimeError("Controller CUDA RNG state count does not match the visible CUDA topology.")
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
    }


def restore_controller_rng_state(state: dict[str, Any]) -> None:
    """Validate a controller RNG snapshot completely, then restore it."""
    if not isinstance(state, dict):
        raise RuntimeError("Invalid controller RNG checkpoint state.")
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    if type(state.get("cuda_available")) is not bool:
        raise RuntimeError("Controller RNG checkpoint has an invalid CUDA availability marker.")
    saved_cuda_count = state.get("cuda_device_count")
    if isinstance(saved_cuda_count, bool) or not isinstance(saved_cuda_count, int) or saved_cuda_count < 0:
        raise RuntimeError("Controller RNG checkpoint has an invalid CUDA device count.")
    if state["cuda_available"] != cuda_available or saved_cuda_count != cuda_device_count:
        raise RuntimeError(
            "Controller RNG checkpoint CUDA topology differs from this process: "
            f"saved=({state['cuda_available']}, {saved_cuda_count}), "
            f"current=({cuda_available}, {cuda_device_count})."
        )

    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_cpu_state = state.get("torch_cpu")
    torch_cuda_states = state.get("torch_cuda")
    try:
        random.Random().setstate(python_state)
        np.random.RandomState().set_state(numpy_state)
    except Exception as exc:
        raise RuntimeError("Controller Python/NumPy RNG checkpoint state is corrupt.") from exc
    if (
        not isinstance(torch_cpu_state, torch.Tensor)
        or torch_cpu_state.device.type != "cpu"
        or torch_cpu_state.dtype != torch.uint8
        or torch_cpu_state.ndim != 1
        or torch_cpu_state.numel() == 0
    ):
        raise RuntimeError("Controller Torch CPU RNG checkpoint state is corrupt.")
    try:
        torch.Generator(device="cpu").set_state(torch_cpu_state)
    except Exception as exc:
        raise RuntimeError("Controller Torch CPU RNG checkpoint state is corrupt.") from exc
    if not isinstance(torch_cuda_states, list) or len(torch_cuda_states) != saved_cuda_count:
        raise RuntimeError("Controller Torch CUDA RNG checkpoint state is topology-incompatible.")
    if any(
        not isinstance(item, torch.Tensor)
        or item.device.type != "cpu"
        or item.dtype != torch.uint8
        or item.ndim != 1
        or item.numel() == 0
        for item in torch_cuda_states
    ):
        raise RuntimeError("Controller Torch CUDA RNG checkpoint state is corrupt.")

    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_cpu_state)
    if cuda_available:
        torch.cuda.set_rng_state_all(torch_cuda_states)


def compute_monitor_old_ref_kl(data: DataProto) -> tuple[float, float, int]:
    """Return strict valid-response-token KL mean, sum and count on the driver."""
    response_mask = data.batch["response_mask"]
    kld = core_algos.compute_stable_low_var_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"])
    if kld.shape != response_mask.shape:
        raise ValueError("monitor_old_ref KL and response_mask shapes differ.")
    # kld is a fresh diagnostic tensor. Mask it in place to avoid materializing
    # two additional BxL float buffers on long-horizon batches.
    kld.mul_(response_mask)
    valid_sum = kld.sum(dtype=torch.float64)
    valid_count = int(response_mask.sum().item())
    if valid_count <= 0 or not torch.isfinite(valid_sum):
        raise ValueError(f"Invalid monitor_old_ref KL aggregate: sum={valid_sum.item()}, count={valid_count}.")
    return float((valid_sum / valid_count).item()), float(valid_sum.item()), valid_count


def strict_optimizer_counter_metrics(metrics: Dict[str, Any]) -> tuple[int, int, int]:
    """Read worker-agreed counters without lossy mean-and-round coercion."""
    agreement = metrics.get("actor/optimizer_counter_agreement", 0.0)
    if (
        isinstance(agreement, (bool, str))
        or not isinstance(agreement, (int, float, np.integer, np.floating))
        or not np.isfinite(float(agreement))
        or float(agreement) != 1.0
    ):
        raise RuntimeError("Actor workers did not attest optimizer counter agreement.")

    def _read(name: str) -> int:
        raw = metrics.get(name)
        if isinstance(raw, (bool, str)) or not isinstance(raw, (int, float, np.integer, np.floating)):
            raise RuntimeError(f"Missing or invalid actor counter {name}: {raw!r}.")
        value = float(raw)
        if not np.isfinite(value) or value < 0 or not value.is_integer():
            raise RuntimeError(f"Actor counter {name} must be an exact nonnegative integer, got {value!r}.")
        return int(value)

    attempted = _read("actor/optimizer_steps_attempted")
    executed = _read("actor/optimizer_steps_executed")
    skipped = _read("actor/optimizer_steps_skipped")
    if executed + skipped != attempted:
        raise RuntimeError(
            f"Invalid actor optimizer-step status: attempted={attempted}, executed={executed}, skipped={skipped}."
        )
    return attempted, executed, skipped


def build_action_step_tensors(
    event_rows: list[Any],
    values: list[list[float]],
    response_mask: torch.Tensor,
    response_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, response_length = response_mask.shape
    if tuple(response_ids.shape) != (bsz, response_length):
        raise ValueError("Response IDs must align exactly with the response mask.")
    if not (len(event_rows) == len(values) == bsz):
        raise ValueError("Action event rows must align exactly with the response batch.")
    def _event_count(row: Any) -> int:
        if isinstance(row, dict) and isinstance(row.get("events"), (list, tuple)):
            return len(row["events"])
        return len(row) if isinstance(row, (list, tuple)) else 0

    max_events = max((_event_count(row) for row in event_rows), default=0)
    spans = torch.zeros((bsz, max_events, 2), dtype=torch.long, device=response_mask.device)
    event_values = torch.zeros((bsz, max_events), dtype=torch.float32, device=response_mask.device)
    event_types = torch.zeros((bsz, max_events), dtype=torch.long, device=response_mask.device)
    for row_idx, (row_events, row_values) in enumerate(zip(event_rows, values)):
        # Invalid ledger rows were already failed closed by the reward worker.
        if not row_events:
            if row_values:
                raise ValueError(f"Invalid action row {row_idx} cannot carry values.")
            continue
        valid_length = int(response_mask[row_idx].sum().item())
        validated = validate_action_event_row(
            row_events,
            valid_length,
            response_token_ids=response_ids[row_idx, :valid_length].tolist(),
            require_token_binding=True,
        )
        if len(validated) != len(row_values):
            raise ValueError(f"Action event/value mismatch on row {row_idx}.")
        for event_idx, (event, value) in enumerate(zip(validated, row_values)):
            start, end = event["decision_start"], event["decision_end"]
            event_type = event["type"]
            if event_type != "point" and float(value) != 0.0:
                raise ValueError(f"{event_type} cannot carry native local action credit.")
            spans[row_idx, event_idx] = torch.tensor((start, end), device=response_mask.device)
            event_values[row_idx, event_idx] = float(value)
            event_types[row_idx, event_idx] = ACTION_TYPE_IDS[event_type]
    return spans, event_values, event_types


def _parse_stepcount_gt_answer(ground_truth: Any) -> Optional[int]:
    """Extract StepCount GT answer from numeric or JSON ground_truth."""
    if ground_truth is None:
        return None
    if isinstance(ground_truth, (int, np.integer)):
        return int(ground_truth)
    if isinstance(ground_truth, (float, np.floating)):
        return int(ground_truth) if float(ground_truth).is_integer() else None

    text = str(ground_truth).strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("count_number", "final_count", "answer", "N", "total_count"):
                value = payload.get(key)
                if value is not None:
                    parsed = _parse_stepcount_gt_answer(value)
                    if parsed is not None:
                        return parsed
            for key in ("point_gts", "point_trajectory", "trajectory_points", "points"):
                value = payload.get(key)
                if isinstance(value, list) and value:
                    return len(value)
        elif isinstance(payload, (int, float, str)):
            parsed = _parse_stepcount_gt_answer(payload)
            if parsed is not None:
                return parsed
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    match = re.search(r"-?\d+", text)
    if match:
        return int(match.group(0))
    return None


def _build_adaptive_max_turns(ground_truth_values: Any, global_max_turns: int, margin: int) -> np.ndarray:
    """Build per-sample cap: min(global cap, GT answer + answer turn + margin)."""
    global_cap = max(1, int(global_max_turns))
    extra_margin = max(0, int(margin))
    caps: List[int] = []
    for gt in ground_truth_values:
        gt_answer = _parse_stepcount_gt_answer(gt)
        if gt_answer is None or gt_answer < 0:
            caps.append(global_cap)
        else:
            caps.append(max(1, min(global_cap, int(gt_answer) + 1 + extra_margin)))
    return np.asarray(caps, dtype=np.int32)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"
    DRGRPO = "drgrpo"
    GRPO_STEP = "grpo_step"
    BOK_GRPO = "bok_grpo"
    BOK_GRPO_STEP = "bok_grpo_step"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    kl_coef_before = kl_ctrl.kl_coef
    data.batch["token_level_rewards"] = token_level_scores - kl_coef_before * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    kl_penalty_mean = current_kl * kl_coef_before

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    metrics = {
        "critic/kl": current_kl,
        "critic/kl_coef": kl_coef_before,
        "critic/kl_penalty": kl_penalty_mean,
        "adaptive_kl/kl_loss": current_kl,
        "adaptive_kl/kl_coef": kl_coef_before,
        "adaptive_kl/kl_coef_next": kl_ctrl.kl_coef,
        "adaptive_kl/kl_penalty": kl_penalty_mean,
        # Alias for dashboards that historically watched actor/kl_loss. When
        # algorithm.use_kl_loss=false, this is the reward-side equivalent KL.
        "actor/kl_loss_equiv": current_kl,
        "actor/kl_coef_equiv": kl_coef_before,
    }
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.DRGRPO:
        advantages, returns = core_algos.compute_drgrpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.GRPO_STEP:
        advantages, returns = core_algos.compute_grpo_step_level_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.BOK_GRPO:
        # Read BoK-GRPO hyperparameters from environment variables
        import os
        bok_tau = float(os.environ.get("BOK_TAU", "0.3"))
        bok_clip = float(os.environ.get("BOK_CLIP", "3.0"))
        bok_uniform_mix = float(os.environ.get("BOK_UNIFORM_MIX", "0.1"))
        # τ annealing: optionally anneal tau from init->final over training
        bok_tau_init = float(os.environ.get("BOK_TAU_INIT", "0"))
        bok_tau_final = float(os.environ.get("BOK_TAU_FINAL", "0"))
        # Read global_step / total_steps from meta_info if available
        _gs = int(data.meta_info.get("global_step", 0)) if hasattr(data, "meta_info") and data.meta_info else 0
        # total_steps: prefer meta_info (auto-computed), fallback to env var
        _ts_meta = int(data.meta_info.get("total_steps", 0)) if hasattr(data, "meta_info") and data.meta_info else 0
        _ts = _ts_meta if _ts_meta > 1 else int(os.environ.get("BOK_TOTAL_STEPS", "1"))
        # Prefer independent binary correctness for routing; retain shaped
        # answer_scores for legacy callers and step-level soft gating.
        _answer_scores = data.batch.get("answer_scores", None)
        _answer_correct_scores = data.batch.get("answer_correct_scores", None)
        _raw_success_scores = data.batch.get("raw_success_scores", None)
        _trajectory_quality_scores = data.batch.get("trajectory_quality_scores", None)
        advantages, returns = core_algos.compute_bok_grpo_advantage(
            token_level_rewards, response_mask, index,
            bok_tau=bok_tau, bok_clip=bok_clip, bok_uniform_mix=bok_uniform_mix,
            bok_tau_init=bok_tau_init, bok_tau_final=bok_tau_final,
            global_step=_gs, total_steps=_ts,
            answer_scores=_answer_scores,
            answer_correct_scores=_answer_correct_scores,
            raw_success_scores=_raw_success_scores,
            trajectory_quality_scores=_trajectory_quality_scores,
        )
    elif adv_estimator == AdvantageEstimator.BOK_GRPO_STEP:
        import os
        bok_tau = float(os.environ.get("BOK_TAU", "0.3"))
        bok_clip = float(os.environ.get("BOK_CLIP", "3.0"))
        bok_uniform_mix = float(os.environ.get("BOK_UNIFORM_MIX", "0.1"))
        bok_tau_init = float(os.environ.get("BOK_TAU_INIT", "0"))
        bok_tau_final = float(os.environ.get("BOK_TAU_FINAL", "0"))
        _gs = int(data.meta_info.get("global_step", 0)) if hasattr(data, "meta_info") and data.meta_info else 0
        _ts_meta = int(data.meta_info.get("total_steps", 0)) if hasattr(data, "meta_info") and data.meta_info else 0
        _ts = _ts_meta if _ts_meta > 1 else int(os.environ.get("BOK_TOTAL_STEPS", "1"))
        _answer_scores = data.batch.get("answer_scores", None)
        _answer_correct_scores = data.batch.get("answer_correct_scores", None)
        _raw_success_scores = data.batch.get("raw_success_scores", None)
        _trajectory_quality_scores = data.batch.get("trajectory_quality_scores", None)
        _point_step_mask = data.batch.get("point_step_mask", None)
        _point_step_value = data.batch.get("point_step_value", None)
        _action_step_span = data.batch.get("action_step_span", None)
        _action_step_value = data.batch.get("action_step_value", None)
        _action_step_type = data.batch.get("action_step_type", None)
        advantages, returns = core_algos.compute_bok_grpo_step_advantage(
            token_level_rewards, response_mask, index,
            bok_tau=bok_tau, bok_clip=bok_clip, bok_uniform_mix=bok_uniform_mix,
            bok_tau_init=bok_tau_init, bok_tau_final=bok_tau_final,
            global_step=_gs, total_steps=_ts,
            answer_scores=_answer_scores,
            point_step_mask=_point_step_mask,
            point_step_value=_point_step_value,
            answer_correct_scores=_answer_correct_scores,
            raw_success_scores=_raw_success_scores,
            trajectory_quality_scores=_trajectory_quality_scores,
            action_step_span=_action_step_span,
            action_step_value=_action_step_value,
            action_step_type=_action_step_type,
        )
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: Any,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.adaptive_actor_kl = bool(config.algorithm.adaptive_actor_kl)
        self.effective_actor_updates = 0
        self._current_step_complete = False
        self._adaptive_ref_identity = None
        self._v37_oom_count_cumulative = 0

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")

        if self.adaptive_actor_kl:
            if not self.use_reference_policy:
                raise ValueError("adaptive_actor_kl requires an enabled reference policy.")
            if config.algorithm.use_kl_loss:
                raise ValueError("adaptive_actor_kl is independent of use_kl_loss; enable only one actor KL mode.")
            if config.algorithm.kl_type != "adaptive":
                raise ValueError("adaptive_actor_kl requires algorithm.kl_type=adaptive.")
            if config.algorithm.kl_penalty != "low_var_kl":
                raise ValueError("adaptive_actor_kl only supports sampled-token low_var_kl.")
            if os.environ.get("EASYR1_ALLOW_ZERO_MM_LOGPROB", "0").lower() in ("1", "true", "yes", "on"):
                raise ValueError("adaptive_actor_kl forbids EASYR1_ALLOW_ZERO_MM_LOGPROB.")
            # Hash once per trainer process. The reference policy is immutable
            # after construction, while hashing multi-shard weights per save is
            # prohibitively expensive.
            self._adaptive_ref_identity = reference_content_identity(config.worker.actor.model.model_path)

        action_event_reward = os.environ.get("ACTION_EVENT_REWARD_ENABLE", "0").lower() in (
            "1", "true", "yes"
        )
        action_event_ledger = os.environ.get("ACTION_EVENT_LEDGER_ENABLE", "0").lower() in (
            "1", "true", "yes"
        )
        if action_event_reward and not action_event_ledger:
            raise ValueError("ACTION_EVENT_REWARD_ENABLE requires ACTION_EVENT_LEDGER_ENABLE=1.")
        if action_event_ledger:
            if config.worker.reward.reward_type != "sequential":
                raise ValueError("ACTION_EVENT_LEDGER_ENABLE requires worker.reward.reward_type=sequential.")
            if not config.worker.rollout.interleaved_point_to_count:
                raise ValueError("ACTION_EVENT_LEDGER_ENABLE requires interleaved_point_to_count rollout.")
            if os.environ.get("V37_ACTION_PARSER_CONTRACT", "0").lower() not in ("1", "true", "yes"):
                raise ValueError("ACTION_EVENT_LEDGER_ENABLE requires V37_ACTION_PARSER_CONTRACT=1.")
            if os.environ.get("V37_ACTION_LEDGER_CONTRACT", "0").lower() not in ("1", "true", "yes"):
                raise ValueError("ACTION_EVENT_LEDGER_ENABLE requires V37_ACTION_LEDGER_CONTRACT=1.")
        if action_event_reward:
            if config.algorithm.adv_estimator != AdvantageEstimator.BOK_GRPO_STEP:
                raise ValueError("ACTION_EVENT_REWARD_ENABLE requires algorithm.adv_estimator=bok_grpo_step.")
            if int(os.environ.get("BOK_CORRECTNESS_FIRST", "0")) != 1:
                raise ValueError("ACTION_EVENT_REWARD_ENABLE requires BOK_CORRECTNESS_FIRST=1.")
            if os.environ.get("BOK_STEP_SIGNAL", "").strip().lower() != "native_action_event":
                raise ValueError("ACTION_EVENT_REWARD_ENABLE requires BOK_STEP_SIGNAL=native_action_event.")

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        # Export total steps so BoK-GRPO τ annealing can read it from env
        import os
        os.environ["BOK_TOTAL_STEPS"] = str(self.training_steps)
        print(f"Total training steps: {self.training_steps}")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        if isinstance(self.val_dataloader, dict):
            all_metrics: Dict[str, Any] = {}
            for suite_name, val_loader in self.val_dataloader.items():
                metric_prefix = f"val/{suite_name}"
                print(f"[ValSuite] step={getattr(self, 'global_step', -1)} suite={suite_name} begin")
                suite_metrics = self._validate_single(
                    val_loader,
                    metric_prefix=metric_prefix,
                    log_generations=False,
                    suite_name=suite_name,
                )
                all_metrics.update(suite_metrics)
                print(f"[ValSuite] step={getattr(self, 'global_step', -1)} suite={suite_name} end")
            return all_metrics

        return self._validate_single(
            self.val_dataloader,
            metric_prefix="val",
            log_generations=True,
            suite_name=None,
        )

    def _validate_single(
        self,
        val_dataloader: StatefulDataLoader,
        metric_prefix: str = "val",
        log_generations: bool = True,
        suite_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        skipped_samples = []  # Track skipped samples due to OOM
        
        for batch_dict in val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            batch_size = input_ids.size(0)

            question_key = None
            for candidate in ("problem", "prompt", "question", "query", "instruction"):
                if candidate in test_batch.non_tensor_batch:
                    question_key = candidate
                    break

            adaptive_max_turns = None
            rollout_cfg = self.config.worker.rollout
            if bool(getattr(rollout_cfg, "interleaved_adaptive_max_turns", False)):
                adaptive_max_turns = _build_adaptive_max_turns(
                    test_batch.non_tensor_batch.get("ground_truth", []),
                    getattr(rollout_cfg, "interleaved_max_turns", 1),
                    getattr(rollout_cfg, "interleaved_adaptive_max_turns_margin", 0),
                )

            if "multi_modal_data" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=[
                        "raw_prompt_ids",
                        "multi_modal_data",
                        *([question_key] if question_key else []),
                    ],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", *([question_key] if question_key else [])],
                )
            if adaptive_max_turns is not None:
                test_gen_batch.non_tensor_batch["adaptive_max_turns"] = adaptive_max_turns

            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info.update({
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "data_format_prompt": self.config.data.format_prompt,
                "global_step": self.global_step,
                "total_steps": self.training_steps,
            })
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            
            # Check GPU memory before batch processing
            exceeds_threshold, usage_ratio = self._check_gpu_memory_usage(threshold=0.99)
            used_per_sample = False
            successful_indices = None
            
            if exceeds_threshold:
                # Formal evidence treats a proactive high-memory fallback as an
                # avoided OOM event. Otherwise a run could silently switch to
                # per-sample validation while still reporting oom_count=0.
                self._v37_oom_count_cumulative += 1
                print(f"Warning: GPU memory usage ({usage_ratio*100:.1f}%) is high before batch processing. "
                      f"Falling back to per-sample processing to avoid OOM...")
                # Try to free some memory
                try:
                    torch.cuda.empty_cache()
                    time.sleep(1)
                except:
                    pass
                # Skip directly to per-sample processing
                used_per_sample = True
                test_output_gen_batch, successful_indices = self._validate_per_sample(
                    test_batch, test_gen_batch, pad_size, skipped_samples
                )
                # Filter test_batch to only include successful samples
                if successful_indices:
                    test_batch = DataProto.concat([test_batch[i:i+1] for i in successful_indices])
                    input_texts = [input_texts[i] for i in successful_indices]
                else:
                    print("Warning: No successful samples in this batch. Skipping.")
                    continue
            else:
                # Try to process the batch, fallback to per-sample processing on OOM
                test_output_gen_batch = None
                max_retries = 2
                retry_delay = 3  # seconds
                
                for retry in range(max_retries):
                    try:
                        test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
                        break
                    except (ActorDiedError, RayTaskError) as e:
                        # Check if it's an ActorDiedError (either directly or wrapped in RayTaskError)
                        is_actor_died = isinstance(e, ActorDiedError)
                        if isinstance(e, RayTaskError):
                            try:
                                cause = e.as_instanceof_cause()
                                is_actor_died = isinstance(cause, ActorDiedError)
                            except:
                                pass
                        
                        if is_actor_died:
                            self._v37_oom_count_cumulative += 1
                            if retry < max_retries - 1:
                                print(f"Warning: Actor died during batch validation (likely OOM). Retrying ({retry + 1}/{max_retries}) after {retry_delay}s...")
                                time.sleep(retry_delay)
                                # Reinitialize workers if needed
                                try:
                                    self.actor_rollout_wg.init_model()
                                except Exception as init_error:
                                    print(f"Warning: Failed to reinitialize workers: {init_error}")
                                continue
                            else:
                                # Fallback to per-sample processing
                                print(f"Warning: Batch processing failed after {max_retries} retries. Falling back to per-sample processing to skip OOM samples...")
                                used_per_sample = True
                                test_output_gen_batch, successful_indices = self._validate_per_sample(
                                    test_batch, test_gen_batch, pad_size, skipped_samples
                                )
                                # Filter test_batch to only include successful samples
                                if successful_indices:
                                    test_batch = DataProto.concat([test_batch[i:i+1] for i in successful_indices])
                                    input_texts = [input_texts[i] for i in successful_indices]
                                break
                        else:
                            raise
            
            if test_output_gen_batch is None or len(test_output_gen_batch) == 0:
                print("Error: Failed to generate sequences even with per-sample fallback. Skipping this batch.")
                continue
            
            # Unpad if needed (per-sample processing already handled padding)
            if not used_per_sample and pad_size > 0:
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size)
            
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info = dict(getattr(test_batch, "meta_info", {}) or {})
            test_batch.meta_info["is_validation"] = True

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                if str(key).startswith("_"):
                    continue
                reward_metrics_lst[key].extend(value)
        
        if skipped_samples:
            print(f"Warning: Skipped {len(skipped_samples)} samples due to OOM during validation.")
        
        if log_generations:
            self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        reduced = reduce_metrics(reward_metrics_lst)
        # Helpful diagnostics: task mix in validation (requires reward fn to emit is_*_task metrics).
        try:
            point_rate = float(reduced.get("is_point_task", 0.0))
            count_rate = float(reduced.get("is_count_task", 0.0))
            traj_rate = float(reduced.get("is_trajectory_task", 0.0))
            print(
                "[ValTaskMix] "
                f"suite={suite_name or 'default'} samples={len(sample_scores)} "
                f"point_rate={point_rate:.3f} count_rate={count_rate:.3f} traj_rate={traj_rate:.3f}"
            )
        except Exception:
            pass

        val_reward_metrics = {f"{metric_prefix}/{key}_reward": value for key, value in reduced.items()}
        # Structured single-line validation summary for automated parsing
        _val_result = {f"{metric_prefix}/reward_score": reward_score, **val_reward_metrics}
        _val_parts = " ".join(f"{k}={v:.4f}" for k, v in sorted(_val_result.items()) if isinstance(v, (int, float)))
        print(f"[ValSummary] step={getattr(self, 'global_step', -1)} suite={suite_name or 'default'} {_val_parts}")
        return _val_result
    
    def _check_gpu_memory_usage(self, threshold: float = 0.95) -> Tuple[bool, float]:
        """Check if GPU memory usage exceeds threshold.
        
        Args:
            threshold: Memory usage threshold (0.95 = 95%)
            
        Returns:
            tuple: (exceeds_threshold, current_usage_ratio)
        """
        if not torch.cuda.is_available():
            return False, 0.0
        
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            used_mem = total_mem - free_mem
            usage_ratio = used_mem / total_mem if total_mem > 0 else 0.0
            exceeds = usage_ratio >= threshold
            return exceeds, usage_ratio
        except Exception as e:
            # If we can't check memory, assume it's safe
            print(f"Warning: Could not check GPU memory: {e}")
            return False, 0.0
    
    def _validate_per_sample(
        self, test_batch: DataProto, test_gen_batch: DataProto, pad_size: int, skipped_samples: List[int]
    ) -> Tuple[DataProto, List[int]]:
        """Process validation samples one by one, skipping those that cause OOM.
        
        Returns:
            tuple: (successful_outputs, successful_indices)
        """
        from ..protocol import DataProto
        
        batch_size = len(test_gen_batch)
        successful_outputs = []
        successful_indices = []
        memory_threshold = 0.99  # Skip if memory usage >= 99%
        
        print(f"Processing {batch_size} samples individually to skip OOM samples...")
        
        for i in range(batch_size):
            # Check GPU memory before processing
            exceeds_threshold, usage_ratio = self._check_gpu_memory_usage(memory_threshold)
            if exceeds_threshold:
                prompt_len = len(test_gen_batch[i].batch['input_ids'][0]) if len(test_gen_batch[i].batch['input_ids']) > 0 else 0
                print(f"Warning: Skipping sample {i} - GPU memory usage ({usage_ratio*100:.1f}%) exceeds threshold ({memory_threshold*100:.1f}%) (prompt length: {prompt_len} tokens)")
                skipped_samples.append(i)
                self._v37_oom_count_cumulative += 1
                # Try to free some memory
                try:
                    torch.cuda.empty_cache()
                    time.sleep(0.5)  # Brief pause to allow memory cleanup
                except:
                    pass
                continue
            
            try:
                # Extract single sample
                single_gen_batch = test_gen_batch[i:i+1]
                single_gen_batch.meta_info = test_gen_batch.meta_info
                
                # Pad to divisor if needed
                single_gen_batch, single_pad_size = pad_dataproto_to_divisor(
                    single_gen_batch, self.actor_rollout_wg.world_size
                )
                
                # Try to generate
                single_output = self.actor_rollout_wg.generate_sequences(single_gen_batch)
                single_output = unpad_dataproto(single_output, single_pad_size)
                
                successful_outputs.append(single_output)
                successful_indices.append(i)
                
                # Clear cache after successful generation
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
                
            except (ActorDiedError, RayTaskError) as e:
                # Check if it's an ActorDiedError
                is_actor_died = isinstance(e, ActorDiedError)
                if isinstance(e, RayTaskError):
                    try:
                        cause = e.as_instanceof_cause()
                        is_actor_died = isinstance(cause, ActorDiedError)
                    except:
                        pass
                
                if is_actor_died:
                    self._v37_oom_count_cumulative += 1
                    try:
                        ids = test_gen_batch[i].batch['input_ids']
                        if ids.dim() == 0:
                            prompt_len = 0
                        elif ids.dim() == 1:
                            prompt_len = ids.shape[0]
                        else:
                            prompt_len = ids.shape[1] if ids.shape[0] > 0 else 0
                    except (TypeError, IndexError, AttributeError):
                        prompt_len = 0
                    # Check memory usage for logging
                    _, usage_ratio = self._check_gpu_memory_usage()
                    print(f"Warning: Skipping sample {i} due to OOM (GPU memory: {usage_ratio*100:.1f}%, prompt length: {prompt_len} tokens)")
                    skipped_samples.append(i)
                    # Reinitialize workers after OOM
                    try:
                        torch.cuda.empty_cache()
                        time.sleep(2)  # Brief delay before reinitializing
                        self.actor_rollout_wg.init_model()
                    except Exception as init_error:
                        print(f"Warning: Failed to reinitialize workers after skipping sample {i}: {init_error}")
                else:
                    # For non-OOM errors, re-raise
                    print(f"Error processing sample {i}: {e}")
                    raise
        
        if not successful_outputs:
            print("Error: All samples failed. Returning empty DataProto.")
            return test_gen_batch[:0], []  # Return empty DataProto with same structure
        
        # Concatenate successful outputs
        result = DataProto.concat(successful_outputs)
        print(f"Successfully processed {len(successful_outputs)}/{batch_size} samples.")
        
        return result, successful_indices

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> bool:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        sealed_checkpoint = (
            self.adaptive_actor_kl
            or os.environ.get("V37_TRAINING_EVIDENCE_REQUIRED", "0") == "1"
            or os.environ.get("V37_FINALIZE_HF_CHECKPOINT", "0") == "1"
        )
        if not sealed_checkpoint:
            # Preserve the historical V36 layout, integer tracker and retention
            # order when every V37/adaptive checkpoint feature is disabled.
            remove_obsolete_ckpt(
                self.config.trainer.save_checkpoint_path,
                self.global_step,
                best_global_step=-1,
                save_limit=self.config.trainer.save_limit,
            )
            folder_path = os.path.join(
                self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}"
            )
            self.actor_rollout_wg.save_checkpoint(os.path.join(folder_path, "actor"))
            if self.use_critic:
                self.critic_wg.save_checkpoint(os.path.join(folder_path, "critic"))
            torch.save(
                self.train_dataloader.state_dict(), os.path.join(folder_path, "dataloader.pt")
            )
            with open(
                os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write(str(self.global_step))
            return True
        if sealed_checkpoint and not self._current_step_complete:
            logger.warning(
                "Skipping sealed checkpoint at global_step=%s because the optimizer step is incomplete; "
                "the last complete checkpoint remains authoritative.",
                self.global_step,
            )
            return False
        save_root = self.config.trainer.save_checkpoint_path
        os.makedirs(save_root, exist_ok=True)
        folder_path = os.path.join(save_root, f"global_step_{self.global_step}")
        required_paths = ["actor", "dataloader.pt"]
        if self.use_critic:
            required_paths.append("critic")
        if self.adaptive_actor_kl:
            required_paths.append(TRAINER_RUNTIME_STATE_FILE)
        finalize_hf_requested = os.environ.get("V37_FINALIZE_HF_CHECKPOINT", "0") == "1"
        if finalize_hf_requested and os.environ.get("V37_RUN_CLASS") != "formal":
            raise RuntimeError(
                "V37_FINALIZE_HF_CHECKPOINT is reserved for formal training."
            )
        # Formal runs may still save resumable shard checkpoints before the
        # terminal step (for example step 10 of a 12-step run).  Only the
        # terminal checkpoint pays the CPU merge cost and becomes evaluable.
        finalize_hf = finalize_hf_requested and self.global_step == self.training_steps
        if finalize_hf:
            required_paths.append("actor/huggingface")

        # A sealed same-step directory is already complete. Re-publish only its
        # tracker entry; never rewrite distributed shards in place.
        if os.path.exists(folder_path):
            manifest = validate_checkpoint_manifest(folder_path, expected_step=self.global_step)
            if not set(required_paths).issubset(manifest["required_paths"]):
                raise RuntimeError("Existing same-step checkpoint does not satisfy this trainer's artifact contract.")
            tracker_payload = {
                "last_global_step": int(self.global_step),
                "manifest_sha256": checkpoint_manifest_hash(folder_path),
            }
            _atomic_write_json(os.path.join(save_root, CHECKPOINT_TRACKER), tracker_payload)
            remove_obsolete_ckpt(
                save_root, self.global_step, best_global_step=-1, save_limit=self.config.trainer.save_limit
            )
            return True

        staging_path = os.path.join(save_root, f".global_step_{self.global_step}.staging-{uuid.uuid4().hex}")
        os.makedirs(staging_path, exist_ok=False)
        try:
            self.actor_rollout_wg.save_checkpoint(os.path.join(staging_path, "actor"))
            if finalize_hf:
                merger = Path(__file__).resolve().parents[2] / "scripts" / "model_merger.py"
                if not merger.is_file():
                    raise RuntimeError(f"V37 checkpoint merger is missing: {merger}")
                merge_environment = dict(os.environ)
                merge_environment.update({
                    "HF_DATASETS_OFFLINE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "WANDB_MODE": "offline",
                    "PYTHONUNBUFFERED": "1",
                })
                merge_memory_policy = os.environ.get(
                    "V37_HF_MERGE_HOST_MEMORY_PREFLIGHT", "off"
                )
                if merge_memory_policy not in {"off", "warn", "error"}:
                    raise RuntimeError(
                        "V37_HF_MERGE_HOST_MEMORY_PREFLIGHT must be off, warn, or error"
                    )
                merge_memory_factor = os.environ.get(
                    "V37_HF_MERGE_HOST_MEMORY_SAFETY_FACTOR", "2.0"
                )
                try:
                    parsed_memory_factor = float(merge_memory_factor)
                except ValueError as exc:
                    raise RuntimeError(
                        "V37_HF_MERGE_HOST_MEMORY_SAFETY_FACTOR must be numeric"
                    ) from exc
                if not np.isfinite(parsed_memory_factor) or parsed_memory_factor < 1.0:
                    raise RuntimeError(
                        "V37_HF_MERGE_HOST_MEMORY_SAFETY_FACTOR must be finite and >= 1.0"
                    )
                merge_command = [
                    sys.executable,
                    str(merger),
                    "--local_dir",
                    os.path.join(staging_path, "actor"),
                    "--host-memory-preflight",
                    merge_memory_policy,
                    "--host-memory-safety-factor",
                    merge_memory_factor,
                ]
                subprocess.run(
                    merge_command,
                    cwd=str(Path(__file__).resolve().parents[2]),
                    env=merge_environment,
                    check=True,
                )
                hf_path = Path(staging_path) / "actor" / "huggingface"
                weight_files = sorted(hf_path.glob("*.safetensors"))
                if not weight_files or any(not item.is_file() or item.is_symlink() for item in weight_files):
                    raise RuntimeError("V37 formal checkpoint merger produced no regular safetensors weights.")
            if self.use_critic:
                self.critic_wg.save_checkpoint(os.path.join(staging_path, "critic"))

            torch.save(self.train_dataloader.state_dict(), os.path.join(staging_path, "dataloader.pt"))

            if self.adaptive_actor_kl:
                trainer_runtime_state = {
                    "version": TRAINER_RUNTIME_STATE_VERSION,
                    "controller": self.kl_ctrl.state_dict(),
                    "horizon_unit": ADAPTIVE_ACTOR_KL_HORIZON_UNIT,
                    "effective_updates": int(self.effective_actor_updates),
                    "ref_identity": self._adaptive_ref_identity,
                    "global_step": int(self.global_step),
                    "step_complete": bool(self._current_step_complete),
                    "worker_world_size": int(self.actor_rollout_wg.world_size),
                    "controller_rng": capture_controller_rng_state(),
                }
                torch.save(
                    trainer_runtime_state,
                    os.path.join(staging_path, TRAINER_RUNTIME_STATE_FILE),
                )

            write_checkpoint_manifest(staging_path, self.global_step, required_paths)
            publish_staged_checkpoint(staging_path, folder_path, self.global_step)

            # Publication order is intentional: checkpoint, then tracker, then
            # retention. Pre-publication failures preserve the previous tracker;
            # retention failures leave the newly tracked checkpoint authoritative.
            tracker_payload = {
                "last_global_step": int(self.global_step),
                "manifest_sha256": checkpoint_manifest_hash(folder_path),
            }
            _atomic_write_json(os.path.join(save_root, CHECKPOINT_TRACKER), tracker_payload)
            remove_obsolete_ckpt(
                save_root, self.global_step, best_global_step=-1, save_limit=self.config.trainer.save_limit
            )
            return True
        finally:
            if os.path.exists(staging_path):
                shutil.rmtree(staging_path, ignore_errors=True)

    def _load_checkpoint(self) -> bool:
        if self.config.trainer.load_checkpoint_path is None:
            return False

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        validate_resume_checkpoint_binding(
            self.config.trainer.load_checkpoint_path,
            config_binding={
                "V37_RUN_CLASS": self.config.trainer.v37_run_class,
                "V37_RESUME_MODE": self.config.trainer.v37_resume_mode,
                "V37_EXPECTED_RESUME_CHECKPOINT_PATH": (
                    self.config.trainer.v37_expected_resume_checkpoint_path
                ),
                "V37_EXPECTED_RESUME_CHECKPOINT_SHA256": (
                    self.config.trainer.v37_expected_resume_checkpoint_sha256
                ),
            },
        )

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        trainer_runtime_state = None
        if self.adaptive_actor_kl:
            manifest = validate_checkpoint_manifest(
                self.config.trainer.load_checkpoint_path, expected_step=self.global_step
            )
            if TRAINER_RUNTIME_STATE_FILE not in manifest["required_paths"]:
                raise RuntimeError(
                    f"adaptive_actor_kl manifest does not require {TRAINER_RUNTIME_STATE_FILE}."
                )
            trainer_state_path = os.path.join(
                self.config.trainer.load_checkpoint_path, TRAINER_RUNTIME_STATE_FILE
            )
            try:
                trainer_runtime_state = torch.load(trainer_state_path, weights_only=False, map_location="cpu")
            except Exception as exc:
                raise RuntimeError(
                    f"adaptive_actor_kl trainer runtime artifact is corrupt: {trainer_state_path}."
                ) from exc
            if (
                not isinstance(trainer_runtime_state, dict)
                or trainer_runtime_state.get("version") != TRAINER_RUNTIME_STATE_VERSION
            ):
                raise RuntimeError("adaptive_actor_kl trainer runtime artifact has an unsupported contract.")
            if trainer_runtime_state.get("horizon_unit") != ADAPTIVE_ACTOR_KL_HORIZON_UNIT:
                raise RuntimeError(
                    "adaptive_actor_kl horizon_unit must be executed_optimizer_updates."
                )
            if (
                trainer_runtime_state.get("global_step") != self.global_step
                or trainer_runtime_state.get("step_complete") is not True
            ):
                raise RuntimeError("adaptive_actor_kl refuses incomplete or mismatched trainer runtime state.")
            saved_world_size = trainer_runtime_state.get("worker_world_size")
            if (
                isinstance(saved_world_size, bool)
                or not isinstance(saved_world_size, int)
                or saved_world_size != int(self.actor_rollout_wg.world_size)
            ):
                raise RuntimeError("adaptive_actor_kl worker topology differs from the trainer runtime artifact.")
            expected_ref = self._adaptive_ref_identity
            if trainer_runtime_state.get("ref_identity") != expected_ref:
                raise RuntimeError("adaptive_actor_kl reference identity differs from the checkpoint.")
            expected = self.kl_ctrl.state_dict()
            controller_state = trainer_runtime_state.get("controller")
            if not isinstance(controller_state, dict) or controller_state.get("type") != expected.get("type"):
                raise RuntimeError("adaptive_actor_kl controller type differs from the checkpoint.")
            for key in ("target", "horizon"):
                try:
                    matches = float(controller_state.get(key)) == float(expected.get(key))
                except (TypeError, ValueError, OverflowError):
                    matches = False
                if not matches:
                    raise RuntimeError(f"adaptive_actor_kl {key} differs from the checkpoint.")
            effective_updates = trainer_runtime_state.get("effective_updates")
            if isinstance(effective_updates, bool) or not isinstance(effective_updates, int) or effective_updates < 0:
                raise RuntimeError("adaptive_actor_kl effective_updates must be an exact nonnegative integer.")
            # Require the native RNG payload here; full validation/restoration
            # happens after every other load operation.
            rng_state = trainer_runtime_state.get("controller_rng")
            if not isinstance(rng_state, dict):
                raise RuntimeError("adaptive_actor_kl trainer runtime artifact is missing controller RNG state.")

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if not os.path.exists(dataloader_path):
            if self.adaptive_actor_kl:
                raise RuntimeError(f"adaptive_actor_kl resume requires {dataloader_path}.")
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")
            dataloader_state_dict = None
        else:
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)

        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        if dataloader_state_dict is not None:
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        if trainer_runtime_state is not None:
            self.kl_ctrl.load_state_dict(trainer_runtime_state["controller"])
            self.effective_actor_updates = trainer_runtime_state["effective_updates"]
            self._current_step_complete = bool(trainer_runtime_state["step_complete"])
            # Restore last: worker/dataloader deserialization is allowed to use
            # controller RNG internally, but validation/training must observe
            # the exact stream captured at checkpoint publication.
            restore_controller_rng_state(trainer_runtime_state["controller_rng"])
        return True

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _finalize_training(
        self,
        val_metrics: Optional[Dict[str, Any]],
        last_validation_step: Optional[int],
        last_checkpoint_step: Optional[int],
        v37_evidence: Optional[TrainingEvidenceRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        """Preserve legacy ordering unless exact adaptive continuation is enabled."""
        needs_terminal_validation = (
            self.val_reward_fn is not None and last_validation_step != self.global_step
        )
        exact_continuation = bool(getattr(self, "adaptive_actor_kl", False))
        if exact_continuation and needs_terminal_validation and last_checkpoint_step != self.global_step:
            if self._save_checkpoint():
                last_checkpoint_step = self.global_step

        if needs_terminal_validation:
            val_metrics = self._validate()
            if v37_evidence is not None:
                v37_evidence.update_latest_oom_count(
                    self._v37_oom_count_cumulative
                )
            self.logger.log(data=val_metrics, step=self.global_step)
            last_validation_step = self.global_step

        if self.val_reward_fn is not None and val_metrics is not None:
            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        # Feature-off V36 and normal periodic validation preserve the established
        # validation-then-checkpoint ordering.
        if last_checkpoint_step != self.global_step:
            self._save_checkpoint()

        return val_metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        # Setup logging - save debug logs to shared CephFS (accessible from all nodes & dev machine)
        debug_log_dir = os.environ.get(
            'TRAINING_DEBUG_LOG_DIR',
            os.path.join(os.getcwd(), 'logs', 'debug')
        )
        os.makedirs(debug_log_dir, exist_ok=True)
        log_file = os.path.join(debug_log_dir, f'training_debug_{int(time.time())}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        logger = logging.getLogger('training_debug')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        # Don't propagate to root logger to avoid console spam
        logger.propagate = False
        
        logger.info("=" * 80)
        logger.info("Starting training loop")
        logger.info(f"Total epochs: {self.config.trainer.total_epochs}, Training steps: {self.training_steps}")
        logger.info("=" * 80)
        
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        v37_evidence = TrainingEvidenceRecorder.from_environment()
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None
        last_validation_step: Optional[int] = None
        last_checkpoint_step: Optional[int] = None

        # load checkpoint before doing anything
        logger.info("Loading checkpoint...")
        try:
            resumed_from_checkpoint = self._load_checkpoint()
            logger.info(f"Checkpoint loaded. Starting from global_step: {self.global_step}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}\n{traceback.format_exc()}")
            raise

        # perform validation before training
        # currently, we only support validation using the reward_function.
        skip_resume_validation = self.adaptive_actor_kl and resumed_from_checkpoint and not self.config.trainer.val_only
        if skip_resume_validation:
            logger.info("Skipping validation-before-train on adaptive resume to preserve the restored RNG stream.")
        if self.val_reward_fn is not None and self.config.trainer.val_before_train and not skip_resume_validation:
            logger.info("Running validation before training...")
            try:
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)
                last_validation_step = self.global_step
                logger.info(f"Validation completed. Metrics: {val_metrics}")
            except Exception as e:
                logger.error(f"Error during validation: {e}\n{traceback.format_exc()}")
                raise
            if self.config.trainer.val_only:
                logger.info("Validation only mode. Exiting.")
                return

        logger.info("Starting training epochs...")
        for epoch_idx in range(self.config.trainer.total_epochs):
            if getattr(self, '_early_stopped', False):
                break
            # Check before constructing/advancing the dataloader iterator. A
            # post-fetch max_steps check consumes one extra batch and stores an
            # incorrect dataloader cursor in the final checkpoint.
            if self.global_step >= self.training_steps:
                logger.info(f"Reached training steps limit ({self.training_steps}). Stopping.")
                break
            if epoch_idx == 0 or (epoch_idx + 1) % 10 == 0:  # Log every 10 epochs
                logger.info(f"Starting epoch {epoch_idx + 1}/{self.config.trainer.total_epochs}")
            try:
                for batch_idx, batch_dict in enumerate(self.train_dataloader):
                    if self.global_step >= self.training_steps:
                        logger.info(f"Reached training steps limit ({self.training_steps}). Stopping.")
                        break
                    self.global_step += 1
                    self._current_step_complete = False
                    monitor_old_ref_kl = None
                    beta_before = None
                    v37_reward_summary = None
                    v37_action_rows_expected = 0
                    v37_action_rows_mapped = 0
                    v37_frontier_groups = 0
                    v37_frontier_mixed_groups = 0
                    v37_allwrong_groups = 0
                    v37_sign_errors = 0
                    v37_optimizer_attempted = None
                    v37_optimizer_executed = None
                    v37_optimizer_skipped = None
                    v37_nonfinite_cumulative = None
                    if getattr(self, '_early_stopped', False):
                        break

                    metrics, timing_raw = {}, {}
                    try:
                        batch: DataProto = DataProto.from_single_dict(batch_dict)
                    except Exception as e:
                        logger.error(f"[Step {self.global_step}] Error creating batch: {e}\n{traceback.format_exc()}")
                        raise

                    question_key = None
                    for candidate in ("problem", "prompt", "question", "query", "instruction"):
                        if candidate in batch.non_tensor_batch:
                            question_key = candidate
                            break

                    adaptive_max_turns = None
                    rollout_cfg = self.config.worker.rollout
                    if bool(getattr(rollout_cfg, "interleaved_adaptive_max_turns", False)):
                        adaptive_max_turns = _build_adaptive_max_turns(
                            batch.non_tensor_batch.get("ground_truth", []),
                            getattr(rollout_cfg, "interleaved_max_turns", 1),
                            getattr(rollout_cfg, "interleaved_adaptive_max_turns_margin", 0),
                        )

                    # pop those keys for generation
                    try:
                        if "multi_modal_data" in batch.non_tensor_batch.keys():
                            gen_batch = batch.pop(
                                batch_keys=["input_ids", "attention_mask", "position_ids"],
                                non_tensor_batch_keys=[
                                    "raw_prompt_ids",
                                    "multi_modal_data",
                                    *([question_key] if question_key else []),
                                ],
                            )
                            gen_batch.meta_info.update({
                                "min_pixels": self.config.data.min_pixels,
                                "max_pixels": self.config.data.max_pixels,
                                "data_format_prompt": self.config.data.format_prompt,
                            })
                        else:
                            gen_batch = batch.pop(
                                batch_keys=["input_ids", "attention_mask", "position_ids"],
                                non_tensor_batch_keys=["raw_prompt_ids", *([question_key] if question_key else [])],
                            )

                        if adaptive_max_turns is not None:
                            gen_batch.non_tensor_batch["adaptive_max_turns"] = adaptive_max_turns

                        with timer("step", timing_raw):
                            # generate a batch
                            # Only log memory every 50 steps or on errors
                            log_memory = (self.global_step % 50 == 0) or (self.global_step == 1)
                            
                            if log_memory and torch.cuda.is_available():
                                try:
                                    free_mem, total_mem = torch.cuda.mem_get_info()
                                    usage_ratio = (total_mem - free_mem) / total_mem if total_mem > 0 else 0.0
                                    logger.info(f"[Step {self.global_step}] GPU memory: {usage_ratio*100:.1f}% ({free_mem/1024**3:.2f}GB free)")
                                except Exception:
                                    pass
                            
                            # Add step info to gen_batch for logging
                            gen_batch.meta_info["global_step"] = self.global_step
                            gen_batch.meta_info["total_steps"] = self.training_steps
                            
                            with timer("gen", timing_raw):  # wg: worker group
                                try:
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                except Exception as e:
                                    logger.error(f"[Step {self.global_step}] Error during sequence generation: {e}\n{traceback.format_exc()}")
                                    raise

                            if self.config.algorithm.adv_estimator == "remax":
                                with timer("gen_max", timing_raw):
                                    gen_baseline_batch = deepcopy(gen_batch)
                                    gen_baseline_batch.meta_info["temperature"] = 0
                                    gen_baseline_batch.meta_info["n"] = 1
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                                    batch = batch.union(gen_baseline_output)
                                    reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(batch))
                                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                                    batch.batch["reward_baselines"] = reward_baseline_tensor
                                    del gen_baseline_batch, gen_baseline_output

                            batch.non_tensor_batch["uid"] = np.array(
                                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                            )
                            configured_n = int(self.config.worker.rollout.n)
                            expected_rollout_size = len(batch.batch) * configured_n
                            actual_rollout_size = len(gen_batch_output.batch)
                            if actual_rollout_size != expected_rollout_size:
                                raise RuntimeError(
                                    "Rollout batch size contract violated: "
                                    f"base={len(batch.batch)}, configured_n={configured_n}, "
                                    f"expected={expected_rollout_size}, generated={actual_rollout_size}. "
                                    "rollout.n must remain invariant within a trainer step."
                                )
                            # repeat to align with repeated responses in rollout
                            batch = batch.repeat(repeat_times=configured_n, interleave=True)
                            batch = batch.union(gen_batch_output)

                            # balance the number of valid tokens on each dp rank.
                            # Note that this breaks the order of data inside the batch.
                            # Please take care when you implement group based adv computation such as GRPO and rloo
                            self._balance_batch(batch, metrics=metrics)

                            # compute global_valid tokens
                            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                            # compute reward
                            with timer("reward", timing_raw):
                                reward_ref = self.reward_fn.compute_reward.remote(batch)

                            # recompute old_log_probs
                            with timer("old", timing_raw):
                                try:
                                    old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
                                    batch = batch.union(old_log_probs)
                                except Exception as e:
                                    logger.error(f"[Step {self.global_step}] Error computing old log probs: {e}\n{traceback.format_exc()}")
                                    raise

                            # compute ref_log_probs
                            if self.use_reference_policy:
                                with timer("ref", timing_raw):
                                    ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                                    batch = batch.union(ref_log_probs)

                            # compute values
                            if self.use_critic:
                                with timer("values", timing_raw):
                                    values = self.critic_wg.compute_values(batch)
                                    batch = batch.union(values)

                            with timer("adv", timing_raw):
                                # get token level scores
                                reward_tensor, reward_metrics = ray.get(reward_ref)
                                batch.batch["token_level_scores"] = reward_tensor
                                if v37_evidence is not None:
                                    v37_reward_summary = v37_reward_counts(
                                        reward_metrics, int(reward_tensor.shape[0])
                                    )
                                _raw_point_step_positions = reward_metrics.pop("_point_step_token_positions", None)
                                _raw_point_step_values = reward_metrics.pop("_point_step_value", None)
                                _raw_action_events = reward_metrics.pop("_action_events", None)
                                _raw_action_values = reward_metrics.pop("_action_event_values", None)
                                _selector_task_scores = reward_metrics.pop("_selector_task_scores", None)
                                if v37_evidence is not None and os.environ.get(
                                    "ACTION_EVENT_REWARD_ENABLE", "0"
                                ) == "1":
                                    v37_action_rows_expected = int(reward_tensor.shape[0])
                                    if _raw_action_events is None or _raw_action_values is None:
                                        raise TrainingEvidenceError(
                                            "progress arm reward omitted row-aligned action evidence"
                                        )
                                if _raw_point_step_positions is not None and len(_raw_point_step_positions) == reward_tensor.shape[0]:
                                    point_step_mask = torch.zeros_like(reward_tensor, dtype=torch.float32)
                                    # Parallel value tensor (progress/stoptiming arms). When the
                                    # reward worker did not emit per-step values, this stays all-zero
                                    # and the estimator falls back to token_level_rewards (legacy).
                                    _has_step_values = (
                                        _raw_point_step_values is not None
                                        and len(_raw_point_step_values) == reward_tensor.shape[0]
                                    )
                                    point_step_value = torch.zeros_like(reward_tensor, dtype=torch.float32)
                                    for _row_idx, _positions in enumerate(_raw_point_step_positions):
                                        if isinstance(_positions, (list, tuple)):
                                            _vals = _raw_point_step_values[_row_idx] if _has_step_values else None
                                            for _k, _pos in enumerate(_positions):
                                                try:
                                                    _pos_int = int(_pos)
                                                except (TypeError, ValueError):
                                                    continue
                                                if 0 <= _pos_int < point_step_mask.shape[1]:
                                                    point_step_mask[_row_idx, _pos_int] = 1.0
                                                    if _vals is not None and _k < len(_vals):
                                                        try:
                                                            point_step_value[_row_idx, _pos_int] = float(_vals[_k])
                                                        except (TypeError, ValueError):
                                                            pass
                                    batch.batch["point_step_mask"] = point_step_mask
                                    if _has_step_values:
                                        batch.batch["point_step_value"] = point_step_value
                                if os.environ.get("ACTION_EVENT_REWARD_ENABLE", "0") == "1" and _raw_action_events is not None:
                                    action_spans, action_values, action_types = build_action_step_tensors(
                                        _raw_action_events,
                                        _raw_action_values,
                                        batch.batch["response_mask"],
                                        batch.batch["responses"],
                                    )
                                    batch.batch["action_step_span"] = action_spans
                                    batch.batch["action_step_value"] = action_values
                                    batch.batch["action_step_type"] = action_types
                                    if v37_evidence is not None and v37_action_rows_expected:
                                        v37_action_rows_mapped = int(reward_tensor.shape[0])
                                if _selector_task_scores is not None:
                                    if len(_selector_task_scores) != reward_tensor.shape[0]:
                                        raise ValueError("Selector task scores are not row aligned.")
                                    metrics["bok/selector_task_score_mean"] = float(
                                        np.mean(_selector_task_scores)
                                    )
                                # Extract per-sample answer_scores before reduce_metrics destroys the list
                                _raw_answer_list = reward_metrics.get("answer", [])
                                if _raw_answer_list and len(_raw_answer_list) == reward_tensor.shape[0]:
                                    batch.batch["answer_scores"] = torch.tensor(_raw_answer_list, dtype=torch.float32, device=reward_tensor.device)
                                _raw_answer_correct_list = reward_metrics.pop(
                                    "_answer_correct_for_routing", None
                                )
                                # Backward compatibility for custom managers
                                # that emit a complete public metric but not the
                                # internal row-aligned routing channel.
                                if _raw_answer_correct_list is None:
                                    _legacy_answer_correct = reward_metrics.get("answer_correct", [])
                                    if len(_legacy_answer_correct) == reward_tensor.shape[0]:
                                        _raw_answer_correct_list = _legacy_answer_correct
                                if (
                                    _raw_answer_correct_list is not None
                                    and len(_raw_answer_correct_list) == reward_tensor.shape[0]
                                ):
                                    batch.batch["answer_correct_scores"] = torch.tensor(
                                        _raw_answer_correct_list,
                                        dtype=torch.float32,
                                        device=reward_tensor.device,
                                    )
                                for metric_name, tensor_name in (
                                    ("raw_success", "raw_success_scores"),
                                    ("trajectory_quality", "trajectory_quality_scores"),
                                ):
                                    raw_values = reward_metrics.get(metric_name, [])
                                    if raw_values and len(raw_values) == reward_tensor.shape[0]:
                                        batch.batch[tensor_name] = torch.tensor(
                                            raw_values, dtype=torch.float32, device=reward_tensor.device
                                        )
                                reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                                metrics.update(reward_metrics)

                                # apply kl penalty if available
                                if self.adaptive_actor_kl:
                                    monitor_old_ref_kl, monitor_kl_sum, monitor_kl_count = compute_monitor_old_ref_kl(batch)
                                    beta_before = float(self.kl_ctrl.kl_coef)
                                    # Adaptive actor KL is applied only in the
                                    # differentiable actor loss.  The task reward
                                    # channel remains untouched by reward-side KL.
                                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                                    batch.meta_info["adaptive_actor_kl"] = True
                                    batch.meta_info["adaptive_actor_kl_coef"] = beta_before
                                    metrics.update({
                                        "adaptive_actor_kl/monitor_old_ref_kl": monitor_old_ref_kl,
                                        "adaptive_actor_kl/monitor_token_sum": monitor_kl_sum,
                                        "adaptive_actor_kl/monitor_token_count": monitor_kl_count,
                                        "adaptive_actor_kl/beta_before": beta_before,
                                        "adaptive_actor_kl/selector_excludes_kl": 1.0,
                                    })
                                elif not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                                    # Legacy V36 route: correctness-first changes
                                    # selection only; it does not disable reward KL.
                                    batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                                    metrics.update(kl_metrics)
                                else:
                                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                                # compute advantages, executed on the driver process
                                batch = compute_advantage(
                                    batch,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma,
                                    lam=self.config.algorithm.lam,
                                )
                                if int(os.environ.get("BOK_CORRECTNESS_FIRST", "0")) > 0:
                                    raw_success = batch.batch["raw_success_scores"] >= 0.5
                                    terminal_adv = VF.masked_mean(
                                        batch.batch["advantages"], batch.batch["response_mask"], dim=-1
                                    )
                                    group_rows = defaultdict(list)
                                    for row_idx, uid in enumerate(batch.non_tensor_batch["uid"]):
                                        group_rows[uid].append(row_idx)
                                    mixed_rows = []
                                    mixed_groups = 0
                                    allwrong_groups = 0
                                    allsuccess_groups = 0
                                    for rows in group_rows.values():
                                        outcomes = raw_success[rows]
                                        if bool(outcomes.all()):
                                            allsuccess_groups += 1
                                        elif bool((~outcomes).all()):
                                            allwrong_groups += 1
                                        else:
                                            mixed_groups += 1
                                            mixed_rows.extend(rows)
                                    sign_errors = 0
                                    for row_idx in mixed_rows:
                                        if raw_success[row_idx] and not terminal_adv[row_idx] > 0:
                                            sign_errors += 1
                                        if not raw_success[row_idx] and not terminal_adv[row_idx] < 0:
                                            sign_errors += 1
                                    metrics.update({
                                        "bok/correctness_mixed_rows": float(len(mixed_rows)),
                                        "bok/correctness_sign_errors": float(sign_errors),
                                        "bok/allwrong_groups": float(allwrong_groups),
                                        "bok/all_raw_success_groups": float(allsuccess_groups),
                                    })
                                    if v37_evidence is not None:
                                        v37_frontier_groups = len(group_rows)
                                        v37_frontier_mixed_groups = mixed_groups
                                        v37_allwrong_groups = allwrong_groups
                                        v37_sign_errors = sign_errors

                            # update critic
                            if self.use_critic:
                                with timer("update_critic", timing_raw):
                                    critic_output = self.critic_wg.update_critic(batch)

                                critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                                metrics.update(critic_metrics)

                            # update actor
                            if self.config.trainer.critic_warmup <= self.global_step:
                                with timer("update_actor", timing_raw):
                                    try:
                                        actor_output = self.actor_rollout_wg.update_actor(batch)
                                    except Exception as e:
                                        logger.error(f"[Step {self.global_step}] Error during actor update: {e}\n{traceback.format_exc()}")
                                        # Log memory after error
                                        if torch.cuda.is_available():
                                            try:
                                                free_mem, total_mem = torch.cuda.mem_get_info()
                                                usage_ratio = (total_mem - free_mem) / total_mem if total_mem > 0 else 0.0
                                                logger.error(f"[Step {self.global_step}] GPU memory after error: {usage_ratio*100:.1f}%")
                                            except:
                                                pass
                                        raise

                                actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                                metrics.update(actor_metrics)
                                if self.adaptive_actor_kl:
                                    attempted, executed, skipped = strict_optimizer_counter_metrics(actor_metrics)
                                    if v37_evidence is not None:
                                        v37_optimizer_attempted = attempted
                                        v37_optimizer_executed = executed
                                        v37_optimizer_skipped = skipped
                                        if actor_metrics.get("actor/nonfinite_counter_agreement") != 1.0:
                                            raise TrainingEvidenceError(
                                                "actor workers did not attest nonfinite counter agreement"
                                            )
                                        raw_nonfinite = actor_metrics.get("actor/nonfinite_grad_count")
                                        if (
                                            isinstance(raw_nonfinite, (bool, str))
                                            or not isinstance(
                                                raw_nonfinite,
                                                (int, float, np.integer, np.floating),
                                            )
                                            or not np.isfinite(float(raw_nonfinite))
                                            or float(raw_nonfinite) < 0
                                            or not float(raw_nonfinite).is_integer()
                                        ):
                                            raise TrainingEvidenceError(
                                                "actor/nonfinite_grad_count must be an exact cumulative integer"
                                            )
                                        v37_nonfinite_cumulative = int(float(raw_nonfinite))
                                    if executed > 0:
                                        self.kl_ctrl.update(current_kl=monitor_old_ref_kl, n_steps=executed)
                                        self.effective_actor_updates += executed
                                    metrics.update({
                                        "adaptive_actor_kl/beta_after": float(self.kl_ctrl.kl_coef),
                                        "adaptive_actor_kl/controller_advanced": float(executed > 0),
                                        "adaptive_actor_kl/effective_updates": float(self.effective_actor_updates),
                                    })

                            if v37_evidence is not None:
                                if (
                                    v37_optimizer_attempted is None
                                    or v37_optimizer_executed is None
                                    or v37_optimizer_skipped is None
                                    or v37_nonfinite_cumulative is None
                                ):
                                    raise TrainingEvidenceError(
                                        "formal V37 lacks optimizer integrity counters"
                                    )
                                require_clean_optimizer_step(
                                    v37_optimizer_attempted,
                                    v37_optimizer_executed,
                                    v37_optimizer_skipped,
                                    v37_nonfinite_cumulative,
                                )
                                if v37_reward_summary is None:
                                    raise TrainingEvidenceError(
                                        "formal V37 step lacks reward evidence"
                                    )
                                if metrics.get("adaptive_actor_kl/selector_excludes_kl") != 1.0:
                                    raise TrainingEvidenceError(
                                        "formal V37 selector KL exclusion was not attested"
                                    )
                                v37_evidence.record_step({
                                    "global_step": self.global_step,
                                    "optimizer_microsteps_attempted": v37_optimizer_attempted,
                                    "optimizer_microsteps_executed": v37_optimizer_executed,
                                    "optimizer_microsteps_skipped": v37_optimizer_skipped,
                                    "nonfinite_count_cumulative": v37_nonfinite_cumulative,
                                    "oom_count_cumulative": self._v37_oom_count_cumulative,
                                    **v37_reward_summary,
                                    "selector_kl_contribution": 0.0,
                                    "action_rows_expected": v37_action_rows_expected,
                                    "action_rows_mapped": v37_action_rows_mapped,
                                    "frontier_group_count": v37_frontier_groups,
                                    "frontier_mixed_group_count": v37_frontier_mixed_groups,
                                    "outcome_allwrong_group_count": v37_allwrong_groups,
                                    "correctness_sign_error_count": v37_sign_errors,
                                })

                            self._current_step_complete = True

                            # validate
                            if (
                                self.val_reward_fn is not None
                                and self.config.trainer.val_freq > 0
                                and self.global_step % self.config.trainer.val_freq == 0
                            ):
                                with timer("validation", timing_raw):
                                    val_metrics = self._validate()

                                if v37_evidence is not None:
                                    v37_evidence.update_latest_oom_count(
                                        self._v37_oom_count_cumulative
                                    )

                                metrics.update(val_metrics)
                                last_validation_step = self.global_step

                            if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                                with timer("save_checkpoint", timing_raw):
                                    checkpoint_saved = self._save_checkpoint()
                                if checkpoint_saved:
                                    last_checkpoint_step = self.global_step

                            # collect metrics
                            num_gpus = self.resource_pool_manager.get_num_gpus()
                            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                            
                            # Ensure step timing is recorded (fallback if not already recorded)
                            if "step" not in timing_raw:
                                # Calculate step time from other timings if available
                                step_time = sum(timing_raw.values()) if timing_raw else 0.0
                                if step_time > 0:
                                    timing_raw["step"] = step_time
                            
                            # Only compute throughout metrics if 'step' timing is available
                            if "step" in timing_raw:
                                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

                            self.logger.log(data=metrics, step=self.global_step)
                            
                            # Print reward-related information after each step
                            reward_info = []
                            if "reward" in metrics:
                                for key, value in metrics.items():
                                    if key.startswith("reward/"):
                                        reward_info.append(f"{key}={value:.4f}")
                            
                            # Also include sequence-level reward metrics
                            if "data/sequence_score_mean" in metrics:
                                reward_info.append(f"sequence_score={metrics['data/sequence_score_mean']:.4f}")
                            if "data/sequence_reward_mean" in metrics:
                                reward_info.append(f"sequence_reward={metrics['data/sequence_reward_mean']:.4f}")
                            
                            if reward_info:
                                logger.info(f"[Step {self.global_step}] Reward: {', '.join(reward_info)}")

                            # ── Watchdog early stop: check stop file ──
                            _stop_file = os.environ.get("STOP_FILE", "")
                            if _stop_file and os.path.exists(_stop_file):
                                logger.warning(f"[Step {self.global_step}] [EARLY_STOP] Stop file detected: {_stop_file}")
                                try:
                                    with open(_stop_file, "r") as _sf:
                                        _stop_reason = _sf.read().strip()[:200]
                                    logger.warning(f"[Step {self.global_step}] [EARLY_STOP] Reason: {_stop_reason}")
                                except Exception:
                                    pass
                                logger.warning(f"[Step {self.global_step}] [EARLY_STOP] Saving checkpoint before exit...")
                                checkpoint_saved = self._save_checkpoint()
                                if checkpoint_saved:
                                    last_checkpoint_step = self.global_step
                                    logger.warning(f"[Step {self.global_step}] [EARLY_STOP] Checkpoint saved. Exiting training loop.")
                                else:
                                    logger.warning(f"[Step {self.global_step}] [EARLY_STOP] Checkpoint skipped; last complete checkpoint remains authoritative.")
                                self._early_stopped = True
                                break  # exit batch loop
                    
                    except Exception as e:
                        logger.error(f"[Step {self.global_step}] Fatal error in training step: {e}\n{traceback.format_exc()}")
                        raise

                    # Stop immediately after the final requested update so the
                    # next batch is never fetched from the dataloader.
                    if self.global_step >= self.training_steps:
                        logger.info(f"Reached training steps limit ({self.training_steps}). Stopping.")
                        break
            except BaseException as e:
                logger.error(f"Error in epoch {epoch_idx + 1}, batch loop: {e}\n{traceback.format_exc()}")
                print(f"Error in epoch {epoch_idx + 1}, batch loop: {e}")
                logger.info("Saving checkpoint due to error...")
                print("Saving checkpoint due to error...")
                try:
                    checkpoint_saved = self._save_checkpoint()
                    if checkpoint_saved:
                        logger.info("Checkpoint saved successfully after error.")
                        print("Checkpoint saved successfully after error.")
                    else:
                        logger.warning("Checkpoint skipped after error; last complete checkpoint remains authoritative.")
                        print("Checkpoint skipped after error; last complete checkpoint remains authoritative.")
                except Exception as save_e:
                    logger.error(f"Failed to save checkpoint during error handling: {save_e}\n{traceback.format_exc()}")
                    print(f"Failed to save checkpoint during error handling: {save_e}")
                raise

        if getattr(self, '_early_stopped', False):
            logger.info(f"Training early-stopped at global_step: {self.global_step}")
        else:
            logger.info(f"Training loop completed. Final global_step: {self.global_step}")

        self._finalize_training(
            val_metrics, last_validation_step, last_checkpoint_step, v37_evidence,
        )
        if v37_evidence is not None:
            checkpoint = Path(self.config.trainer.save_checkpoint_path) / f"global_step_{self.global_step}"
            validate_checkpoint_manifest(str(checkpoint), expected_step=self.global_step)
            runtime_path = checkpoint / TRAINER_RUNTIME_STATE_FILE
            if not runtime_path.is_file():
                raise TrainingEvidenceError(
                    f"formal final checkpoint lacks {TRAINER_RUNTIME_STATE_FILE}"
                )
            checkpoint_id = (
                f"{os.environ.get('V37_ARM')}-seed{os.environ.get('V37_SEED')}-{checkpoint.name}"
            )
            v37_evidence.finalize(
                completed=(
                    not getattr(self, "_early_stopped", False)
                    and self.global_step == self.training_steps == v37_evidence.expected_steps
                ),
                checkpoint_id=checkpoint_id,
                checkpoint_path=checkpoint,
                checkpoint_sha256=v37_tree_sha256(checkpoint),
                kl_recoverable=bool(self.adaptive_actor_kl),
            )
