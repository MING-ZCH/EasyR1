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
The main entry point to run the PPO algorithm
"""

import os
from typing import Literal, Optional, Union

import numpy as np
import psutil
import torch
import torch.distributed as dist

_H20_SIGFPE_WORKAROUND = (
    os.environ.get("STEPCOUNT_HARDWARE_PROFILE", "").strip().lower() == "h20"
    or os.environ.get("STEPCOUNT_ENABLE_H20_SIGFPE_WORKAROUND", "0") == "1"
)

# H20 SIGFPE protection: force TF32 + cublaslt in worker processes only
# when the H20 workaround is explicitly active. H200 does not need this and
# PyTorch logs an experimental-feature warning for preferred_blas_library.
if _H20_SIGFPE_WORKAROUND:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.backends.cuda.preferred_blas_library("cublaslt")
    except (AttributeError, RuntimeError):
        pass

from copy import deepcopy
from accelerate import init_empty_weights
from codetiming import Timer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from ..models.monkey_patch import apply_ulysses_patch
from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from ..utils.dataset import process_image
from ..utils.ray_environment import validate_v37_remote_environment
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.torch_functional import AnyPrecisionAdamW, get_constant_schedule_with_warmup
from .config import ActorConfig, CriticConfig, FSDPConfig, ModelConfig, OptimConfig, RefConfig, WorkerConfig
from .rollout import vLLMRollout
from .sharding_manager import FSDPVLLMShardingManager
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


def _adaptive_actor_scheduler_should_step(metrics: dict) -> bool:
    """Advance adaptive LR only when this RPC executed an optimizer update."""
    raw = metrics.get("actor/optimizer_steps_executed")
    if raw is None:
        raise RuntimeError("adaptive actor metrics are missing optimizer_steps_executed")
    values = np.asarray(raw, dtype=object).reshape(-1)
    if values.size != 1:
        raise RuntimeError("adaptive actor optimizer_steps_executed must be one nonnegative integer")
    value = values[0]
    if (
        isinstance(value, (bool, str))
        or not isinstance(value, (int, float, np.integer, np.floating))
    ):
        raise RuntimeError("adaptive actor optimizer_steps_executed must be one nonnegative integer")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise RuntimeError("adaptive actor optimizer_steps_executed must be one nonnegative integer")
    return int(numeric) > 0


def _reconcile_spike_cooldown_lr(actor, optimizer, scheduler_advanced: bool) -> float:
    """Reapply cooldown only after a scheduler update changed optimizer LR."""
    cooldown = int(getattr(actor, "_spike_cooldown_remaining", 0))
    original_lrs = getattr(actor, "_original_lrs", None)
    if cooldown > 0 and original_lrs is not None and scheduler_advanced:
        scheduled_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        actor._original_lrs = scheduled_lrs
        factor = float(actor._spike_lr_factor)
        for group, scheduled_lr in zip(optimizer.param_groups, scheduled_lrs):
            group["lr"] = scheduled_lr * factor
    return float(optimizer.param_groups[0]["lr"])


def _set_cuda_device_from_local_rank() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return None
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RAY_LOCAL_RANK", "0")) or 0)
    except Exception:
        local_rank = 0
    # Ray often narrows CUDA_VISIBLE_DEVICES to one GPU per actor. In that case
    # the process-local CUDA index is 0 even when LOCAL_RANK is 1..7.
    device_idx = local_rank if local_rank < device_count else 0
    torch.cuda.set_device(device_idx)
    return int(torch.cuda.current_device())


def _dist_barrier():
    if torch.cuda.is_available():
        try:
            dist.barrier(device_ids=[torch.cuda.current_device()])
            return
        except TypeError:
            pass
    dist.barrier()


class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        validate_v37_remote_environment("FSDP worker")
        self.config = config
        self.role = role
        cuda_device = _set_cuda_device_from_local_rank()

        if not dist.is_initialized():
            if cuda_device is not None:
                try:
                    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", cuda_device))
                except TypeError:
                    dist.init_process_group(backend="nccl")
            else:
                dist.init_process_group(backend="nccl")

        # improve numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_critic = self.role == "critic"
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        self._cache = {}

        self._use_param_offload = False
        self._use_optimizer_offload = False
        if self._is_actor:
            self._use_param_offload = self.config.actor.offload.offload_params
            self._use_optimizer_offload = self.config.actor.offload.offload_optimizer
            self._init_config(self.config.actor, "actor")
        elif self._is_critic:
            self._use_param_offload = self.config.critic.offload.offload_params
            self._use_optimizer_offload = self.config.critic.offload.offload_optimizer
            self._init_config(self.config.critic, "critic")
        elif self._is_ref:  # NOTE: it seems that manual offload is slower than FSDP offload
            self._use_param_offload = self.config.ref.offload.offload_params
            self._init_config(self.config.ref, "ref")

    def _init_config(
        self, config: Union[ActorConfig, CriticConfig, RefConfig], role: Literal["actor", "critic", "ref"]
    ):
        world_size = dist.get_world_size()
        fsdp_size = config.fsdp.fsdp_size
        if fsdp_size <= 0 or fsdp_size >= world_size:
            self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        else:  # hsdp
            self.device_mesh = init_device_mesh(
                "cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp")
            )

        if config.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(
                    world_size // config.ulysses_sequence_parallel_size,
                    config.ulysses_sequence_parallel_size,
                ),
                mesh_dim_names=("dp", "sp"),
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        if not hasattr(config, "global_batch_size"):  # ref model
            return

        if self.config.rollout.n > 1:
            config.global_batch_size *= self.config.rollout.n
            self.print_rank0(f"{role} will use global batch size {config.global_batch_size}.")

        config.global_batch_size_per_device = (
            config.global_batch_size // self.device_mesh.size() * config.ulysses_sequence_parallel_size
        )
        if config.global_batch_size_per_device == 0:
            raise ValueError(f"{role} global batch size * ulysses size must be larger than num gpus.")

        if config.global_batch_size_per_device % config.micro_batch_size_per_device_for_update != 0:
            raise ValueError(f"{role} global batch size per device must be divisible by the micro batch size.")

        if (
            config.fsdp.enable_cpu_offload
            and config.global_batch_size_per_device != config.micro_batch_size_per_device_for_update
        ):
            # 提供更详细的错误信息和解决方案
            error_msg = (
                f"{role} cannot use FSDP's CPU offload when gradient accumulation is enabled.\n"
                f"  - global_batch_size_per_device: {config.global_batch_size_per_device}\n"
                f"  - micro_batch_size_per_device_for_update: {config.micro_batch_size_per_device_for_update}\n"
                f"  - gradient_accumulation_steps: {config.global_batch_size_per_device // config.micro_batch_size_per_device_for_update}\n"
                f"Solutions:\n"
                f"  1. Disable CPU offload: set {role}.fsdp.enable_cpu_offload=false\n"
                f"  2. Disable gradient accumulation: set {role}.micro_batch_size_per_device_for_update={config.global_batch_size_per_device}\n"
                f"  3. Use parameter offload instead: set {role}.offload.offload_params=true (recommended)"
            )
            raise ValueError(error_msg)

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool = False,
    ) -> None:
        self.tokenizer = get_tokenizer(
            model_config.tokenizer_path,
            trust_remote_code=model_config.trust_remote_code,
            use_fast=True,
        )
        self.processor = get_processor(
            model_config.tokenizer_path,
            trust_remote_code=model_config.trust_remote_code,
            use_fast=True,
        )
        self.model_config = AutoConfig.from_pretrained(
            model_config.model_path,
            trust_remote_code=model_config.trust_remote_code,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_config.override_config,
        )

        try:
            self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
        except Exception:
            self.generation_config = GenerationConfig.from_model_config(self.model_config)

        self.print_rank0(f"Model config: {self.model_config}")

        if padding_free:
            apply_ulysses_patch(self.model_config.model_type)
            self.print_rank0("Ulysses patch applied!")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor or self._is_critic else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        if self._is_critic:
            auto_class = AutoModelForTokenClassification
        elif type(self.model_config) in AutoModelForVision2Seq._model_mapping.keys():
            auto_class = AutoModelForVision2Seq
        else:
            auto_class = AutoModelForCausalLM

        # Determine best available attention implementation
        try:
            import flash_attn  # noqa: F401
            _attn_impl = "flash_attention_2"
        except ImportError:
            # sdpa can segfault with qwen2_5_vl on transformers 4.55+
            # Use eager as the safest fallback
            _attn_impl = "eager"
        self.print_rank0(f"Using attention implementation: {_attn_impl}")

        if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
            model = auto_class.from_pretrained(
                model_config.model_path,
                config=self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation=_attn_impl,
                device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                # transformers >= 4.49: from_config() no longer accepts
                # torch_dtype, attn_implementation, or trust_remote_code as kwargs.
                # Try progressively stripped-down calls.
                from_config_kwargs = dict(
                    torch_dtype=torch_dtype,
                    attn_implementation=_attn_impl,
                    trust_remote_code=model_config.trust_remote_code,
                )
                model = None
                while from_config_kwargs is not None:
                    try:
                        model = auto_class.from_config(self.model_config, **from_config_kwargs)
                        break
                    except TypeError as e:
                        # Remove the offending kwarg and retry
                        bad_kwarg = None
                        err_msg = str(e)
                        for kw in list(from_config_kwargs.keys()):
                            if kw in err_msg:
                                bad_kwarg = kw
                                break
                        if bad_kwarg:
                            del from_config_kwargs[bad_kwarg]
                        else:
                            # Can't identify which kwarg is wrong; try bare call
                            from_config_kwargs = None
                if model is None:
                    model = auto_class.from_config(self.model_config)

        assert isinstance(model, PreTrainedModel)  # lint
        model.tie_weights()  # avoid hanging
        model = model.to(torch_dtype)
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if not (self._is_actor or self._is_critic):
            model.requires_grad_(False)

        if model_config.freeze_vision_tower:
            if hasattr(model, "visual"):
                model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            else:
                self.print_rank0("No vision tower found.")

        _dist_barrier()
        print_model_size(model)
        print_gpu_memory_usage("After huggingface model init")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}.")

        if self.device_mesh.ndim == 2:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            else:
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        else:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.FULL_SHARD
            else:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.enable_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None

        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None

        self.fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        print_gpu_memory_usage("After FSDP module init")

        if self._is_actor or self._is_critic:
            if optim_config.strategy == "adamw":
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                    fused=True,
                )
            elif optim_config.strategy == "adamw_bf16":
                self.optimizer = AnyPrecisionAdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {optim_config.strategy} not supported.")

            num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
            )
            print_gpu_memory_usage("After optimizer init")
        else:
            self.optimizer, self.lr_scheduler = None, None

    def _build_rollout(self) -> None:
        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        assert self.world_size % tp_size == 0, (
            f"rollout world size: {self.world_size} is not divisible by tp size: {tp_size}"
        )
        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        self.rollout = vLLMRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
        )
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
        )
        print_gpu_memory_usage("After vllm init")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._is_critic:
            model_config = self.config.critic.model
            fsdp_config = self.config.critic.fsdp
            optim_config = self.config.critic.optim
            padding_free = self.config.critic.padding_free
            role = "critic"
        elif self._is_actor:
            model_config = self.config.actor.model
            fsdp_config = self.config.actor.fsdp
            optim_config = self.config.actor.optim
            padding_free = self.config.actor.padding_free
            role = "actor"
        elif self._is_ref:
            model_config = self.config.actor.model
            fsdp_config = self.config.ref.fsdp
            optim_config = None
            padding_free = self.config.ref.padding_free
            role = "ref"
        else:
            raise ValueError(f"Unknown role {role}.")

        if self._is_actor or self._is_critic or self._is_ref:
            self._build_model_optimizer(
                model_config=model_config,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                padding_free=padding_free,
            )
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage(f"After offload {role} optimizer during init")

        # If this worker also serves rollout (vLLM), free cached GPU memory before
        # initializing vLLM KV cache. This is important when FSDP offload frees tensors
        # but the CUDA caching allocator keeps large reserved segments.
        if self._is_rollout and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print_gpu_memory_usage("After empty_cache before vllm init")

        if self._is_actor:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
            )

        if self._is_critic:
            from .critic.dp_critic import DataParallelPPOCritic  # lazy import

            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._is_rollout:
            self._build_rollout()

        if self._is_ref:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.fsdp_module,
            )

        if self._is_actor or self._is_critic:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str):
        assert self._is_actor or self._is_critic
        
        # Clear cache to free up memory before saving
        torch.cuda.empty_cache()
        
        if self._use_param_offload:
            try:
                load_fsdp_model(self.fsdp_module)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error when loading model for checkpoint saving: {e}. Trying to save from current state.")
                # If load failed, we might want to ensure everything is offloaded to be consistent, 
                # or just proceed. If some are on GPU and some on CPU, get_state_dict might handle it 
                # or fail. But failing here is better than crashing the whole save process if we can avoid it.
                # Let's try to offload everything to CPU to be safe if load failed.
                try:
                    offload_fsdp_model(self.fsdp_module)
                except:
                    pass

        self.checkpoint_manager.save_checkpoint(path)
        _dist_barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.load_checkpoint(path)
        _dist_barrier()

        # Override optimizer LR if config LR differs from checkpoint (e.g., V24 uses 1.5e-6 but resumes from V21 with 1e-6)
        if self.optimizer is not None:
            if self._is_actor:
                target_lr = self.config.actor.optim.lr
            elif self._is_critic:
                target_lr = self.config.critic.optim.lr
            else:
                target_lr = None

            if target_lr is not None:
                current_lr = self.optimizer.param_groups[0]["lr"]
                if abs(current_lr - target_lr) > 1e-10:
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = target_lr
                    # Also reset lr_scheduler base_lrs so future scheduling uses the new LR
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.base_lrs = [target_lr for _ in self.lr_scheduler.base_lrs]
                    if dist.get_rank() == 0:
                        print(f"[LR-Override] Checkpoint had lr={current_lr:.2e}, config specifies lr={target_lr:.2e}. "
                              f"Overriding optimizer LR to {target_lr:.2e}.")

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:  # avoid OOM in resuming
            offload_fsdp_optimizer(self.optimizer)

    def preprocess_multi_modal_data(self, data: DataProto):
        # inplace load & process image data
        min_pixels = data.meta_info["min_pixels"]
        max_pixels = data.meta_info["max_pixels"]
        multi_modal_data_copy = deepcopy(data.non_tensor_batch["multi_modal_data"])

        processed_images = []
        for multi_modal_data in multi_modal_data_copy:
            processed_per_query_images = []
            for image in multi_modal_data['image']:
                processed_per_query_images.append(
                    process_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
                )
            processed_images.append(processed_per_query_images)

        # Note: Using the alternative (commented) code below to process images can lead to subtle resize issues:
        # For example: an image with size (656, 369) should be resized to (682, 383) with `min_pixels=512 ** 2`,
        # however, it will produce an image with size (683, 383) when using the following for loop.
        # (But it works normally when directly applying `process_image` to this image).
        # This behavior is unexpected and difficult to explain.
        # The code above works normally.

        # images = [multi_modal_data['image'] for multi_modal_data in multi_modal_data_copy]
        # for i, per_query_images in enumerate(images):
        #     for j, image in enumerate(per_query_images):
        #         images[i][j] = process_image(image, min_pixels=min_pixels, max_pixels=max_pixels)

        multi_modal_inputs = np.array([
            dict(self.processor.image_processor(images=per_query_images, videos=None))
            for per_query_images in processed_images
        ], dtype=object)
        data.non_tensor_batch["multi_modal_inputs"] = multi_modal_inputs

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if "multi_modal_inputs" in self._cache:
            data.non_tensor_batch['multi_modal_inputs'] = deepcopy(self._cache['multi_modal_inputs'])
        elif "multi_modal_data" in data.non_tensor_batch:
            self.preprocess_multi_modal_data(data)

        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                try:
                    metrics = self.actor.update_policy(data=data)
                except ValueError as e:
                    # 捕获 "Image features and image tokens do not match" 错误
                    if "Image features and image tokens do not match" in str(e):
                        allow_zero_fallback = os.environ.get(
                            "EASYR1_ALLOW_ZERO_MM_LOGPROB", "0"
                        ).strip().lower() in {"1", "true", "yes", "on"}
                        if not allow_zero_fallback:
                            raise RuntimeError(
                                "Multimodal image/token mismatch reached actor update. "
                                "Refusing to return zero/default metrics because that silently "
                                "corrupts RL training."
                            ) from e
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            "EASYR1_ALLOW_ZERO_MM_LOGPROB=1: returning legacy default metrics "
                            f"for image features/tokens mismatch. Error: {e}"
                        )
                        # 返回默认的 metrics，避免 Actor 崩溃
                        metrics = {
                            "actor/pg_loss": 0.0,
                            "actor/pg_clipfrac_higher": 0.0,
                            "actor/pg_clipfrac_lower": 0.0,
                            "actor/entropy_loss": 0.0,
                            "actor/ppo_kl": 0.0,
                            "actor/grad_norm": 0.0,
                        }
                    else:
                        raise

            delta_time = timer.last if timer.last is not None else 0.0
            global_num_tokens_raw = data.meta_info.get("global_token_num", 0)
            # estimate_flops 需要接收一个 list，而不是求和后的整数
            # 确保 batch_seqlens 是一个 list
            if isinstance(global_num_tokens_raw, list):
                batch_seqlens = global_num_tokens_raw
            elif isinstance(global_num_tokens_raw, (int, float)):
                # 如果是单个数字，转换为 list
                batch_seqlens = [int(global_num_tokens_raw)]
            else:
                batch_seqlens = []
            
            if len(batch_seqlens) > 0 and delta_time > 0:
                estimated_flops, promised_flops = self.flops_counter.estimate_flops(batch_seqlens, delta_time)
                metrics["perf/mfu_actor"] = (
                    estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
                )
            else:
                metrics["perf/mfu_actor"] = 0.0
            
            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            adaptive_actor_kl = bool(getattr(self.config.actor, "adaptive_actor_kl", False))
            scheduler_advanced = (
                _adaptive_actor_scheduler_should_step(metrics) if adaptive_actor_kl else True
            )
            if scheduler_advanced:
                self.lr_scheduler.step()
            # A skipped adaptive update leaves the already-reduced cooldown LR
            # untouched. Reapply the factor only when scheduler.step() actually
            # replaced optimizer LR with a newly scheduled baseline.
            lr = _reconcile_spike_cooldown_lr(self.actor, self.optimizer, scheduler_advanced)

            metrics["actor/lr"] = lr
            metrics["actor/lr_scheduler_advanced"] = 1.0 if scheduler_advanced else 0.0

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info.
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            # after parameters sync with rollout, offload actor model to CPU
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)

            # load image data
            cached_multi_modal_data = None
            if "multi_modal_data" in prompts.non_tensor_batch:
                cached_multi_modal_data = deepcopy(prompts.non_tensor_batch["multi_modal_data"])
                min_pixels = prompts.meta_info['min_pixels']
                max_pixels = prompts.meta_info['max_pixels']
                processed_images = []
                for i, multi_modal_data in enumerate(prompts.non_tensor_batch["multi_modal_data"]):
                    for j, image in enumerate(multi_modal_data["image"]):
                        multi_modal_data['image'][j] = process_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
                    processed_images.append(multi_modal_data)
                prompts.non_tensor_batch["multi_modal_data"] = processed_images

            output = self.rollout.generate_sequences(prompts=prompts)

            if cached_multi_modal_data is not None:
                sampling_n = prompts.meta_info.get("n", self.config.rollout.n)
                # restore multi_modal_data
                output.non_tensor_batch["multi_modal_data"] = cached_multi_modal_data
                if sampling_n > 1:
                    output.non_tensor_batch["multi_modal_data"] = np.repeat(
                        output.non_tensor_batch["multi_modal_data"], repeats=sampling_n, axis=0,
                    )

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs(self, data: DataProto):
        assert self._is_actor
        self._cache.clear()
        if "multi_modal_data" in data.non_tensor_batch:
            self.preprocess_multi_modal_data(data)
            # create cache for multi_modal_inputs
            self._cache['multi_modal_inputs'] = deepcopy(data.non_tensor_batch['multi_modal_inputs'])

        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output}, meta_info={"temperature": self.config.rollout.temperature}
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_probs(self, data: DataProto):
        # The `self._cache` is empty here since cached `multi_modal_inputs` is only saved in the actor's _cache,
        # not in the ref_policy's or critic's caches.
        assert self._is_ref
        if "multi_modal_inputs" in self._cache:
            data.non_tensor_batch['multi_modal_inputs'] = deepcopy(self._cache['multi_modal_inputs'])
        elif "multi_modal_data" in data.non_tensor_batch:
            self.preprocess_multi_modal_data(data)

        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_probs": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._is_critic
        # The `self._cache` is empty here since cached `multi_modal_inputs` is only saved in the actor's _cache,
        # not in the ref_policy's or critic's caches.
        if "multi_modal_inputs" in self._cache:
            data.non_tensor_batch['multi_modal_inputs'] = deepcopy(self._cache['multi_modal_inputs'])
        elif "multi_modal_data" in data.non_tensor_batch:
            self.preprocess_multi_modal_data(data)

        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        # The `self._cache` is empty here since cached `multi_modal_inputs` is only saved in the actor's _cache,
        # not in the ref_policy's or critic's caches.
        if "multi_modal_inputs" in self._cache:
            data.non_tensor_batch['multi_modal_inputs'] = deepcopy(self._cache['multi_modal_inputs'])
        elif "multi_modal_data" not in data.non_tensor_batch:
            self.preprocess_multi_modal_data(data)

        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            delta_time = timer.last if timer.last is not None else 0.0
            global_num_tokens_raw = data.meta_info.get("global_token_num", 0)
            # estimate_flops 需要接收一个 list，而不是求和后的整数
            # 确保 batch_seqlens 是一个 list
            if isinstance(global_num_tokens_raw, list):
                batch_seqlens = global_num_tokens_raw
            elif isinstance(global_num_tokens_raw, (int, float)):
                # 如果是单个数字，转换为 list
                batch_seqlens = [int(global_num_tokens_raw)]
            else:
                batch_seqlens = []
            
            if len(batch_seqlens) > 0 and delta_time > 0:
                estimated_flops, promised_flops = self.flops_counter.estimate_flops(batch_seqlens, delta_time)
            else:
                estimated_flops, promised_flops = 0.0, 1.0
            metrics["perf/mfu_critic"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info.
            output = DataProto(
                non_tensor_batch={
                    metric: np.array([value] if np.isscalar(value) else value) for metric, value in metrics.items()
                }
            )

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output
