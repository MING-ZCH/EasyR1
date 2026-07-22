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
from dataclasses import asdict
from typing import Optional, Union

import torch
import torch.distributed as dist
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import save_file
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

from .checkpoint_manager import BaseCheckpointManager


def _dist_barrier():
    if torch.cuda.is_available():
        try:
            dist.barrier(device_ids=[torch.cuda.current_device()])
            return
        except TypeError:
            pass
    dist.barrier()


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin],
    ):
        super().__init__(model, optimizer, lr_scheduler, processing_class)

    @staticmethod
    def _adaptive_runtime_required(actor_state_loader) -> bool:
        owner = getattr(actor_state_loader, "__self__", None)
        config = getattr(owner, "config", None)
        return bool(getattr(config, "adaptive_actor_kl", False))

    @staticmethod
    def _assert_runtime_agreement(state, label: str) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, state)
        if any(item != gathered[0] for item in gathered[1:]):
            raise RuntimeError(f"Distributed {label} checkpoint state disagrees across ranks.")

    @staticmethod
    def _assert_all_ranks(condition: bool, label: str) -> None:
        if not dist.is_available() or not dist.is_initialized():
            if not condition:
                raise RuntimeError(label)
            return
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, bool(condition))
        if not all(gathered):
            raise RuntimeError(label)

    def load_checkpoint(self, path: Optional[str] = None):
        if path is None:
            return

        # every rank download its own checkpoint
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        print(f"[rank-{self.rank}]: Loading model from {os.path.abspath(model_path)}.")
        print(f"[rank-{self.rank}]: Loading optimizer from {os.path.abspath(optim_path)}.")
        print(f"[rank-{self.rank}]: Loading extra_state from {os.path.abspath(extra_path)}.")
        model_state_dict = torch.load(model_path, weights_only=False)
        optim_state_dict = torch.load(optim_path, weights_only=False)
        extra_state_dict = torch.load(extra_path, weights_only=False)

        actor_state_loader = getattr(self.optimizer, "_easy_r1_actor_state_loader", None)
        adaptive_runtime = self._adaptive_runtime_required(actor_state_loader)
        actor_runtime_state = extra_state_dict.get("actor_runtime_state")
        self._assert_runtime_agreement(actor_runtime_state, "actor runtime")

        rollout_state_loader = getattr(self.model, "_easy_r1_rollout_state_loader", None)
        rollout_runtime_state = extra_state_dict.get("rollout_runtime_state")
        if adaptive_runtime:
            runtime_complete = (
                actor_runtime_state is not None
                and rollout_state_loader is not None
                and rollout_runtime_state is not None
                and "rng" in extra_state_dict
            )
            self._assert_all_ranks(
                runtime_complete,
                "adaptive_actor_kl checkpoint is missing actor, rollout, or process RNG runtime state on a rank.",
            )

        state_dict_options = StateDictOptions(cpu_offload=True)
        set_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optim_state_dict,
            options=state_dict_options,
        )
        self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])

        # recover random state
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

        if actor_state_loader is not None:
            actor_state_loader(actor_runtime_state)
        if rollout_state_loader is not None and rollout_runtime_state is not None:
            rollout_state_loader(rollout_runtime_state)

    def save_checkpoint(self, path: str, save_model_only: bool = False):
        path = self.local_mkdir(path)
        _dist_barrier()

        # every rank will save its own model and optim shard
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

        state_dict_options = StateDictOptions(cpu_offload=True)
        if save_model_only:
            model_state_dict = get_model_state_dict(self.model, options=state_dict_options)
            print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}.")
            torch.save(model_state_dict, model_path)
        else:
            model_state_dict, optim_state_dict = get_state_dict(self.model, self.optimizer, options=state_dict_options)
            extra_state_dict = {
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "rng": self.get_rng_state(),
            }
            actor_state_getter = getattr(self.optimizer, "_easy_r1_actor_state_getter", None)
            actor_runtime_state = None
            if actor_state_getter is not None:
                actor_runtime_state = actor_state_getter()
            self._assert_runtime_agreement(actor_runtime_state, "actor runtime")
            if actor_runtime_state is not None:
                extra_state_dict["actor_runtime_state"] = actor_runtime_state
            rollout_state_getter = getattr(self.model, "_easy_r1_rollout_state_getter", None)
            adaptive_runtime = bool(
                isinstance(actor_runtime_state, dict)
                and actor_runtime_state.get("config_identity", {}).get("adaptive_actor_kl", False)
            )
            if adaptive_runtime and rollout_state_getter is None:
                raise RuntimeError("adaptive_actor_kl cannot checkpoint the rollout generation RNG state.")
            if rollout_state_getter is not None:
                rollout_runtime_state = None
                rollout_error = None
                try:
                    rollout_runtime_state = rollout_state_getter()
                except RuntimeError as exc:
                    rollout_error = str(exc)
                if dist.is_available() and dist.is_initialized():
                    rollout_errors = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(rollout_errors, rollout_error)
                else:
                    rollout_errors = [rollout_error]
                if adaptive_runtime and any(error is not None for error in rollout_errors):
                    raise RuntimeError(
                        f"adaptive_actor_kl could not checkpoint rollout RNG state on every rank: {rollout_errors}."
                    )
                if rollout_runtime_state is not None:
                    extra_state_dict["rollout_runtime_state"] = rollout_runtime_state
            print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}.")
            print(f"[rank-{self.rank}]: Saving optimizer to {os.path.abspath(optim_path)}.")
            print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}.")
            torch.save(model_state_dict, model_path)
            torch.save(optim_state_dict, optim_path)
            torch.save(extra_state_dict, extra_path)

        # wait for everyone to dump to local
        _dist_barrier()

        if self.rank == 0:
            hf_path = os.path.join(path, "huggingface")
            os.makedirs(hf_path, exist_ok=True)
            assert isinstance(self.model._fsdp_wrapped_module, (PreTrainedModel, PeftModel))
            self.model._fsdp_wrapped_module.config.save_pretrained(hf_path)
            self.model._fsdp_wrapped_module.generation_config.save_pretrained(hf_path)
            self.processing_class.save_pretrained(hf_path)

        if isinstance(self.model._fsdp_wrapped_module, PeftModel):
            lora_path = os.path.join(path, "lora_adapter")
            peft_config = {}
            if self.rank == 0:
                os.makedirs(lora_path, exist_ok=True)
                peft_config = asdict(self.model._fsdp_wrapped_module.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])

            sharded_lora_weights = get_peft_model_state_dict(
                self.model._fsdp_wrapped_module, state_dict=model_state_dict
            )
            cuda_device = torch.device("cuda")
            lora_weights = {
                name: sharded_weight.to(cuda_device).full_tensor().detach().cpu()
                if isinstance(sharded_weight, DTensor)
                else sharded_weight.detach().cpu()
                for name, sharded_weight in sharded_lora_weights.items()
            }
            torch.cuda.empty_cache()
            if self.rank == 0:
                save_file(lora_weights, os.path.join(lora_path, "adapter_model.safetensors"))
                with open(os.path.join(lora_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                    json.dump(peft_config, f, ensure_ascii=False, indent=4)

            _dist_barrier()
            if self.rank == 0:
                print(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_path}")

        _dist_barrier()
