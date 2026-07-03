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

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
            Role.RefPolicy: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

        RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        # 从环境变量中获取 NCCL 超时设置，如果没有则使用默认值
        nccl_timeout = os.environ.get("NCCL_TIMEOUT", "1800")  # 默认 30 分钟
        torch_distributed_timeout = os.environ.get("TORCH_DISTRIBUTED_TIMEOUT", "1800")
        ray_address = os.environ.get("RAY_ADDRESS", "").strip()
        ray_address_candidates = os.environ.get("RAY_ADDRESS_CANDIDATES", "").strip()
        
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "PYTHONUNBUFFERED": "1",
                # NCCL 超时配置（确保传递到 Ray worker）
                "NCCL_TIMEOUT": nccl_timeout,
                "TORCH_DISTRIBUTED_TIMEOUT": torch_distributed_timeout,
                "NCCL_ASYNC_ERROR_HANDLING": os.environ.get("NCCL_ASYNC_ERROR_HANDLING", "1"),
                # CPU 线程限制（避免占用过多 CPU 资源影响 NCCL 通信）
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "8"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "8"),
                "TORCH_NUM_THREADS": os.environ.get("TORCH_NUM_THREADS", "8"),
                # vLLM configuration (must be passed to Ray workers)
                "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "1"),
                "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND", "XFORMERS"),
                # LD_LIBRARY_PATH for cusparselt
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
                # NCCL IB ECE fix (ibv_set_ece Invalid argument on some RDMA drivers)
                "NCCL_IB_DISABLE_ECE": os.environ.get("NCCL_IB_DISABLE_ECE", "1"),
                # NCCL multi-node settings (from cluster config)
                "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE", "0"),
                "NCCL_IB_GID_INDEX": os.environ.get("NCCL_IB_GID_INDEX", "3"),
                "NCCL_IB_SL": os.environ.get("NCCL_IB_SL", "3"),
                "NCCL_IB_TC": os.environ.get("NCCL_IB_TC", "160"),
                "NCCL_IB_QPS_PER_CONNECTION": os.environ.get("NCCL_IB_QPS_PER_CONNECTION", "4"),
                "NCCL_IB_TIMEOUT": os.environ.get("NCCL_IB_TIMEOUT", "22"),
                "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME", "bond1"),
                "NCCL_IB_HCA": os.environ.get("NCCL_IB_HCA", ""),
                "NCCL_NET_GDR_LEVEL": os.environ.get("NCCL_NET_GDR_LEVEL", "2"),
                "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE", "0"),
                "NCCL_PXN_DISABLE": os.environ.get("NCCL_PXN_DISABLE", "1"),
            }
        }
        # Pass ALL StepCount/trajectory/reward/BoK/grad-safety env vars to Ray workers.
        # Without this, remote actors (FSDPWorker/reward workers) can't see launch-script
        # overrides for these knobs and silently fall back to the hardcoded os.getenv(...)
        # defaults baked into the reading code (dp_actor.py, core_algos.py, reward/function.py)
        # -- this is NOT a config error on the launch-script side, it's a missing whitelist
        # entry here. Confirmed bug (2026-07-02): GRAD_SPIKE_*/GRAD_NONFINITE_* were exported
        # by v35's launch script (skip-bad-update, never-touch-LR spec) but never reached the
        # actor process, which silently ran on dp_actor.py's built-in defaults instead
        # (threshold=5.0x, lr_factor=0.1, brake_max=6/3) -- reproducing the exact v34 LR-crush
        # failure mode (permanent LR halving after repeated spikes/nonfinite events) that the
        # launch script was explicitly designed to avoid.
        #   GRAD_SPIKE_ / GRAD_NONFINITE_ : gradient-spike & nonfinite-grad protection knobs
        #                                   (dp_actor.py DataParallelPPOActor.__init__)
        #   VCRL_                         : Variance-based Curriculum RL advantage shaping
        #                                   (core_algos.py, same os.getenv-without-whitelist
        #                                    bug class as BOK_, found during this audit)
        for key, val in os.environ.items():
            if key.startswith(("STEPCOUNT_", "TRAJ_", "EASYR1_", "INTERLEAVED_",
                               "BOK_", "PROCESS_REWARD_", "POLICY_LOSS_",
                               "GRAD_SPIKE_", "GRAD_NONFINITE_", "VCRL_")):
                runtime_env["env_vars"][key] = val
        # Also pass critical H20 SIGFPE fixes + other os.getenv-only knobs read inside remote
        # actors (REWARD_NUM_WORKERS: workers/reward/function.py reward-computation thread pool
        # size; same missing-whitelist bug class found alongside GRAD_SPIKE_/VCRL_ above).
        for key in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "DISABLE_ADDMM_CUDA_LT",
                    "REWARD_NUM_WORKERS"):
            if key in os.environ:
                runtime_env["env_vars"][key] = os.environ[key]
        # Pass proxy + wandb vars to Ray actors (Runner needs proxy for wandb online).
        # Ray is started WITHOUT proxy (Phase 1: fixes worker registration gRPC hang),
        # but actors need proxy injected via runtime_env for network-dependent ops.
        for key in ("http_proxy", "https_proxy", "no_proxy",
                     "WANDB_API_KEY", "WANDB_MODE", "WANDB_DIR"):
            if key in os.environ:
                runtime_env["env_vars"][key] = os.environ[key]
        if ray_address:
            addrs = []
            if ray_address_candidates:
                addrs.extend([x.strip() for x in ray_address_candidates.split(",") if x.strip()])
            if ray_address not in addrs:
                addrs.insert(0, ray_address)

            last_error = None
            for addr in addrs:
                print(f"[EasyR1] Connecting to existing Ray cluster: {addr}")
                try:
                    ray.init(address=addr, runtime_env=runtime_env)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    print(f"[EasyR1] Ray connect failed at {addr}: {e}")

            if last_error is not None:
                hint = (
                    "\n[EasyR1][RayConnectHint] Failed to connect remote Ray.\n"
                    "- If this driver runs outside Kubernetes, '*.svc.cluster.local' may be unreachable.\n"
                    "- Prefer Ray Client endpoint 'ray://<head-host>:10001' for remote submit.\n"
                    "- Use routable IP/hostname or port-forward to 10001.\n"
                    f"- Tried addresses: {addrs}\n"
                )
                raise RuntimeError(hint) from last_error
        else:
            print("[EasyR1] RAY_ADDRESS is empty, starting local Ray runtime.")
            ray.init(runtime_env=runtime_env)

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))


if __name__ == "__main__":
    main()
