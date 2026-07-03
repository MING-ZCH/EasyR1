#!/bin/bash
export NCCL_DEBUG=INFO
export NCCL_IB_TC=106
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CROSS_NIC=0
export TORCH_DISTRIBUTED_TIMEOUT=1800
LOG_DIR="/weiyang/250010212/chenhaoz/logs/train"
mkdir -p ${LOG_DIR}
export output_path='./'
export ckpt_path='./ckpt'
apt-get update && apt-get install -y libgl1-mesa-glx

# Ensure vLLM uses a stable libcudart path (avoid deleted torchvision.libs)
export VLLM_CUDART_SO_PATH="/weiyang/250010212/miniconda3/envs/stepcount/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
export LD_LIBRARY_PATH="/weiyang/250010212/miniconda3/envs/stepcount/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"

set -x

export PYTHONUNBUFFERED=1
export WANDB_API_KEY='5c554b68aed1a458465f44705a27102d8579c1cb'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16
# 强制Python垃圾回收，配合empty_cache使用
export PYTHONHASHSEED=0

# StepCount mask reward: treat earlier turns as already-pointed (required for xxx_N multi-turn data)
export STEPCOUNT_MASKS_METADATA=${STEPCOUNT_MASKS_METADATA:-/weiyang/share/250010212/StepCount-RL_Masks/masks_metadata.json}
export STEPCOUNT_MASKS_DIR=${STEPCOUNT_MASKS_DIR:-/weiyang/share/250010212/StepCount-RL_Masks/masks}
export STEPCOUNT_MASK_REQUIRE=${STEPCOUNT_MASK_REQUIRE:-1}
export STEPCOUNT_MASK_PREFILL_BY_TURN=${STEPCOUNT_MASK_PREFILL_BY_TURN:-1}
export STEPCOUNT_MASK_LOG_CONFIG=${STEPCOUNT_MASK_LOG_CONFIG:-1}
export STEPCOUNT_MASK_DEBUG=${STEPCOUNT_MASK_DEBUG:-1}
export STEPCOUNT_MASK_DEBUG_EVERY=${STEPCOUNT_MASK_DEBUG_EVERY:-50}

MODEL_PATH=/weiyang/share/250010212/ckpt/StepCount-7b-SFT-10k-high-without-reasoning

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/weiyang/share/250010212/StepCountQA-RL-Easy_to_Hard_Small \
    data.val_files=/weiyang/share/250010212/StepCountQA-RL-Eval \
    data.format_prompt=./examples/format_prompt/StepCount_format.jinja \
    data.max_prompt_length=7500 \
    data.max_response_length=1500 \
    data.shuffle=true \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.optim.lr_warmup_ratio=0.1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=./examples/reward_function/StepCount_mask_reward.py:compute_score \
    trainer.experiment_name=StepCount-7B-SFT-10k-high_without_point_reasoning_easy_to_hard_small_StepCount_mask_reward_v4_grpo_n16_shuffle_$(date +%Y%m%d_%H%M) \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=1 \
    trainer.save_checkpoint_path=/weiyang/share/250010212/EasyR1/easy_r1/StepCount-7B-SFT-10k-high_without_point_reasoning_easy_to_hard_small_StepCount_mask_reward_v4_grpo_n16_shuffle_$(date +%Y%m%d_%H%M) \
    trainer.n_gpus_per_node=8 2>&1 | tee "${LOG_DIR}/training_without_point_reasoning_easy_to_hard_small_StepCount_mask_reward_v4_grpo_n16_shuffle_$(date +%Y%m%d_%H%M%S).log"