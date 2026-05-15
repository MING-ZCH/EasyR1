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

set -x

export PYTHONUNBUFFERED=1
export WANDB_API_KEY='5c554b68aed1a458465f44705a27102d8579c1cb'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/weiyang/250010212/checkpoints/StepCount-7b-SFT-16k/checkpoint-3099

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/weiyang/250010212/data/StepCountQA-RL@train \
    data.val_files=/weiyang/250010212/data/StepCountQA-RL@test \
    data.format_prompt=./examples/format_prompt/StepCount_format.jinja \
    data.max_prompt_length=12000 \
    data.max_response_length=1500 \
    data.shuffle=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=./examples/reward_function/StepCount.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_7b_StepCount_full_grpo_shuffle_$(date +%Y%m%d_%H%M) \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=8 2>&1 | tee "${LOG_DIR}/training_full_shuffle_$(date +%Y%m%d_%H%M%S).log"