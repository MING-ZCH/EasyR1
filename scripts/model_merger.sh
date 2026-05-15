#!/bin/bash
set -x

LOG_DIR="/mnt/shared-storage-user/zhangchenhao/work/MetaphorStar/logs/merge"
mkdir -p ${LOG_DIR}
export PYTHONUNBUFFERED=1

MODEL_PATH=/mnt/shared-storage-user/zhangchenhao/work/EasyR1/checkpoints/easy_r1/qwen25_vl_7b_II-Bench_reasoning_grpo_format03/global_step_50/actor
python3 /mnt/shared-storage-user/zhangchenhao/work/EasyR1/scripts/model_merger.py \
    --local_dir ${MODEL_PATH} \
    2>&1 | tee "${LOG_DIR}/qwen25_vl_7b_II-Bench_reasoning_grpo_format03_step_50.log"
