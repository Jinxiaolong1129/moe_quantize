#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
nohup python quantize_gptq_deepseek_layer.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits "moe.shared_4.other_2+other_block.4" \
    > run_log/pingzhi_exp1/moe.shared_4.other_2+other_block.4.out &


export CUDA_VISIBLE_DEVICES=5
nohup python quantize_gptq_deepseek_layer.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits "moe.shared_8.other_2+other_block.4" \
    > run_log/pingzhi_exp1/moe.shared_8.other_2+other_block.4.out &


export CUDA_VISIBLE_DEVICES=6
nohup python quantize_gptq_deepseek_layer.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits "moe.shared_3.other_2+other_block.4" \
    > run_log/pingzhi_exp1/moe.shared_3.other_2+other_block.4.out &


export CUDA_VISIBLE_DEVICES=7
nohup python quantize_gptq_deepseek_layer.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits "moe.shared_2.other_2+other_block.4" \
    > run_log/pingzhi_exp1/moe.shared_2.other_2+other_block.4.out &