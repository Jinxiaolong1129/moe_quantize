#!/bin/bash

# "moe.shared_2.other_2+other_block.4"
# "moe.shared_4.other_2+other_block.4"
# "moe.shared_4.top30_4.other_2+other_block.4"
# 'moe.shared_4.top25_4.other_2+other_block.4+startlayer_5'
# 'moe.shared_4.top25_4.other_2+other_block.4+dejavu_5'


export CUDA_VISIBLE_DEVICES=0
nohup python quantize_gptq_deepseek_layer.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits "moe.shared_2.other_2+other_block.4" \
    > run_log/pingzhi_exp_main_table/moe.shared_2.other_2+other_block.4.out &

export CUDA_VISIBLE_DEVICES=0
nohup python quantize_gptq_deepseek_layer.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits "moe.shared_4.other_2+other_block.4" \
    > run_log/pingzhi_exp_main_table/moe.shared_4.other_2+other_block.4.out &


export CUDA_VISIBLE_DEVICES=1
nohup python quantize_gptq_deepseek_layer.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits "moe.shared_4.top30_4.other_2+other_block.4" \
    > run_log/pingzhi_exp_main_table/moe.shared_4.top30_4.other_2+other_block.4.out &


export CUDA_VISIBLE_DEVICES=1
nohup python quantize_gptq_deepseek_layer.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits "moe.shared_4.top25_4.other_2+other_block.4+startlayer_5" \
    > run_log/pingzhi_exp_main_table/moe.shared_4.top25_4.other_2+other_block.4+startlayer_5.out &

export CUDA_VISIBLE_DEVICES=2
nohup python quantize_gptq_deepseek_layer.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits "moe.shared_4.top25_4.other_2+other_block.4+dejavu_5" \
    > run_log/pingzhi_exp_main_table/moe.shared_4.top25_4.other_2+other_block.4+dejavu_5.out &
