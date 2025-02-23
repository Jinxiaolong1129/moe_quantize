#!/bin/bash

# "moe.shared_4.other_2+other_block.4+alpha50" -> 3.06
# "moe.shared_4.other_2+other_block.4+alpha55" -> 3.15
# "moe.shared_4.other_2+other_block.4+alpha60" -> 3.25


export CUDA_VISIBLE_DEVICES=0
nohup python quantize_gptq_deepseek_layer.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits "moe.shared_4.other_2+other_block.4+alpha50" \
    > run_log/pingzhi_exp_main_table/moe.shared_4.other_2+other_block.4+alpha50.out &

export CUDA_VISIBLE_DEVICES=1
nohup python quantize_gptq_deepseek_layer.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits "moe.shared_4.other_2+other_block.4+alpha55" \
    > run_log/pingzhi_exp_main_table/moe.shared_4.other_2+other_block.4+alpha55.out &

export CUDA_VISIBLE_DEVICES=2
nohup python quantize_gptq_deepseek_layer.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits "moe.shared_4.other_2+other_block.4+alpha60" \
    > run_log/pingzhi_exp_main_table/moe.shared_4.other_2+other_block.4+alpha60.out &
