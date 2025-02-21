#!/bin/bash

quant_model_path=(
    'deepseek-moe-16b-base-gptq_w_bit_moe.shared_2.other_2+other_block.4'
    'deepseek-moe-16b-base-gptq_w_bit_moe.shared_3.other_2+other_block.4'
    'deepseek-moe-16b-base-gptq_w_bit_moe.shared_4.other_2+other_block.4'
    'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.other_2+other_block.4'
)

echo "Bash start running..."

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

for i in "${!quant_model_path[@]}"; do
    echo "Running ${quant_model_path[$i]} on CUDA device"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

    nohup python lm_eval.py \
        --model_name 'deepseek-ai/deepseek-moe-16b-base' \
        --quant_model_path autogptq_deepseek-ai/${quant_model_path[$i]} \
        --is_quantized > "run_log/gptq_eval/log_${quant_model_path[$i]}.log" 2>&1

    wait
done

