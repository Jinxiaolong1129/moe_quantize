#!/bin/bash


export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/awq/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=5,6,7
nohup python lm_eval_awq.py \
    --model_path quantized_deepseek-moe-16b-base-awq-w_bit4-group_size64 \
    --is_quantized > lm_eval_awq_quantized_deepseek-moe-16b-base-awq-w_bit4-group_size64.log &

echo "quantized_deepseek-moe-16b-base-awq-w_bit4-group_size64"


wait


export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/awq/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=5,6,7
nohup python lm_eval_awq.py \
    --model_path quantized_deepseek-moe-16b-base-awq-w_bit2-group_size64 \
    --is_quantized > lm_eval_awq_quantized_deepseek-moe-16b-base-awq-w_bit2-group_size64.log &

echo "quantized_deepseek-moe-16b-base-awq-w_bit2-group_size64"

wait



export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/awq/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=5,6,7
nohup python lm_eval_awq.py \
    --model_path quantized_deepseek-moe-16b-base-awq-w_bit8-group_size64 \
    --is_quantized > lm_eval_awq_quantized_deepseek-moe-16b-base-awq-w_bit8-group_size64.log &

echo "quantized_deepseek-moe-16b-base-awq-w_bit8-group_size64"



