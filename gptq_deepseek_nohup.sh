#!/bin/bash

export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH

export DEBUG=0

# Command 1
export CUDA_VISIBLE_DEVICES=3
nohup python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits all_4 > log_cuda_3_all_4.out &

# Command 2
export CUDA_VISIBLE_DEVICES=4
nohup python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits all_2 > log_cuda_4_all_2.out &

# Command 3
export CUDA_VISIBLE_DEVICES=5
nohup python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits moe.all_mlp.2+other_block.4 > log_cuda_5_moe.all_mlp.2+other_block.4.out &

# Command 4
export CUDA_VISIBLE_DEVICES=6
nohup python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits moe.shared_4.other.2+other_block_4 > log_cuda_6_moe.shared_4.other.2+other_block_4.out &

# Command 5
export CUDA_VISIBLE_DEVICES=7
nohup python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits moe.shared_2.other.4+other_block_4 > log_cuda_7_moe.shared_2.other.4+other_block_4.out &

# Command 6
export CUDA_VISIBLE_DEVICES=7
nohup python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits all_8 > log_cuda_7_all_8.out &

# Command 7
export CUDA_VISIBLE_DEVICES=6
nohup python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits moe.all_mlp.4+other_block.8 > log_cuda_6_moe.all_mlp.4+other_block.8.out &

# Command 8
export CUDA_VISIBLE_DEVICES=5
nohup python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits moe.shared_4.other.2+other_block.8 > log_cuda_5_moe.shared_4.other.2+other_block.8.out &

# Command 9
export CUDA_VISIBLE_DEVICES=4
nohup python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits moe.shared_2.other.4+other_block.8 > log_cuda_4_moe.shared_2.other.4+other_block.8.out &






# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/awq/:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=5
# nohup python quantize_awq_deepseek.py --bits 4 > quantize_awq_deepseek_bit.4.log &



# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/awq/:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=6
# nohup python quantize_awq_deepseek.py --bits 2 > quantize_awq_deepseek_bit.2.log &



# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/awq/:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=7
# nohup python quantize_awq_deepseek.py --bits 8 > quantize_awq_deepseek_bit.8.log &



export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4
nohup python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --nsamples 512 \
    --group_size 64 \
    --bits moe.shared_8.top1_8.other_2+other_block.4 > run_log/log_cuda_4_moe.shared_8.top1_8.other_2+other_block.4.out &
