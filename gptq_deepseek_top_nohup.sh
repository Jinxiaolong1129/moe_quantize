#!/bin/bash

# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH

# bits=(
#     "moe.shared_8.top1_8.other_2+other_block.4"
#     "moe.shared_8.top1_4.other_2+other_block.4"
#     "moe.shared_8.top2_8.other_2+other_block.4"
#     "moe.shared_8.top2_4.other_2+other_block.4"
#     "moe.shared_4.top2_8.other_2+other_block.4"
#     "moe.shared_4.top2_4.other_2+other_block.4"
#     "moe.shared_8.top1_8.other_2+other_block.8"
#     "moe.shared_8.top1_4.other_2+other_block.8"
#     "moe.shared_8.top2_8.other_2+other_block.8"
#     "moe.shared_8.top2_4.other_2+other_block.8"
#     "moe.shared_4.top2_8.other_2+other_block.8"
#     "moe.shared_4.top2_4.other_2+other_block.8"
# )

# gpus=(4 5 6 7)
# gpu_idx=0

# for bit in "${bits[@]}"; do
#     export CUDA_VISIBLE_DEVICES=${gpus[$gpu_idx]}
#     nohup python quantize_gptq_deepseek.py \
#         --model_name deepseek-ai/deepseek-moe-16b-base \
#         --nsamples 512 \
#         --group_size 64 \
#         --bits "$bit" > run_log/log_cuda_${gpus[$gpu_idx]}_"$bit".out &
    
#     # Increment and cycle the gpu_idx to use the next GPU
#     let gpu_idx=(gpu_idx+1)%${#gpus[@]}
# done


# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH

# export CUDA_VISIBLE_DEVICES=4
# nohup python quantize_gptq_deepseek.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --nsamples 512 \
#     --group_size 64 \
#     --bits moe.shared_8.top1_8.other_2+other_block.4 > run_log/log_cuda_4_moe.shared_8.top1_8.other_2+other_block.4.out &

# export CUDA_VISIBLE_DEVICES=5
# nohup python quantize_gptq_deepseek.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --nsamples 512 \
#     --group_size 64 \
#     --bits moe.shared_8.top1_4.other_2+other_block.4 > run_log/log_cuda_5_moe.shared_8.top1_4.other_2+other_block.4.out &

# export CUDA_VISIBLE_DEVICES=6
# nohup python quantize_gptq_deepseek.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --nsamples 512 \
#     --group_size 64 \
#     --bits moe.shared_8.top2_8.other_2+other_block.4 > run_log/log_cuda_6_moe.shared_8.top2_8.other_2+other_block.4.out &

# export CUDA_VISIBLE_DEVICES=7
# nohup python quantize_gptq_deepseek.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --nsamples 512 \
#     --group_size 64 \
#     --bits moe.shared_8.top2_4.other_2+other_block.4 > run_log/log_cuda_7_moe.shared_8.top2_4.other_2+other_block.4.out &

# export CUDA_VISIBLE_DEVICES=0
# nohup python quantize_gptq_deepseek.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --nsamples 512 \
#     --group_size 64 \
#     --bits moe.shared_4.top2_8.other_2+other_block.4 > run_log/log_cuda_0_moe.shared_4.top2_8.other_2+other_block.4.out &


# export CUDA_VISIBLE_DEVICES=0
# nohup python quantize_gptq_deepseek.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --nsamples 512 \
#     --group_size 64 \
#     --bits moe.shared_4.top2_4.other_2+other_block.4 > run_log/log_cuda_0_moe.shared_4.top2_4.other_2+other_block.4.out &

# export CUDA_VISIBLE_DEVICES=6
# nohup python quantize_gptq_deepseek.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --nsamples 512 \
#     --group_size 64 \
#     --bits moe.shared_8.top1_8.other_2+other_block.8 > run_log/log_cuda_6_moe.shared_8.top1_8.other_2+other_block.8.out &

# export CUDA_VISIBLE_DEVICES=7
# nohup python quantize_gptq_deepseek.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --nsamples 512 \
#     --group_size 64 \
#     --bits moe.shared_8.top1_4.other_2+other_block.8 > run_log/log_cuda_7_moe.shared_8.top1_4.other_2+other_block.8.out &

# export CUDA_VISIBLE_DEVICES=4
# nohup python quantize_gptq_deepseek.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --nsamples 512 \
#     --group_size 64 \
#     --bits moe.shared_8.top2_8.other_2+other_block.8 > run_log/log_cuda_4_moe.shared_8.top2_8.other_2+other_block.8.out &

# export CUDA_VISIBLE_DEVICES=5
# nohup python quantize_gptq_deepseek.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --nsamples 512 \
#     --group_size 64 \
#     --bits moe.shared_8.top2_4.other_2+other_block.8 > run_log/log_cuda_5_moe.shared_8.top2_4.other_2+other_block.8.out &


# # ===============
# # # no run
# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=1
# nohup python quantize_gptq_deepseek.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --nsamples 512 \
#     --group_size 64 \
#     --bits moe.shared_4.top2_8.other_2+other_block.8 > run_log/log_cuda_6_moe.shared_4.top2_8.other_2+other_block.8.out &

# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=2
# nohup python quantize_gptq_deepseek.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --nsamples 512 \
#     --group_size 64 \
#     --bits moe.shared_4.top2_4.other_2+other_block.8 > run_log/log_cuda_7_moe.shared_4.top2_4.other_2+other_block.8.out &
