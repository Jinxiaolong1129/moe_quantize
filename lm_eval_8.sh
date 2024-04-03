#!/bin/bash

export DEBUG=0
export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2

nohup python lm_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-base-gptq_w_bit_moe.shared_4.other.2+other_block.8 \
    --bits moe.shared_4.other.2+other_block.8 \
    --is_quantized > eval_output_moe_shared_4_other_2+other_block_8.log &

echo "moe_shared_4_other_2+other_block_8 done"


wait


nohup python lm_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-base-gptq_w_bit_moe.shared_2.other.4+other_block.8 \
    --bits moe.shared_2.other.4+other_block.8 \
    --is_quantized > eval_output_moe_shared_2_other_4+other_block_8.log &

echo "moe_shared_2_other_4+other_block_8 done"


wait


# ======= below finish =========


# nohup python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-base-gptq_w_bit_all_4 \
#     --bits all_4 \
#     --is_quantized > eval_output_all_4.log &

# echo "all_4 done"


# wait

# export DEBUG=0
# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=3,4

# nohup python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-base-gptq_w_bit_all_2 \
#     --bits all_2 \
#     --is_quantized > eval_output_all_2.log &

# echo "all_2 done"


# wait

# nohup python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-base-gptq_w_bit_all_8 \
#     --bits all_8 \
#     --is_quantized > eval_output_all_8.log &



# wait


# nohup python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-base-gptq_w_bit_moe.all_mlp.2+other_block.4 \
#     --bits moe.all_mlp.2+other_block.4 \
#     --is_quantized > eval_output_moe_all_mlp_2+other_block_4.log &

# echo "moe_all_mlp_2+other_block_4 done"


# wait


# nohup python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-base-gptq_w_bit_moe.shared_4.other.2+other_block_4 \
#     --bits moe.shared_4.other.2+other_block_4 \
#     --is_quantized > eval_output_moe_shared_4_other_2+other_block_4.log &

# echo "moe_shared_4_other_2+other_block_4 done"


# wait


# nohup python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-base-gptq_w_bit_moe.shared_2.other.4+other_block_4 \
#     --bits moe.shared_2.other.4+other_block_4 \
#     --is_quantized > eval_output_moe_shared_2_other_4+other_block_4.log &

# echo "moe_shared_2_other_4+other_block_4 done"


# wait


# nohup python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-base-gptq_w_bit_moe.all_mlp.4+other_block.8 \
#     --bits moe.all_mlp.4+other_block.8 \
#     --is_quantized > eval_output_moe_all_mlp_4+other_block_8.log &

# echo "moe_all_mlp_4+other_block_8 done"


# wait


#  model scale up
#  parameter scale up