
#!/bin/bash
export PYTHONPATH=auto_gptq/:$PYTHONPATH

quant_model_path=(
    # NOTE running on 6,7,0
    'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+alpha10'
    'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+alpha20'
    'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+alpha30'
    'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+alpha40'
    'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+alpha50'

    # NOTE running on 3,4,5
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+alpha60'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+alpha70'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+alpha80'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+alpha90'
)

echo "Bash start running..."

# export CUDA_VISIBLE_DEVICES=3,4,5
export CUDA_VISIBLE_DEVICES=6,7,0

for i in "${!quant_model_path[@]}"; do
    echo "Running ${quant_model_path[$i]} on CUDA device"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

    nohup python lm_eval_combination.py \
        --model_name 'deepseek-ai/deepseek-moe-16b-base' \
        --quant_model_path autogptq_deepseek-ai/${quant_model_path[$i]} \
        --is_quantized > "run_log/gptq_eval_combination/log_${quant_model_path[$i]}.log" 2>&1

    wait
done







# export DEBUG=0
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0
# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-chat \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-chat-gptq_w_bit_moe.shared_4.other.2+other_block.8 \
#     --bits moe.shared_4.other.2+other_block.8 \
#     --is_quantized 


# export DEBUG=0
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0
# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-chat \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-chat-gptq_w_bit_moe.shared_2.other.4+other_block.8 \
#     --bits moe.shared_2.other.4+other_block.8 \
#     --is_quantized 

# autogptq_deepseek-ai