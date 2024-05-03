
#!/bin/bash

export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/lm_eval/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/lm_eval/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH

quant_model_path=(
    # NOTE running on 0,1,2
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_4.top2_4.other_2+other_block.4'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_4.top2_8.other_2+other_block.8'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top1_8.other_2+other_block.8'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top2_8.other_2+other_block.8'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_4.top2_4.other_2+other_block.8'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top2_4.other_2+other_block.8'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top2_8.other_2+other_block.4'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top2_4.other_2+other_block.4'

    # NOTE running on 3,4,5
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top1_8.other_2+other_block.4'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top1_4.other_2+other_block.4'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top1_4.other_2+other_block.8'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_4.top2_8.other_2+other_block.4'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top5_4.other_2+other_block.4'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top10_4.other_2+other_block.4'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top15_4.other_2+other_block.4'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top20_4.other_2+other_block.4'

    # NOTE running on 6,7
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top30_4.other_2+other_block.4'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top35_4.other_2+other_block.4'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top45_4.other_2+other_block.4'
    

    # NOTE running on 0,1,2
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top15_4.other_2+other_block.8'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top2_8.other_2+other_block.8'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top5_4.other_2+other_block.8'

    # NOTE running on 3,4,5
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.8'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top45_4.other_2+other_block.8'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top35_4.other_2+other_block.8'

    # # NOTE running on 6,7
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top20_4.other_2+other_block.8'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top30_4.other_2+other_block.8'
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top10_4.other_2+other_block.8'


    # NOTE running on 0,2,3
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+endlayer_25' 
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+startlayer_25' 
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+startlayer_15' 
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+endlayer_15' 

    # NOTE running on 4,6,7
    'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+endlayer_27' 
    'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+startlayer_27' 
    'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+endlayer_5' 
    'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+startlayer_5'
    
    
    # NOTE running on 3,4,5
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+endlayer_10' 
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+startlayer_10' 
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+endlayer_20' 
    # 'deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+startlayer_20' 

)

echo "Bash start running..."

export CUDA_VISIBLE_DEVICES=4,6,7

for i in "${!quant_model_path[@]}"; do
    echo "Running ${quant_model_path[$i]} on CUDA device"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

    nohup python lm_eval.py \
        --model_name 'deepseek-ai/deepseek-moe-16b-base' \
        --quant_model_path autogptq_deepseek-ai/${quant_model_path[$i]} \
        --is_quantized > "run_log/gptq_eval/log_${quant_model_path[$i]}.log" 2>&1

    wait
done


# # Run commands
# for i in "${!quant_model_path[@]}"; do
#     export CUDA_VISIBLE_DEVICES=$((i % 8))
#     nohup python lm_eval.py \
#         --quant_model_path autogptq_deepseek-ai/${quant_model_path[$i]} \
#         --bits ${bits[$i]} \
#         --is_quantized > "output_${bits[$i]}.log" 2>&1 &
#     echo "Running ${quant_model_path[$i]} with ${bits[$i]}"
# done



# export DEBUG=0
# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0
# python lm_eval.py \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-chat-gptq_w_bit_all_4 \
#     --bits all_4 \
#     --is_quantized 


# export DEBUG=0
# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0
# python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-chat \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-chat-gptq_w_bit_all_2 \
#     --bits all_2 \
#     --is_quantized 



# export DEBUG=0
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0
# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-chat \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-chat-gptq_w_bit_moe.all_mlp.2+other_block.4 \
#     --bits moe.all_mlp.2+other_block.4 \
#     --is_quantized 



# export DEBUG=0
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0
# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-chat \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-chat-gptq_w_bit_moe.shared_4.other.2+other_block_4 \
#     --bits moe.shared_4.other.2+other_block_4 \
#     --is_quantized 



# # no finish
# export DEBUG=0
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0
# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-chat \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-chat-gptq_w_bit_moe.shared_2.other.4+other_block_4 \
#     --bits moe.shared_2.other.4+other_block_4 \
#     --is_quantized 



# export DEBUG=0
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0
# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-chat \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-chat-gptq_w_bit_all_8 \
#     --bits all_8 \
#     --is_quantized 


# export DEBUG=0
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0
# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# python lm_eval.py \
#     --model_name deepseek-ai/deepseek-moe-16b-chat \
#     --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-chat-gptq_w_bit_moe.all_mlp.4+other_block.8 \
#     --bits moe.all_mlp.4+other_block.8 \
#     --is_quantized 


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