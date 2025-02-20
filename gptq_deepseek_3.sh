
#!/bin/bash

# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=5,6,7
# export DEBUG=0

# python quantize_gptq_deepseek.py \
#     --model_name deepseek-ai/deepseek-moe-16b-base \
#     --nsamples 512 \
#     --group_size 64 \
#     --bits all_3


# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=5,6,7
export DEBUG=0

nohup python lm_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-base \
    --quant_model_path autogptq_deepseek-ai/deepseek-moe-16b-base-gptq_w_bit_moe.shared_8.top25_4.other_2+other_block.4+startlayer_20 \
    --bits moe.shared_8.top25_4.other_2+other_block.4+startlayer_20 \
    --is_quantized >> eval_output_moe_moe.shared_8.top25_4.other_2+other_block.4+startlayer_20.log &

echo "moe.shared_8.top25_4.other_2+other_block.4+startlayer_20 done"

