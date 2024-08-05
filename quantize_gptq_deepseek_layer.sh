# !/bin/bash

export PYTHONPATH=auto_gptq/:$PYTHONPATH

bits=(
moe.shared_4.other_4+other_block.4+startlayer_4
moe.shared_4.other_4+other_block.4+startlayer_8

moe.shared_4.other_4+other_block.4+endlayer_4
moe.shared_4.other_4+other_block.4+endlayer_8
)

gpus=(1 2 4 5)
gpu_idx=0

for bit in "${bits[@]}"; do
    export CUDA_VISIBLE_DEVICES=${gpus[$gpu_idx]}
    nohup python quantize_gptq_deepseek_layer.py \
        --model_name deepseek-ai/deepseek-moe-16b-base \
        --nsamples 512 \
        --group_size 64 \
        --bits "$bit" > run_log/desc_act_True/log_cuda_${gpus[$gpu_idx]}_"$bit".out &
    
    let gpu_idx=(gpu_idx+1)%${#gpus[@]}
done
