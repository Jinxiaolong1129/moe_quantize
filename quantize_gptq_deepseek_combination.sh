# !/bin/bash

export PYTHONPATH=auto_gptq/:$PYTHONPATH

bits=(
moe.shared_8.top25_4.other_2+other_block.4+alpha10
moe.shared_8.top25_4.other_2+other_block.4+alpha20
moe.shared_8.top25_4.other_2+other_block.4+alpha30
moe.shared_8.top25_4.other_2+other_block.4+alpha40
moe.shared_8.top25_4.other_2+other_block.4+alpha50
moe.shared_8.top25_4.other_2+other_block.4+alpha60
moe.shared_8.top25_4.other_2+other_block.4+alpha70
moe.shared_8.top25_4.other_2+other_block.4+alpha80
moe.shared_8.top25_4.other_2+other_block.4+alpha90
)

gpus=(3 4 5 6 7)
gpu_idx=0

for bit in "${bits[@]}"; do
    export CUDA_VISIBLE_DEVICES=${gpus[$gpu_idx]}
    nohup python quantize_gptq_deepseek_combination.py \
        --model_name deepseek-ai/deepseek-moe-16b-base \
        --nsamples 512 \
        --group_size 64 \
        --bits "$bit" > run_log/combination/log_cuda_${gpus[$gpu_idx]}_"$bit".out &
    
    let gpu_idx=(gpu_idx+1)%${#gpus[@]}
done
