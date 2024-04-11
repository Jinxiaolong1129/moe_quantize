#!/bin/bash

export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH

topx_values=(1 2 5 10 15 20 25 30 35 40)

# gpus=(4 5 6 7)
gpus=(0 1 2 3)
gpu_idx=0
task_count=0

for topx in "${topx_values[@]}"; do
    bit="moe.shared_4.top${topx}_8.other_2+other_block.8"
    
    # Assign a GPU from the list
    export CUDA_VISIBLE_DEVICES=${gpus[$gpu_idx]}
    
    # Run the experiment with nohup
    nohup python quantize_gptq_deepseek.py \
        --model_name deepseek-ai/deepseek-moe-16b-base \
        --nsamples 512 \
        --group_size 64 \
        --bits "$bit" > run_log/gptq/log_${bit}.out &
    
    echo "Started experiment with top${topx} | bits ${bit} on GPU ${gpus[$gpu_idx]}"
    
    # Increment and cycle the gpu_idx to use the next GPU
    let gpu_idx=(gpu_idx+1)%${#gpus[@]}
    let task_count=(task_count+1)
    
    # If the number of tasks equals the number of GPUs, wait for the tasks to complete
    if [ "$task_count" -eq "${#gpus[@]}" ]; then
        wait # Wait for all background jobs to finish
        task_count=0 # Reset task counter
    fi
done

wait
