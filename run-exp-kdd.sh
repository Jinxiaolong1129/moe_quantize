DEBUG=0 CUDA_VISIBLE_DEVICES=0,3,4,5,6,7 CUDA_LAUNCH_BLOCKING=1 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.frequency_top5_per_layer \
    --is_quantized

DEBUG=0 CUDA_VISIBLE_DEVICES=0,3,4,5,6,7 CUDA_LAUNCH_BLOCKING=1 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.frequency_top6_per_layer \
    --is_quantized