DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4 \
      --bits_dict_overwrite results/top_0.25_linear_weight.pt \
      --bits_name main_2.attn_4.top_0.25_linear_weight &

DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4 \
      --bits_dict_overwrite results/random_0.25_linear_weight_seed42.pt \
      --bits_name main_2.attn_4.random_0.25_linear_weight_seed42 &

wait

DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.top_0.25_linear_weight \
    --is_quantized &

DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4 \
      --bits_dict_overwrite results/random_0.25_linear_weight_seed43.pt \
      --bits_name main_2.attn_4.random_0.25_linear_weight_seed43 &

wait

DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.random_0.25_linear_weight_seed42 \
    --is_quantized &

DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4 \
      --bits_dict_overwrite results/random_0.25_linear_weight_seed44.pt \
      --bits_name main_2.attn_4.random_0.25_linear_weight_seed44 &

wait

DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.random_0.25_linear_weight_seed43 \
    --is_quantized

DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.random_0.25_linear_weight_seed44 \
    --is_quantized