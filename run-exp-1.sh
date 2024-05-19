DEBUG=0 CUDA_VISIBLE_DEVICES=3,4,5 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.top8_cos_sim \
    --is_quantized &

DEBUG=0 CUDA_VISIBLE_DEVICES=7 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_6_4.layer_13_4.layer_18_4.layer_24_4.layer_22_4.layer_7_4.layer_28_4.layer_31_4 \
      --bits_name main_2.attn_4.random_8_layers_seed42 &

wait

DEBUG=0 CUDA_VISIBLE_DEVICES=3,4,5 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.random_8_layers_seed42 \
    --is_quantized &

DEBUG=0 CUDA_VISIBLE_DEVICES=7 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_4_4.layer_23_4.layer_7_4.layer_5_4.layer_29_4.layer_2_4.layer_14_4.layer_13_4 \
      --bits_name main_2.attn_4.random_8_layers_seed43 &

wait

DEBUG=0 CUDA_VISIBLE_DEVICES=3,4,5 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.random_8_layers_seed43 \
    --is_quantized &

DEBUG=0 CUDA_VISIBLE_DEVICES=7 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_20_4.layer_15_4.layer_9_4.layer_19_4.layer_3_4.layer_5_4.layer_16_4.layer_8_4 \
      --bits_name main_2.attn_4.random_8_layers_seed44 &

wait

DEBUG=0 CUDA_VISIBLE_DEVICES=3,4,5 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.random_8_layers_seed44 \
    --is_quantized