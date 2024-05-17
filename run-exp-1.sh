DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_0_4.layer_1_4.layer_2_4.layer_3_4.layer_4_4.layer_5_4.layer_6_4.layer_7_4 \
      --bits_name main_2.attn_4.first8_blocks

DEBUG=0 CUDA_VISIBLE_DEVICES=3,4,5 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.first8_blocks \
    --is_quantized

DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_24_4.layer_25_4.layer_26_4.layer_27_4.layer_28_4.layer_29_4.layer_30_4.layer_31_4 \
      --bits_name main_2.attn_4.last8_blocks

DEBUG=0 CUDA_VISIBLE_DEVICES=3,4,5 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.last8_blocks \
    --is_quantized