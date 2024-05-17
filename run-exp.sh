DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_0_4.layer_1_4.layer_2_4.layer_3_4 \
      --bits_name main_2.attn_4.first4_blocks

DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.first4_blocks \
    --is_quantized

DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_28_4.layer_29_4.layer_30_4.layer_31_4 \
      --bits_name main_2.attn_4.last4_blocks

DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.last4_blocks \
    --is_quantized