

DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_16.attn_16 \
      --bits_name main_16.attn_16

DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_16.attn_16 \
    --is_quantized &
