DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_2.exp_l20e5_4.exp_l31e4_4.exp_l21e2_4.exp_l31e3_4.exp_l5e7_4.exp_l4e5_4.exp_l10e7_4.exp_l28e4_4 \
      --bits_name main_2.attn_2.random_8_experts_seed44

DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_2.random_8_experts_seed44 \
    --is_quantized

DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_2.exp_l6e0_4.exp_l16e5_4.exp_l19e3_4.exp_l22e6_4.exp_l0e2_4.exp_l6e2_4.exp_l24e2_4.exp_l11e6_4.exp_l1e7_4.exp_l18e7_4.exp_l15e7_4.exp_l31e5_4.exp_l21e3_4.exp_l20e6_4.exp_l0e6_4.exp_l21e7_4.exp_l19e7_4.exp_l20e7_4.exp_l0e7_4.exp_l27e1_4.exp_l3e7_4.exp_l24e7_4.exp_l26e5_4 \
      --bits_name main_2.attn_2.random_23_experts_seed42

DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_2.random_23_experts_seed42 \
    --is_quantized