DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_2.exp_l4e0_4.exp_l10e0_4.exp_l3e6_4.exp_l7e1_4.exp_l8e5_4.exp_l23e6_4.exp_l24e6_4.exp_l16e5_4.exp_l25e5_4.exp_l26e4_4.exp_l13e7_4.exp_l8e6_4.exp_l2e1_4.exp_l1e3_4.exp_l0e7_4.exp_l11e6_4.exp_l28e6_4.exp_l16e4_4.exp_l21e2_4.exp_l16e5_4.exp_l27e2_4.exp_l0e7_4.exp_l26e6_4 \
      --bits_name main_2.attn_2.random_23_experts_seed43

DEBUG=0 CUDA_VISIBLE_DEVICES=3,4,5 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_2.random_23_experts_seed43 \
    --is_quantized

DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_2.exp_l20e5_4.exp_l31e4_4.exp_l21e2_4.exp_l31e3_4.exp_l5e7_4.exp_l4e5_4.exp_l10e7_4.exp_l28e4_4.exp_l22e1_4.exp_l22e3_4.exp_l18e5_4.exp_l21e1_4.exp_l20e3_4.exp_l2e7_4.exp_l9e6_4.exp_l11e7_4.exp_l19e4_4.exp_l21e5_4.exp_l0e0_4.exp_l27e7_4.exp_l24e7_4.exp_l30e5_4.exp_l5e1_4 \
      --bits_name main_2.attn_2.random_23_experts_seed44

DEBUG=0 CUDA_VISIBLE_DEVICES=3,4,5 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_2.random_23_experts_seed44 \
    --is_quantized