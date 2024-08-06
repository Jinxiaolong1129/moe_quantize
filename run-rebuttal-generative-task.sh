DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e5_4.exp_l0e2_4.exp_l1e4_4.exp_l1e3_4.exp_l2e0_4.exp_l2e1_4.exp_l3e2_4.exp_l3e5_4.exp_l4e3_4.exp_l4e2_4.exp_l5e1_4.exp_l5e2_4.exp_l6e5_4.exp_l6e0_4.exp_l7e5_4.exp_l7e6_4.exp_l8e5_4.exp_l8e0_4.exp_l9e0_4.exp_l9e3_4.exp_l10e1_4.exp_l10e2_4.exp_l11e1_4.exp_l11e4_4.exp_l12e1_4.exp_l12e2_4.exp_l13e7_4.exp_l13e4_4.exp_l14e4_4.exp_l14e5_4.exp_l15e1_4.exp_l15e2_4.exp_l16e7_4.exp_l16e6_4.exp_l17e2_4.exp_l17e5_4.exp_l18e0_4.exp_l18e2_4.exp_l19e0_4.exp_l19e4_4.exp_l20e6_4.exp_l20e7_4.exp_l21e6_4.exp_l21e3_4.exp_l22e0_4.exp_l22e3_4.exp_l23e1_4.exp_l23e0_4.exp_l24e2_4.exp_l24e1_4.exp_l25e1_4.exp_l25e5_4.exp_l26e3_4.exp_l26e2_4.exp_l27e5_4.exp_l27e0_4.exp_l28e1_4.exp_l28e3_4.exp_l29e2_4.exp_l29e6_4.exp_l30e7_4.exp_l30e4_4.exp_l31e7_4.exp_l31e3_4 \
      --bits_name main_2.attn_4.frequency_top2_per_layer &

DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e6_4.exp_l0e3_4.exp_l1e4_4.exp_l1e1_4.exp_l2e7_4.exp_l2e3_4.exp_l3e4_4.exp_l3e1_4.exp_l4e7_4.exp_l4e3_4.exp_l5e4_4.exp_l5e5_4.exp_l6e1_4.exp_l6e6_4.exp_l7e5_4.exp_l7e2_4.exp_l8e7_4.exp_l8e6_4.exp_l9e0_4.exp_l9e6_4.exp_l10e1_4.exp_l10e5_4.exp_l11e3_4.exp_l11e7_4.exp_l12e6_4.exp_l12e3_4.exp_l13e3_4.exp_l13e2_4.exp_l14e5_4.exp_l14e7_4.exp_l15e1_4.exp_l15e0_4.exp_l16e7_4.exp_l16e2_4.exp_l17e4_4.exp_l17e6_4.exp_l18e1_4.exp_l18e3_4.exp_l19e4_4.exp_l19e7_4.exp_l20e7_4.exp_l20e4_4.exp_l21e2_4.exp_l21e7_4.exp_l22e4_4.exp_l22e3_4.exp_l23e0_4.exp_l23e1_4.exp_l24e6_4.exp_l24e0_4.exp_l25e5_4.exp_l25e1_4.exp_l26e5_4.exp_l26e0_4.exp_l27e5_4.exp_l27e2_4.exp_l28e1_4.exp_l28e7_4.exp_l29e0_4.exp_l29e7_4.exp_l30e6_4.exp_l30e1_4.exp_l31e4_4.exp_l31e3_4 \
      --bits_name main_2.attn_4.random2_per_layer_seed42 &

DEBUG=0 CUDA_VISIBLE_DEVICES=2,3,4 python lm_eval_gptq-generative.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 &