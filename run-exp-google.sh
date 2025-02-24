export HF_HOME=./cache/huggingface

DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2,3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_0_4.layer_1_4.layer_2_4.layer_3_4.layer_4_4.layer_5_4.layer_6_4.layer_7_4.layer_8_4.layer_9_4.layer_10_4.layer_11_4.keyword__w2__4 \
      --bits_name main_2.attn_4.first_12_layer_and_all_w2 &

DEBUG=0 CUDA_VISIBLE_DEVICES=4,5,6,7 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_0_4.layer_1_4.layer_2_4.layer_3_4.layer_4_4.layer_5_4.layer_6_4.layer_7_4.layer_8_4.layer_9_4.layer_10_4.layer_11_4.layer_12_4.layer_13_4.layer_14_4.layer_15_4.keyword__w2__4 \
      --bits_name main_2.attn_4.first_16_layer_and_all_w2 &

wait

DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_LAUNCH_BLOCKING=1 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.first_12_layer_and_all_w2 \
    --is_quantized

DEBUG=0 CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 CUDA_LAUNCH_BLOCKING=1 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.first_16_layer_and_all_w2 \
    --is_quantized