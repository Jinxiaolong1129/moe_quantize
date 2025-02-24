export HF_HOME=./cache/huggingface

DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_0_4.layer_1_4.layer_2_4.layer_3_4.keyword__w2__4 \
      --bits_name main_2.attn_4.first_4_layer_and_all_w2 &

DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_0_4.layer_1_4.layer_2_4.layer_3_4.layer_4_4.layer_5_4.keyword__w2__4 \
      --bits_name main_2.attn_4.first_6_layer_and_all_w2 &

wait

DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_LAUNCH_BLOCKING=1 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.first_4_layer_and_all_w2 \
    --is_quantized

DEBUG=0 CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 CUDA_LAUNCH_BLOCKING=1 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.first_6_layer_and_all_w2 \
    --is_quantized