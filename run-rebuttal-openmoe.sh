#export PYTHONPATH=/data2/pzli/moe_quantize/optimum/onnxruntime/:/data2/pzli/moe_quantize/optimum/:/data2/pzli/moe_quantize/auto_gptq/:$PYTHONPATH

#DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
#      --model_name mistralai/Mixtral-8x7B-v0.1 \
#      --bits main_16.attn_16 \
#      --bits_name main_16.attn_16
#
#DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2 python lm_eval_gptq.py \
#    --model_name mistralai/Mixtral-8x7B-v0.1 \
#    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_16.attn_16 \
#    --is_quantized &
#DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2 python lm_eval_gptq.py \
#    --model_name mistralai/Mixtral-8x7B-v0.1


DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_openmoe.py \
      --model_name OrionZheng/openmoe-8b \
      --bits main_4.attn_4 \
      --bits_name main_4.attn_4 &

# 2.41 bits
DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_openmoe.py \
      --model_name OrionZheng/openmoe-8b \
      --bits main_4.attn_4.layer_5_2.layer_11_2.layer_17_2.layer_23_2 \
      --bits_name main_4.attn_4.layer_5_2.layer_11_2.layer_17_2.layer_23_2

# todo

DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_openmoe.py \
      --model_name OrionZheng/openmoe-8b \
      --bits main_4.attn_4.layer_17_2.layer_23_2 \
      --bits_name main_4.attn_4.first_2 &

DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_openmoe.py \
      --model_name OrionZheng/openmoe-8b \
      --bits main_4.attn_4.layer_5_2.layer_23_2 \
      --bits_name main_4.attn_4.predicted_2  &

DEBUG=0 CUDA_VISIBLE_DEVICES=2,3 python lm_eval_gptq_openmoe.py \
    --model_name OrionZheng/openmoe-8b \
    --quant_model_path autogptq_OrionZheng/openmoe-8b-gptq_w_bit_main_4.attn_4 \
    --is_quantized &

DEBUG=0 CUDA_VISIBLE_DEVICES=4,5 python lm_eval_gptq_openmoe.py \
    --model_name OrionZheng/openmoe-8b \
    --quant_model_path autogptq_OrionZheng/openmoe-8b-gptq_w_bit_main_4.attn_4.layer_5_2.layer_11_2.layer_17_2.layer_23_2 \
    --is_quantized &