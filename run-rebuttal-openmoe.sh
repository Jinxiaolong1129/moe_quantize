#export PYTHONPATH=/data2/pzli/moe_quantize/optimum/onnxruntime/:/data2/pzli/moe_quantize/optimum/:/data2/pzli/moe_quantize/auto_gptq/:$PYTHONPATH
#
#DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_openmoe.py \
#      --model_name openmoe-checkpoints/openmoe-8b-native-pt \
#      --bits main_4.attn_4 \
#      --bits_name main_4.attn_4 &
##
### 2.41 bits
#DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_openmoe.py \
#      --model_name openmoe-checkpoints/openmoe-8b-native-pt \
#      --bits main_4.attn_4.layer_5_2.layer_11_2.layer_17_2.layer_23_2 \
#      --bits_name main_4.attn_4.layer_5_2.layer_11_2.layer_17_2.layer_23_2 &

# 3.20 bits
#DEBUG=0 CUDA_VISIBLE_DEVICES=2 python quantize_gptq_openmoe.py \
#      --model_name openmoe-checkpoints/openmoe-8b-native-pt \
#      --bits main_4.attn_4.layer_17_2.layer_23_2 \
#      --bits_name main_4.attn_4.first_2 &

#DEBUG=0 CUDA_VISIBLE_DEVICES=6 python quantize_gptq_openmoe.py \
#      --model_name openmoe-checkpoints/openmoe-8b-native-pt \
#      --bits main_4.attn_4.layer_5_2.layer_11_2.layer_23_2 \
#      --bits_name main_4.attn_4.predicted_1 &

#DEBUG=0 CUDA_VISIBLE_DEVICES=7 python quantize_gptq_openmoe.py \
#      --model_name openmoe-checkpoints/openmoe-8b-native-pt \
#      --bits main_4.attn_4.layer_11_2.layer_17_2.layer_23_2 \
#      --bits_name main_4.attn_4.first_1  &

#DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_openmoe.py \
#      --model_name openmoe-checkpoints/openmoe-8b-native-pt \
#      --bits main_4.attn_4.layer_5_2.layer_23_2 \
#      --bits_name main_4.attn_4.predicted_2  &
#
#DEBUG=0 CUDA_VISIBLE_DEVICES=6,7 python lm_eval_gptq_openmoe.py \
#    --model_name openmoe-checkpoints/openmoe-8b-native-pt \
#    --quant_model_path autogptq_openmoe-checkpoints/openmoe-8b-native-pt-gptq_w_bit_main_4.attn_4.first_2 \
#    --is_quantized &
#
#DEBUG=0 CUDA_VISIBLE_DEVICES=6,7 python lm_eval_gptq_openmoe.py \
#    --model_name openmoe-checkpoints/openmoe-8b-native-pt \
#    --quant_model_path autogptq_openmoe-checkpoints/openmoe-8b-native-pt-gptq_w_bit_main_4.attn_4.predicted_2 \
#    --is_quantized &

#DEBUG=0 CUDA_VISIBLE_DEVICES=5,6,7 python lm_eval_gptq_openmoe.py \
#    --model_name openmoe-checkpoints/openmoe-8b-native-pt \
#    --quant_model_path autogptq_openmoe-checkpoints/openmoe-8b-native-pt-gptq_w_bit_main_4.attn_4.first_1 \
#    --is_quantized
#
#DEBUG=0 CUDA_VISIBLE_DEVICES=5,6,7 python lm_eval_gptq_openmoe.py \
#    --model_name openmoe-checkpoints/openmoe-8b-native-pt \
#    --quant_model_path autogptq_openmoe-checkpoints/openmoe-8b-native-pt-gptq_w_bit_main_4.attn_4.predicted_1 \
#    --is_quantized

# todo: (without MMLU)
#   1) 800B model evaluation on 2&4 bits MoE;
#   2) full model evaluation on 2&4 bits MoE;
#   3) collect group size 64 results

DEBUG=0 CUDA_VISIBLE_DEVICES=0,1 python lm_eval_gptq_openmoe.py \
    --model_name openmoe-checkpoints/openmoe-8b-native-pt \
    --quant_model_path autogptq_openmoe-checkpoints/openmoe-8b-native-pt-gptq_w_bit_main_4.attn_4 \
    --is_quantized &

DEBUG=0 CUDA_VISIBLE_DEVICES=2,3 python lm_eval_gptq_openmoe.py \
    --model_name openmoe-checkpoints/openmoe-8b-native-pt \
    --quant_model_path autogptq_openmoe-checkpoints/openmoe-8b-native-pt-gptq_w_bit_main_4.attn_4.layer_5_2.layer_11_2.layer_17_2.layer_23_2 \
    --is_quantized &