device_idx=0

# MOST USED
for bits in 2 4
do
    DEBUG=0 CUDA_VISIBLE_DEVICES=$device_idx python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_$bits.exp_l0e5_8.exp_l1e4_8.exp_l2e0_8.exp_l3e2_8.exp_l4e3_8.exp_l5e1_8.exp_l6e5_8.exp_l7e5_8.exp_l8e5_8.exp_l9e0_8.exp_l10e1_8.exp_l11e1_8.exp_l12e1_8.exp_l13e7_8.exp_l14e4_8.exp_l15e1_8.exp_l16e7_8.exp_l17e2_8.exp_l18e0_8.exp_l19e0_8.exp_l20e6_8.exp_l21e6_8.exp_l22e0_8.exp_l23e1_8.exp_l24e2_8.exp_l25e1_8.exp_l26e3_8.exp_l27e5_8.exp_l28e1_8.exp_l29e2_8.exp_l30e7_8.exp_l31e7_8 \
      --bits_name main_$bits.most_used &

    device_idx=$((device_idx+1))
done

# ONE KEY EXPERT
for bits in 2 4
do
    DEBUG=0 CUDA_VISIBLE_DEVICES=$device_idx python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_$bits.exp_l1e3_8 \
      --bits_name main_$bits.exp_l1e3_8 &

    device_idx=$((device_idx+1))
done

# MASSIVE EXPERTS
for bits in 2 4
do
    DEBUG=0 CUDA_VISIBLE_DEVICES=$device_idx python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_$bits.exp_l0e0_8.exp_l1e3_8.exp_l2e0_8.exp_l3e4_8.exp_l4e5_8.exp_l5e6_8.exp_l6e4_8.exp_l7e0_8.exp_l8e1_8.exp_l9e7_8.exp_l10e4_8.exp_l11e5_8.exp_l12e2_8.exp_l13e0_8.exp_l14e2_8.exp_l15e3_8.exp_l16e0_8.exp_l17e6_8.exp_l18e0_8.exp_l19e6_8.exp_l20e3_8.exp_l21e6_8.exp_l22e3_8.exp_l23e3_8.exp_l24e2_8.exp_l25e2_8.exp_l26e0_8.exp_l27e4_8.exp_l28e2_8.exp_l29e3_8.exp_l30e0_8.exp_l31e0_8 \
      --bits_name main_$bits.most_mass &

    device_idx=$((device_idx+1))
done

# TOP-2 MASSIVE EXPERTS
for bits in 2 4
do
    DEBUG=0 CUDA_VISIBLE_DEVICES=$device_idx python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_$bits.exp_l0e0_8.exp_l1e3_8 \
      --bits_name main_$bits.most_mass_top-2-layer &

    device_idx=$((device_idx+1))
done

# All Task-specific (44 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.exp_l30e7_8.exp_l31e3_8.exp_l28e4_8.exp_l31e1_8.exp_l29e7_8.exp_l27e2_8.exp_l23e1_8.exp_l8e0_8.exp_l2e0_8.exp_l24e1_8.exp_l12e0_8.exp_l21e0_8.exp_l20e6_8.exp_l12e1_8.exp_l26e4_8.exp_l11e1_8.exp_l31e2_8.exp_l29e6_8.exp_l6e5_8.exp_l17e1_8.exp_l31e7_8.exp_l0e5_8.exp_l0e2_8.exp_l15e2_8.exp_l19e0_8.exp_l4e3_8.exp_l16e3_8.exp_l11e4_8.exp_l18e2_8.exp_l3e2_8.exp_l16e7_8.exp_l10e1_8.exp_l2e1_8.exp_l17e5_8.exp_l1e4_8.exp_l26e3_8.exp_l4e2_8.exp_l9e3_8.exp_l13e4_8.exp_l25e4_8.exp_l14e4_8.exp_l20e2_8.exp_l27e5_8.exp_l24e2_8 \
      --bits_name main_2.task_specific_avg_3bits

# Massive experts in each layer + Task-specific for left (12 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.exp_l0e0_8.exp_l1e3_8.exp_l2e0_8.exp_l3e4_8.exp_l4e5_8.exp_l5e6_8.exp_l6e4_8.exp_l7e0_8.exp_l8e1_8.exp_l9e7_8.exp_l10e4_8.exp_l11e5_8.exp_l12e2_8.exp_l13e0_8.exp_l14e2_8.exp_l15e3_8.exp_l16e0_8.exp_l17e6_8.exp_l18e0_8.exp_l19e6_8.exp_l20e3_8.exp_l21e6_8.exp_l22e3_8.exp_l23e3_8.exp_l24e2_8.exp_l25e2_8.exp_l26e0_8.exp_l27e4_8.exp_l28e2_8.exp_l29e3_8.exp_l30e0_8.exp_l31e0_8.exp_l30e7_8.exp_l31e3_8.exp_l28e4_8.exp_l31e1_8.exp_l29e7_8.exp_l27e2_8.exp_l23e1_8.exp_l8e0_8.exp_l24e1_8.exp_l12e0_8.exp_l21e0_8.exp_l20e6_8 \
      --bits_name main_2.most_mass_each_layer_and_task_specific_avg_3bits

# Attention 8 bits, others 4 bits
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_4.attn_8 \
      --bits_name attn_8_others_4
# Attention 2 bits, others 4 bits
DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_4.attn_2 \
      --bits_name attn_2_others_4

# Attention 4 bits + Left Task-specific (42 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l30e7_8.exp_l31e3_8.exp_l28e4_8.exp_l31e1_8.exp_l29e7_8.exp_l27e2_8.exp_l23e1_8.exp_l8e0_8.exp_l2e0_8.exp_l24e1_8.exp_l12e0_8.exp_l21e0_8.exp_l20e6_8.exp_l12e1_8.exp_l26e4_8.exp_l11e1_8.exp_l31e2_8.exp_l29e6_8.exp_l6e5_8.exp_l17e1_8.exp_l31e7_8.exp_l0e5_8.exp_l0e2_8.exp_l15e2_8.exp_l19e0_8.exp_l4e3_8.exp_l16e3_8.exp_l11e4_8.exp_l18e2_8.exp_l3e2_8.exp_l16e7_8.exp_l10e1_8.exp_l2e1_8.exp_l17e5_8.exp_l1e4_8.exp_l26e3_8.exp_l4e2_8.exp_l9e3_8.exp_l13e4_8.exp_l25e4_8.exp_l14e4_8.exp_l20e2_8 \
      --bits_name main_2.attn_4.task_specific_avg_3bits

# Attention 4 bits + Massive experts in each layer + Task-specific for left (12 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e0_8.exp_l1e3_8.exp_l2e0_8.exp_l3e4_8.exp_l4e5_8.exp_l5e6_8.exp_l6e4_8.exp_l7e0_8.exp_l8e1_8.exp_l9e7_8.exp_l10e4_8.exp_l11e5_8.exp_l12e2_8.exp_l13e0_8.exp_l14e2_8.exp_l15e3_8.exp_l16e0_8.exp_l17e6_8.exp_l18e0_8.exp_l19e6_8.exp_l20e3_8.exp_l21e6_8.exp_l22e3_8.exp_l23e3_8.exp_l24e2_8.exp_l25e2_8.exp_l26e0_8.exp_l27e4_8.exp_l28e2_8.exp_l29e3_8.exp_l30e0_8.exp_l31e0_8.exp_l30e7_8.exp_l31e3_8.exp_l28e4_8.exp_l31e1_8.exp_l29e7_8.exp_l27e2_8.exp_l23e1_8.exp_l8e0_8.exp_l24e1_8.exp_l12e0_8 \
      --bits_name main_2.attn_4.most_mass_each_layer_and_task_specific_avg_3bits


# All WANDA (44 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.exp_l31e1_8.exp_l31e5_8.exp_l30e5_8.exp_l30e3_8.exp_l30e7_8.exp_l29e2_8.exp_l29e7_8.exp_l29e1_8.exp_l29e6_8.exp_l28e0_8.exp_l28e1_8.exp_l30e2_8.exp_l31e7_8.exp_l29e0_8.exp_l28e3_8.exp_l29e3_8.exp_l30e4_8.exp_l28e2_8.exp_l28e4_8.exp_l28e5_8.exp_l28e7_8.exp_l29e4_8.exp_l28e6_8.exp_l31e2_8.exp_l27e4_8.exp_l27e5_8.exp_l30e0_8.exp_l27e0_8.exp_l27e2_8.exp_l27e6_8.exp_l26e6_8.exp_l30e6_8.exp_l31e3_8.exp_l26e0_8.exp_l27e1_8.exp_l27e7_8.exp_l30e1_8.exp_l26e4_8.exp_l26e3_8.exp_l26e1_8.exp_l29e5_8.exp_l25e4_8.exp_l26e7_8.exp_l26e2_8 \
      --bits_name main_2.wanda_avg_3bits

# Attention 4 bits + Left WANDA (42 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l31e1_8.exp_l31e5_8.exp_l30e5_8.exp_l30e3_8.exp_l30e7_8.exp_l29e2_8.exp_l29e7_8.exp_l29e1_8.exp_l29e6_8.exp_l28e0_8.exp_l28e1_8.exp_l30e2_8.exp_l31e7_8.exp_l29e0_8.exp_l28e3_8.exp_l29e3_8.exp_l30e4_8.exp_l28e2_8.exp_l28e4_8.exp_l28e5_8.exp_l28e7_8.exp_l29e4_8.exp_l28e6_8.exp_l31e2_8.exp_l27e4_8.exp_l27e5_8.exp_l30e0_8.exp_l27e0_8.exp_l27e2_8.exp_l27e6_8.exp_l26e6_8.exp_l30e6_8.exp_l31e3_8.exp_l26e0_8.exp_l27e1_8.exp_l27e7_8.exp_l30e1_8.exp_l26e4_8.exp_l26e3_8.exp_l26e1_8.exp_l29e5_8.exp_l25e4_8 \
      --bits_name main_2.attn_4.wanda_avg_3bits

# Attention 2 bits + Massive experts in each layer + WANDA for left (12 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.exp_l0e0_8.exp_l1e3_8.exp_l2e0_8.exp_l3e4_8.exp_l4e5_8.exp_l5e6_8.exp_l6e4_8.exp_l7e0_8.exp_l8e1_8.exp_l9e7_8.exp_l10e4_8.exp_l11e5_8.exp_l12e2_8.exp_l13e0_8.exp_l14e2_8.exp_l15e3_8.exp_l16e0_8.exp_l17e6_8.exp_l18e0_8.exp_l19e6_8.exp_l20e3_8.exp_l21e6_8.exp_l22e3_8.exp_l23e3_8.exp_l24e2_8.exp_l25e2_8.exp_l26e0_8.exp_l27e4_8.exp_l28e2_8.exp_l29e3_8.exp_l30e0_8.exp_l31e0_8.exp_l31e1_8.exp_l31e5_8.exp_l30e5_8.exp_l30e3_8.exp_l30e7_8.exp_l29e2_8.exp_l29e7_8.exp_l29e1_8.exp_l29e6_8.exp_l28e0_8.exp_l28e1_8.exp_l30e_8 \
      --bits_name main_2.most_mass_each_layer_and_wanda_avg_3bits

# Attention 4 bits + Massive experts in each layer + WANDA for left (10 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e0_8.exp_l1e3_8.exp_l2e0_8.exp_l3e4_8.exp_l4e5_8.exp_l5e6_8.exp_l6e4_8.exp_l7e0_8.exp_l8e1_8.exp_l9e7_8.exp_l10e4_8.exp_l11e5_8.exp_l12e2_8.exp_l13e0_8.exp_l14e2_8.exp_l15e3_8.exp_l16e0_8.exp_l17e6_8.exp_l18e0_8.exp_l19e6_8.exp_l20e3_8.exp_l21e6_8.exp_l22e3_8.exp_l23e3_8.exp_l24e2_8.exp_l25e2_8.exp_l26e0_8.exp_l27e4_8.exp_l28e2_8.exp_l29e3_8.exp_l30e0_8.exp_l31e0_8.exp_l31e1_8.exp_l31e5_8.exp_l30e5_8.exp_l30e3_8.exp_l30e7_8.exp_l29e2_8.exp_l29e7_8.exp_l29e1_8.exp_l29e6_8.exp_l28e0_8 \
      --bits_name main_2.attn_4.most_mass_each_layer_and_wanda_avg_3bits