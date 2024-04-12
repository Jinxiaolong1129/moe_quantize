device_idx=0


# MOST USED
#for bits in 2 4
#do
#    DEBUG=0 CUDA_VISIBLE_DEVICES=$device_idx python quantize_gptq_mixtral.py \
#      --model_name mistralai/Mixtral-8x7B-v0.1 \
#      --bits main_$bits.exp_l0e5_8.exp_l1e4_8.exp_l2e0_8.exp_l3e2_8.exp_l4e3_8.exp_l5e1_8.exp_l6e5_8.exp_l7e5_8.exp_l8e5_8.exp_l9e0_8.exp_l10e1_8.exp_l11e1_8.exp_l12e1_8.exp_l13e7_8.exp_l14e4_8.exp_l15e1_8.exp_l16e7_8.exp_l17e2_8.exp_l18e0_8.exp_l19e0_8.exp_l20e6_8.exp_l21e6_8.exp_l22e0_8.exp_l23e1_8.exp_l24e2_8.exp_l25e1_8.exp_l26e3_8.exp_l27e5_8.exp_l28e1_8.exp_l29e2_8.exp_l30e7_8.exp_l31e7_8 \
#      --bits_name main_$bits.most_used &
#
#    device_idx=$((device_idx+1))
#done

# ONE KEY EXPERT
#for bits in 2 4
#do
#    DEBUG=0 CUDA_VISIBLE_DEVICES=$device_idx python quantize_gptq_mixtral.py \
#      --model_name mistralai/Mixtral-8x7B-v0.1 \
#      --bits main_$bits.exp_l1e3_8 &
#
#    device_idx=$((device_idx+1))
#done

# MASSIVE EXPERTS
#for bits in 2 4
#do
#    DEBUG=0 CUDA_VISIBLE_DEVICES=$device_idx python quantize_gptq_mixtral.py \
#      --model_name mistralai/Mixtral-8x7B-v0.1 \
#      --bits main_$bits.exp_l0e0_8.exp_l1e3_8.exp_l2e0_8.exp_l3e4_8.exp_l4e5_8.exp_l5e6_8.exp_l6e4_8.exp_l7e0_8.exp_l8e1_8.exp_l9e7_8.exp_l10e4_8.exp_l11e5_8.exp_l12e2_8.exp_l13e0_8.exp_l14e2_8.exp_l15e3_8.exp_l16e0_8.exp_l17e6_8.exp_l18e0_8.exp_l19e6_8.exp_l20e3_8.exp_l21e6_8.exp_l22e3_8.exp_l23e3_8.exp_l24e2_8.exp_l25e2_8.exp_l26e0_8.exp_l27e4_8.exp_l28e2_8.exp_l29e3_8.exp_l30e0_8.exp_l31e0_8 \
#      --bits_name main_$bits.most_mass &
#
#    device_idx=$((device_idx+1))
#done

# TOP-8 MASSIVE EXPERTS
for bits in 2 4
do
    DEBUG=0 CUDA_VISIBLE_DEVICES=$device_idx python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_$bits.exp_l0e0_8.exp_l1e3_8.exp_l2e0_8.exp_l3e4_8 \
      --bits_name main_$bits.most_mass_top-16-layer &

    device_idx=$((device_idx+1))
done