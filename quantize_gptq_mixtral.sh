#for bits in 2 4 8
#do
#    DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py --model_name mistralai/Mixtral-8x7B-v0.1 --bits main_$bits.exp_l1e3_16
#done

device_idx=5
for bits in 2 4 8
do
    DEBUG=0 CUDA_VISIBLE_DEVICES=$device_idx python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_$bits.exp_l0e5_16.exp_l1e4_16.exp_l2e0_16.exp_l3e2_16.exp_l4e3_16.exp_l5e1_16.exp_l6e5_16.exp_l7e5_16.exp_l8e5_16.exp_l9e0_16.exp_l10e1_16.exp_l11e1_16.exp_l12e1_16.exp_l13e7_16.exp_l14e4_16.exp_l15e1_16.exp_l16e7_16.exp_l17e2_16.exp_l18e0_16.exp_l19e0_16.exp_l20e6_16.exp_l21e6_16.exp_l22e0_16.exp_l23e1_16.exp_l24e2_16.exp_l25e1_16.exp_l26e3_16.exp_l27e5_16.exp_l28e1_16.exp_l29e2_16.exp_l30e7_16.exp_l31e7_16 \
      --bits_name main_$bits.most_used &

    device_idx=$((device_idx+1))
done