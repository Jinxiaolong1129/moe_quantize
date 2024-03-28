export DEBUG=0
export CUDA_VISIBLE_DEVICES=0,1,2

for bits in 8 4 2
do
python lm_eval_awq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path quantized_Mixtral-8x7B-v0.1-awq-w_bit.$bits-group_size.64 >> output_model_bits_$bits.out
done
