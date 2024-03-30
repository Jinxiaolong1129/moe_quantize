export CUDA_VISIBLE_DEVICES=0,1,2
export DEBUG=0

#for bits in 2 4 8
#do
#    python quantize_gptq_mixtral.py --model_name mistralai/Mixtral-8x7B-v0.1 --all_bits $bits
#done

for bits in 2 4
do
    python quantize_gptq_mixtral.py --model_name mistralai/Mixtral-8x7B-v0.1 --bits main_$bits.exp_l1e3_16
done