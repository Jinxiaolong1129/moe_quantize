export CUDA_VISIBLE_DEVICES=0,1,2


for bits in 2 4 8
do
    python quantize_gptq_mixtral.py --model_name mistralai/Mixtral-8x7B-v0.1 --all_bits $bits
done