export CUDA_VISIBLE_DEVICES=0,1,2


for bits in all_2 all_4 all_8
do
    python quantize_gptq_mixtral.py \
        --bits $bits \
        --model_name mistralai/Mixtral-8x7B-v0.1
done