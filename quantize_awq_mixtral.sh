export CUDA_VISIBLE_DEVICES=4


for bits in 8 4 2
do
    python quantize_awq_mixtral.py --bits $bits --group_size 64 &
done