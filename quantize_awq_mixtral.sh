export CUDA_VISIBLE_DEVICES=4,5,6


for bits in 2 4 8
do
    python quantize_mixtral.py --bits $bits --group_size 64
done