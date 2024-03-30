export CUDA_VISIBLE_DEVICES=3,4,5
export DEBUG=0

python quantize_gptq_mixtral.py --model_name mistralai/Mixtral-8x7B-v0.1 --bits main_8.exp_l1e3_16

#for bits in 2 4 8
#do
#    python quantize_gptq_mixtral.py --model_name mistralai/Mixtral-8x7B-v0.1 --bits main_$bits.exp_l1e3_16
#done