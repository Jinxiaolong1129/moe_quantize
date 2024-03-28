export DEBUG=0
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2

for bits in 8 4 2
do
python lm_eval_awq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path quantized_Mixtral-8x7B-v0.1-awq-w_bit.$bits-group_size.64
done
