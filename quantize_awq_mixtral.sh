device_id=0
for bits in 8 4
do
    CUDA_VISIBLE_DEVICES=$device_id python quantize_awq_mixtral.py --bits $bits --group_size 64 &
    device_id=$((device_id+1))
done