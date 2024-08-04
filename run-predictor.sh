for id in 5 11 17 23
do
    CUDA_VISIBLE_DEVICES=0 python run-openmoe-predictor.py --ffn_block_id=$id
done