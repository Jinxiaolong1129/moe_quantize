export CUDA_VISIBLE_DEVICES=0 && python collect-deepseek-predictor-data.py --function_name collect_deepseek_predictor_test_data & \
export CUDA_VISIBLE_DEVICES=1 && python collect-deepseek-predictor-data.py --function_name collect_deepseek_ffn_with_residual_predictor_train_data & \
export CUDA_VISIBLE_DEVICES=2 && python collect-deepseek-predictor-data.py --function_name collect_deepseek_ffn_with_residual_predictor_test_data
