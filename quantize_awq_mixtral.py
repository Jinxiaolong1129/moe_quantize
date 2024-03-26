import os
os.environ['HF_HOME'] = '/home/LeiFeng/xiaolong/moe_quantize/hf_cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,6,7'

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from argparse import ArgumentParser



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=64)

    args = parser.parse_args()

    if args.is_quantized:
        w_bit = args.bits
        group_size = args.group_size

        model_path = "mistralai/Mixtral-8x7B-v0.1"
        quant_path = f'/home/LeiFeng/xiaolong/moe_quantize/quantized_mistral-instruct-v0.2-awq-w_bit.{w_bit}-group_size.{group_size}'

        quant_config = { "zero_point": True, "q_group_size": group_size, "w_bit": w_bit, "version": "GEMM" }

        # TODO AWQ 
        model = AutoAWQForCausalLM.from_pretrained(
            model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Quantize
        model.quantize(tokenizer, quant_config=quant_config)

        # Save quantized model
        model.save_quantized(quant_path)
        tokenizer.save_pretrained(quant_path)

        print(f'Model is quantized and saved at "{quant_path}"')
    else:
        model_path = "mistralai/Mixtral-8x7B-v0.1"
        model = AutoAWQForCausalLM.from_pretrained(
            model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print(f'Model is loaded from "{model_path}"')




# nohup python quantize_mixtral.py --bits 4 --group_size 64 > awq_quantize_mixtral_all_4bit_group64.log 2>&1 &
# nohup python quantize_mixtral.py --bits 2 --group_size 64 > awq_quantize_mixtral_all_2bit_group64.log 2>&1 &

# nohup python quantize_mixtral.py --bits 8 --group_size 64 > awq_quantize_mixtral_all_8bit_group64.log 2>&1 & 768953

# nohup python quantize_mixtral.py --bits 4 --group_size 128 > awq_quantize_mixtral_all_4bit_group128.log 2>&1 & 769848
# nohup python quantize_mixtral.py --bits 2 --group_size 128 > awq_quantize_mixtral_all_2bit_group128.log 2>&1 & 770554
