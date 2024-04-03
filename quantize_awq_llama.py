import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=64)

    args = parser.parse_args()


    w_bit = args.bits
    group_size = args.group_size

    model_path = "meta-llama/Llama-2-7b-hf"
    quant_path = f'/home/LeiFeng/xiaolong/moe_quantize/llama-2-7b-awq-w_bit.{w_bit}-group_size.{group_size}'

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

# nohup python quantize_awq_llama.py --bits 4 --group_size 128 > awq_quantize_llama_all_4bit_group128.log 2>&1 &