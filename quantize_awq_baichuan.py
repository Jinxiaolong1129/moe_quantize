import os

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from argparse import ArgumentParser
import logging


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--is_quantized", type=bool, default=True)
    args = parser.parse_args()

    if args.is_quantized:
        w_bit = args.bits
        group_size = args.group_size

        model_path = "baichuan-inc/Baichuan2-7B-Base"
        quant_path = f'/home/LeiFeng/xiaolong/moe_quantize/quantized_baichuan-awq-w_bit.{w_bit}-group_size.{group_size}'

        quant_config = { "zero_point": True, "q_group_size": group_size, "w_bit": w_bit, "version": "GEMM" }

        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Console handler
                        logging.FileHandler(f'quantize_awq_deepseek-moe-16b-base_bits.{w_bit}_group.{group_size}.log') 
                    ])

        logger = logging.getLogger(__name__)

        # TODO AWQ 
        model = AutoAWQForCausalLM.from_pretrained(
            model_path, **{"low_cpu_mem_usage": True, "use_cache": False, "safetensors":False}
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Quantize
        model.quantize(tokenizer, quant_config=quant_config)

        # Save quantized model
        model.save_quantized(quant_path)
        tokenizer.save_pretrained(quant_path)

        print(f'Model is quantized and saved at "{quant_path}"')


# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/awq/:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=7
# nohup python quantize_awq_baichun.py --bits 4 > quantize_awq_baichun_bit.4.log &



