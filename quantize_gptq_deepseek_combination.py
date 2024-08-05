import sys
print(sys.path)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
from argparse import ArgumentParser

from transformers import AutoTokenizer, TextGenerationPipeline
import logging
from datasets import load_dataset

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig, AutoGPTQForCausalLM_mixed_precision, BaseQuantizeConfig_mixed_precision
from auto_gptq import moe_quantize_config
import logging
import csv
import time
import os
from lm_eval_combination import lm_eval_gptq


def get_Pile_dataset(tokenizer, seqlen: int, nsamples: int, split: str = "train"):
    # data = load_dataset("json", data_files='data/minipile/val.jsonl.zst', split="train")

    data = load_dataset('mit-han-lab/pile-val-backup')['validation']

    text = "".join([" \n" if s == "" else s for s in data["text"][:1000]])

    enc = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return dataset

    

def main():
    parser = ArgumentParser()
    parser.add_argument("--bits", type=str, default='all_4')
    parser.add_argument("--model_name", type=str, default='deepseek-ai/deepseek-moe-16b-base')
    parser.add_argument("--quantized_model_file_base_name", type=str, default=None)
    parser.add_argument("--quant_path", type=str, default=None)
    
    parser.add_argument("--nsamples", type=int, default=512)
    parser.add_argument("--seqlen", type=int, default=512)  
    parser.add_argument("--group_size", type=int, default=128)    
    
    
    parser.add_argument("--n_ctx", type=int, default=512, help="Context size.")
    parser.add_argument("--n_batch", type=int, default=512, help="Batch size.")
    parser.add_argument("--dataset_path", type=str, default="wikitext", help="Path to the dataset.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use.")
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column in the dataset containing the text.",
    )
    parser.add_argument(
        "--per_gpu_max_memory",
        type=int,
        default=None,
        help="Max memory used in each GPU.",
    )
    parser.add_argument("--cpu_max_memory", type=int, default=None, help="Mx memory used in CPU.")
    parser.add_argument("--is_quantized", action="store_true", help="Is the model GPTQ quantized?")
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="Whether to use safetensors model file",
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Wheter to use fast tokenizer")
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to use remote code")
    parser.add_argument(
        "--disable_exllama",
        action="store_true",
        help="Whether to use disable exllama kernel",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f"run_log/gptq/quantize_gptq_deepseek_{args.bits}.log"
    )
        
    args_dict = vars(args)
    logging.info("Command-line arguments: %s", args_dict)

    model_name = args.model_name    
    quant_path = f'autogptq_{model_name}-gptq_w_bit_{args.bits}'
    quantized_model_file_base_name = f'{model_name.split("/")[-1]}-gptq_w_bit_{args.bits}'
    
    logging.info(f"Quantized model will be saved to {quant_path}")
    logging.info(f"Quantized model file base name: {quantized_model_file_base_name}")
    
    deeepseek_bit = moe_quantize_config(args)
    logging.info(f"Quantization config: {deeepseek_bit}")
    print(f"Quantization config:\n {deeepseek_bit}")
    
    quantize_config = BaseQuantizeConfig_mixed_precision(
        bits=deeepseek_bit,  # quantize model to 4-bit
        group_size=args.group_size,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
        model_file_base_name = quantized_model_file_base_name
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_dataset = get_Pile_dataset(tokenizer=tokenizer, seqlen=args.seqlen, nsamples=args.nsamples, split="train")

    model = AutoGPTQForCausalLM_mixed_precision.from_pretrained(model_name, quantize_config, torch_dtype=torch.float16, trust_remote_code=True)

    logging.info(f"Quantization dataset loaded with {args.nsamples} samples")
    logging.info(f"Quantization dataset loaded with {args.seqlen} seq len")
    
    logging.info(f"Quantizing model to {args.bits}-bit")
    logging.info(f"Quantization config: {deeepseek_bit}")
    
    logging.info(f"Quantized begin!!!!")
    model.quantize(quantization_dataset)
    logging.info(f"Quantized finish!!!!")

    model.save_quantized(quant_path)
    logging.info(f"Quantized model saved to {quant_path}")

    lm_eval_gptq(args)

    
    

    
if __name__ == "__main__":
    main()
    
    
    

# =========================
# total_params_moe 570949632
# total_bits_moe 2249195520
# total_params_self_attn 33554432
# total_bits_self_attn 268435456
# total_params 671744000
# Bits: moe.shared_4.other.2+other_block.8
# MoE Average Bit: 3.9393939393939394
# Self-Attention Average Bit: 8.0
# Average Bit: 4.548682926829268
# =========================







