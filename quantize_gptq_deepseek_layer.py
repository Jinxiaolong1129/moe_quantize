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
from auto_gptq import deepseek_quantize_config

import logging
import csv
import time
import os


def get_Pile_dataset(tokenizer, seqlen: int, nsamples: int, split: str = "train"):
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


def average_bit():
    parser = ArgumentParser()
    parser.add_argument("--bits", type=str, default='moe.all_mlp.2+other_block.4')
    parser.add_argument("--model_name", type=str, default='deepseek-ai/deepseek-moe-16b-base')
        
    parser.add_argument("--nsamples", type=int, default=512)
    parser.add_argument("--seqlen", type=int, default=512)  
    parser.add_argument("--group_size", type=int, default=128)    
    
    args = parser.parse_args()
    args_dict = vars(args)
    logging.info("Command-line arguments: %s", args_dict)

    model_name = args.model_name    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
    
    log_data = []
    
    eval_bits = [args.bits]

    eval_bits = [
        "moe.shared_2.other_2+other_block.4",
        "moe.shared_4.other_2+other_block.4",
        "moe.shared_4.top30_4.other_2+other_block.4",
        'moe.shared_4.top25_4.other_2+other_block.4+startlayer_5',
        'moe.shared_4.top25_4.other_2+other_block.4+dejavu_5',
        "moe.shared_4.other_2+other_block.4+alpha30",
        "moe.shared_4.other_2+other_block.4+alpha40",
        "moe.shared_4.other_2+other_block.4+alpha50",
        "moe.shared_4.other_2+other_block.4+alpha55",
        "moe.shared_4.other_2+other_block.4+alpha60",
        "moe.shared_4.other_2+other_block.4+alpha70",
    ]
        
    print(f"len(eval_bits): {len(eval_bits)}")
    
    
    for bits in eval_bits:
        args.bits = bits
        
        deeepseek_bit = deepseek_quantize_config(args)
        
        total_bits_moe = 0
        total_params_moe = 0
        total_bits_self_attn = 0
        total_params_self_attn = 0
        total_bits = 0
        total_params = 0
        
        # for name, module in model.named_modules():
        #     if hasattr(module, 'weight'):
        #         weights_count[name] = [module.weight.numel(), module.weight.shape]
                
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight.data
                num_params = weight.numel()  # Total number of parameters in the module
                
                if name in deeepseek_bit:
                    bit = deeepseek_bit[name]
                    total_bits += num_params * bit  # Accumulate total bits for all specified modules
                    total_params += num_params
                    
                if ('experts' in name or 'shared_experts' in name) and name in deeepseek_bit:
                    bit = deeepseek_bit[name]
                    total_bits_moe += num_params * bit
                    total_params_moe += num_params
                    
                    # print(f'name {name} | bit {bit}')
                    # print(f'total_bits_moe {total_bits_moe} | num_params {num_params} | bit {bit}')
                elif 'self_attn' in name and name in deeepseek_bit:
                    bit = deeepseek_bit[name]
                    total_bits_self_attn += num_params * bit
                    total_params_self_attn += num_params
            
            
        # print('=========================')
        # print(f'total_params_moe {total_params_moe}')
        # print(f'total_bits_moe {total_bits_moe}')
        # print(f'total_params_self_attn {total_params_self_attn}')
        # print(f'total_bits_self_attn {total_bits_self_attn}')
        # print(f'total_params {total_params}')
        
        
        average_bit_moe = total_bits_moe / total_params_moe if total_params_moe > 0 else 0
        average_bit_self_attn = total_bits_self_attn / total_params_self_attn if total_params_self_attn > 0 else 0
        average_bit = total_bits / total_params if total_params > 0 else 0
        print(f"Bits: {bits}")
        print(f"MoE Average Bit: {average_bit_moe}")
        print(f"Self-Attention Average Bit: {average_bit_self_attn}")
        print(f"Average Bit: {average_bit}")
        print('=========================')
        
        data = {
            "Bits": bits,
            "MoE Average Bit": average_bit_moe,
            "Self-Attention Average Bit": average_bit_self_attn,
            "Average Bit": average_bit
        }
        
        # Add the data to the list
        log_data.append(data)
    
    fieldnames = ["Bits", "MoE Average Bit", "Self-Attention Average Bit", "Average Bit"]

    # Open a CSV file to write the data
    with open('deepseek_bits_data.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()
        
        # Write the log data
        writer.writerows(log_data)

    print("Log data has been saved to log_data.csv.")
    return average_bit

def main():
    parser = ArgumentParser()
    parser.add_argument("--bits", type=str, default='all_4')
    parser.add_argument("--model_name", type=str, default='deepseek-ai/deepseek-moe-16b-base')
    parser.add_argument("--quantized_model_file_base_name", type=str, default=None)
    parser.add_argument("--quant_path", type=str, default=None)
    parser.add_argument("--nsamples", type=int, default=512)
    parser.add_argument("--seqlen", type=int, default=512)  
    parser.add_argument("--group_size", type=int, default=128)    
    parser.add_argument("--save_path", type=str, default=None)
    
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
    model_specific_path = f'autogptq_{model_name}-gptq_w_bit_{args.bits}'
    
    if args.save_path:
        quant_path = os.path.join(args.save_path, model_specific_path)
        logging.info(f"Using combined save path: {quant_path}")
    else:
        quant_path = model_specific_path
        logging.info(f"Using default save path: {quant_path}")
        
        
    quantized_model_file_base_name = f'{model_name.split("/")[-1]}-gptq_w_bit_{args.bits}'
    
    logging.info(f"Quantized model will be saved to {quant_path}")
    logging.info(f"Quantized model file base name: {quantized_model_file_base_name}")

    deeepseek_bit = deepseek_quantize_config(args)
    
    logging.info(f"Quantization config: {deeepseek_bit}")
    print(f"Quantization config:\n {deeepseek_bit}")
    
    
    quantize_config = BaseQuantizeConfig_mixed_precision(
        bits=deeepseek_bit,  # quantize model to 4-bit
        group_size=args.group_size,  # it is recommended to set the value to 128
        desc_act=True,  # set to False can significantly speed up inference but the perplexity may slightly bad
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



    
if __name__ == "__main__":
    # average_bit()
    main()
    
    
    
    
    
# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=0,1
# python quantize_gptq_deepseek.py \


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