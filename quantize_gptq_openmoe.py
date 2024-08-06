import os
import random
import re
import sys
from argparse import ArgumentParser

import torch

# sys.path.append("/data2/pzli/moe_quantize/optimum/")  # Add the path to Python's search path
print(sys.path)
os.environ['HF_TOKEN'] = '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'

os.environ['HF_HOME'] = '/data2/pzli/hf_cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
from transformers import AutoTokenizer
from datasets import load_dataset
from auto_gptq import BaseQuantizeConfig_mixed_precision
from auto_gptq.modeling.hf_openmoe import HFOpenMoeGPTQForCausalLM
import logging


def get_wikitext2(tokenizer, seqlen: int, nsamples: int, split: str = "train"):
    if split == "train":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif split == "validation":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # length of 288059 should be enough
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


def openmoe_quantize_config(bits_config_str: str):
    openmoe_bit = dict()
    # The main weight bits
    main_bits = re.search(r"main_(\d+)", bits_config_str)
    if main_bits is None:
        raise ValueError(f"Invalid bits config string: {bits_config_str}")
    main_bits = int(main_bits.group(1))
    moe_block_bit_dict = {}
    for i in range(4):
        key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
        if "attn" in bits_config_str:
            attn_bits = re.search(r"attn_(\d+)", bits_config_str)[1]
            moe_block_bit_dict[key] = int(attn_bits)
        else:
            moe_block_bit_dict[key] = main_bits

    for part in ['gate_proj', 'up_proj', 'down_proj']:
        key = f"mlp.{part}"
        moe_block_bit_dict[key] = main_bits

    for block_num in range(0, 24):
        for layer in moe_block_bit_dict:
            key = f'model.layers.{block_num}' + '.' + layer
            openmoe_bit[key] = moe_block_bit_dict[layer]

    for moe_block_num in [5, 11, 17, 23]:
        for part in ["gate_proj", "up_proj", "down_proj"]:
            for expert_id in range(32):
                key = f'model.layers.{moe_block_num}.mlp.experts.{expert_id}.{part}'
                openmoe_bit[key] = main_bits
        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f'model.layers.{moe_block_num}.extra_mlp.{part}'
            openmoe_bit[key] = main_bits

    # fixme: Expert weights are stacked in OpenMoE.
    # # Special expert bits, e.g. "exp_l1e3_16": 16-bit for expert 3 in layer 1
    # special_expert_bits = re.findall(r"exp_l(\d+)e(\d+)_(\d+)", bits_config_str)
    # for layer, expert, bits in special_expert_bits:
    #     for part in ['w1', 'w2', 'w3']:
    #         key = f"model.layers.{int(layer)}.block_sparse_moe.experts.{int(expert)}.{part}"
    #         mixtral_bit[key] = int(bits)

    # Special layer bits, e.g. "layer_16_4": 4-bit for layer 16
    special_layer_bits = re.findall(r"layer_(\d+)_(\d+)", bits_config_str)
    for layer, bits in special_layer_bits:
        print(f"Applying {bits}-bit to layer {layer}")
        for key in openmoe_bit:
            if f"model.layers.{int(layer)}.mlp.experts" in key:
                openmoe_bit[key] = int(bits)

    # Special module name keywords, e.g. "keyword__gate_proj__4": 4-bit for all gate_proj modules
    special_module_bits = re.findall(r"keyword__(\w+)__(\d+)", bits_config_str)
    for module_key, bits in special_module_bits:
        print(f"Applying {bits}-bit to module {module_key}")
        for key in openmoe_bit:
            if module_key in key:
                openmoe_bit[key] = int(bits)

    return openmoe_bit


def main():
    parser = ArgumentParser()
    parser.add_argument("--bits", type=str)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--nsamples", type=int, default=1024)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--bits_name", type=str, default=None)
    parser.add_argument("--bits_dict_overwrite", type=str, default=None)

    args = parser.parse_args()

    bits_name = str(args.bits) if args.bits_name is None else args.bits_name

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f"./quantize_gptq_openmoe_{bits_name}.log"
    )

    args_dict = vars(args)
    logging.info("Command-line arguments: %s", args_dict)

    model_name = args.model_name
    quant_path = f'autogptq_{model_name}-gptq_w_bit_{bits_name}'
    quantized_model_file_base_name = f'{model_name.split("/")[-1]}-gptq_w_bit_{bits_name}'

    openmoe_bits = openmoe_quantize_config(args.bits)

    if args.bits_dict_overwrite is not None:
        overwrite_bits = torch.load(args.bits_dict_overwrite)
        print(f"Overwrite bits from {args.bits_dict_overwrite}")
        openmoe_bits.update(overwrite_bits)

    print("====== First 32 bits config items ======")
    for i, (k, v) in enumerate(openmoe_bits.items()):
        if i >= 32:
            break
        print(f"{k}: {v}")
    print()

    quantize_config = BaseQuantizeConfig_mixed_precision(
        bits={k: v for k, v in openmoe_bits.items() if v != 16},  # quantize model to 4-bit
        group_size=args.group_size,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
        model_file_base_name=quantized_model_file_base_name,
        damp_percent=0.1
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = HFOpenMoeGPTQForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        quantize_config=quantize_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Calculate the average bit-width of the model
    total_bits = 0
    total_num_params = 0
    for name, param in model.model.named_parameters():
        bits = None
        for bit_key in openmoe_bits:
            if bit_key in name:
                bits = openmoe_bits[bit_key]
                break
        if bits is None:
            continue
        num_params = param.numel()
        total_bits += bits * num_params
        total_num_params += num_params

    average_bits = total_bits / total_num_params
    print(f"Average bit-width of the model w/ {bits_name}: {average_bits:.2f}")

    quantization_dataset = get_wikitext2(tokenizer=tokenizer, seqlen=2048, nsamples=args.nsamples, split="train")
    model.quantize(quantization_dataset)
    model.save_quantized(quant_path)
    logging.info(f"Quantized model saved to {quant_path}")


if __name__ == "__main__":
    main()
