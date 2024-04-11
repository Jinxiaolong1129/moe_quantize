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


def get_Pile_dataset(tokenizer, seqlen: int, nsamples: int, split: str = "train"):
    # data = load_dataset("json", data_files='data/minipile/val.jsonl.zst', split="train")

    custom_cache_dir = '/home/LeiFeng/xiaolong/moe_quantize/data/minipile/'
    data = load_dataset('mit-han-lab/pile-val-backup', cache_dir=custom_cache_dir)['validation']

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


# def moe_quantize_config(args):
#     if args.bits == 'all_4':
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 4

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 4

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 4

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 4, 
#             'model.layers.0.self_attn.k_proj': 4,
#             'model.layers.0.self_attn.v_proj': 4,
#             'model.layers.0.self_attn.o_proj': 4,
#             'model.layers.0.mlp.gate_proj': 4,
#             'model.layers.0.mlp.up_proj': 4,
#             'model.layers.0.mlp.down_proj': 4
#         }
        
#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit
        
#     if args.bits == 'all_2':
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 2

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 2

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 2

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 2, 
#             'model.layers.0.self_attn.k_proj': 2,
#             'model.layers.0.self_attn.v_proj': 2,
#             'model.layers.0.self_attn.o_proj': 2,
#             'model.layers.0.mlp.gate_proj': 2,
#             'model.layers.0.mlp.up_proj': 2,
#             'model.layers.0.mlp.down_proj': 2
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit
    
#     if args.bits == 'all_8':
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 8

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 8

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 8

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 8, 
#             'model.layers.0.self_attn.k_proj': 8,
#             'model.layers.0.self_attn.v_proj': 8,
#             'model.layers.0.self_attn.o_proj': 8,
#             'model.layers.0.mlp.gate_proj': 8,
#             'model.layers.0.mlp.up_proj': 8,
#             'model.layers.0.mlp.down_proj': 8
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit

#     if args.bits == 'moe.all_mlp.2+other_block.4':
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 4

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 2

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 2

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 4, 
#             'model.layers.0.self_attn.k_proj': 4,
#             'model.layers.0.self_attn.v_proj': 4,
#             'model.layers.0.self_attn.o_proj': 4,
#             'model.layers.0.mlp.gate_proj': 4,
#             'model.layers.0.mlp.up_proj': 4,
#             'model.layers.0.mlp.down_proj': 4
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit

#     if args.bits == 'moe.shared_4.other.2+other_block_4':
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 4

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 2

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 4

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 4, 
#             'model.layers.0.self_attn.k_proj': 4,
#             'model.layers.0.self_attn.v_proj': 4,
#             'model.layers.0.self_attn.o_proj': 4,
#             'model.layers.0.mlp.gate_proj': 4,
#             'model.layers.0.mlp.up_proj': 4,
#             'model.layers.0.mlp.down_proj': 4
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit

#     if args.bits == "moe.shared_2.other.4+other_block_4":
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 4

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 4

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 2

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 4, 
#             'model.layers.0.self_attn.k_proj': 4,
#             'model.layers.0.self_attn.v_proj': 4,
#             'model.layers.0.self_attn.o_proj': 4,
#             'model.layers.0.mlp.gate_proj': 4,
#             'model.layers.0.mlp.up_proj': 4,
#             'model.layers.0.mlp.down_proj': 4
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit
    
#     if args.bits == "moe.all_mlp.4+other_block.8":
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 8

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 4

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 4

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 8, 
#             'model.layers.0.self_attn.k_proj': 8,
#             'model.layers.0.self_attn.v_proj': 8,
#             'model.layers.0.self_attn.o_proj': 8,
#             'model.layers.0.mlp.gate_proj': 8,
#             'model.layers.0.mlp.up_proj': 8,
#             'model.layers.0.mlp.down_proj': 8
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit
    
#     if args.bits == 'moe.shared_4.other.2+other_block.8':
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 8

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 2

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 4

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 8, 
#             'model.layers.0.self_attn.k_proj': 8,
#             'model.layers.0.self_attn.v_proj': 8,
#             'model.layers.0.self_attn.o_proj': 8,
#             'model.layers.0.mlp.gate_proj': 8,
#             'model.layers.0.mlp.up_proj': 8,
#             'model.layers.0.mlp.down_proj': 8
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit
    
#     if args.bits == 'moe.shared_2.other.4+other_block.8':
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 8

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 4

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 2

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 8, 
#             'model.layers.0.self_attn.k_proj': 8,
#             'model.layers.0.self_attn.v_proj': 8,
#             'model.layers.0.self_attn.o_proj': 8,
#             'model.layers.0.mlp.gate_proj': 8,
#             'model.layers.0.mlp.up_proj': 8,
#             'model.layers.0.mlp.down_proj': 8
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit


#     if args.bits == 'moe.shared_8.top1_8.other_2+other_block.4':
#         # other block(attention) 4bit
#         # moe shared 8bit, top1 8bit, other 2bit
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 4

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 4

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 2

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 4, 
#             'model.layers.0.self_attn.k_proj': 4,
#             'model.layers.0.self_attn.v_proj': 4,
#             'model.layers.0.self_attn.o_proj': 4,
#             'model.layers.0.mlp.gate_proj': 4,
#             'model.layers.0.mlp.up_proj': 4,
#             'model.layers.0.mlp.down_proj': 4
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit

#     if args.bits == 'moe.shared_8.top1_4.other_2+other_block.4':
#         # other block(attention) 4bit
#         # moe shared 8bit, top1 8bit, other 2bit
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 4

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 4

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 2

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 4, 
#             'model.layers.0.self_attn.k_proj': 4,
#             'model.layers.0.self_attn.v_proj': 4,
#             'model.layers.0.self_attn.o_proj': 4,
#             'model.layers.0.mlp.gate_proj': 4,
#             'model.layers.0.mlp.up_proj': 4,
#             'model.layers.0.mlp.down_proj': 4
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit

#     if args.bits == 'moe.shared_4.top2_8.other_2+other_block.4':
#         # other block(attention) 4bit
#         # moe shared 8bit, top1 8bit, other 2bit
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 4

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 4

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 2

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 4, 
#             'model.layers.0.self_attn.k_proj': 4,
#             'model.layers.0.self_attn.v_proj': 4,
#             'model.layers.0.self_attn.o_proj': 4,
#             'model.layers.0.mlp.gate_proj': 4,
#             'model.layers.0.mlp.up_proj': 4,
#             'model.layers.0.mlp.down_proj': 4
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit

#     if args.bits == 'moe.shared_4.top2_8.other_2+other_block.4':
#         # other block(attention) 4bit
#         # moe shared 8bit, top1 8bit, other 2bit
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 4

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 4

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 2

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 4, 
#             'model.layers.0.self_attn.k_proj': 4,
#             'model.layers.0.self_attn.v_proj': 4,
#             'model.layers.0.self_attn.o_proj': 4,
#             'model.layers.0.mlp.gate_proj': 4,
#             'model.layers.0.mlp.up_proj': 4,
#             'model.layers.0.mlp.down_proj': 4
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit


#     if args.bits == 'moe.shared_8.top1_8.other_2+other_block.8':
#         # other block(attention) 4bit
#         # moe shared 8bit, top1 8bit, other 2bit
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 4

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 4

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 2

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 4, 
#             'model.layers.0.self_attn.k_proj': 4,
#             'model.layers.0.self_attn.v_proj': 4,
#             'model.layers.0.self_attn.o_proj': 4,
#             'model.layers.0.mlp.gate_proj': 4,
#             'model.layers.0.mlp.up_proj': 4,
#             'model.layers.0.mlp.down_proj': 4
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit

#     if args.bits == 'moe.shared_8.top1_4.other_2+other_block.8':
#         # other block(attention) 4bit
#         # moe shared 8bit, top1 8bit, other 2bit
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 4

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 4

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 2

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 4, 
#             'model.layers.0.self_attn.k_proj': 4,
#             'model.layers.0.self_attn.v_proj': 4,
#             'model.layers.0.self_attn.o_proj': 4,
#             'model.layers.0.mlp.gate_proj': 4,
#             'model.layers.0.mlp.up_proj': 4,
#             'model.layers.0.mlp.down_proj': 4
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit

#     if args.bits == 'moe.shared_4.top2_8.other_2+other_block.8':
#         # other block(attention) 4bit
#         # moe shared 8bit, top1 8bit, other 2bit
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 4

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 4

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 2

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 4, 
#             'model.layers.0.self_attn.k_proj': 4,
#             'model.layers.0.self_attn.v_proj': 4,
#             'model.layers.0.self_attn.o_proj': 4,
#             'model.layers.0.mlp.gate_proj': 4,
#             'model.layers.0.mlp.up_proj': 4,
#             'model.layers.0.mlp.down_proj': 4
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit

#     if args.bits == 'moe.shared_4.top2_8.other_2+other_block.8':
#         # other block(attention) 4bit
#         # moe shared 8bit, top1 8bit, other 2bit
#         moe_block_bit_dict = {}

#         for i in range(4):
#             key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
#             moe_block_bit_dict[key] = 4

#         for i in range(64):
#             for part in ['gate_proj', 'up_proj', 'down_proj']:
#                 key = f"mlp.experts.{i}.{part}"
#                 moe_block_bit_dict[key] = 4

#         for part in ['gate_proj', 'up_proj', 'down_proj']:
#             key = f"mlp.shared_experts.{part}"
#             moe_block_bit_dict[key] = 2

#         deeepseek_bit = {
#             'model.layers.0.self_attn.q_proj': 4, 
#             'model.layers.0.self_attn.k_proj': 4,
#             'model.layers.0.self_attn.v_proj': 4,
#             'model.layers.0.self_attn.o_proj': 4,
#             'model.layers.0.mlp.gate_proj': 4,
#             'model.layers.0.mlp.up_proj': 4,
#             'model.layers.0.mlp.down_proj': 4
#         }

#         for block_num in range(1, 28):
#             for layer in moe_block_bit_dict:
#                 key = f'model.layers.{block_num}' + '.' + layer
#                 deeepseek_bit[key] = moe_block_bit_dict[layer]
#         return deeepseek_bit


#     raise ValueError("Invalid bits")


def average_bit():
    parser = ArgumentParser()
    parser.add_argument("--bits", type=str, default='moe.all_mlp.2+other_block.4')
    parser.add_argument("--model_name", type=str, default='deepseek-ai/deepseek-moe-16b-base')
    
    args = parser.parse_args()
    args_dict = vars(args)
    logging.info("Command-line arguments: %s", args_dict)

    model_name = args.model_name    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
    
    log_data = []
    
    def extract_bits(filename):
        """Extract the bits part from the filename."""
        try:
            return filename.split("_w_bit_")[1].split("_pile")[0]
        except IndexError:
            return None

    eval_bits = []

    for filename in os.listdir('autogptq_eval_result/deepseek-moe-16b-base'):
        if filename.startswith("eval_result_deepseek-moe-16b-base-gptq_w_bit_") and filename.endswith("_pile"):
            bits = extract_bits(filename)
            eval_bits.append(bits)

    print(f"len(eval_bits): {len(eval_bits)}")
    
    
    for bits in eval_bits:
        args.bits = bits
        
        deeepseek_bit = moe_quantize_config(args)
        
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
    
    

def main():
    parser = ArgumentParser()
    parser.add_argument("--bits", type=str, default='all_4')
    
    parser.add_argument("--model_name", type=str, default='deepseek-ai/deepseek-moe-16b-base')
    
    parser.add_argument("--quantized_model_file_base_name", type=str, default=None)
    parser.add_argument("--quant_path", type=str, default=None)
    
    parser.add_argument("--nsamples", type=int, default=512)
    parser.add_argument("--seqlen", type=int, default=512)  

    parser.add_argument("--group_size", type=int, default=128)    
    
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



    
if __name__ == "__main__":
    # average_bit()
    main()
    
    
    
    
    
# export PYTHONPATH=/home/LeiFeng/xiaolong/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/xiaolong/moe_quantize/auto_gptq/:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=0,1
# python quantize_gptq_deepseek.py \


# (2883584*4*64+5167168*2)/(2883584*64+5167168)
# 189716544

# 5167168*2
# 2883584*8+2883584*2*63
# (2883584*8+2883584*2*63+5167168*2)/(5167168+2883584*64)

# (2883584*8+2883584*4*63+5167168*4)/(5167168+2883584*64)

# (2883584*4+2883584*2*63+5167168*2)/(5167168+2883584*64)




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