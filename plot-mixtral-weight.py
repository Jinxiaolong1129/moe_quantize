# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/4/29
import os

import torch
from fire import Fire
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import MixtralForCausalLM


# plt.rcParams['font.family'] = 'Times New Roman'


@torch.no_grad()
def block_wise_weight_boxplot(save_dir="./results/"):
    model = MixtralForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.float16, device_map="auto"
    )

    block_flatten_weight = []
    for block in tqdm(model.model.layers):
        ffn = block.block_sparse_moe
        ffn_weight = torch.cat(
            [exp.w1.weight.data.flatten() for exp in ffn.experts] + (
                [exp.w2.weight.data.flatten() for exp in ffn.experts]) + (
                [exp.w3.weight.data.flatten() for exp in ffn.experts])
        ).flatten().cpu()
        block_flatten_weight.append(ffn_weight)

    block_flatten_weight = torch.stack(block_flatten_weight)

    plt.boxplot(block_flatten_weight.abs(), positions=range(len(block_flatten_weight)), showfliers=True)
    plt.yscale("log")
    plt.xlabel("Block")
    plt.ylabel("Weight Value")
    plt.save(block_flatten_weight, os.path.join(save_dir, "block_wise_weight_boxplot.png"))


@torch.no_grad()
def expert_wise_weight_boxplot(save_dir="./results/"):
    model = MixtralForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.float16, device_map="auto"
    )
    save_dir = os.path.join(save_dir, "expert_wise_weight_boxplot")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    block_max_weight = []
    for block_id, block in enumerate(tqdm(model.model.layers)):
        ffn = block.block_sparse_moe
        expert_flatten_weight = torch.stack([
            torch.cat([exp.w1.weight.data.flatten(), exp.w2.weight.data.flatten(), exp.w3.weight.data.flatten()])
            for exp in ffn.experts
        ]).cpu()
        block_max_weight.append(expert_flatten_weight.abs().max().item())
        # plt.boxplot(expert_flatten_weight.abs(), positions=range(len(expert_flatten_weight)), showfliers=True)
        # plt.yscale("log")
        # plt.xlabel("Expert")
        # plt.ylabel("Weight Value")
        # plt.savefig(os.path.join(save_dir, f"block_{block_id}.png"))
        # plt.clf()
    print(f"Max weight magnitude per block: {block_max_weight}")


def main(level: str = "expert"):
    if level == "block":
        block_wise_weight_boxplot()
    elif level == "expert":
        expert_wise_weight_boxplot()
    else:
        raise ValueError(f"Invalid level: {level}")


if __name__ == "__main__":
    Fire(main)
