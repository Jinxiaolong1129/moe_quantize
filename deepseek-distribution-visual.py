# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/15
import os
from itertools import chain

import torch
from datasets import load_dataset
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    set_seed,
    MixtralForCausalLM,
    AutoTokenizer,
    default_data_collator
)
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
set_seed(42)


def visual_deepseek_routing_top_trace(
    save_dir: str,
):
    file_path = '/home/LeiFeng/xiaolong/moe_quantize/save/routing-count.pt'
    expert_routed_distribution = torch.load(file_path, map_location=torch.device('cpu')).numpy()

    num_hidden_layers = expert_routed_distribution.shape[0]
    n_routed_experts = expert_routed_distribution.shape[1]
    
    # Heatmap for the overall distribution
    plt.figure(figsize=(16, 12))
    sns.heatmap(expert_routed_distribution, annot=False, cmap="viridis",
                xticklabels=[f"E {i+1}" for i in range(n_routed_experts)],
                yticklabels=[f"B {i+1}" for i in range(num_hidden_layers)])
    
    
    plt.title("MoE Activation Frequency Heatmap")
    plt.xlabel("Experts")
    plt.ylabel("MoE Blocks")
    plt.savefig(os.path.join(save_dir, "moe_activation_frequency_heatmap.png"))

    # Calculating the total distribution across all blocks
    total_distribution = np.sum(expert_routed_distribution, axis=0)

    # Plotting the total distribution graph
    plt.figure(figsize=(18, 6))
    plt.bar(range(n_routed_experts), total_distribution, color=sns.color_palette("coolwarm", n_routed_experts))
    plt.title("Total Expert Activation Frequency Across All Blocks")
    plt.xlabel("Expert")
    plt.ylabel("Total Frequency")
    plt.xticks(range(n_routed_experts), [f"{i+1}" for i in range(n_routed_experts)], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "total_expert_activation_frequency.png"))
    print('Done')
    
    
    
if __name__ == "__main__":
    visual_deepseek_routing_top_trace(save_dir = 'save/')
    
    
    
# python deepseek-distribution.py --save_dir save/ --batch_size 32