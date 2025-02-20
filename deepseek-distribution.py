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
    AutoTokenizer,
    default_data_collator
)
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
set_seed(42)


def deepseek_routing_top_trace(
        save_dir: str,
        batch_size: int
):
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")
    model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-moe-16b-base', 
                                                torch_dtype=torch.bfloat16, 
                                                device_map="auto",
                                                trust_remote_code=True)
    
    dataset = load_dataset("JeanKaddour/minipile", split="train")
    dataset = dataset.shuffle(seed=42).select(range(2048))
    column_names = dataset.column_names

    dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=16384),
        batched=True,
        num_proc=8,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    block_size = 1024

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(
        group_texts,
        batched=True,
        num_proc=8,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        shuffle=True,
    )


    config = model.config

    expert_routed_distribution = torch.zeros(config.num_hidden_layers-1, config.n_routed_experts)

    for batch in tqdm(data_loader, desc=f"Dumping routing distribution"):
        batch = {k: v.cuda() for k, v in batch.items()}
        if "labels" in batch:
            batch.pop("labels")
        with torch.no_grad():
            outputs = model(**batch, output_router_logits=True)
        all_router_logits = outputs.router_logits
        all_router_logits = torch.stack(
            all_router_logits)  # of shape (num_hidden_layers, num_tokens, num_local_experts)
        
        selected_experts = torch.topk(all_router_logits, k=config.num_experts_per_tok, dim=-1, sorted=False)[1].reshape(
            config.num_hidden_layers-1, -1)
        
        for layer_idx in range(config.num_hidden_layers-1):
            unique, counts = torch.unique(selected_experts[layer_idx], return_counts=True)
            expert_routed_distribution[layer_idx, unique.cpu()] += counts.cpu()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(expert_routed_distribution, os.path.join(save_dir, f"deepseek-routing-count.pt"))
    print(f"Saved routing distribution to {os.path.join(save_dir, 'deepseek-routing-count.pt')}")


if __name__ == "__main__":
    deepseek_routing_top_trace(save_dir = 'save/', batch_size = 32)
    




# CUDA_VISIBLE_DEVICES=4,5,6,7 python deepseek-distribution.py --save_dir save/ --batch_size 32