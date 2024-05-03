# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/15
import os
import random

import torch
from datasets import load_dataset, Dataset
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator
)

set_seed(42)

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


def dump_mixtral_massive_token_top_routing_count(
        save_dir: str = "./save",
        batch_size: int = 1,
        top_k: int = 2,
        activation_threshold: float = 1000,
):
    model = AutoModelForCausalLM.from_pretrained(
        'deepseek-ai/deepseek-moe-16b-base', device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-moe-16b-base')
    dataset = get_Pile_dataset(tokenizer=tokenizer, seqlen=4096, nsamples=512, split="train")
    data_loader = DataLoader(
        Dataset.from_list(dataset),
        batch_size=batch_size,
        collate_fn=default_data_collator,
        shuffle=True,
    )
    
    config = model.config
    num_experts = 64
    num_layers = 27
    
    expert_routed_counter = torch.zeros(num_layers, num_experts, dtype=torch.int)

    for batch in tqdm(data_loader, desc=f"Dumping routing distribution"):
        batch = {k: v.cuda() for k, v in batch.items()}
        if "labels" in batch:
            batch.pop("labels")
        if batch_size == 1:
            for k, v in batch.items():
                batch[k] = v.squeeze(0)
        with torch.no_grad():
            outputs = model(**batch, output_router_logits=True, output_hidden_states=True)
        all_router_logits = torch.stack(outputs.router_logits)  # (num_hidden_layers, num_tokens, num_local_experts)
        all_hidden_states = torch.stack(outputs.hidden_states[1:])[1:]  # (num_hidden_layers, num_tokens, hidden_size)

        # tokens that with any activation larger than activation_threshold
        massive_tokens = (all_hidden_states.abs() >= activation_threshold).any(dim=-1).reshape(num_layers,
                                                                                            -1)  # (num_hidden_layers, num_tokens)
        selected_experts = torch.topk(all_router_logits, top_k, dim=-1)[1].reshape(num_layers, -1,
                                                                                top_k)  # (num_hidden_layers, num_tokens, top_k)

        for layer_idx in range(num_layers):
            # unique, counts = torch.unique(selected_experts[layer_idx], return_counts=True)
            # expert_routed_counter[layer_idx, unique.cpu()] += counts.cpu()
            massive_token_selected_experts = selected_experts[layer_idx][massive_tokens[layer_idx]]
            unique, counts = torch.unique(massive_token_selected_experts, return_counts=True)
            expert_routed_counter[layer_idx, unique.cpu()] += counts.cpu()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(
        expert_routed_counter, os.path.join(save_dir, f"mass-routing-count-top{top_k}-{activation_threshold}.pt")
    )


if __name__ == "__main__":
    dump_mixtral_massive_token_top_routing_count()