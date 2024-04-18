# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/4/14
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.append("/home/LeiFeng/pingzhi/moe_quantize/optimum/")  # Add the path to Python's search path
import random

from transformers import AutoTokenizer
from transformers.models.mixtral.modeling_mixtral import MixtralBlockSparseTop2MLP
from datasets import load_dataset
from fire import Fire


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


def mixtral_task_specific_expert_pruning_inference(
        global_ranking: bool = True, batch_size: int = 8
):
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", device_map="auto", torch_dtype=torch.float16
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    dataset = get_wikitext2(tokenizer=tokenizer, seqlen=4096, nsamples=128, split="train")

    # Add hook to MixtralBLockSparseTop2MLP
    activation = {}
    all_hooks = []

    def get_activation(name):
        def hook(model, input, output):
            if name in activation:
                activation[name].append(input[0].detach())
            else:
                activation[name] = [input[0].detach()]

        return hook

    for name, module in model.named_modules():
        if isinstance(module, MixtralBlockSparseTop2MLP):
            for linear in ["w1", "w2", "w3"]:
                hook = getattr(module, linear).register_forward_hook(get_activation(f"{name}.{linear}"))
                all_hooks.append(hook)

    for i in tqdm(range(0, len(dataset), batch_size), desc="Inference"):
        input_ids = torch.cat([d["input_ids"] for d in dataset[i: i + batch_size]], dim=0)
        attention_mask = torch.cat([d["attention_mask"] for d in dataset[i: i + batch_size]], dim=0)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    for hook in all_hooks:
        hook.remove()

    input_channel_norm = {}
    for name, module in model.named_modules():
        if isinstance(module, MixtralBlockSparseTop2MLP):
            for linear in ["w1", "w2", "w3"]:
                input_dim = getattr(module, linear).in_features
                activation[f"{name}.{linear}"] = torch.cat([
                    act.reshape(-1, input_dim) for act in activation[f"{name}.{linear}"]
                ], dim=0)
                input_channel_norm[f"{name}.{linear}"] = torch.linalg.norm(activation[f"{name}.{linear}"], ord=2, dim=0)
                del activation[f"{name}.{linear}"]

    expert_wanda_score = {}
    for name, module in model.named_modules():
        if isinstance(module, MixtralBlockSparseTop2MLP):
            _scores = []
            for linear in ["w1", "w2", "w3"]:
                wanda_score = getattr(module, linear).weight.abs() * input_channel_norm[f"{name}.{linear}"]
                _scores.append(wanda_score)
            expert_wanda_score[name] = torch.stack(_scores).mean()

    config = model.config
    num_experts = config.num_local_experts
    num_layers = config.num_hidden_layers

    expert_ranking = []
    for layer in range(num_layers):
        for expert_id in range(num_experts):
            wanda_score = expert_wanda_score[f"model.layers.{layer}.block_sparse_moe.experts.{expert_id}"]
            expert_ranking.append((f"exp_l{layer}e{expert_id}", wanda_score.item()))
    expert_ranking.sort(key=lambda x: x[1], reverse=True)
    print([exp[0] for exp in expert_ranking])

    # all_router_logits = torch.cat(all_router_logits, dim=1).to(torch.float32)
    # all_router_weights = F.softmax(all_router_logits, dim=-1)
    # expert_proficiency = all_router_weights.mean(dim=1)  # of shape (num_hidden_layers, num_local_experts)
    #
    # # Sort experts by proficiency
    # config = model.config
    # num_experts = config.num_local_experts
    # num_layers = config.num_hidden_layers
    #
    #
    # # an expert at (layer, expert_id) is denoted as "exp_l{layer}e{expert_id}"
    # if global_ranking:
    #     expert_ranking = []
    #     for layer in range(num_layers):
    #         for expert_id in range(num_experts):
    #             expert_ranking.append((f"exp_l{layer}e{expert_id}", expert_proficiency[layer, expert_id].item()))
    #     expert_ranking.sort(key=lambda x: x[1], reverse=True)
    #     print([exp[0] for exp in expert_ranking])
    # else:
    #     for layer in range(num_layers):
    #         expert_proficiency_layer = expert_proficiency[layer]
    #         expert_ranking = expert_proficiency_layer.argsort(descending=True)
    #         print(f"Layer {layer}:", [f"exp_l{layer}e{expert_id}" for expert_id in expert_ranking])


if __name__ == "__main__":
    Fire(mixtral_task_specific_expert_pruning_inference)
