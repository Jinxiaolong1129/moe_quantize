# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/5/21
import os.path
import random
from tqdm import tqdm
import torch
from fire import Fire
from transformers import (
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator
)

def dump_deepseek_linear_weight_outlier_metric(
        save_dir: str = "./save",
):
    model = AutoModelForCausalLM.from_pretrained(
        'deepseek-ai/deepseek-moe-16b-base', device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    ).eval()
    name_to_score = {}
    for name, module in tqdm(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            with torch.no_grad():
                abs_weight = weight.abs()
                score = abs_weight.max(dim=0).values / abs_weight.mean(dim=0)
                score = score.max().item()
                name_to_score[name] = score

    torch.save(name_to_score, os.path.join(save_dir, "deepseek_linear_weight_outlier_metric.pt"))
    print(f"Dumped to {os.path.join(save_dir, 'deepseek_linear_weight_outlier_metric.pt')}")


if __name__ == "__main__":
    Fire(dump_deepseek_linear_weight_outlier_metric)
    