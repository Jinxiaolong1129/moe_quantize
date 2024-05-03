import random

import torch
from datasets import load_dataset
from fire import Fire
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

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



def mixtral_task_specific_expert_pruning_inference(
        global_ranking: bool = True, batch_size: int = 8
):
    model = AutoModelForCausalLM.from_pretrained(
        'deepseek-ai/deepseek-moe-16b-base', device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-moe-16b-base')
    dataset = get_Pile_dataset(tokenizer=tokenizer, seqlen=4096, nsamples=512, split="train")

    save_path = "save/expert_task_specific_score.pt"
    if not os.path.exists(save_path):
        all_router_logits = []

        for i in tqdm(range(0, len(dataset), batch_size), desc="Inference"):
            input_ids = torch.cat([d["input_ids"] for d in dataset[i: i + batch_size]], dim=0)
            attention_mask = torch.cat([d["attention_mask"] for d in dataset[i: i + batch_size]], dim=0)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=True)
            router_logits = torch.stack(outputs.router_logits, dim=0)
            # of shape (num_hidden_layers, num_tokens, num_local_experts)
            all_router_logits.append(router_logits)

        all_router_logits = torch.cat(all_router_logits, dim=1).to(torch.float32)
        all_router_weights = F.softmax(all_router_logits, dim=-1)
        expert_proficiency = all_router_weights.mean(dim=1)  # of shape (num_hidden_layers, num_local_experts)
        torch.save(expert_proficiency, "save/expert_task_specific_score.pt")
    else:
        expert_proficiency = torch.load(save_path)
    # Sort experts by proficiency
    num_experts = 64
    num_layers = 27

    global_ranking = False
    # an expert at (layer, expert_id) is denoted as "exp_l{layer}e{expert_id}"
    if global_ranking:
        expert_ranking = []
        for layer in range(num_layers):
            for expert_id in range(num_experts):
                expert_ranking.append((f"exp_l{layer}e{expert_id}", expert_proficiency[layer, expert_id].item()))
        expert_ranking.sort(key=lambda x: x[1], reverse=True)
        print([exp[0] for exp in expert_ranking])
    else:
        for layer in range(num_layers):
            expert_proficiency_layer = expert_proficiency[layer]
            expert_ranking = expert_proficiency_layer.argsort(descending=True)
            print(f"Layer {layer}:", [f"exp_l{layer}e{expert_id}" for expert_id in expert_ranking])

    torch.save(expert_proficiency, "save/expert_task_specific_score.pt")


if __name__ == "__main__":
    mixtral_task_specific_expert_pruning_inference(global_ranking=True, batch_size = 8)
    