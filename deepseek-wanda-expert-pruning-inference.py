import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import random

from transformers import AutoTokenizer

from datasets import load_dataset
from fire import Fire

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

def mixtral_task_specific_expert_pruning_inference(batch_size: int = 8):
    model = AutoModelForCausalLM.from_pretrained(
        'deepseek-ai/deepseek-moe-16b-base', device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-moe-16b-base')
    dataset = get_Pile_dataset(tokenizer=tokenizer, seqlen=4096, nsamples=512, split="train")

    # Add hook to MixtralBLockSparseTop2MLP
    activation = {}
    all_hooks = []

    last_cuda_device = torch.cuda.device_count() - 1

    def get_activation(name):
        def hook(model, input, output):
            if name in activation:
                activation[name].append(input[0].detach().cpu())
            else:
                activation[name] = [input[0].detach().cpu()]

        return hook
    # DeepseekMLP = None
    for name, module in model.named_modules():
        if type(module).__name__ == 'DeepseekMLP':
            # print(name, module)
            for linear in ["gate_proj", "up_proj", "down_proj"]:
                hook = getattr(module, linear).register_forward_hook(get_activation(f"{name}.{linear}"))
                all_hooks.append(hook)

    for i in tqdm(range(0, len(dataset), batch_size), desc="Inference"):
        input_ids = torch.cat([d["input_ids"] for d in dataset[i: i + batch_size]], dim=0)
        attention_mask = torch.cat([d["attention_mask"] for d in dataset[i: i + batch_size]], dim=0)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # break
        
    for hook in all_hooks:
        hook.remove()

    input_channel_norm = {}
    for name, module in model.named_modules():
        if type(module).__name__ == 'DeepseekMLP':
            for linear in ["gate_proj", "up_proj", "down_proj"]:
                input_dim = getattr(module, linear).in_features
                activation[f"{name}.{linear}"] = torch.cat([
                    act.reshape(-1, input_dim) for act in activation[f"{name}.{linear}"]
                ], dim=0).cuda(last_cuda_device)
                input_channel_norm[f"{name}.{linear}"] = torch.linalg.norm(activation[f"{name}.{linear}"], ord=2, dim=0)
                del activation[f"{name}.{linear}"]

    expert_wanda_score = {}
    for name, module in model.named_modules():
        if type(module).__name__ == 'DeepseekMLP':
            _scores = []
            for linear in ["gate_proj", "up_proj", "down_proj"]:
                with torch.no_grad():
                    wanda_score = getattr(module, linear).weight.abs().cuda(last_cuda_device) * input_channel_norm[f"{name}.{linear}"]
                _scores.append(wanda_score.mean())
            expert_wanda_score[name] = sum(_scores).cpu().item()

    print(expert_wanda_score)

    num_experts = 64
    num_layers = 27

    expert_ranking = []
    for layer in range(1, num_layers+1):
        for expert_id in range(num_experts):
            wanda_score = expert_wanda_score[f"model.layers.{layer}.mlp.experts.{expert_id}"]
            expert_ranking.append((f"exp_l{layer}e{expert_id}", wanda_score))
    expert_ranking.sort(key=lambda x: x[1], reverse=True)
    print([exp[0] for exp in expert_ranking])

    # save wanda score
    torch.save(expert_wanda_score, "save/expert_wanda_score.pt")

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
    mixtral_task_specific_expert_pruning_inference(batch_size=8)