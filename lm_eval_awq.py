import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks

from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model, GPTQQuantizer_deepseek
import torch
import random
from argparse import ArgumentParser

from transformers import AutoTokenizer, TextGenerationPipeline
import logging
from datasets import load_dataset

from awq import AutoAWQForCausalLM

DEEPSEEK_MODEL_COMPONENTS = [
    'model.embed_tokens', 'model.layers.0', 'model.layers.1', 'model.layers.2', 'model.layers.3', 'model.layers.4', 'model.layers.5', 'model.layers.6', 'model.layers.7', 'model.layers.8',
    'model.layers.9', 'model.layers.10', 'model.layers.11', 'model.layers.12', 'model.layers.13', 'model.layers.14', 'model.layers.15', 'model.layers.16', 'model.layers.17', 'model.layers.18',
    'model.layers.19', 'model.layers.20', 'model.layers.21', 'model.layers.22', 'model.layers.23', 'model.layers.24', 'model.layers.25', 'model.layers.26', 'model.layers.27', 'model.norm', 'lm_head'
]


LM_EVAL_TASK_KWARGS_DICT = {
    # "winogrande": {"task": "winogrande", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    # "copa": {"task": "copa", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    # "openbookqa": {"task": "openbookqa", "num_fewshot": 0, "batch_size": 128, "metric": "acc_norm"},    
    # "hellaswag": {"task": "hellaswag", "num_fewshot": 0, "batch_size": 128, "metric": "acc_norm"},
    # "lambada_openai": {"task": "lambada_openai", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    # "rte": {"task": "rte", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    # "piqa": {"task": "piqa", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    
    "mmlu": {"task": "mmlu", "num_fewshot": 5, "batch_size": 8, "metric": "acc"},
}




def create_device_map(components):
    num_gpus = torch.cuda.device_count()
    device_map = {}
    if num_gpus == 0:
        print("No GPU found. Please check your system.")
        return device_map

    part_size = len(components) // num_gpus
    for i, component in enumerate(components):
        device_id = i // part_size
        device_id = min(device_id, num_gpus - 1)
        device_map[component] = device_id
    print(f"Device map: {device_map}")
    return device_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--bits", type=str)
    parser.add_argument("--is_quantized", action="store_true", help="Is the model GPTQ quantized?", default=True)

    args = parser.parse_args()
    
    if args.is_quantized:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # args.model_path = 'llama-2-7b-awq-w_bit.4-group_size.128'
        # args.model_path = 'quantized_deepseek-moe-16b-base-awq-w_bit4-group_size64'
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if 'deepseek' in args.model_path:
            device_map = create_device_map(DEEPSEEK_MODEL_COMPONENTS)
        else:
            device_map = 'auto'
            
        model = AutoAWQForCausalLM.from_quantized(args.model_path, quant_file='', fuse_layers=False, device_map=device_map)
    else:
        if 'deepseek' in args.model_path:
            device_map = create_device_map(DEEPSEEK_MODEL_COMPONENTS)
        else:
            device_map = 'auto'
        model_name = "deepseek-ai/deepseek-moe-16b-base"
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
    for name, param in model.model.named_parameters():
        print(name, param.dtype)

    total_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print("Total number of parameters:", total_params)

    total_memory = sum(p.element_size() * p.numel() for p in model.model.parameters() if p.requires_grad)
    print("Total memory used by model parameters (in bytes):", total_memory)

    all_metrics = {}
    
    save_file_path = os.path.join(f"{args.model_path}", f"eval_result_{args.model_path}")
    
    for task_kwargs in LM_EVAL_TASK_KWARGS_DICT.values():
        print(f"Evaluating task: {task_kwargs['task']}")
        task_name = task_kwargs["task"]
        initialize_tasks(verbosity="ERROR")
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=task_kwargs["batch_size"],
        )
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=task_name,
            num_fewshot=task_kwargs["num_fewshot"],
            batch_size=task_kwargs["batch_size"],
            log_samples=False,
        )
        metric = task_kwargs["metric"]
        for key, value in results["results"][task_name].items():
            if key.startswith(metric + ","):
                all_metrics[f"{task_name}_{metric}"] = value
                
                
        print(">>>>> Results <<<<<")
        print(f"Metrics: {all_metrics}")
        with open(save_file_path, 'w') as file:
            json.dump(all_metrics, file, indent=4)
            
    print(">>>>> Results <<<<<")
    print(f"Metrics: {all_metrics}")



