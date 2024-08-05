import sys

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from transformers import AutoTokenizer
import logging



LM_EVAL_TASK_KWARGS_DICT = {
    "winogrande": {"task": "winogrande", "num_fewshot": 0, "batch_size": 16, "metric": "acc"},
    "copa": {"task": "copa", "num_fewshot": 0, "batch_size": 16, "metric": "acc"},
    "openbookqa": {"task": "openbookqa", "num_fewshot": 0, "batch_size": 16, "metric": "acc_norm"},
    "hellaswag": {"task": "hellaswag", "num_fewshot": 0, "batch_size": 16, "metric": "acc_norm"},
    
    # "lambada_openai": {"task": "lambada_openai", "num_fewshot": 0, "batch_size": 16, "metric": "acc"},
    # "rte": {"task": "rte", "num_fewshot": 0, "batch_size": 16, "metric": "acc"},
    
    "piqa": {"task": "piqa", "num_fewshot": 0, "batch_size": 16, "metric": "acc"},
    "mmlu": {"task": "mmlu", "num_fewshot": 5, "batch_size": 16, "metric": "acc"},
}

class eval_config():
    def __init__(self, model_name='deepseek-ai/deepseek-moe-16b-base', 
                quant_model_path=None, 
                bits=None, 
                n_ctx=512, 
                n_batch=512, 
                dataset_path="wikitext", 
                dataset_name=None, split="test", 
                text_column="text", 
                per_gpu_max_memory=None, 
                cpu_max_memory=None, 
                is_quantized=False, 
                use_safetensors=False, 
                use_fast_tokenizer=False, 
                trust_remote_code=False, 
                disable_exllama=False):
        
        self.model_name = model_name
        self.quant_model_path = quant_model_path
        self.bits = bits
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.split = split
        self.text_column = text_column
        self.per_gpu_max_memory = per_gpu_max_memory
        self.cpu_max_memory = cpu_max_memory
        self.is_quantized = is_quantized
        self.use_safetensors = use_safetensors
        self.use_fast_tokenizer = use_fast_tokenizer
        self.trust_remote_code = trust_remote_code
        self.disable_exllama = disable_exllama



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
    parser.add_argument("--model_name", type=str, default='deepseek-ai/deepseek-moe-16b-base')
    parser.add_argument("--n_ctx", type=int, default=512, help="Context size.")
    parser.add_argument("--n_batch", type=int, default=512, help="Batch size.")
    parser.add_argument("--dataset_path", type=str, default="wikitext", help="Path to the dataset.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use.")
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column in the dataset containing the text.",
    )
    parser.add_argument(
        "--per_gpu_max_memory",
        type=int,
        default=None,
        help="Max memory used in each GPU.",
    )
    parser.add_argument("--cpu_max_memory", type=int, default=None, help="Mx memory used in CPU.")
    parser.add_argument("--is_quantized", action="store_true", help="Is the model GPTQ quantized?")
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="Whether to use safetensors model file",
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Wheter to use fast tokenizer")
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to use remote code")
    parser.add_argument(
        "--disable_exllama",
        action="store_true",
        help="Whether to use disable exllama kernel",
    )
    args = parser.parse_args()

    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f"run_log/no_quantize/eval_{args.model_name}.log"
    )
    
    logging.info(f"Model name: {args.model_name}")
    logging.info(f'Eval dataset {LM_EVAL_TASK_KWARGS_DICT}')
    logging.info(f'Logging filename: {f"run_log/no_quantize/eval_{args.model_name}.log"}')
    
        
    save_file_path = os.path.join(f"autogptq_eval_result", f"eval_result_{args.model_name}_pile.log")
    print(save_file_path)
    # os.makedirs(save_file_path, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)


    model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    
    
    all_metrics = {}
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as file:
            all_metrics = json.load(file)

    for task_kwargs in LM_EVAL_TASK_KWARGS_DICT.values():
        logging.info(f"Evaluating task: {task_kwargs['task']}")
        task_name = task_kwargs["task"]
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=task_kwargs["batch_size"],
        )
        initialize_tasks(verbosity="ERROR")
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

        with open(save_file_path, 'w') as file:
            json.dump(all_metrics, file, indent=4)
            
    logging.info(">>>>> Results <<<<<")
    if args.is_quantized:
        logging.info(f"Quantization on {args.model_name}")
    else:
        logging.info(f"No quantization on {args.model_name}")
    logging.info(f"Metrics: {all_metrics}")


# CUDA_VISIBLE_DEVICES=0,1,2 python lm_eval_no_quantize.py
# CUDA_VISIBLE_DEVICES=0,1,2 nohup python lm_eval_no_quantize.py --model_name="deepseek-ai/deepseek-moe-16b-base" > run_log/no_quantize/eval_deepseek.log 2>&1 &



# /home/ycuser01/xiaolong/moe_quantize/quantize_gptq_llama_moe.py