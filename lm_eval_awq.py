import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from awq import AutoAWQForCausalLM
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks

LM_EVAL_TASK_KWARGS_DICT = {
    "winogrande": {"task": "winogrande", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    "copa": {"task": "copa", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    "openbookqa": {"task": "openbookqa", "num_fewshot": 0, "batch_size": 128, "metric": "acc_norm"},
    "hellaswag": {"task": "hellaswag", "num_fewshot": 0, "batch_size": 128, "metric": "acc_norm"},
    "lambada_openai": {"task": "lambada_openai", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    "rte": {"task": "rte", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    "piqa": {"task": "piqa", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    "mmlu": {"task": "mmlu", "num_fewshot": 5, "batch_size": 16, "metric": "acc"},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
    parser.add_argument("--model_name", type=str, default='deepseek-ai/deepseek-moe-16b-chat')
    parser.add_argument("--quant_model_path", type=str)
    parser.add_argument("--bits", type=str)
    parser.add_argument(
        "--per_gpu_max_memory",
        type=int,
        default=None,
        help="Max memory used in each GPU.",
    )
    parser.add_argument("--cpu_max_memory", type=int, default=None, help="Mx memory used in CPU.")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_memory = {}
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update({i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())})
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    model = AutoAWQForCausalLM.from_quantized(
        args.model_name, args.quant_model_path, fuse_layers=False
    )

    save_file_path = os.path.join(f"{args.quant_model_path.split('/')[0]}",
                                  f"eval_result_{args.quant_model_path.split('/')[-1]}")
    all_metrics = {}
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as file:
            all_metrics = json.load(file)

    for task_kwargs in LM_EVAL_TASK_KWARGS_DICT.values():
        print(f"Evaluating task: {task_kwargs['task']}")
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

    print(">>>>> Results <<<<<")
    print(f"Quantization on {args.model_name}")

    print(f"Metrics: {all_metrics}")
