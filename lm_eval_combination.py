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
from optimum.gptq import GPTQQuantizer, load_quantized_model, GPTQQuantizer_deepseek
import torch
import random
from argparse import ArgumentParser

from transformers import AutoTokenizer, TextGenerationPipeline
import logging
from datasets import load_dataset

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig, AutoGPTQForCausalLM_mixed_precision, BaseQuantizeConfig_mixed_precision


LM_EVAL_TASK_KWARGS_DICT = {
    "winogrande": {"task": "winogrande", "num_fewshot": 0, "batch_size": 16, "metric": "acc"},
    "copa": {"task": "copa", "num_fewshot": 0, "batch_size": 16, "metric": "acc"},
    "openbookqa": {"task": "openbookqa", "num_fewshot": 0, "batch_size": 16, "metric": "acc_norm"},
    "hellaswag": {"task": "hellaswag", "num_fewshot": 0, "batch_size": 64, "metric": "acc_norm"},
    
    # "lambada_openai": {"task": "lambada_openai", "num_fewshot": 0, "batch_size": 16, "metric": "acc"},
    # "rte": {"task": "rte", "num_fewshot": 0, "batch_size": 16, "metric": "acc"},
    
    "piqa": {"task": "piqa", "num_fewshot": 0, "batch_size": 16, "metric": "acc"},
    "mmlu": {"task": "mmlu", "num_fewshot": 5, "batch_size": 16, "metric": "acc"},
}


def lm_eval_gptq(args, model):
    logging.info(f"Model name: {args.model_name}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=args.use_fast_tokenizer)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # if args.is_quantized:
    #     args.quantized_model_file_base_name = f'{args.model_name.split("/")[-1]}-gptq_w_bit_{args.bits}'
        
    #     max_memory = {}
    #     if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
    #         if torch.cuda.is_available():
    #             max_memory.update({i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())})
    #     if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
    #         max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    #     if not max_memory:
    #         max_memory = None

    #     if args.use_safetensors:
    #         logging.info(
    #             "The argument --use_safetensors is deprecrated and will be removed in the next release. It is now the default behavior."
    #         )

    #     model = AutoGPTQForCausalLM_mixed_precision.from_quantized(
    #         args.quant_path,
    #         low_cpu_mem_usage=True,
    #         device_map="auto",
    #         max_memory=max_memory,
    #         model_basename=args.quantized_model_file_base_name,
    #         use_safetensors=True,
    #         trust_remote_code=True,
    #         inject_fused_mlp=False,
    #         inject_fused_attention=False,
    #         # disable_exllama=args.disable_exllama,
    #     )

    save_file_path = os.path.join(f"{args.quant_path.split('/')[0]}", f"eval_result_{args.quant_path.split('/')[-1]}_pile")
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










if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
    parser.add_argument("--model_name", type=str, default='deepseek-ai/deepseek-moe-16b-base')
    parser.add_argument("--quant_model_path", type=str)
    parser.add_argument("--bits", type=str)
    
    # parser.add_argument("--model_basename", type=str, default=None, help="Model file's basename.")
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

    args.bits = args.quant_model_path.split('w_bit_')[-1]
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f"run_log/gptq/eval_quantize_gptq_deepseek_{args.bits}.log"
    )
    
    logging.info(f"Model name: {args.model_name}")
    logging.info(f'Eval dataset {LM_EVAL_TASK_KWARGS_DICT}')
    logging.info(f'Quantized model path: {args.quant_model_path}')
    logging.info(f'Bits: {args.bits}')
    logging.info(f'Logging filename: {f"run_log/gptq/eval_quantize_gptq_deepseek_{args.bits}.log"}')
    
    
        
    # save_file_path = os.path.join(f"{args.quant_model_path.split('/')[0]}", f"eval_result_{args.quant_model_path.split('/')[-1]}_pile")
    save_file_path = os.path.join(f"autogptq_eval_result/combination/{args.model_name.split('/')[1]}", f"eval_result_{args.quant_model_path.split('/')[-1]}_pile")
    os.makedirs(f"autogptq_eval_result/combination/{args.model_name.split('/')[1]}", exist_ok=True)
    
    if args.is_quantized:
        args.quantized_model_file_base_name = f'{args.model_name.split("/")[-1]}-gptq_w_bit_{args.bits}'
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=args.use_fast_tokenizer)
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

        if args.use_safetensors:
            logging.info(
                "The argument --use_safetensors is deprecrated and will be removed in the next release. It is now the default behavior."
            )

        model = AutoGPTQForCausalLM_mixed_precision.from_quantized(
            args.quant_model_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=max_memory,
            model_basename=args.quantized_model_file_base_name,
            use_safetensors=True,
            trust_remote_code=True,
            inject_fused_mlp=False,
            inject_fused_attention=False,
            # disable_exllama=args.disable_exllama,
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

