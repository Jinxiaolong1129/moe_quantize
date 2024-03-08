import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from pickle import NONE
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from datasets import load_dataset


def get_calib_dataset(
    data,
    tokenizer=None,
    n_samples=512,
    block_size=512,
    split="train",
    text_column="text",
):
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        else:
            dataset = load_dataset(data, split=split)

        dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words."
            )
    else:
        raise NotImplementedError(
            "Either pass a string to a huggingface dataset or a list"
            "that is preprocessed with one sample of text per element"
            " or a list of list of int for tokenized words."
        )

    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]


if __name__ == "__main__":


    custom_cache_dir = "/data4/share/xiaolong/switch_transformer"
    models = ["google/switch-base-8"]

    for model_name in models:
        print(f"Running analysis on {model_name}")

        model_name = "meta-llama/Llama-2-7b-chat-hf"
        custom_cache_dir = "/data4/share/xiaolong/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)
        dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
        gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)

        quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=gptq_config,cache_dir=custom_cache_dir)









        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=custom_cache_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)

        input_text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)

        outputs = model.generate(input_ids)
        print(tokenizer.decode(outputs[0]))

        samples = get_calib_dataset(
            data='pileval',
            tokenizer=tokenizer,
            n_samples=32,
            block_size=512,
            split='train',
            text_column='text',
        )
        samples = torch.cat(samples, dim=0) # [65 512]
        samples = samples.to(device)
        # inputs = tokenizer(samples, max_length=512, truncation=True, padding="max_length", return_tensors="pt")

        # Forward pass through the model
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(samples)

        print(f"Analysis on {model_name} complete")
