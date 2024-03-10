import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['HF_HOME'] = '/data3/user/jin509/hf_cache'

from pickle import NONE
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from datasets import load_dataset


import torch
from transformers import AutoConfig

from awq import AutoAWQForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
print(f'Using GPUs: {os.environ["CUDA_VISIBLE_DEVICES"]}')
print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from moduleformer import ModuleFormerForCausalLM, ModuleFormerConfig, ModuleFormerForSequenceClassification
AutoConfig.register("moduleformer", ModuleFormerConfig)
AutoModelForCausalLM.register(ModuleFormerConfig, ModuleFormerForCausalLM)
AutoModelForSequenceClassification.register(ModuleFormerConfig, ModuleFormerForSequenceClassification)

tokenizer = AutoTokenizer.from_pretrained('ibm/MoLM-350M-4B')
model = AutoModelForCausalLM.from_pretrained('ibm/MoLM-350M-4B')

# def get_calib_dataset(
#     data,
#     tokenizer=None,
#     n_samples=512,
#     block_size=512,
#     split="train",
#     text_column="text",
# ):
#     if isinstance(data, str):
#         if data == "pileval":
#             dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
#         else:
#             dataset = load_dataset(data, split=split)

#         dataset = dataset.shuffle(seed=42)

#     elif isinstance(data, list):
#         if isinstance(data[0], str):
#             dataset = [{text_column: text} for text in data]
#         elif isinstance(data[0][0], int):
#             dataset = data
#         else:
#             raise NotImplementedError(
#                 "Either pass a string to a huggingface dataset or a list"
#                 "that is preprocessed with one sample of text per element"
#                 " or a list of list of int for tokenized words."
#             )
#     else:
#         raise NotImplementedError(
#             "Either pass a string to a huggingface dataset or a list"
#             "that is preprocessed with one sample of text per element"
#             " or a list of list of int for tokenized words."
#         )

#     samples = []
#     n_run = 0
#     for data in dataset:
#         if isinstance(data, list):
#             line_encoded = data
#         else:
#             line = data[text_column]
#             line = line.strip()
#             line_encoded = tokenizer.encode(line)
#         if len(line_encoded) > 512:
#             continue
#         sample = torch.tensor([line_encoded])
#         if sample.numel() == 0:
#             continue
#         samples.append(sample)
#         n_run += 1
#         if n_run == n_samples:
#             break
#     # now concatenate all samples and split according to block size
#     cat_samples = torch.cat(samples, dim=1)
#     n_split = cat_samples.shape[1] // block_size
#     return [
#         cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
#     ], []




def get_calib_dataset(
    data,
    tokenizer,
    n_samples=512,
    block_size=512,
    split="train",
    text_column="text",
):
    # Load and shuffle the dataset
    if isinstance(data, str):
        dataset = load_dataset(data, split=split) if data != "pileval" else load_dataset("mit-han-lab/pile-val-backup", split="validation")
        dataset = dataset.shuffle(seed=42)
    elif isinstance(data, list) and isinstance(data[0], str):
        dataset = [{text_column: text} for text in data]
    else:
        raise NotImplementedError("Data should be a dataset name or a list of strings.")

    # Initialize lists to store encoder and decoder samples
    encoder_samples, decoder_samples = [], []

    # Iterate over the dataset to prepare encoder and decoder inputs
    for n_run, item in enumerate(dataset):
        if n_run >= n_samples:
            break

        # Tokenize the text
        text = item[text_column].strip()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if 0 < len(tokens) <= block_size:
            # Prepare encoder input
            encoder_input = torch.tensor([tokens])
            encoder_samples.append(encoder_input)

            # Prepare decoder input: shift right and add <sos> token at the beginning
            decoder_input = torch.tensor([[0] + tokens[:-1]])
            decoder_samples.append(decoder_input)

    # Concatenate all samples and split according to block size
    encoder_samples = torch.cat(encoder_samples, dim=1)
    decoder_samples = torch.cat(decoder_samples, dim=1)

    # Split samples into blocks
    n_split = encoder_samples.shape[1] // block_size
    encoder_blocks = [encoder_samples[:, i * block_size: (i + 1) * block_size] for i in range(n_split)]
    decoder_blocks = [decoder_samples[:, i * block_size: (i + 1) * block_size] for i in range(n_split)]

    return encoder_blocks, decoder_blocks




if __name__ == "__main__":

    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    # custom_cache_dir = "/data4/share/xiaolong/Llama-2-7b-chat-hf"
    # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)
    # dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
    # gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)

    # quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=gptq_config,cache_dir=custom_cache_dir)

    custom_cache_dir = "/data4/share/xiaolong/switch_transformer"
    models = ["google/switch-base-8"]

    for model_name in models:
        print(f"Running analysis on {model_name}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)
        
        # token1 = tokenizer.bos_token
        # token2 = tokenizer.eos_token
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=custom_cache_dir)
        # model = model.to(device)


        # input_text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
        # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)

        # model.generate(input_ids)


        encoder_input_id, decoder_input_id = get_calib_dataset(
            data='pileval',
            tokenizer=tokenizer,
            n_samples=32,
            block_size=512,
            split='train',
            text_column='text',
        )

        encoder_input = torch.cat(encoder_input_id, dim=0).to(device)
        decoder_input = torch.cat(decoder_input_id, dim=0).to(device)

        # outputs = model(input_ids=encoder_input, decoder_input_ids=decoder_input)

        w_bit = 4


        model_path = "google/switch-base-8"


        # model_path = "/data4/share/xiaolong/switch_transformer/models--google--switch-base-8"
        # cache_dir = "/data4/share/xiaolong/switch_transformer/models--google--switch-base-8"
        quant_path = f'/data4/share/xiaolong/switch_transformer-awq-w_bit_{w_bit}'


        print(f'Quantizing with w_bit={w_bit}')
        quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": w_bit, "version": "GEMM" }

        print(f'Saving quantized model at "{quant_path}"')


        # Load model
        model = AutoAWQForSeq2SeqLM.from_pretrained(
            model_path=model_path, safetensors=False, **{"low_cpu_mem_usage": True, "use_cache": True}
        )


        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # TODO (xiaolong): quantize start here
        model.quantize(tokenizer, quant_config=quant_config)



        # Save quantized model
        model.save_quantized(quant_path)
        tokenizer.save_pretrained(quant_path)

        print(f'Model is quantized and saved at "{quant_path}"')


        # inputs = tokenizer(samples, max_length=512, truncation=True, padding="max_length", return_tensors="pt")

        # # Forward pass through the model
        # model.eval()  # Set the model to evaluation mode
        # with torch.no_grad():  # Disable gradient calculation
        #     outputs = model(samples)

        print(f"Analysis on {model_name} complete")
