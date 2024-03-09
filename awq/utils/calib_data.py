import torch
import logging
from typing import List, Union
from datasets import load_dataset


def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = "pileval",
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
    logging.debug(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]



def get_calib_dataset_switchtransformer(
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


