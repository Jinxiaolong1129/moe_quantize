# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/4/29
import os.path
import random

import torch
from colossalai.moe.layers import SparseMLP
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm

from openmoe import OpenMoeForCausalLM, OpenMoeTokenizer


@torch.no_grad()
def collect_openmoe_ffn_predictor_train_data(
        seq_len=1024,
        num_samples=400,
        save_dir="/data2/pzli/qmoe_data/ffn_input_output_pairs"
):
    model = OpenMoeForCausalLM.from_pretrained(
        "OrionZheng/openmoe-8b", torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = OpenMoeTokenizer.from_pretrained("OrionZheng/openmoe-8b")

    block_ffn_input_output_pair = {}

    def _custom_ffn_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_token = hidden_states.detach().clone().cpu()
        original_output = self._original_forward(hidden_states)
        output_token = original_output[0].detach().clone().cpu()
        with torch.no_grad():
            block_ffn_input_output_pair[self._module_name].append((input_token, output_token))
        return original_output

    for name, module in model.named_modules():
        if isinstance(module, SparseMLP):
            block_ffn_input_output_pair[name] = []
            module._original_forward = module.forward
            module._module_name = name
            module.forward = _custom_ffn_forward.__get__(module, type(module))

    model.eval()

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "".join([" \n" if s == "" else s for s in data["text"][-2000:]])
    encoded_text = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(num_samples):
        i = random.randint(0, encoded_text.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = encoded_text.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    for i, data in enumerate(tqdm(dataset)):
        with torch.no_grad():
            model(**data)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for key, pairs in block_ffn_input_output_pair.items():
        torch.save(pairs, f"{save_dir}/{key}.pt")
        print(f"Saved at {save_dir}/{key}.pt")


if __name__ == "__main__":
    Fire(collect_openmoe_ffn_predictor_train_data)