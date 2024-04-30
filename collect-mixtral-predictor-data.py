# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/4/29

import torch
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralSparseMoeBlock


@torch.no_grad()
def collect_mixtral_predictor_data(
        seq_len=1024,
        num_samples=400,
        chunk_size=100000,
        save_dir="/data/data4/pingzhi/data"
):
    model = MixtralForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    config = model.config

    def _custom_ffn_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_token = hidden_states.detach().clone().cpu()
        original_output = self._original_forward(hidden_states)
        output_token = original_output.detach().clone().cpu()
        with torch.no_grad():
            block_ffn_input_output_pair[self._module_name].append((input_token, output_token))
        return original_output

    block_ffn_input_output_pair = {}
    for name, module in model.named_modules():
        if isinstance(module, MixtralSparseMoeBlock):
            block_ffn_input_output_pair[name] = []
            module._original_forward = module.forward
            module._module_name = name
            module.forward = _custom_ffn_forward.__get__(model, type(module))

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

    num_tokens = len(block_ffn_input_output_pair[list(block_ffn_input_output_pair.keys())[0]])
    print(f"Total number of tokens: {num_tokens}")
    print(f"Chunk size: {chunk_size}")
    print(f"Number of chunks: {num_tokens // chunk_size}")
    for start_id in range(0, num_tokens, chunk_size):
        chunk = {}
        for name, input_output_pair in block_ffn_input_output_pair.items():
            chunk[name] = input_output_pair[start_id:start_id + chunk_size]
        torch.save(chunk, f"{save_dir}/chunk_{start_id}.pt")
        print(f"Saved chunk {start_id}/{num_tokens}")


if __name__ == "__main__":
    Fire(collect_mixtral_predictor_data)
