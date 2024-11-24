import os.path
import random
from typing import Tuple, Optional

import torch
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralSparseMoeBlock, MixtralDecoderLayer
from transformers import AutoModelForCausalLM


def get_Pile_dataset(tokenizer, seqlen: int, nsamples: int, split: str = "train"):
    # custom_cache_dir = '/home/LeiFeng/xiaolong/moe_quantize/data/minipile/'
    data = load_dataset('mit-han-lab/pile-val-backup')['validation']

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


@torch.no_grad()
def collect_deepseek_ffn_predictor_train_data(
        seq_len=4096,
        num_samples=400,
        save_dir="data/deepseek/ffn_input_output_pairs"
):
    model = AutoModelForCausalLM.from_pretrained(
        'deepseek-ai/deepseek-moe-16b-base', device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-moe-16b-base')

    def _custom_ffn_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_token = hidden_states.detach().clone().cpu()
        original_output = self._original_forward(hidden_states)
        output_token = original_output[0].detach().clone().cpu()
        with torch.no_grad():
            block_ffn_input_output_pair[self._module_name].append((input_token, output_token))
        return original_output

    block_ffn_input_output_pair = {}
    for name, module in model.named_modules():
        if type(module).__name__ == 'DeepseekMoE':
            block_ffn_input_output_pair[name] = []
            module._original_forward = module.forward
            module._module_name = name
            module.forward = _custom_ffn_forward.__get__(module, type(module))

    model.eval()
    
    dataset = get_Pile_dataset(tokenizer=tokenizer, seqlen=seq_len, nsamples=num_samples, split="train")

    for i, data in enumerate(tqdm(dataset)):
        with torch.no_grad():
            data = {key: value.to(model.device) for key, value in data.items()}
            model(**data)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for key, pairs in block_ffn_input_output_pair.items():
        torch.save(pairs, f"{save_dir}/{key}.pt")
        print(f"Saved at {save_dir}/{key}.pt")


@torch.no_grad()
def collect_deepseek_predictor_test_data(
        seq_len=4096,
        num_samples=128,
        save_dir="data/deepseek/ffn_input_output_pairs/testset",
        dataset_name: str = "wikitext"
):
    model = AutoModelForCausalLM.from_pretrained(
        'deepseek-ai/deepseek-moe-16b-base', device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-moe-16b-base')

    def _custom_ffn_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_token = hidden_states.detach().clone().cpu()
        original_output = self._original_forward(hidden_states)
        output_token = original_output[0].detach().clone().cpu()
        with torch.no_grad():
            block_ffn_input_output_pair[self._module_name].append((input_token, output_token))
        return original_output

    block_ffn_input_output_pair = {}
    for name, module in model.named_modules():
        if type(module).__name__ == 'DeepseekMoE':
            
            block_ffn_input_output_pair[name] = []
            module._original_forward = module.forward
            module._module_name = name
            module.forward = _custom_ffn_forward.__get__(module, type(module))

    model.eval()

    dataset = get_Pile_dataset(tokenizer=tokenizer, seqlen=seq_len, nsamples=num_samples, split="train")

    for i, data in enumerate(tqdm(dataset)):
        with torch.no_grad():
            data = {key: value.to(model.device) for key, value in data.items()}
            model(**data)
            
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for key, pairs in block_ffn_input_output_pair.items():
        torch.save(pairs, f"{save_dir}/{key}.pt")
        print(f"Saved at {save_dir}/{key}.pt")


@torch.no_grad()
def collect_deepseek_ffn_with_residual_predictor_train_data(
        seq_len=1024,
        num_samples=400,
        save_dir="data/deepseek/ffn_input_output_pairs_with_residual/"
        
):
    model = AutoModelForCausalLM.from_pretrained(
        'deepseek-ai/deepseek-moe-16b-base', device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-moe-16b-base')

    def _custom_decoder_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            output_router_logits: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual.to(hidden_states.device) + hidden_states

        # Fully Connected
        input_token = hidden_states.detach().clone().cpu()  # added
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states
        output_token = hidden_states.detach().clone().cpu()  # added
        with torch.no_grad():
            block_ffn_input_output_pair[self._module_name].append((input_token, output_token))  # added

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

    block_ffn_input_output_pair = {}
    for name, module in model.named_modules():
        if type(module).__name__ == 'DeepseekDecoderLayer':
            block_ffn_input_output_pair[name] = []
            module._module_name = name
            module.forward = _custom_decoder_forward.__get__(module, type(module))

    model.eval()

    dataset = get_Pile_dataset(tokenizer=tokenizer, seqlen=seq_len, nsamples=num_samples, split="train")

    for i, data in enumerate(tqdm(dataset)):
        with torch.no_grad():
            model(**data)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for key, pairs in block_ffn_input_output_pair.items():
        torch.save(pairs, f"{save_dir}/{key}.pt")
        print(f"Saved at {save_dir}/{key}.pt")


@torch.no_grad()
def collect_deepseek_ffn_with_residual_predictor_test_data(
        seq_len=4096,
        num_samples=128,
        save_dir="data/deepseek/ffn_input_output_pairs_with_residual/testset"
):
    model = AutoModelForCausalLM.from_pretrained(
        'deepseek-ai/deepseek-moe-16b-base', device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-moe-16b-base')

    def _custom_decoder_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            output_router_logits: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual.to(hidden_states.device) + hidden_states

        # Fully Connected
        input_token = hidden_states.detach().clone().cpu()  # added
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states
        output_token = hidden_states.detach().clone().cpu()  # added
        with torch.no_grad():
            block_ffn_input_output_pair[self._module_name].append((input_token, output_token))  # added

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

    block_ffn_input_output_pair = {}
    for name, module in model.named_modules():
        if type(module).__name__ == 'DeepseekDecoderLayer':
            block_ffn_input_output_pair[name] = []
            module._module_name = name
            module.forward = _custom_decoder_forward.__get__(module, type(module))

    model.eval()

    dataset = get_Pile_dataset(tokenizer=tokenizer, seqlen=seq_len, nsamples=num_samples, split="train")

    for i, data in enumerate(tqdm(dataset)):
        with torch.no_grad():
            model(**data)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for key, pairs in block_ffn_input_output_pair.items():
        torch.save(pairs, f"{save_dir}/{key}.pt")
        print(f"Saved at {save_dir}/{key}.pt")


@torch.no_grad()
def collect_deepseek_ffn_mse_loss(
        seq_len=4096,
        num_samples=128,
        save_dir="results/deepseek/ffn_mse_loss"
):
    model = AutoModelForCausalLM.from_pretrained(
        'deepseek-ai/deepseek-moe-16b-base', device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-moe-16b-base')

    # def _custom_decoder_forward(
    #         self,
    #         hidden_states: torch.Tensor,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         position_ids: Optional[torch.LongTensor] = None,
    #         past_key_value: Optional[Tuple[torch.Tensor]] = None,
    #         output_attentions: Optional[bool] = False,
    #         output_router_logits: Optional[bool] = False,
    #         use_cache: Optional[bool] = False,
    #         **kwargs
    # ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    #     residual = hidden_states
    #
    #     hidden_states = self.input_layernorm(hidden_states)
    #
    #     # Self Attention
    #     hidden_states, self_attn_weights, present_key_value = self.self_attn(
    #         hidden_states=hidden_states,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_value=past_key_value,
    #         output_attentions=output_attentions,
    #         use_cache=use_cache,
    #     )
    #     hidden_states = residual.to(hidden_states.device) + hidden_states
    #
    #     # Fully Connected
    #     input_token = hidden_states.detach().clone().cpu()  # added
    #     residual = hidden_states
    #     hidden_states = self.post_attention_layernorm(hidden_states)
    #     hidden_states, router_logits = self.block_sparse_moe(hidden_states)
    #     hidden_states = residual + hidden_states
    #     output_token = hidden_states.detach().clone().cpu()  # added
    #     with torch.no_grad():
    #         block_ffn_input_output_pair[self._module_name].append(
    #             torch.nn.functional.mse_loss(input_token.float(), output_token.float(), reduction="mean").item()
    #         )  # added
    #
    #     outputs = (hidden_states,)
    #
    #     if output_attentions:
    #         outputs += (self_attn_weights,)
    #
    #     if use_cache:
    #         outputs += (present_key_value,)
    #
    #     if output_router_logits:
    #         outputs += (router_logits,)
    #
    #     return outputs

    def _custom_ffn_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_token = hidden_states.detach().clone().cpu()
        original_output = self._original_forward(hidden_states)
        output_token = original_output[0].detach().clone().cpu()
        with torch.no_grad():
            block_ffn_input_output_pair[self._module_name].append(
                torch.nn.functional.mse_loss(input_token.float(), output_token.float(), reduction="mean").item()
            )
        return original_output

    block_ffn_input_output_pair = {}
    for name, module in model.named_modules():
        if type(module).__name__ == 'DeepSeekMoE':
            block_ffn_input_output_pair[name] = []
            module._module_name = name
            module.forward = _custom_ffn_forward.__get__(module, type(module))

    model.eval()

    dataset = get_Pile_dataset(tokenizer=tokenizer, seqlen=seq_len, nsamples=num_samples, split="train")

    for i, data in enumerate(tqdm(dataset)):
        with torch.no_grad():
            model(**data)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    block_ffn_input_output_pair = {
        key: torch.tensor(value).mean().item()
        for key, value in block_ffn_input_output_pair.items()
    }
    print(block_ffn_input_output_pair)
    print(list(block_ffn_input_output_pair.values()))


def main(function_name = None):
    if function_name is None:
        raise ValueError("Please specify a function name to run.")
    elif function_name == 'collect_deepseek_ffn_predictor_train_data':
        collect_deepseek_ffn_predictor_train_data()
    elif function_name == 'collect_deepseek_predictor_test_data':
        collect_deepseek_predictor_test_data()
    elif function_name == 'collect_deepseek_ffn_with_residual_predictor_train_data':
        collect_deepseek_ffn_with_residual_predictor_train_data()
    elif function_name == 'collect_deepseek_ffn_with_residual_predictor_test_data':
        collect_deepseek_ffn_with_residual_predictor_test_data()
        

if __name__ == "__main__":
    Fire(collect_deepseek_ffn_mse_loss)