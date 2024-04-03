import sys
sys.path.append("/home/LeiFeng/xiaolong/moe_quantize/optimum/")  # Add the path to Python's search path
print(sys.path)


import os
# os.environ['HF_HOME'] = '/home/LeiFeng/xiaolong/moe_quantize/hf_cache'
# os.makedirs(os.environ['HF_HOME'], exist_ok=True)


from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model, GPTQQuantizer_deepseek
import torch

from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging


moe_block_bit_dict = {}

for i in range(4):
    key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
    moe_block_bit_dict[key] = 4

for i in range(64):
    for part in ['gate_proj', 'up_proj', 'down_proj']:
        key = f"mlp.experts.{i}.{part}"
        moe_block_bit_dict[key] = 4

for part in ['gate_proj', 'up_proj', 'down_proj']:
    key = f"mlp.shared_experts.{part}"
    moe_block_bit_dict[key] = 4

deeepseek_bit = {
    'model.layers.0.self_attn.q_proj': 2, 
    'model.layers.0.self_attn.k_proj': 2,
    'model.layers.0.self_attn.v_proj': 2,
    'model.layers.0.self_attn.o_proj': 2,
    'model.layers.0.mlp.gate_proj': 4,
    'model.layers.0.mlp.up_proj': 4,
    'model.layers.0.mlp.down_proj': 4
}

for block_num in range(1, 28):
    for layer in moe_block_bit_dict:
        key = f'model.layers.{block_num}' + '.' + layer
        deeepseek_bit[key] = moe_block_bit_dict



w_bit = 4

model_name = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"
quant_path = f'/home/LeiFeng/xiaolong/moe_quantize/quantized_LLaMA-MoE-v1-3_5B-2_8-gptq_w_bit_{w_bit}'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)

modules_in_block_to_quantize = []

quantizer = GPTQQuantizer(bits=w_bit, dataset="wikitext2", 
                        #   block_name_to_quantize = "model.decoder.layers", 
                          model_seqlen = 4096)


quantized_model = quantizer.quantize_model(model, tokenizer)

quantizer.save(model, quant_path)

# load quantized model to the first GPU
# model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")


model = AutoModelForCausalLM.from_quantized(quant_path, device="cuda:0")

# download quantized model from Hugging Face Hub and load to the first GPU
# model = AutoGPTQForCausalLM.from_quantized(repo_id, device="cuda:0", use_safetensors=True, use_triton=False)

# inference with model.generate

output = model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device))

# print(tokenizer.decode())

# or you can also use pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

print(pipeline("auto-gptq is")[0]["generated_text"])