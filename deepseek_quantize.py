
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ['HF_HOME'] = '/data3/user/jin509/hf_cache'


import torch
from transformers import AutoConfig

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
print(f'Using GPUs: {os.environ["CUDA_VISIBLE_DEVICES"]}')
print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')

w_bit = 4

models = ["deepseek-ai/deepseek-moe-16b-chat"]

custom_cache_dir = "/data3/user/jin509/deepseek"

if not os.path.exists(custom_cache_dir):
    os.makedirs(custom_cache_dir)

for model_name in models:
    print(f"Running analysis on {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda:8")  
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, 
                                                    cache_dir=custom_cache_dir, trust_remote_code=True).to(device)
    break

# model_path = "google/switch-base-8"
# custom_cache_dir = "/data4/share/xiaolong/switch_transformer"
# quant_path = f'/data4/share/xiaolong/switch_transformer-w_bit_{w_bit}'

# model = AutoModelForSeq2SeqLM.from_pretrained(model_path, cache_dir=custom_cache_dir, device_map="auto")


model_path = 'llama-moe/LLaMA-MoE-v1-3_5B-2_8'
quant_path = f'/data4/share/xiaolong/llama-moe/LLaMA-MoE-v1-3_5B-2_8-w_bit_{w_bit}'
custom_cache_dir = "/data3/user/jin509/llama-moe"


print(f'Quantizing with w_bit={w_bit}')
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": w_bit, "version": "GEMM" }

print(f'Saving quantized model at "{quant_path}"')

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path=model_path, safetensors=False, **{"low_cpu_mem_usage": True, "use_cache": True, "cache_dir": custom_cache_dir }
)



tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')