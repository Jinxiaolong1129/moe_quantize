import torch
from transformers import AwqConfig, AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-2-7b-hf"

quantization_config = AwqConfig(
    bits=4,
    fuse_max_seq_len=512,
    do_fuse=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config).to(0)


model.quantize(tokenizer)


print(model)
print('Done')