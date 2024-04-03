# Load model directly
from transformers import AutoModelForCausalLM
from transformers import AutoConfig

# cache_dir = '/home/LeiFeng/xiaolong/moe_quantize/baichuan/'
# model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-7B-Base", trust_remote_code=True,
#                                             cache_dir=cache_dir)

model_dir = 'baichuan/models--baichuan-inc--Baichuan2-7B-Base/snapshots/f9d4d8dd2f7a3dbede3bda3b0cf0224e9272bbe5'
config = AutoConfig.from_pretrained(
    model_dir, trust_remote_code=True
)
print(config)
print('Done')