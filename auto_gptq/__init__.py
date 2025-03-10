from .modeling import AutoGPTQForCausalLM, BaseQuantizeConfig, AutoGPTQForCausalLM_mixed_precision, BaseQuantizeConfig_mixed_precision
from .utils.exllama_utils import exllama_set_max_input_length
from .utils.peft_utils import get_gptq_peft_model

from .utils.mixed_precision import moe_quantize_config, moe_quantize_config_layer
from .utils.bit_config_deepseek import deepseek_quantize_config
__version__ = "0.8.0.dev0"
