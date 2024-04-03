import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM


# from transformers.models.llama.modeling_llama import (
#     LlamaDecoderLayer as OldLlamaDecoderLayer,
#     LlamaForCausalLM as OldLlamaForCausalLM,
# )


class DeepSeekAWQForCausalLM(BaseAWQForCausalLM):
    # NOTE (xiaolong): This is a custom model, so we need to define the model type in layer_type, you can check mixtral
    layer_type = "LlamaDecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):

        # NOTE (xiaolong): define the layers for scaling, and the input_feat is the input feature for the model, layers contains the order of the layers in the model
        if not hasattr(module.mlp, 'experts'):
        # deepseek first layer is non-moe based model, so we need to define the layers for the first layer
            layers = []

            # attention input
            layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[
                        module.self_attn.q_proj,
                        module.self_attn.k_proj,
                        module.self_attn.v_proj,
                    ],
                    inp=input_feat["self_attn.q_proj"],
                    module2inspect=module.self_attn,
                    kwargs=module_kwargs,
                )
            )

            # attention out
            # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
            if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=module.self_attn.v_proj,
                        layers=[module.self_attn.o_proj],
                        inp=input_feat["self_attn.o_proj"],
                    )
                )

            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=module.mlp,
                )
            )
            layers.append(
                dict(
                    prev_op=module.mlp.up_proj,
                    layers=[module.mlp.down_proj],
                    inp=input_feat[f"mlp.down_proj"],
                )
            )

        elif hasattr(module.mlp, 'experts'):

            layers = []

            # attention input
            layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[
                        module.self_attn.q_proj,
                        module.self_attn.k_proj,
                        module.self_attn.v_proj,
                    ],
                    inp=input_feat["self_attn.q_proj"],
                    module2inspect=module.self_attn,
                    kwargs=module_kwargs,
                )
            )

            # attention out
            # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
            if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=module.self_attn.v_proj,
                        layers=[module.self_attn.o_proj],
                        inp=input_feat["self_attn.o_proj"],
                    )
                )

            # linear 1
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    # layers=[module.mlp.gate_proj, module.mlp.gate_proj],
                    layers=[
                        w
                        for expert in module.mlp.experts
                        for w in [expert.gate_proj, expert.up_proj]
                    ],
                    inp=input_feat["block_sparse_moe"],
                    module2inspect=module.mlp,
                )
            )

            # linear out
            for i, expert in enumerate(module.mlp.experts):
                layers.append(
                    dict(
                        prev_op=expert.up_proj,
                        layers=[expert.down_proj],
                        inp=input_feat[f"mlp.experts.{i}.down_proj"],
                    )
                )
                
            # shared expert gate_proj up_proj
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.shared_experts.gate_proj, module.mlp.shared_experts.up_proj],
                    inp=input_feat["block_sparse_moe"],
                    module2inspect=module.mlp,
                )
            )

            # shared expert down_proj
            layers.append(
                dict(
                    prev_op=module.mlp.shared_experts.up_proj,
                    layers=[module.mlp.shared_experts.down_proj],
                    inp=input_feat[f"mlp.shared_experts.down_proj"],
                )
            )
                
        return layers

