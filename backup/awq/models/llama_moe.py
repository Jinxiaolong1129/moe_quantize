import tqdm
import torch
from typing import List, Tuple
from .base import BaseAWQForCausalLM


# LlamaAWQForCausalLM
class LlamaMOEAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "LlamaMoEDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["gate"]

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
    def get_layers_for_scaling(
        module, input_feat, module_kwargs
    ):
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
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        # linear in
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[
                    w
                    for expert in module.block_sparse_moe.experts
                    for w in [expert.w1, expert.w3]
                ],
                inp=input_feat["block_sparse_moe"],
                module2inspect=module.block_sparse_moe,
            )
        )

        # linear out
        for i, expert in enumerate(module.block_sparse_moe.experts):
            layers.append(
                dict(
                    prev_op=expert.w3,
                    layers=[expert.w2],
                    inp=input_feat[f"block_sparse_moe.experts.{i}.w2"],
                )
            )

        return layers

