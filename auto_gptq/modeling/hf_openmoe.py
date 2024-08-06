# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/8/5
from ._base import BaseGPTQForCausalLM_mixed_precision


class HFOpenMoeGPTQForCausalLM(BaseGPTQForCausalLM_mixed_precision):
    model_type = "openmoe"
    layer_type = "HFOpenMoeDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]

    moe_1_list = []
    moe_2_list = []

    for part in ['gate_proj', 'up_proj']:
        for i in range(32):
            key = f"mlp.experts.{i}.{part}"
            moe_1_list.append(key)

    for part in ['gate_proj', 'up_proj']:
        key = f"extra_mlp.{part}"
        moe_1_list.append(key)

    for i in range(32):
        for part in ['down_proj']:
            key = f"mlp.experts.{i}.{part}"
            moe_2_list.append(key)

    for part in ['down_proj']:
        key = f"extra_mlp.{part}"
        moe_2_list.append(key)

    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        moe_1_list,
        moe_2_list
    ]

    normal_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]


__all__ = ["HFOpenMoeGPTQForCausalLM"]
