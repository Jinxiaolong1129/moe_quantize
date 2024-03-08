import tqdm
from typing import List, Tuple
from .base import BaseAWQForSeq2SeqLM

from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersBlock as OldSwitchTransformersBlock,
    SwitchTransformersModel as OldSwitchTransformersModel,
)
from awq.modules.fused.norm import FasterTransformerRMSNorm



# NOTE (xiaolong): switch transformer is Seq2SeqLM, so the layers are different, need to check the layers in the model
# NOTE (xiaolong): we need to define BaseAWQForSeq2SeqLM in base.py


class SwitchAWQ(BaseAWQForSeq2SeqLM):
    layer_type = "SwitchTransformersBlock"
    max_seq_len_key = "max_position_embeddings"


    @staticmethod
    def get_model_layers(model: OldSwitchTransformersModel):
        model.decoder.block
        model.encoder.block
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldSwitchTransformersBlock):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldSwitchTransformersModel, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldSwitchTransformersBlock, input_feat, module_kwargs):
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
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers

