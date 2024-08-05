import tqdm
from typing import List, Tuple
from .base import BaseAWQForSeq2SeqLM
import torch
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
    modules_to_not_convert = ["mlp.router.classifier"]


    @staticmethod
    def get_model_layers(model: OldSwitchTransformersModel):
        layers = [model.encoder.block + model.decoder.block]
        return layers

    @staticmethod
    def get_act_for_scaling(module: OldSwitchTransformersBlock):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldSwitchTransformersModel, device: str):
        # TODO just for the encoder 
        model.encoder.embed_tokens = model.encoder.embed_tokens.to(device)
        model.decoder.embed_tokens = model.decoder.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldSwitchTransformersBlock, input_feat, module_kwargs):
        # TODO (xiaolong): just for encoder
        if not module.is_decoder: # encoder
            if not module.is_sparse:
                layers = [] 

                # attention input
                layers.append(
                    dict(
                        prev_op=module.layer[0].layer_norm,
                        layers=[
                            module.layer[0].SelfAttention.q,
                            module.layer[0].SelfAttention.k,
                            module.layer[0].SelfAttention.v
                        ],
                        inp=input_feat["layer.0.SelfAttention.q"],
                        module2inspect=module.layer[0].SelfAttention,
                        kwargs=module_kwargs,
                    )
                )

                # attention out
                # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
                if module.layer[0].SelfAttention.v.weight.shape == module.layer[0].SelfAttention.o.weight.shape:
                    layers.append(
                        dict(
                            prev_op= module.layer[0].SelfAttention.v,
                            layers=[ module.layer[0].SelfAttention.o ],
                            inp=input_feat["layer.0.SelfAttention.o"],
                        )
                    )

                # linear 1
                layers.append(
                    dict(
                        prev_op=module.layer[1].layer_norm,
                        layers=[module.layer[1].mlp.wi],
                        inp=input_feat["layer.1.mlp.wi"],
                        module2inspect=module.layer[1].mlp,
                    )
                )

                # linear 2
                layers.append(
                    dict(
                        prev_op=module.layer[1].mlp.wi,
                        layers=[module.layer[1].mlp.wo],
                        inp=input_feat["layer.1.mlp.wo"],
                    )
                )

                return layers
            else:
                layers = [] 

                # attention input
                layers.append(
                    dict(
                        prev_op=module.layer[0].layer_norm,
                        layers=[
                            module.layer[0].SelfAttention.q,
                            module.layer[0].SelfAttention.k,
                            module.layer[0].SelfAttention.v
                        ],
                        inp=input_feat["layer.0.SelfAttention.q"],
                        module2inspect=module.layer[0].SelfAttention,
                        kwargs=module_kwargs,
                    )
                )

                # attention out
                # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
                if module.layer[0].SelfAttention.v.weight.shape == module.layer[0].SelfAttention.o.weight.shape:
                    layers.append(
                        dict(
                            prev_op= module.layer[0].SelfAttention.v,
                            layers=[ module.layer[0].SelfAttention.o ],
                            inp=input_feat["layer.0.SelfAttention.o"],
                        )
                    )
                
                layers.append(
                    dict(
                        prev_op=module.layer[1].layer_norm,
                        layers=[
                            w
                            for expert in module.layer[1].mlp.experts
                            for w in [module.layer[1].mlp.experts[expert].wi]
                        ],
                        inp=input_feat["mlp"],
                        module2inspect=module.layer[1].mlp,
                    )
                )

                for i, expert in enumerate(module.layer[1].mlp.experts):
                    layers.append(
                        dict(
                            prev_op=module.layer[1].mlp.experts[expert].wi,
                            layers=[module.layer[1].mlp.experts[expert].wo],
                            inp=input_feat[f"layer.1.mlp.experts.expert_{i}.wo"],
                        )
                    )

                return layers
        else:
            if not module.is_sparse:
                layers = [] 
                # attention input
                layers.append(
                    dict(
                        prev_op=module.layer[0].layer_norm,
                        layers=[
                            module.layer[0].SelfAttention.q,
                            module.layer[0].SelfAttention.k,
                            module.layer[0].SelfAttention.v
                        ],
                        inp=input_feat["layer.0.SelfAttention.q"],
                        module2inspect=module.layer[0].SelfAttention,
                        kwargs=module_kwargs,
                    )
                )

                # attention out
                # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
                if module.layer[0].SelfAttention.v.weight.shape == module.layer[0].SelfAttention.o.weight.shape:
                    layers.append(
                        dict(
                            prev_op= module.layer[0].SelfAttention.v,
                            layers=[ module.layer[0].SelfAttention.o ],
                            inp=input_feat["layer.0.SelfAttention.o"],
                        )
                    )

                layers.append(
                    dict(
                        prev_op=module.layer[1].layer_norm,
                        layers=[
                            module.layer[1].EncDecAttention.q,
                            module.layer[1].EncDecAttention.k,
                            module.layer[1].EncDecAttention.v
                        ],
                        inp=input_feat["layer.1.EncDecAttention.q"],
                        module2inspect=module.layer[1].EncDecAttention,
                        kwargs=module_kwargs,
                    )
                )

                # attention out
                # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
                if module.layer[1].EncDecAttention.v.weight.shape == module.layer[1].EncDecAttention.o.weight.shape:
                    layers.append(
                        dict(
                            prev_op= module.layer[1].EncDecAttention.q,
                            layers=[ module.layer[1].EncDecAttention.k ],
                            inp=input_feat["layer.1.EncDecAttention.o"],
                        )
                    )


                # linear 1
                layers.append(
                    dict(
                        prev_op=module.layer[2].layer_norm,
                        layers=[module.layer[2].mlp.wi],
                        inp=input_feat["layer.2.mlp.wi"],
                        module2inspect=module.layer[2].mlp,
                    )
                )

                # linear 2
                layers.append(
                    dict(
                        prev_op=module.layer[2].mlp.wi,
                        layers=[module.layer[2].mlp.wo],
                        inp=input_feat["layer.2.mlp.wo"],
                    )
                )

                return layers
            else:
                layers = [] 
                # attention input
                layers.append(
                    dict(
                        prev_op=module.layer[0].layer_norm,
                        layers=[
                            module.layer[0].SelfAttention.q,
                            module.layer[0].SelfAttention.k,
                            module.layer[0].SelfAttention.v
                        ],
                        inp=input_feat["layer.0.SelfAttention.q"],
                        module2inspect=module.layer[0].SelfAttention,
                        kwargs=module_kwargs,
                    )
                )

                # attention out
                # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
                if module.layer[0].SelfAttention.v.weight.shape == module.layer[0].SelfAttention.o.weight.shape:
                    layers.append(
                        dict(
                            prev_op= module.layer[0].SelfAttention.v,
                            layers=[ module.layer[0].SelfAttention.o ],
                            inp=input_feat["layer.0.SelfAttention.o"],
                        )
                    )

                layers.append(
                    dict(
                        prev_op=module.layer[1].layer_norm,
                        layers=[
                            module.layer[1].EncDecAttention.q,
                            module.layer[1].EncDecAttention.k,
                            module.layer[1].EncDecAttention.v
                        ],
                        inp=input_feat["layer.1.EncDecAttention.q"],
                        module2inspect=module.layer[1].EncDecAttention,
                        kwargs=module_kwargs,
                    )
                )

                # attention out
                # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
                if module.layer[1].EncDecAttention.v.weight.shape == module.layer[1].EncDecAttention.o.weight.shape:
                    layers.append(
                        dict(
                            prev_op= module.layer[1].EncDecAttention.q,
                            layers=[ module.layer[1].EncDecAttention.k ],
                            inp=input_feat["layer.1.EncDecAttention.o"],
                        )
                    )


                layers.append(
                    dict(
                        prev_op=module.layer[2].layer_norm,
                        layers=[
                            w
                            for expert in module.layer[2].mlp.experts
                            for w in [module.layer[2].mlp.experts[expert].wi]
                        ],
                        inp=input_feat["mlp"],
                        module2inspect=module.layer[2].mlp,
                    )
                )

                for i, expert in enumerate(module.layer[2].mlp.experts):
                    layers.append(
                        dict(
                            prev_op=module.layer[2].mlp.experts[expert].wi,
                            layers=[module.layer[2].mlp.experts[expert].wo],
                            inp=input_feat[f"layer.2.mlp.experts.expert_{i}.wo"],
                        )
                    )

                return layers            
