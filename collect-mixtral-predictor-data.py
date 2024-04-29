# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/4/29
import os

import torch
from fire import Fire
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import MixtralForCausalLM
from copy import deepcopy
plt.rcParams['font.family'] = 'Times New Roman'


@torch.no_grad()
def collect_mixtral_predictor_data(save_dir="/data/data4/pingzhi/data"):
    model = MixtralForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.bfloat16, device_map="auto"
    )

    # block_flatten_weight = []
    # for block in tqdm(model.model.layers):
    #     ffn = block.block_sparse_moe
    #     ffn_weight = torch.cat(
    #         [exp.w1.weight.data.flatten() for exp in ffn.experts] + (
    #             [exp.w2.weight.data.flatten() for exp in ffn.experts]) + (
    #             [exp.w3.weight.data.flatten() for exp in ffn.experts])
    #     ).flatten().cpu()
    #     block_flatten_weight.append(ffn_weight)



    def _custom_ffn_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass




if __name__ == "__main__":
    Fire(collect_mixtral_predictor_data)
