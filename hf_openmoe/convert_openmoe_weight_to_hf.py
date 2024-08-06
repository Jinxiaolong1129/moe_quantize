# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/8/6
import os
from copy import deepcopy

from fire import Fire

from hf_openmoe import HFOpenMoeForCausalLM, HFOpenMoeConfig, HFOpenMoeTokenizer
from openmoe import OpenMoeForCausalLM as OriginalOpenMoeForCausalLM


def convert_openmoe_weight_to_hf_openmoe(input_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = HFOpenMoeConfig.from_pretrained(input_dir)  # this should work
    tokenizer = HFOpenMoeTokenizer.from_pretrained(input_dir)

    original_model = OriginalOpenMoeForCausalLM.from_pretrained(input_dir)
    state_dict = deepcopy(original_model.state_dict())

    hf_openmoe_model = HFOpenMoeForCausalLM(config)

    # rename
    moe_blocks = [i for i in range(config.num_hidden_layers) if (i + 1) % config.moe_layer_interval == 0]

    for block_id in moe_blocks:
        # gate
        state_dict[f"model.layers.{block_id}.mlp.gate.weight"] = state_dict.pop(
            f"model.layers.{block_id}.mlp.gate_weight")
        # experts
        for expert_id in range(config.num_experts):
            state_dict[f"model.layers.{block_id}.mlp.experts.{expert_id}.gate_proj.weight"] = state_dict[
                f"model.layers.{block_id}.mlp.experts.wi_gate"][expert_id].t()
            state_dict[f"model.layers.{block_id}.mlp.experts.{expert_id}.up_proj.weight"] = state_dict[
                f"model.layers.{block_id}.mlp.experts.wi_up"][expert_id].t()
            state_dict[f"model.layers.{block_id}.mlp.experts.{expert_id}.down_proj.weight"] = state_dict[
                f"model.layers.{block_id}.mlp.experts.wo"][expert_id].t()
        state_dict.pop(f"model.layers.{block_id}.mlp.experts.wi_gate")
        state_dict.pop(f"model.layers.{block_id}.mlp.experts.wi_up")
        state_dict.pop(f"model.layers.{block_id}.mlp.experts.wo")

    print(f"Saving converted checkpoints...")
    hf_openmoe_model.load_state_dict(state_dict)
    hf_openmoe_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved at {output_dir}")


if __name__ == "__main__":
    Fire(convert_openmoe_weight_to_hf_openmoe)
