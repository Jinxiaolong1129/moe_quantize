import torch
import argparse
import re

BITS = [
    # NOTE other block 4bit
    'moe.shared_8.top1_8.other_2+other_block.4',
    'moe.shared_8.top1_4.other_2+other_block.4',
    'moe.shared_8.top2_8.other_2+other_block.4',
    'moe.shared_8.top2_4.other_2+other_block.4',
    'moe.shared_4.top2_8.other_2+other_block.4',
    'moe.shared_4.top2_4.other_2+other_block.4',

    # NOTE other block 8bit
    'moe.shared_8.top1_8.other_2+other_block.8',
    'moe.shared_8.top1_4.other_2+other_block.8',
    'moe.shared_8.top2_8.other_2+other_block.8',
    'moe.shared_8.top2_4.other_2+other_block.8',
    'moe.shared_4.top2_8.other_2+other_block.8',
    'moe.shared_4.top2_4.other_2+other_block.8',
    
    # FIXME wait for test
    'moe.shared_4.top1_8.other_2+other_block.4',
    'moe.shared_4.top1_4.other_2+other_block.4',
    
    'moe.shared_4.top1_8.other_2+other_block.8',
    'moe.shared_4.top1_4.other_2+other_block.8',
]


TEST = [
    # NOTE other block 4bit
    # FIXME 'moe.shared_8.top{x}_8.other_2+other_block.4',
    
    # FIXME 'moe.shared_8.top{x}_4.other_2+other_block.4',
    
    # 'moe.shared_8.top{x}_4.other_2+other_block.4',
    
]

def get_top_expert(matrix):
    sorted_values, sorted_indices = torch.sort(matrix, dim=1,descending=True)

    sorted_expert_indices_by_block = {}

    # Fill the dictionary
    for i in range(matrix.shape[0]):  
        # i+1 : moe block from 2nd
        sorted_expert_indices_by_block[i+1] = sorted_indices[i].tolist()
    return sorted_expert_indices_by_block



# distribution_matrix = torch.load('/home/LeiFeng/xiaolong/moe_quantize/save/routing-count.pt')
# sorted_expert_indices_by_block = get_top_expert(distribution_matrix)
    
    



def parse_config_string(config_string):
    # Initialize the config dictionary with default values
    config_dict = {
        "moe.shared_experts": 0,
        "moe.experts": 0,
        "moe.experts.top_index": 0,
        "moe.experts.top": 0,
        "attention": 0,
    }

    # Extract the shared_experts value
    shared_experts_match = re.search(r"shared_(\d+)", config_string)
    if shared_experts_match:
        config_dict["moe.shared_experts"] = int(shared_experts_match.group(1))

    # Extract the experts value
    experts_match = re.search(r"other_(\d+)", config_string)
    if experts_match:
        config_dict["moe.experts"] = int(experts_match.group(1))

    # Extract the top_index and top values
    top_index_match = re.search(r"top(\d+)_", config_string)
    top_match = re.search(r"top\d+_(\d+)", config_string)
    if top_index_match and top_match:
        config_dict["moe.experts.top_index"] = int(top_index_match.group(1))
        config_dict["moe.experts.top"] = int(top_match.group(1))

    # Extract the attention value
    attention_match = re.search(r"other_block\.(\d+)", config_string)
    if attention_match:
        config_dict["attention"] = int(attention_match.group(1))

    return config_dict




def generate_expert_topx_moe_quantize_bit_config(bit_config_str):
    distribution_matrix = None
    sorted_expert_indices_by_block = get_top_expert(distribution_matrix)
    
    bit_config_dict = parse_config_string(bit_config_str)  # replace args.bits with your actual pattern string
    return bit_config_dict
    
    
def generate_bit_config():
    config_dict = {}
    for x in [5,10,15,20,25,30]:
        bit_config_str = f'moe.shared_8.top{x}_8.other_2+other_block.4',
        bit_config_dict = generate_expert_topx_moe_quantize_bit_config(bit_config_str)
        config_dict[x] = bit_config_dict
        
    return config_dict


def generate_deeepseek_bit(bit_config_str):
    bit_config_dict = generate_expert_topx_moe_quantize_bit_config(bit_config_str)
    
    moe_block_bit_dict = {}

    for i in range(4):
        key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
        moe_block_bit_dict[key] = bit_config_dict['attention']

    deeepseek_bit = {
        'model.layers.0.self_attn.q_proj': bit_config_dict['attention'], 
        'model.layers.0.self_attn.k_proj': bit_config_dict['attention'],
        'model.layers.0.self_attn.v_proj': bit_config_dict['attention'],
        'model.layers.0.self_attn.o_proj': bit_config_dict['attention'],
        'model.layers.0.mlp.gate_proj': bit_config_dict['attention'],
        'model.layers.0.mlp.up_proj': bit_config_dict['attention'],
        'model.layers.0.mlp.down_proj': bit_config_dict['attention']
    }
    
    for i in range(64):
        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.experts.{i}.{part}"
            moe_block_bit_dict[key] = bit_config_dict['moe.experts']

    for part in ['gate_proj', 'up_proj', 'down_proj']:
        key = f"mlp.shared_experts.{part}"
        moe_block_bit_dict[key] = bit_config_dict['moe.shared_experts']


    top_index = bit_config_dict['moe.experts.top_index']
    for block_num in range(1, 28):
        for layer in moe_block_bit_dict:
            topk = sorted_expert_indices_by_block[block_num][0:top_index]
            if 'mlp.experts' in layer:
                moe_index = layer.split('.')[2]
                if int(moe_index) in topk:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = bit_config_dict['moe.experts.top']
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = bit_config_dict['moe.experts']
            else:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
    return deeepseek_bit

    
    

def moe_quantize_config(args):

    # only 2,4,8
    if args.bits == 'all_4':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 4

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 4

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }
        
        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit
        
    if args.bits == 'all_2':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 2

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 2

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 2, 
            'model.layers.0.self_attn.k_proj': 2,
            'model.layers.0.self_attn.v_proj': 2,
            'model.layers.0.self_attn.o_proj': 2,
            'model.layers.0.mlp.gate_proj': 2,
            'model.layers.0.mlp.up_proj': 2,
            'model.layers.0.mlp.down_proj': 2
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit

    if args.bits == 'all_3':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 3

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 3

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 3

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 3, 
            'model.layers.0.self_attn.k_proj': 3,
            'model.layers.0.self_attn.v_proj': 3,
            'model.layers.0.self_attn.o_proj': 3,
            'model.layers.0.mlp.gate_proj': 3,
            'model.layers.0.mlp.up_proj': 3,
            'model.layers.0.mlp.down_proj': 3
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit
    
    if args.bits == 'all_8':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 8

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 8

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 8

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 8, 
            'model.layers.0.self_attn.k_proj': 8,
            'model.layers.0.self_attn.v_proj': 8,
            'model.layers.0.self_attn.o_proj': 8,
            'model.layers.0.mlp.gate_proj': 8,
            'model.layers.0.mlp.up_proj': 8,
            'model.layers.0.mlp.down_proj': 8
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit

    # no special expert bit
    if args.bits == 'moe.all_mlp.2+other_block.4':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 2

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit

    if args.bits == 'moe.shared_4.other.2+other_block_4':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 4

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit

    if args.bits == "moe.shared_2.other.4+other_block_4":
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 4

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 2

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit
    
    if args.bits == "moe.all_mlp.4+other_block.8":
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 8

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 4

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 4

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 8, 
            'model.layers.0.self_attn.k_proj': 8,
            'model.layers.0.self_attn.v_proj': 8,
            'model.layers.0.self_attn.o_proj': 8,
            'model.layers.0.mlp.gate_proj': 8,
            'model.layers.0.mlp.up_proj': 8,
            'model.layers.0.mlp.down_proj': 8
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit
    
    if args.bits == 'moe.shared_4.other.2+other_block.8':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 8

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 4

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 8, 
            'model.layers.0.self_attn.k_proj': 8,
            'model.layers.0.self_attn.v_proj': 8,
            'model.layers.0.self_attn.o_proj': 8,
            'model.layers.0.mlp.gate_proj': 8,
            'model.layers.0.mlp.up_proj': 8,
            'model.layers.0.mlp.down_proj': 8
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit
    
    if args.bits == 'moe.shared_2.other.4+other_block.8':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 8

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 4

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 2

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 8, 
            'model.layers.0.self_attn.k_proj': 8,
            'model.layers.0.self_attn.v_proj': 8,
            'model.layers.0.self_attn.o_proj': 8,
            'model.layers.0.mlp.gate_proj': 8,
            'model.layers.0.mlp.up_proj': 8,
            'model.layers.0.mlp.down_proj': 8
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit

    args.bits = 'moe.shared_28'

    distribution_matrix = None
    sorted_expert_indices_by_block = get_top_expert(distribution_matrix)
    
    # top moe expert bit
    # other_block 4bit
    if args.bits == 'moe.shared_8.top1_8.other_2+other_block.4':
        # other block(attention) 4bit
        # moe shared 8bit, top1 8bit, other 2bit
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 8

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                # top1 
                top1 = sorted_expert_indices_by_block[block_num][0]
                if 'mlp.experts' in layer:
                    moe_index = layer.split('.')[2]
                    if int(moe_index) == int(top1):
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 8
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit

    
    if args.bits == 'moe.shared_8.top1_4.other_2+other_block.4':
        # other block(attention) 4bit
        # moe shared 8bit, top1 8bit, other 2bit
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 8

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                # top1 
                top1 = sorted_expert_indices_by_block[block_num][0]
                if 'mlp.experts' in layer:
                    moe_index = layer.split('.')[2]
                    if int(moe_index) == int(top1):
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 4
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit


    if args.bits == 'moe.shared_8.top2_8.other_2+other_block.4':
        # other block(attention) 4bit
        # moe shared 8bit, top1 8bit, other 2bit
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 8

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                # top1 
                top2 = sorted_expert_indices_by_block[block_num][0:2]
                if 'mlp.experts' in layer:
                    moe_index = layer.split('.')[2]
                    if int(moe_index) in top2:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 8
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit


    if args.bits == 'moe.shared_8.top2_4.other_2+other_block.4':
        # other block(attention) 4bit
        # moe shared 8bit, top1 8bit, other 2bit
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 8

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                # top1 
                top2 = sorted_expert_indices_by_block[block_num][0:2]
                if 'mlp.experts' in layer:
                    moe_index = layer.split('.')[2]
                    if int(moe_index) in top2:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 4
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit


    if args.bits == 'moe.shared_4.top2_8.other_2+other_block.4':
        # other block(attention) 4bit
        # moe shared 8bit, top1 8bit, other 2bit
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 4

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                # top1 
                top2 = sorted_expert_indices_by_block[block_num][0:2]
                if 'mlp.experts' in layer:
                    moe_index = layer.split('.')[2]
                    if int(moe_index) in top2:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 8
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit


    if args.bits == 'moe.shared_4.top2_4.other_2+other_block.4':
        # other block(attention) 4bit
        # moe shared 8bit, top1 8bit, other 2bit
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 4

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                # top1 
                top2 = sorted_expert_indices_by_block[block_num][0:2]
                if 'mlp.experts' in layer:
                    moe_index = layer.split('.')[2]
                    if int(moe_index) in top2:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 4
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit



    # top moe expert bit
    # other_block 4bit
    if args.bits == 'moe.shared_8.top1_8.other_2+other_block.8':
        # other block(attention) 4bit
        # moe shared 8bit, top1 8bit, other 2bit
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 8

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 8, 
            'model.layers.0.self_attn.k_proj': 8,
            'model.layers.0.self_attn.v_proj': 8,
            'model.layers.0.self_attn.o_proj': 8,
            'model.layers.0.mlp.gate_proj': 8,
            'model.layers.0.mlp.up_proj': 8,
            'model.layers.0.mlp.down_proj': 8
        }
        
        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 8



        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                # top1 
                top1 = sorted_expert_indices_by_block[block_num][0]
                if 'mlp.experts' in layer:
                    moe_index = layer.split('.')[2]
                    if int(moe_index) == int(top1):
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 8
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit

    
    if args.bits == 'moe.shared_8.top1_4.other_2+other_block.8':
        # other block(attention) 4bit
        # moe shared 8bit, top1 8bit, other 2bit
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 8

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 8

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 8, 
            'model.layers.0.self_attn.k_proj': 8,
            'model.layers.0.self_attn.v_proj': 8,
            'model.layers.0.self_attn.o_proj': 8,
            'model.layers.0.mlp.gate_proj': 8,
            'model.layers.0.mlp.up_proj': 8,
            'model.layers.0.mlp.down_proj': 8
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                # top1 
                top1 = sorted_expert_indices_by_block[block_num][0]
                if 'mlp.experts' in layer:
                    moe_index = layer.split('.')[2]
                    if int(moe_index) == int(top1):
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 4
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit


    if args.bits == 'moe.shared_8.top2_8.other_2+other_block.8':
        # other block(attention) 4bit
        # moe shared 8bit, top1 8bit, other 2bit
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 8

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 8

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 8, 
            'model.layers.0.self_attn.k_proj': 8,
            'model.layers.0.self_attn.v_proj': 8,
            'model.layers.0.self_attn.o_proj': 8,
            'model.layers.0.mlp.gate_proj': 8,
            'model.layers.0.mlp.up_proj': 8,
            'model.layers.0.mlp.down_proj': 8
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                # top1 
                top2 = sorted_expert_indices_by_block[block_num][0:2]
                if 'mlp.experts' in layer:
                    moe_index = layer.split('.')[2]
                    if int(moe_index) in top2:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 8
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit


    if args.bits == 'moe.shared_8.top2_4.other_2+other_block.8':
        # other block(attention) 4bit
        # moe shared 8bit, top1 8bit, other 2bit
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 8

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 8

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 8, 
            'model.layers.0.self_attn.k_proj': 8,
            'model.layers.0.self_attn.v_proj': 8,
            'model.layers.0.self_attn.o_proj': 8,
            'model.layers.0.mlp.gate_proj': 8,
            'model.layers.0.mlp.up_proj': 8,
            'model.layers.0.mlp.down_proj': 8
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                # top1 
                top2 = sorted_expert_indices_by_block[block_num][0:2]
                if 'mlp.experts' in layer:
                    moe_index = layer.split('.')[2]
                    if int(moe_index) in top2:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 4
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit


    if args.bits == 'moe.shared_4.top2_8.other_2+other_block.8':
        # other block(attention) 4bit
        # moe shared 8bit, top1 8bit, other 2bit
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 8

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 4

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 8, 
            'model.layers.0.self_attn.k_proj': 8,
            'model.layers.0.self_attn.v_proj': 8,
            'model.layers.0.self_attn.o_proj': 8,
            'model.layers.0.mlp.gate_proj': 8,
            'model.layers.0.mlp.up_proj': 8,
            'model.layers.0.mlp.down_proj': 8
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                # top1 
                top2 = sorted_expert_indices_by_block[block_num][0:2]
                if 'mlp.experts' in layer:
                    moe_index = layer.split('.')[2]
                    if int(moe_index) in top2:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 8
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit


    if args.bits == 'moe.shared_4.top2_4.other_2+other_block.8':
        # other block(attention) 4bit
        # moe shared 8bit, top1 8bit, other 2bit
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 8

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 4

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 8, 
            'model.layers.0.self_attn.k_proj': 8,
            'model.layers.0.self_attn.v_proj': 8,
            'model.layers.0.self_attn.o_proj': 8,
            'model.layers.0.mlp.gate_proj': 8,
            'model.layers.0.mlp.up_proj': 8,
            'model.layers.0.mlp.down_proj': 8
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                # top1 
                top2 = sorted_expert_indices_by_block[block_num][0:2]
                if 'mlp.experts' in layer:
                    moe_index = layer.split('.')[2]
                    if int(moe_index) in top2:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 4
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit


    # top k
    pattern = r'^moe\.shared_\d+\.top\d+_\d+\.other_\d+\+other_block\.\d+$'
    if re.match(pattern, args.bits):
        deeepseek_bit = generate_deeepseek_bit(args.bits)
        return deeepseek_bit
    
    raise ValueError("Invalid bits")


def moe_quantize_config_layer(args):
    # moe.shared_8.top25_4.other_2+other_block.4+startlayer_5
    pattern = r"^moe\.shared_8\.top25_4\.other_2\+other_block\.4\+"

    if bool(re.match(pattern, args.bits)):
        
        
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 8

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }

        pattern = r"(startlayer|endlayer)_(\d+)"

        match = re.search(pattern, args.bits)
        num_layer = int(match.group(2)) if match else None

        if 'startlayer' in args.bits:
            for block_num in range(1, num_layer+1):
                for layer in moe_block_bit_dict:
                    top25 = sorted_expert_indices_by_block[block_num][0:25]
                    if 'mlp.experts' in layer:
                        moe_index = layer.split('.')[2]
                        if int(moe_index) in top25:
                            key = f'model.layers.{block_num}' + '.' + layer
                            deeepseek_bit[key] = 2
                        else:
                            key = f'model.layers.{block_num}' + '.' + layer
                            deeepseek_bit[key] = 2
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = moe_block_bit_dict[layer]
                        
                        
            for block_num in range(num_layer+1, 28):
                for layer in moe_block_bit_dict:
                    top25 = sorted_expert_indices_by_block[block_num][0:25]
                    if 'mlp.experts' in layer:
                        moe_index = layer.split('.')[2]
                        if int(moe_index) in top25:
                            key = f'model.layers.{block_num}' + '.' + layer
                            deeepseek_bit[key] = 4
                        else:
                            key = f'model.layers.{block_num}' + '.' + layer
                            deeepseek_bit[key] = 2
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = moe_block_bit_dict[layer]

        elif 'endlayer' in args.bits:
            blocks = list(range(1, 28))
            for block_num in blocks[:-num_layer]:
                for layer in moe_block_bit_dict:
                    top25 = sorted_expert_indices_by_block[block_num][0:25]
                    if 'mlp.experts' in layer:
                        moe_index = layer.split('.')[2]
                        if int(moe_index) in top25:
                            key = f'model.layers.{block_num}' + '.' + layer
                            deeepseek_bit[key] = 4
                        else:
                            key = f'model.layers.{block_num}' + '.' + layer
                            deeepseek_bit[key] = 2
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = moe_block_bit_dict[layer]
                        
                        
            for block_num in blocks[-num_layer:]:
                for layer in moe_block_bit_dict:
                    top25 = sorted_expert_indices_by_block[block_num][0:25]
                    if 'mlp.experts' in layer:
                        moe_index = layer.split('.')[2]
                        if int(moe_index) in top25:
                            key = f'model.layers.{block_num}' + '.' + layer
                            deeepseek_bit[key] = 2
                        else:
                            key = f'model.layers.{block_num}' + '.' + layer
                            deeepseek_bit[key] = 2
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = moe_block_bit_dict[layer]

        return deeepseek_bit


    
    

if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the bits argument
    parser.add_argument('--bits', type=str, default='moe.shared_8.top1_8.other_2+other_block.4')

    # Parse the arguments
    args = parser.parse_args()
    
    config_str = 'moe.shared_8.top10_4.other_2+other_block.4'
    args.bits = config_str

    bit = moe_quantize_config(args)
    print(bit)







# {
#     moe.shared_experts: 8,
#     moe.experts: 4,
#     moe.experts.top_index : 4,
#     moe.experts.top : 8,
#     attention: 2,
# }