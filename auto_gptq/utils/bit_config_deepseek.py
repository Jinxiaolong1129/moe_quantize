import torch
import argparse
import re
import random

def get_top_expert(matrix):
    sorted_values, sorted_indices = torch.sort(matrix, dim=1,descending=True)

    sorted_expert_indices_by_block = {}

    # Fill the dictionary
    for i in range(matrix.shape[0]):  
        # i+1 : moe block from 2nd
        sorted_expert_indices_by_block[i+1] = sorted_indices[i].tolist()
    return sorted_expert_indices_by_block



distribution_matrix = torch.load('save/deepseek-routing-count.pt')
sorted_expert_indices_by_block = get_top_expert(distribution_matrix)
    





def generate_deeepseek_bit_topk(bit_config_str):
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


    bit_config_dict = parse_config_string(bit_config_str)
    
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


def generate_deeepseek_bit_topk_random(bit_config_str):
    def parse_config_string(config_string):
        config_dict = {
            "moe.shared_experts": 0,
            "moe.experts": 0,
            "moe.experts.random_top_index": 0,
            "moe.experts.random_top": 0,
            "attention": 0,
        }

        shared_experts_match = re.search(r"shared_(\d+)", config_string)
        if shared_experts_match:
            config_dict["moe.shared_experts"] = int(shared_experts_match.group(1))

        experts_match = re.search(r"other_(\d+)", config_string)
        if experts_match:
            config_dict["moe.experts"] = int(experts_match.group(1))

        random_top_index_match = re.search(r"random(\d+)_", config_string)
        random_top_match = re.search(r"random\d+_(\d+)", config_string)
        
        if random_top_index_match and random_top_match:
            config_dict["moe.experts.random_top_index"] = int(random_top_index_match.group(1))
            config_dict["moe.experts.random_top"] = int(random_top_match.group(1))

        attention_match = re.search(r"other_block\.(\d+)", config_string)
        if attention_match:
            config_dict["attention"] = int(attention_match.group(1))

        return config_dict

    bit_config_dict = parse_config_string(bit_config_str)
    
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


    random_top_index = bit_config_dict['moe.experts.random_top_index']
    for block_num in range(1, 28):
        for layer in moe_block_bit_dict:
            random_topk = random.sample(range(64), random_top_index)
            if 'mlp.experts' in layer:
                moe_index = layer.split('.')[2]
                if int(moe_index) in random_topk:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = bit_config_dict['moe.experts.random_top']
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = bit_config_dict['moe.experts']
            else:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
    return deeepseek_bit


# moe.shared_4.other_4+other_block.4+startlayer_4
def generate_deeepseek_start_end_layer(bit_config_str):
    def parse_config_string(config_string):
        config_dict = {
            "moe.shared_experts": 0,
            "moe.experts": 0,
            "moe.experts.startlayer": 0,
            "moe.experts.endlayer": 0,
            "moe.experts.randomlayer": 0,
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

        attention_match = re.search(r"other_block\.(\d+)", config_string)
        if attention_match:
            config_dict["attention"] = int(attention_match.group(1))

        layer = re.search(r"(startlayer|endlayer|randomlayer)_(\d+)", config_string)
        if 'startlayer' in layer.group(1):
            config_dict["moe.experts.startlayer"] = int(layer.group(2))
        elif 'endlayer' in layer.group(1):
            config_dict["moe.experts.endlayer"] = int(layer.group(2))
        elif 'randomlayer' in layer.group(1):
            config_dict["moe.experts.randomlayer"] = int(layer.group(2))
        return config_dict



    bit_config_dict = parse_config_string(bit_config_str)
    
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

    if 'startlayer' in bit_config_str:
        num_layer = bit_config_dict['moe.experts.startlayer']
        for block_num in range(1, num_layer+1):
            for layer in moe_block_bit_dict:
                if 'mlp.experts' in layer:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = 4
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]  
                                        
        for block_num in range(num_layer+1, 28):
            for layer in moe_block_bit_dict:
                if 'mlp.experts' in layer:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]  
                    
                    
    elif 'endlayer' in bit_config_str:
        num_layer = bit_config_dict['moe.experts.endlayer']
        blocks = list(range(1, 28))
        for block_num in blocks[:-num_layer]:
            for layer in moe_block_bit_dict:
                if 'mlp.experts' in layer:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]
                    
                    
        for block_num in blocks[-num_layer:]:
            for layer in moe_block_bit_dict:
                if 'mlp.experts' in layer:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = 4
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]  
                    
                    
    elif 'randomlayer' in bit_config_str:
        num_layer = bit_config_dict['moe.experts.randomlayer']
        blocks = list(range(1, 28))
        high_bit_layers = random.sample(blocks, num_layer)
        
        for block_num in blocks:
            for layer in moe_block_bit_dict:
                if 'mlp.experts' in layer:
                    key = f'model.layers.{block_num}.{layer}'
                    deeepseek_bit[key] = 4 if block_num in high_bit_layers else 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]  
                    
    return deeepseek_bit



# moe.shared_4.other_4+other_block.4+dejavu_4
def generate_deeepseek_dejavu_layer(bit_config_str):
    dejavu_predict_residual = [27, 26,  1,  2,  3,  8,  5,  6,  7,  9, 10,  4, 14, 12, 11, 13, 15, 16, 17, 25, 18, 20, 19, 21, 22, 23, 24]
    
    def parse_config_string(config_string):
        config_dict = {
            "moe.shared_experts": 0,
            "moe.experts": 0,
            "moe.experts.dejavu": 0,
            "attention": 0,
        }

        shared_experts_match = re.search(r"shared_(\d+)", config_string)
        if shared_experts_match:
            config_dict["moe.shared_experts"] = int(shared_experts_match.group(1))

        # Extract the experts value
        experts_match = re.search(r"other_(\d+)", config_string)
        if experts_match:
            config_dict["moe.experts"] = int(experts_match.group(1))

        attention_match = re.search(r"other_block\.(\d+)", config_string)
        if attention_match:
            config_dict["attention"] = int(attention_match.group(1))

        layer = re.search(r"(dejavu)_(\d+)", config_string)
        if 'dejavu' in layer.group(1):
            config_dict["moe.experts.dejavu"] = int(layer.group(2))

        return config_dict

    bit_config_dict = parse_config_string(bit_config_str)
    
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

    if 'dejavu' in bit_config_str:
        num_layer = bit_config_dict['moe.experts.dejavu']
        blocks = list(range(1, 28))
        high_bit_layers = dejavu_predict_residual[:num_layer]
        
        for block_num in blocks:
            for layer in moe_block_bit_dict:
                if 'mlp.experts' in layer:
                    key = f'model.layers.{block_num}.{layer}'
                    deeepseek_bit[key] = 4 if block_num in high_bit_layers else 2
                else:
                    key = f'model.layers.{block_num}' + '.' + layer
                    deeepseek_bit[key] = moe_block_bit_dict[layer]  
                    
    return deeepseek_bit


# moe.shared_4.other_4+other_block.4+outlier_top25
# moe.shared_4.other_4+other_block.4+outlier_random25
def generate_deeepseek_outlier(bit_config_str):
    path = 'save/deepseek_linear_weight_outlier_metric.pt'
    outlier_data = torch.load(path)
    filtered_data = {k: v for k, v in outlier_data.items() if 'mlp.experts' in k}
    outlier_data = dict(sorted(filtered_data.items(), key=lambda item: item[1], reverse=True))
    
    def parse_config_string(config_string):
        config_dict = {
            "moe.shared_experts": 0,
            "moe.experts": 0,
            "moe.experts.outlier_percent": 0,
            "attention": 0,
        }

        shared_experts_match = re.search(r"shared_(\d+)", config_string)
        if shared_experts_match:
            config_dict["moe.shared_experts"] = int(shared_experts_match.group(1))

        experts_match = re.search(r"other_(\d+)", config_string)
        if experts_match:
            config_dict["moe.experts"] = int(experts_match.group(1))

        attention_match = re.search(r"other_block\.(\d+)", config_string)
        if attention_match:
            config_dict["attention"] = int(attention_match.group(1))

        percent = re.search(r"(outlier)_(?:top|random)(\d+)", config_string)
        if 'outlier' in percent.group(1):
            config_dict["moe.experts.outlier_percent"] = int(percent.group(2))

        return config_dict

    bit_config_dict = parse_config_string(bit_config_str)
    
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

    if 'outlier' in bit_config_str:
        percent = bit_config_dict['moe.experts.outlier_percent']
        blocks = list(range(1, 28))
        k = int(len(outlier_data) * percent / 100)
        
        if 'top' in bit_config_str:
            high_bit_ffn_layers = list(outlier_data.keys())[:k]
            for block_num in blocks:
                for layer in moe_block_bit_dict:
                    if 'mlp.experts' in layer:
                        key = f'model.layers.{block_num}.{layer}'
                        if key in high_bit_ffn_layers:
                            deeepseek_bit[key] = 4 
                        else:
                            deeepseek_bit[key] = 2
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = moe_block_bit_dict[layer]  
        elif 'random' in bit_config_str:
            random_keys = random.sample(list(outlier_data.keys()), k)
            for block_num in blocks:
                for layer in moe_block_bit_dict:
                    if 'mlp.experts' in layer:
                        key = f'model.layers.{block_num}.{layer}'
                        if key in random_keys:
                            deeepseek_bit[key] = 4 
                        else:
                            deeepseek_bit[key] = 2
                    else:
                        key = f'model.layers.{block_num}' + '.' + layer
                        deeepseek_bit[key] = moe_block_bit_dict[layer] 

    return deeepseek_bit






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

    
    
# moe.shared_8.top25_4.other_2+other_block.4
# moe.shared_8.top25_4.other_2+other_block.4+alpha10
# moe.shared_8.top25_4.other_2+other_block.4+alpha20
# moe.shared_8.top25_4.other_2+other_block.4+alpha30
def generate_deeepseek_combination_alpha(bit_config_str):
    path = 'save/deepseek_linear_weight_outlier_metric.pt'
    outlier_data = torch.load(path)
    filtered_data = {k: v for k, v in outlier_data.items() if 'mlp.experts' in k}
    outlier_data = dict(sorted(filtered_data.items(), key=lambda item: item[1], reverse=True))
    
    def parse_config_string(config_string):
        config_dict = {
            "moe.shared_experts": 0,
            "moe.experts": 0,
            "moe.experts.top_index": 0,
            "moe.experts.top": 0,
            "moe.experts.alpha": 0,
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


        experts_match = re.search(r"other_(\d+)", config_string)
        if experts_match:
            config_dict["moe.experts"] = int(experts_match.group(1))

        attention_match = re.search(r"other_block\.(\d+)", config_string)
        if attention_match:
            config_dict["attention"] = int(attention_match.group(1))


        percent = re.search(r"alpha(\d+)", config_string)
        if percent:
            config_dict["moe.experts.alpha"] = int(percent.group(1))

        return config_dict



    bit_config_dict = parse_config_string(bit_config_str)
    
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


    if 'alpha' in bit_config_str:
        alpha = bit_config_dict['moe.experts.alpha']
        blocks = list(range(1, 28))
        # k = int(len(outlier_data) * alpha / 100)
        # list(outlier_data.keys())[:k]
        
        # select top [25-int(25*alpha)]
        # select outlier not in top [25-int(25*alpha)] and int(25*alpha)*27*3个 outlier

        top_index = bit_config_dict['moe.experts.top_index'] - int(bit_config_dict['moe.experts.top_index']*alpha/100)
        topk = {}
        for block_num in range(1, 28):
            topk[f'block_{block_num}'] = sorted_expert_indices_by_block[block_num][0:top_index]
        
        outlier_alpha = []
        assert len(topk) == 27
        total_limit = int(bit_config_dict['moe.experts.top_index'] * alpha / 100) * len(topk)
        current_count = 0
        for key, value in outlier_data.items():
            expert = int(re.search(r"experts\.(\d+)", key).group(1))
            layer = int(re.search(r"layers\.(\d+)", key).group(1))
            
            if f'block_{layer}' in topk and expert not in topk[f'block_{layer}']:
                outlier_alpha.append(key)
                
            current_count += 1
            if current_count >= total_limit:
                break
            
            
        blocks = list(range(1, 28))
        for block_num in blocks:
            for layer in moe_block_bit_dict:
                topk_moe = topk[f'block_{block_num}']
                if 'mlp.experts' in layer:
                    expert_index = layer.split('.')[2]
                    key = f'model.layers.{block_num}' + '.' + layer
                    
                    if int(expert_index) in topk_moe:
                        deeepseek_bit[key] = bit_config_dict['moe.experts.top']
                    elif key in outlier_alpha:
                        deeepseek_bit[key] = bit_config_dict['moe.experts.top']
                    else:
                        deeepseek_bit[key] = bit_config_dict['moe.experts']
                else:
                    # attention
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
        
    # top k
    pattern = r'^moe\.shared_\d+\.top\d+_\d+\.other_\d+\+other_block\.\d+$'
    if re.match(pattern, args.bits):
        deeepseek_bit = generate_deeepseek_bit_topk(args.bits)
        return deeepseek_bit
    # random top k
    pattern = r'^moe\.shared_\d+\.random\d+_\d+\.other_\d+\+other_block\.\d+$'
    if re.match(pattern, args.bits):
        deeepseek_bit = generate_deeepseek_bit_topk_random(args.bits)
        return deeepseek_bit
    
    # moe.shared_4.other_4+other_block.4+startlayer_4
    pattern = r"^moe\.shared_(\d+)\.other_(\d+)\+other_block\.(\d+)\+(startlayer|endlayer|randomlayer)_(\d+)$"
    if re.match(pattern, args.bits):
        deeepseek_bit = generate_deeepseek_start_end_layer(args.bits)
        return deeepseek_bit
    
    # moe.shared_4.other_4+other_block.4+dejavu_4
    pattern = r"^moe\.shared_(\d+)\.other_(\d+)\+other_block\.(\d+)\+(dejavu)_(\d+)$"
    if re.match(pattern, args.bits):
        deeepseek_bit = generate_deeepseek_dejavu_layer(args.bits)
        return deeepseek_bit
    
    # moe.shared_4.other_4+other_block.4+outlier_top25
    # moe.shared_4.other_4+other_block.4+outlier_random25
    pattern = r"^moe\.shared_(\d+)\.other_(\d+)\+other_block\.(\d+)\+outlier_[a-zA-Z]+(\d+)$"
    if re.match(pattern, args.bits):
        deeepseek_bit = generate_deeepseek_outlier(args.bits)
        return deeepseek_bit

    # moe.shared_8.top25_4.other_2+other_block.4+alpha30
    pattern = r"^moe\.shared_\d+\.top\d+_\d+\.other_\d+\+other_block\.(\d+)\+alpha(\d+)$"
    if re.match(pattern, args.bits):
        deeepseek_bit = generate_deeepseek_combination_alpha(args.bits)
        return deeepseek_bit

    raise ValueError("Invalid bits")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=str, default=None)
    args = parser.parse_args()
    
    config_str = 'moe.shared_8.top25_4.other_2+other_block.4+alpha30'
    args.bits = config_str

    bit = generate_deeepseek_combination_alpha(config_str)
    print(bit)







# {
#     moe.shared_experts: 8,
#     moe.experts: 4,
#     moe.experts.top_index : 4,
#     moe.experts.top : 8,
#     attention: 2,
# }