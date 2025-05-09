import torch


def generate_config_7b(start, end):
    gen_single_layer =lambda layer_num: [
        (f'model.layers.{layer_num}.self_attn.q_proj.weight', torch.Size([4096, 4096]), torch.float16),
        (f'model.layers.{layer_num}.self_attn.k_proj.weight', torch.Size([4096, 4096]), torch.float16),
        (f'model.layers.{layer_num}.self_attn.v_proj.weight', torch.Size([4096, 4096]), torch.float16),
        (f'model.layers.{layer_num}.self_attn.o_proj.weight', torch.Size([4096, 4096]), torch.float16),
        (f'model.layers.{layer_num}.self_attn.rotary_emb.inv_freq', torch.Size([64]), torch.float32),
        (f'model.layers.{layer_num}.mlp.gate_proj.weight', torch.Size([11008, 4096]), torch.float16),
        (f'model.layers.{layer_num}.mlp.down_proj.weight', torch.Size([4096, 11008]), torch.float16),
        (f'model.layers.{layer_num}.mlp.up_proj.weight', torch.Size([11008, 4096]), torch.float16),
        (f'model.layers.{layer_num}.input_layernorm.weight', torch.Size([4096]), torch.float16),
        (f'model.layers.{layer_num}.post_attention_layernorm.weight', torch.Size([4096]), torch.float16),
    ]
    ret = []
    for i in range(start, end):
        ret += gen_single_layer(i)
    return ret


def generate_data(spec, device_):
    res = {}
    for (name, size, dtype) in spec:
        res[name] = torch.zeros(size, dtype=dtype, device=device_)
    return res


def get_weights_llama7b(fn, device):
    data001 = [('model.embed_tokens.weight', torch.Size([32000, 4096]), torch.float16)] + generate_config_7b(0, 24)
    data002 = generate_config_7b(24, 32) + [
        ('model.norm.weight', torch.Size([4096]), torch.float16),
        ('lm_head.weight', torch.Size([32000, 4096]), torch.float16)
    ]
    all_data = {
        "pytorch_model-00001-of-00002.bin": generate_data(data001, device),
        "pytorch_model-00002-of-00002.bin": generate_data(data002, device),
        "model-00001-of-00002.safetensors": generate_data(data001, device),
        "model-00002-of-00002.safetensors": generate_data(data002, device),
    }
    return all_data[fn]


def get_dummy_weights(weight_dir, fn, device='cpu'):
    if 'llama' in weight_dir.lower():
        if '7b' in weight_dir.lower():
            return get_weights_llama7b(fn, device)
        else:
            raise NotImplementedError(f"Only llama-7B dummy weights is allowed, got {weight_dir}")
    else:
        raise NotImplementedError(f"Only llama dummy weights is allowed, got {weight_dir}")
