import torch
import os
import gc
from safetensors import safe_open
from .dummy_weights import get_dummy_weights


def load_func(file_, use_safetensors=False, pre_post_layer=None, transformer_layer_list=None, weight_dir=None):
    # fix bug for 多线程加载的时候，每个线程内部的cuda device 会切回 0， 修改后来保证不会出现bug
    import torch.distributed as dist    
    tp_rank = dist.get_rank()
    torch.cuda.set_device(tp_rank)

    if use_safetensors:
        try:
            weights = safe_open(os.path.join(weight_dir, file_), 'pt', 'cpu')
            weights = {k: weights.get_tensor(k) for k in weights.keys()}
        except:
            print("model file is not readable, loading dummy weights...")
            weights = get_dummy_weights(weight_dir, file_, 'cpu')
    else:
        try:
            weights = torch.load(os.path.join(weight_dir, file_), 'cpu')
        except:
            print("model file is not readable, loading dummy weights...")
            weights = get_dummy_weights(weight_dir, file_, 'cpu')

    if pre_post_layer is not None:
        pre_post_layer.load_hf_weights(weights)
    if transformer_layer_list is not None:
        for layer in transformer_layer_list:
            layer.load_hf_weights(weights)
    del weights
    gc.collect()


def load_hf_weights(data_type, weight_dir, pre_post_layer=None, transformer_layer_list=None, weight_dict=None):
    data_type = torch.float16 if data_type == 'fp16' else torch.float32
    if pre_post_layer is not None:
        assert pre_post_layer.data_type_ == data_type, "type is not right"
    if transformer_layer_list is not None:
        assert transformer_layer_list[0].data_type_ == data_type, "type is not right"
    if weight_dict:
        if pre_post_layer is not None:
            pre_post_layer.load_hf_weights(weight_dict)
        if transformer_layer_list is not None:
            for layer in transformer_layer_list:
                layer.load_hf_weights(weight_dict)
        del weight_dict
        return
    use_safetensors = True
    files = os.listdir(weight_dir)
    candidate_files = list(filter(lambda x : x.endswith('.safetensors'), files))
    if len(candidate_files) == 0:
        use_safetensors = False
        candidate_files = list(filter(lambda x : x.endswith('.bin'), files))
    assert len(candidate_files) != 0, "can only support pytorch tensor and safetensors format for weights."
    from functools import partial
    from multiprocessing.pool import ThreadPool as Pool
    partial_func = partial(load_func, use_safetensors=use_safetensors, pre_post_layer=pre_post_layer, transformer_layer_list=transformer_layer_list, weight_dir=weight_dir)  # noqa
    worker = os.environ.get('LOADWORKER', 1)
    with Pool(worker) as p:
        _ = p.map(partial_func, candidate_files)
    return
