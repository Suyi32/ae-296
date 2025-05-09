import os
import json
import torch
from typing import final

from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.req_manager import ReqManager
from lightllm.common.infer_utils import init_req_to_token_indexes
from lightllm.common.build_utils import repair_config
from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req
from lightllm.models.lora_cpu import Lora_CPU
from lightllm.models.lora_gpu import Lora_GPU
torch.backends.cudnn.enabled = True

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)

class TpPartBaseModel:
    # weight class
    pre_and_post_weight_class = None
    transformer_weight_class = None

    # infer class
    pre_layer_infer_class = None
    post_layer_infer_class = None
    transformer_layer_infer_class = None

    # infer state class
    infer_state_class = InferStateInfo

    def __init__(self, kvargs):
        self.tp_rank_ = kvargs["tp_rank"]
        self.world_size_ = kvargs["world_size"]
        self.weight_dir_ = kvargs["weight_dir"]
        self.max_total_token_num = kvargs["max_total_token_num"]
        self.load_way = kvargs.get("load_way", "HF")
        self.mode = kvargs.get("mode", [])
        self.weight_dict = kvargs.get("weight_dict", None)
        self.finetune_config = kvargs.get("finetune_config", None)
        self.max_req_num = kvargs.get("max_req_num", 1000)
        self.max_seq_length = kvargs.get("max_seq_length", 1024 * 5)
        self.lora_rank = kvargs.get("lora_rank", 64)
        self.lora_num = kvargs.get("lora_num", 1)
        self.max_prefill_batch_size = kvargs.get("max_prefill_batch_size", 2)
        self.token_per_lora = kvargs.get("token_per_lora", 16)

        self._init_config()
        self._verify_must()
        self._verify_params()
        self._init_weights()
        self._init_mem_manager()
        self._init_req_manager()
        self._init_infer_layer()
        self._init_some_value()
        self._init_custom()
        return
    
    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), 'r') as json_file:
            self.config = json.load(json_file)
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config['vocab_size'] = self.finetune_config.vocab_size
        return
    
    @final
    def _verify_must(self):
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return
    
    def _verify_params(self):
        assert self.load_way == "HF", "only support HF format weights"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        return

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(self.tp_rank_, self.world_size_, torch.float16, network_config=self.config, mode=self.mode)
        self.trans_layers_weight = [
            self.transformer_weight_class(i, self.tp_rank_, self.world_size_, torch.float16, network_config=self.config, mode=self.mode)
            for i in range(self.config["n_layer"])
        ]
        load_hf_weights(
            "fp16",
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict)
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return 
    
    def _init_mem_manager(self):
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        self.mem_manager = MemoryManager(self.max_total_token_num, 
                            dtype=torch.float16,
                            head_num=self.config["num_attention_heads"] // self.world_size_,
                            head_dim=self.config["n_embed"] // self.config["num_attention_heads"],
                            layer_num=self.config["n_layer"])
        return
    
    def _init_req_manager(self):
        self.req_manager = ReqManager(self.max_req_num, 
                                      self.max_seq_length,
                                      self.mem_manager)
        return 
    
    def _init_infer_layer(self):

        ### LoRA Starts ###
        configJson = {}
        configJson["tp_degree"] = self.world_size_
        configJson["tp_rank"] = self.tp_rank_
        configJson["batch_size"] = self.max_req_num
        configJson["lora_rank"] = self.lora_rank
        configJson["lora_num"] = self.lora_num
        configJson["n_layers"] = self.config["n_layer"]
        configJson["hidden_dim"] = self.config["hidden_size"]
        configJson["max_prefill_batch_size"] = self.max_prefill_batch_size
        configJson["token_per_lora"] = self.token_per_lora
        configJson["mode"] = self.mode

        self.gpu_lora = None
        self.cpu_lora = None
        if "gpu_lora_bmm" in self.mode or "gpu_lora" in self.mode:
            self.gpu_lora = Lora_GPU(configJson)
            self.gpu_lora.prepare_lora_prefill(self.max_req_num) 
            self.gpu_lora.merge_lora_prefill( self.max_req_num )
        elif "cpu_lora" in self.mode:
            self.gpu_lora = Lora_GPU(configJson)
            self.cpu_lora = Lora_CPU(configJson, torch.cuda.current_device())
        elif "aaas-gpu" in self.mode or "aaas-cached" in self.mode:
            self.gpu_lora = Lora_GPU(configJson)
        elif "aaas" in self.mode:
            self.gpu_lora = Lora_GPU(configJson, shared=True)
            self.cpu_lora = Lora_CPU(configJson, torch.cuda.current_device())
        ### LoRA Ends ###

        self.pre_infer = self.pre_layer_infer_class(tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode)
        self.post_infer = self.post_layer_infer_class(tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode)
        self.layers_infer = [
            self.transformer_layer_infer_class(
                i,
                tp_rank=self.tp_rank_,
                world_size=self.world_size_,
                network_config=self.config,
                mode=self.mode,
                gpu_lora=self.gpu_lora,
                cpu_lora=self.cpu_lora) for i in range(
                self.config["n_layer"])]
        return
    
    def _init_some_value(self):
        self.head_dim_ = self.config["n_embed"] // self.config["num_attention_heads"]
        self.tp_k_head_num_ = self.config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.layers_num = self.config["n_layer"]
        self.vocab_size = self.config["vocab_size"]
        return
    
    def _init_custom(self):
        pass


    @torch.no_grad()
    def forward(
            self,
            batch_size,
            total_token_num,
            max_len_in_batch,
            input_ids : torch.Tensor,
            lora_ids: torch.Tensor,
            b_req_idx : torch.Tensor,
            b_start_loc : torch.Tensor,
            b_seq_len : torch.Tensor,
            time_breakdown : dict,
            is_prefill=True,
            aaas_mode=None):
        if is_prefill:
            return self._prefill(batch_size, total_token_num, max_len_in_batch, input_ids, lora_ids, b_req_idx, b_start_loc, b_seq_len, time_breakdown, aaas_mode)
        else:
            return self._decode(batch_size, total_token_num, max_len_in_batch, input_ids, lora_ids, b_req_idx, b_start_loc, b_seq_len, time_breakdown, aaas_mode)

    
    def _prefill(self, batch_size, total_token_num, max_len_in_batch, input_ids, lora_ids, b_req_idx, b_start_loc, b_seq_len, time_breakdown, aaas_mode):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = True
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (input_ids.shape[0] == total_token_num)
        assert (b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.lora_ids = lora_ids
        infer_state.aaas_mode = aaas_mode

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager
        infer_state.prefill_mem_index = self.mem_manager.alloc(infer_state.total_token_num)
        infer_state.prefill_key_buffer = torch.empty((infer_state.total_token_num, self.tp_k_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
        infer_state.prefill_value_buffer = torch.empty((infer_state.total_token_num, self.tp_v_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
        init_req_to_token_indexes(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len,
                                   max_len_in_batch, infer_state.prefill_mem_index)

        infer_state.init_some_extra_state(self, batch_size, total_token_num, max_len_in_batch, 
                                          input_ids, self.req_manager.req_to_token_indexs, b_req_idx,
                                          b_start_loc, b_seq_len, True)
        predict_logics = self._context_forward(input_ids, infer_state, time_breakdown)
        return predict_logics
    
    def _decode(self, batch_size, total_token_num, max_len_in_batch, input_ids, lora_ids, b_req_idx, b_start_loc, b_seq_len, time_breakdown, aaas_mode):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.lora_ids = lora_ids
        infer_state.aaas_mode = aaas_mode

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        alloc_mem = self.mem_manager.alloc_contiguous(batch_size)
        if alloc_mem is not None:
            infer_state.decode_is_contiguous = True
            infer_state.decode_mem_index = alloc_mem[0]
            infer_state.decode_mem_start = alloc_mem[1]
            infer_state.decode_mem_end = alloc_mem[2]
            copy_kv_index_to_req(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len, infer_state.decode_mem_index)
        else:
            infer_state.decode_is_contiguous = False
            alloc_mem = self.mem_manager.alloc(batch_size)
            infer_state.decode_mem_index = alloc_mem
            infer_state.decode_key_buffer = torch.empty((batch_size, self.tp_k_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            infer_state.decode_value_buffer = torch.empty((batch_size, self.tp_v_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            copy_kv_index_to_req(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len, infer_state.decode_mem_index)

        infer_state.init_some_extra_state(self, batch_size, total_token_num, max_len_in_batch, input_ids,
                                          self.req_manager.req_to_token_indexs, b_req_idx, b_start_loc, b_seq_len, False)
        predict_logics = self._token_forward(input_ids, infer_state, time_breakdown)
        return predict_logics
    
    @final
    def _context_forward(self, input_ids, infer_state: InferStateInfo, time_breakdown: dict):
        # print("!!!DEBUG: basemodel", infer_state.lora_ids)
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.context_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.layers_num):
            input_embs = self.layers_infer[i].context_forward(input_embs, infer_state, self.trans_layers_weight[i], time_breakdown)
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight, return_logics=True)
        return predict_logics

    @final
    def _token_forward(self, input_ids, infer_state: InferStateInfo, time_breakdown):
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.token_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.layers_num):
            input_embs = self.layers_infer[i].token_forward(input_embs, infer_state, self.trans_layers_weight[i], time_breakdown)
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight, return_logics=True)
        return predict_logics
