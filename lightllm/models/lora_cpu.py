import torch
import numpy as np
import math
from multiprocessing import shared_memory
import time
import torch.nn as nn
import tensor_cp
torch.manual_seed(0)

import os
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)

PROMPT_MAX_LENGTH = 256


def init_shm_helper(shm_name_folder, shm_names):
    all_shms = {}
    for name in shm_names:
        shm_name_files = [ item for item in os.listdir(shm_name_folder) if ".shm" in item and name in item ]
        assert len(shm_name_files) == 1, "{}".format(len(shm_name_files))
        shm_filename = shm_name_files[0]
        shm_name = shm_filename[:-4].split("-")[1]
        all_shms[name] = shared_memory.SharedMemory(name=shm_name)
    return all_shms
        

# lora_shm_op
class Lora_CPU():
    def __init__(self, configJson, gpu_id):

        logging.critical("PID: {}. Current device in lora_cpu: {}".format(os.getpid(), gpu_id))

        self.batch_size = configJson["batch_size"]
        ## the lora number read here is 3 * lora, since each lora will need 3 cpu to do the computation for qkv
        self.lora_num = configJson["lora_num"]
        self.tp_degree = configJson["tp_degree"]
        self.hidden_dim = configJson["hidden_dim"]
        self.lora_rank = configJson['lora_rank']
        self.cpu_capacity = configJson['token_per_lora']
        self.gpu_id = gpu_id

        self.out_size = (self.hidden_dim//self.tp_degree)
        self.collect_start_index = (self.out_size) * gpu_id
        self.collect_end_index = (self.out_size) * (gpu_id + 1)

        self.n_layers = configJson["n_layers"]

        shm_name_folder = os.getenv("shm_name_folder")
        # attach to shared memory
        # shm array, qkvo
        self.all_shms = init_shm_helper(shm_name_folder, ["shm_array", "shm_out_q", "shm_out_k", "shm_out_v", "shm_out_o"])
        self.shared_array = np.ndarray((self.lora_num, PROMPT_MAX_LENGTH, self.hidden_dim) , dtype=np.half, buffer=self.all_shms['shm_array'].buf)
        self.shared_out_q = np.ndarray((self.lora_num, PROMPT_MAX_LENGTH, self.hidden_dim) , dtype=np.half, buffer=self.all_shms['shm_out_q'].buf)
        self.shared_out_k = np.ndarray((self.lora_num, PROMPT_MAX_LENGTH, self.hidden_dim) , dtype=np.half, buffer=self.all_shms['shm_out_k'].buf)
        self.shared_out_v = np.ndarray((self.lora_num, PROMPT_MAX_LENGTH, self.hidden_dim) , dtype=np.half, buffer=self.all_shms['shm_out_v'].buf)
        self.shared_out_o = np.ndarray((self.lora_num, PROMPT_MAX_LENGTH, self.hidden_dim) , dtype=np.half, buffer=self.all_shms['shm_out_o'].buf)

        # get the lock
        shm_name_files = [ item for item in os.listdir(shm_name_folder) if ".shm" in item and "lock" in item ]
        shm_array_filename = shm_name_files[0]
        shm_array_name = shm_array_filename[:-4].split("-")[1]
        self.lock = shared_memory.SharedMemory(name=shm_array_name)
        self.lock_buf = self.lock.buf
        # self.lock_buf[0] = 0

        # signal

        # self.numbers_o = torch.range(0, 100, dtype=torch.int8, device="cuda")
        # self.number_o_ptr = self.numbers_o.untyped_storage().data_ptr()
        self.numbers = torch.range(0, 1024, dtype=torch.int16, device="cuda")
        self.number_ptr = self.numbers.untyped_storage().data_ptr()

        shm_name_files = sorted( [ item for item in os.listdir(shm_name_folder) if ".shm" in item and "shm_signal" in item ] )
        logging.critical("PID: {}. shm_name_files: {}".format(os.getpid(), shm_name_files))

        shm_name_files = [ item for item in os.listdir(shm_name_folder) if ".shm" in item and "shm_signal" in item ]
        shm_array_filename = shm_name_files[0]
        shm_signal_name = shm_array_filename[:-4].split("-")[1]
        self.np_shm_signal = shared_memory.SharedMemory(name=shm_signal_name)
        
        self.shm_signal = torch.from_numpy(np.ndarray((3, self.lora_num), dtype=np.int16, buffer=self.np_shm_signal.buf))
        self.signal_ptr = self.shm_signal.untyped_storage().data_ptr()
        tensor_cp.register_pinned_memory(self.shm_signal.untyped_storage().data_ptr(), self.lora_num * 3 * 2)
        logging.critical(f"is pinned: {self.shm_signal.is_pinned()}")
        # tkn length
        shm_name_files = [ item for item in os.listdir(shm_name_folder) if ".shm" in item and "shm_tkn_len" in item ]
        assert len(shm_name_files) == 1, "{}".format(len(shm_name_files))
        shm_tkn_len_filename = shm_name_files[0]
        shm_tkn_len_name = shm_tkn_len_filename[:-4].split("-")[1]
        self.shm_tkn_len = shared_memory.SharedMemory(name=shm_tkn_len_name)
        self.shared_tkn_len = self.shm_tkn_len.buf

        if self.gpu_id == 0:
            self.shared_tkn_len[0] = 1

        self.torch_res =  torch.from_numpy(self.shared_array)
        tensor_cp.register_pinned_memory(self.torch_res.untyped_storage().data_ptr(), self.shared_array.nbytes)
        self.torch_q =  torch.from_numpy(self.shared_out_q)
        tensor_cp.register_pinned_memory(self.torch_q.untyped_storage().data_ptr(), self.shared_out_q.nbytes)
        self.torch_k =  torch.from_numpy(self.shared_out_k)
        tensor_cp.register_pinned_memory(self.torch_k.untyped_storage().data_ptr(), self.shared_out_k.nbytes)
        self.torch_v =  torch.from_numpy(self.shared_out_v)
        tensor_cp.register_pinned_memory(self.torch_v.untyped_storage().data_ptr(), self.shared_out_v.nbytes)
        self.torch_o =  torch.from_numpy(self.shared_out_o)
        tensor_cp.register_pinned_memory(self.torch_o.untyped_storage().data_ptr(), self.shared_out_o.nbytes)
        logging.critical(f"pinned: {self.torch_res.is_pinned()}")
        self.torch_res_ptr = self.torch_res.untyped_storage().data_ptr()
        self.torch_q_ptr = self.torch_q.untyped_storage().data_ptr()
        self.torch_k_ptr = self.torch_k.untyped_storage().data_ptr()
        self.torch_v_ptr = self.torch_v.untyped_storage().data_ptr()
        self.torch_o_ptr = self.torch_o.untyped_storage().data_ptr()
        # tensor_cp.register_shm_signals(0)
        tensor_cp.create_stream(0)
        # tensor_cp.register_lock(42)
        self.active_cpu = 0
        
    

    def invoke_lora_async_cpp(self, hidden_states, tkn_num, layer_id, lora_inds=None):
        # if self.gpu_id != 0:
        #     return 0
        # indicate which lora to use
        if lora_inds == None:
            lora_inds = range(len(self.signal_list))
        
        tid = tensor_cp.invoke_cpu_lora(
            hidden_states, tkn_num, lora_inds, self.batch_size,
            self.hidden_dim, self.tp_degree, self.torch_res_ptr,
            self.number_ptr, self.gpu_id, layer_id, self.n_layers
        )
        return tid

    def collect_cpp(self, tid):
        return tensor_cp.collect(tid)

    def invoke_lora_async(self, hidden_states, tkn_num, layer_id, lora_inds=None, is_o=False):
        '''
        hidden_states: the input hidden states: (B, L, H)
        tkn_num: L
        '''
        
        # indicate which lora to use
        if lora_inds == None:
            lora_inds = range(3 * self.lora_num)

        # assume all tokens need to sent to cpu in prefill(do not consider decode yet).
        batch_size = hidden_states.shape[0]//tkn_num
        cpu_per_request = (tkn_num-1) // self.cpu_capacity + 1
        cpu_lora_num = batch_size * cpu_per_request
        assert cpu_lora_num * 3 <= self.shm_signal.shape[1] * 3, \
            "Not enough CPU LoRAs: {}/{}. batch_size: {}, tkn_num: {}, cpu_capacity: {}, cpu_per_request: {}".format( 
            self.shm_signal.shape[1] * 3, cpu_lora_num * 3, batch_size, tkn_num, self.cpu_capacity, cpu_per_request
        )
        self.active_cpu = cpu_lora_num

        # set sig to -1.
        # guard
        if self.gpu_id == 0:
            for shm_id in lora_inds:
                
                self.shm_signal[shm_id%3, shm_id//3] = 1
            self.lock_buf[0] = (layer_id) % self.n_layers + 1
        else:
            while self.lock_buf[0] != (layer_id + 1):
                pass
        
        if self.gpu_id != 0:
            return
        # here we assume that in prefill, all the hidden_states need to be send to CPU. Therefore, we can simply send all data using one kernel. Then send all signals(better use one kernel instead of a bunch of kernel)
        
        # set tkn length
        
        self.shared_tkn_len[0] = min(tkn_num, self.cpu_capacity)
        # for each batch, do one copy
        for b in range(batch_size):
            # offset by batch
            tensor_cp.cudaDtoHAsync(hidden_states, self.torch_res_ptr + 2 * b * PROMPT_MAX_LENGTH * self.hidden_dim, b * tkn_num * self.hidden_dim, 2 * tkn_num * self.hidden_dim)

        # send 3 signals, 0-1 reserved. 2-201 for q, 202-401 for k, 402-601 for v.

        tensor_cp.cudaSignalDtoHAsync(self.numbers, self.signal_ptr, 2, 2 * cpu_lora_num)
        tensor_cp.cudaSignalDtoHAsync(self.numbers, self.signal_ptr + self.lora_num * 2, 202, 2 * cpu_lora_num)
        tensor_cp.cudaSignalDtoHAsync(self.numbers, self.signal_ptr + self.lora_num * 4, 402, 2 * cpu_lora_num)

    def collect_qkv(self, tkn_num, batch_size, lora_inds=None):

        if lora_inds == None:
            lora_inds = range(3 * self.lora_num)

        # logging.critical(f"{self.batch_size}, {tkn_num}, {self.out_size}")
        # it should be B * L * rank
        # Then it will be send back to base to do up projection.
        # In this case, each GPU need all the information about the tensor.
        # the up projection will be blr,br(h/tp)-> bl(h/tp)

        q_lora_res = torch.zeros(batch_size, tkn_num, self.out_size, dtype=torch.float16, device="cuda")
        k_lora_res = torch.zeros(batch_size, tkn_num, self.out_size, dtype=torch.float16, device="cuda")
        v_lora_res = torch.zeros(batch_size, tkn_num, self.out_size, dtype=torch.float16, device="cuda")

        # for each qkv lora, it computes partial results, which is 1 * tkn_num * r,
        # then we need to place it into the correct place in the result query.

        # for all loras
        #   for each qkv
        #     place 1 * tkn_num * r data into the location.
        # return the qkv results.
        # set tkn length
        last_cpu_tkn_num = tkn_num % self.cpu_capacity if tkn_num % self.cpu_capacity > 0 else self.cpu_capacity
        # last_cpu_tkn_num = min(tkn_num % self.cpu_capacity, self.cpu_capacity)
        cpu_per_request = (tkn_num-1) // self.cpu_capacity + 1
        cpu_lora_num = batch_size * cpu_per_request
        # while checker:
        #     for i in range(cpu_lora_num * 3):
        #         if (checker & (1 << i)) and self.shm_signal[i%3, i//3] == 0:
        #             checker -= (1 << i)
        checker = (1 << (cpu_lora_num * 3)) - 1
        while checker:
            for b in range(batch_size):
                for offset in range(cpu_per_request):
                    i = b*cpu_per_request + offset
                    if (checker & (1 << i)) and self.shm_signal[0, i] == 0:
                        tensor_cp.cudaHtoD2D(
                            self.torch_q_ptr + 2 * self.hidden_dim * (offset * self.cpu_capacity + b * PROMPT_MAX_LENGTH) + 2 * (self.out_size) * self.gpu_id,
                            q_lora_res,
                            self.hidden_dim * 2,
                            self.out_size * 2,
                            self.out_size * 2,
                            last_cpu_tkn_num if offset == (cpu_per_request - 1) else self.cpu_capacity,
                            self.out_size * (offset * self.cpu_capacity + b * tkn_num)
                        )
                        # torch.cuda.synchronize()
                        # logging.critical(f"cpu_offset: {2 * self.hidden_dim * (offset * self.cpu_capacity + b * PROMPT_MAX_LENGTH) + 2 * (self.out_size) * self.gpu_id}, gpu_offset: {2 * self.out_size * (offset * self.cpu_capacity + b * tkn_num)}, tkn: {last_cpu_tkn_num if offset == (cpu_per_request - 1) else self.cpu_capacity}, shm: {self.torch_q[0,0,0]}, gpu: {q_lora_res[0,0,0]}, out size:{self.out_size}, hidden_dim: {self.hidden_dim}")
                        # tensor_cp.cudaHtoDAsync(
                        #     self.torch_q_ptr + 2 * self.out_size * (offset * self.cpu_capacity + t + b * PROMPT_MAX_LENGTH) + 2 * (self.out_size) * self.gpu_id,
                        #     q_lora_res,
                        #     2 * self.out_size * (offset * self.cpu_capacity + t + b * tkn_num) + 2 * (self.out_size) * self.gpu_id,
                        #     self.out_size * 2 * last_cpu_tkn_num if offset == (cpu_per_request - 1) else self.cpu_capacity)
                        checker ^= (1 << i)

                    if (checker & (1 << (i+self.active_cpu))) and self.shm_signal[1, i] == 0:

                        tensor_cp.cudaHtoD2D(
                            self.torch_k_ptr + 2 * self.hidden_dim * (offset * self.cpu_capacity + b * PROMPT_MAX_LENGTH) + 2 * (self.out_size) * self.gpu_id,
                            k_lora_res,
                            self.hidden_dim * 2,
                            self.out_size * 2,
                            self.out_size * 2,
                            last_cpu_tkn_num if offset == (cpu_per_request - 1) else self.cpu_capacity,
                            self.out_size * (offset * self.cpu_capacity + b * tkn_num)
                        )
                        checker ^= (1 << (i+1*self.active_cpu))
                        # tensor_cp.cudaHtoDAsync(
                        #     self.torch_k_ptr + 2 * self.out_size * (offset * self.cpu_capacity + t + b * PROMPT_MAX_LENGTH) + 2 * (self.out_size) * self.gpu_id,
                        #     k_lora_res,
                        #     2 * self.out_size * (offset * self.cpu_capacity + t + b * tkn_num) + 2 * (self.out_size) * self.gpu_id,
                        #     self.out_size * 2 * last_cpu_tkn_num if offset == (cpu_per_request - 1) else self.cpu_capacity)
                    if (checker & (1 << (i+2*self.active_cpu))) and self.shm_signal[2, i] == 0:

                        tensor_cp.cudaHtoD2D(
                            self.torch_v_ptr + 2 * self.hidden_dim * (offset * self.cpu_capacity + b * PROMPT_MAX_LENGTH) + 2 * (self.out_size) * self.gpu_id,
                            v_lora_res,
                            self.hidden_dim * 2,
                            self.out_size * 2,
                            self.out_size * 2,
                            last_cpu_tkn_num if offset == (cpu_per_request - 1) else self.cpu_capacity,
                            self.out_size * (offset * self.cpu_capacity + b * tkn_num)
                        )
                        checker ^= (1 << (i+2*self.active_cpu))
                        # tensor_cp.cudaHtoDAsync(
                        #     self.torch_v_ptr + 2 * self.out_size * (offset * self.cpu_capacity + t + b * PROMPT_MAX_LENGTH) + 2 * (self.out_size) * self.gpu_id,
                        #     v_lora_res,
                        #     2 * self.out_size * (offset * self.cpu_capacity + t + b * tkn_num) + 2 * (self.out_size) * self.gpu_id,
                        #     self.out_size * 2 * last_cpu_tkn_num if offset == (cpu_per_request - 1) else self.cpu_capacity)
        return q_lora_res, k_lora_res, v_lora_res

    def collect_o(self, tkn_num, lora_inds=None):
        if lora_inds == None:
            lora_inds = range(len(self.signal_list))

        # logging.critical(f"{self.batch_size}, {tkn_num}, {self.lora_rank//self.tp_degree}")
        o_lora_res = torch.zeros(self.batch_size, tkn_num, self.lora_rank//self.tp_degree, dtype=torch.float16, device="cuda")
        checker = (1 << len(lora_inds)) - 1
        while checker:
            for i, e in enumerate(lora_inds):
                if (checker & (1 << i)) and self.signal_list[i][0] == 0:
                    for j in range(tkn_num):
                        tensor_cp.cudaHtoDAsync(self.torch_o_ptr + 2 * ((self.lora_rank//self.tp_degree) * self.gpu_id + i * PROMPT_MAX_LENGTH * self.lora_rank + j * self.lora_rank), 
                                                o_lora_res,
                                                (self.lora_rank//self.tp_degree) * self.gpu_id + e * tkn_num * self.lora_rank + j * self.lora_rank, 
                                                self.lora_rank//self.tp_degree)
                    checker -= (1 << i)
        return o_lora_res

    def __del__(self):
        for shm in self.all_shms:
            shm.close()
        # self.shm_array.unlink()
        self.shm_tkn_len.close()
        # self.shm_tkn_len.unlink()
        for item in self.shm_signal_list:
            item.close()
            # item.unlink()
        # tensor_cp.dealloc_resource()

