import uuid
import asyncio
import numpy as np
from typing import List
from ..io_struct import Batch, Req
from lightllm.utils.infer_utils import  calculate_time
from lightllm.server.io_struct import Req
from lightllm.server.io_struct import ReqRunStatus

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

class ReqQueue:

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size, router_token_ratio, router_max_new_token_len) -> None:
        self.max_total_tokens = max_total_tokens
        assert batch_max_tokens is not None
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        self.waiting_req_list: List[Req] = []
        self.router_token_ratio = router_token_ratio
        self.router_max_new_token_len = router_max_new_token_len
        self.pause_req_dict = {} # 用于保存队列中被暂停的请求，暂停原因为 ReqRunStatus.PAUSED_AND_KVKEEP  ReqRunStatus.PAUSED_AND_OFFLOAD
        self.pause_req_used_tokens = 0
        
    def append(self, req):
        self.waiting_req_list.append(req)
        return
    
    def back_to_wait_list(self, req_list:List[Req]):
        for req in req_list:
            if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
                self.pause_req_dict[req.request_id] = req
        self.waiting_req_list = req_list + self.waiting_req_list
        self.recalcu_pause_req_used_tokens()
        return 

    def _init_cache_list(self, current_batch:Batch, is_busy):
        self.cache_pause_reqs_used_tokens = self.pause_req_used_tokens
        self.cache_pause_reqs_num = len(self.pause_req_dict) 
        if current_batch is not None:
            self.cache_len_list = [req.get_tuple_tokens(is_busy, self.router_max_new_token_len) for req in current_batch.reqs]
        else:
            self.cache_len_list = []

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req:Req, is_busy):
        self.cache_len_list.append(req.get_tuple_tokens(is_busy, self.router_max_new_token_len)) # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])
        
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
            self.cache_pause_reqs_used_tokens -= req.get_used_tokens()
            self.cache_pause_reqs_num -= 1
        if need_max_token_num < self.max_total_tokens - self.cache_pause_reqs_used_tokens and len(self.cache_len_list) + self.cache_pause_reqs_num <= self.running_max_req_size:
            return True
        else:
            return False
    
    #@calculate_time(show=True, min_cost_ms=10)
    def generate_new_batch(self, current_batch:Batch, max_prefill_batch_size=999):
        # 如果当前已经被调度的请求数量超过了上限，直接不调度新的请求了。
        # if current_batch is not None and len(current_batch.reqs) + len(self.pause_req_dict) >= self.running_max_req_size:
        #     return None
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None

        # we add an admission control to ensure batch size not overflow; compute how many new requests are allowed.
        if current_batch is not None: 
            new_req_number = min(self.running_max_req_size - len(current_batch.reqs), max_prefill_batch_size)
        else:
            new_req_number = min(self.running_max_req_size, max_prefill_batch_size)

        is_busy = True # disable this feature. The default value of self.router_token_ratio is 0.0
        self._init_cache_list(current_batch, is_busy)
        can_run_list = []
        new_batch_prefill_need_tokens = 0
        aborted_count = 0
        for req_id, req in enumerate(self.waiting_req_list):
            # print("!!!DEBUG: in generate new batch:", req.get_lora_id())
            if req_id >= new_req_number:
                # can not host more requests
                aborted_count += 1
                continue
            if req.aborted and req.req_status == ReqRunStatus.WAIT_IN_QUEUE: 
                # 由于管理的复杂性，只有没有被调度运行过的请求可以因为abort直接在队列中忽略掉. 
                # 暂停的请求需要恢复后，由 router manager 部分来过滤。暂时保持这种处理方法, 否则会导致管理token的泄漏
                aborted_count += 1
                continue
            req_prefill_need_tokens = req.get_prefill_need_tokens()
            if self._can_add_new_req(req, is_busy) and new_batch_prefill_need_tokens + req_prefill_need_tokens <= self.batch_max_tokens:
                can_run_list.append(req)
                new_batch_prefill_need_tokens += req_prefill_need_tokens
                if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
                    self.pause_req_dict.pop(req.request_id)
            else:
                break

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            # We should not remove the aborted requests, because they are never served
            self.waiting_req_list = self.waiting_req_list[len(can_run_list):]
            # 生成新 batch 以后，更新一下状态
            self.recalcu_pause_req_used_tokens()
            return new_batch
        else:
            return None
        
    def recalcu_pause_req_used_tokens(self):
        used_tokens = 0
        for req_id, req_obj in self.pause_req_dict.items():
            used_tokens += req_obj.get_used_tokens()
        self.pause_req_used_tokens = used_tokens
        return self.pause_req_used_tokens

