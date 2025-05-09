import time
from multiprocessing import shared_memory
import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
from typing import Dict, List, Optional
from ..sampling_params import SamplingParams
from ..io_struct import Req, Batch
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import ReqQueue
from rpyc.utils.classic import obtain
from lightllm.utils.infer_utils import calculate_time
from ..io_struct import BatchTokenIdOut, AbortReq, ReqRunStatus
from .stats import Stats
from .pause_strategy import Fcfs, select_paused_reqs
import os
import logging
import numpy as np
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class RouterManager:

    def __init__(self, args, router_port, detokenization_port, model_rpc_ports):
        self.args = args
        self.model_weightdir = args.model_dir
        self.world_size = args.tp
        self.load_way = args.load_way
        self.mode = args.mode
        self.max_total_token_num = args.max_total_token_num
        self.max_prefill_batch_size = args.max_prefill_batch_size
        self.token_per_lora = args.token_per_lora
        self.req_queue = ReqQueue(args.max_total_token_num, 
                                  args.batch_max_tokens,
                                  args.running_max_req_size,
                                  args.router_token_ratio,
                                  args.router_max_new_token_len)
        
        self.pause_strategy = Fcfs()

        self.running_batch: Batch = None
        self.eos_id = args.eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = 0
        
        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")
        
        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")
        self.model_rpc_ports = model_rpc_ports

        self.stats_tool = Stats(not args.disable_log_stats, args.log_stats_interval)
        # only aaas or aaas-gpu needs LoRA modules
        self.lora_rank = args.lora_rank
        if 'aaas' in self.mode:
            logging.critical("LoRA Num: {}".format( args.lora_num ))
            self.lora_num = args.lora_num

            shm_name_folder = os.getenv("shm_name_folder")
            print("router manager shm_name_folder", shm_name_folder)
            shm_name_files = [ item for item in os.listdir(shm_name_folder) if ".shm" in item and "progress" in item ]
            assert len(shm_name_files) == 1, "{}".format(len(shm_name_files))
            shm_filename = shm_name_files[0]
            shm_name = shm_filename[:-4].split("-")[1]
            print("self.shm_progress shm name", shm_name)
            self.shm_progress = shared_memory.SharedMemory(name=shm_name)
            self.progress = np.ndarray((self.world_size, self.max_prefill_batch_size), dtype=np.int8, buffer=self.shm_progress.buf)

    async def wait_to_model_ready(self):
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
            kvargs = {
                "rank_id" : rank_id,
                "world_size" : self.world_size,
                "weight_dir" : self.model_weightdir,
                "load_way" : self.load_way,
                "max_total_token_num" : self.max_total_token_num,
                "mode" : self.mode,
                "max_req_num" : self.args.running_max_req_size + 8,
                "max_seq_length" : self.args.max_req_total_len + 8, # 留一点余量
                "nccl_port" : self.args.nccl_port,
                "max_prefill_batch_size": self.max_prefill_batch_size,
                "token_per_lora": self.token_per_lora,
                "lora_rank": self.lora_rank,
            }
            if 'aaas' in self.mode:
                kvargs["lora_num"] = self.lora_num
            init_model_ret.append(self.model_rpcs[rank_id].init_model(kvargs))

        await asyncio.gather(*init_model_ret)
        return

    def add_req(
        self,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        request_id: str,
        lora_id: int
    ):
        # print("!!!DEBUG: New request: loraID =", lora_id, type(lora_id))
        req = Req(request_id, prompt_ids, sampling_params, lora_id)
        self.req_queue.append(req)
        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
        return

    async def abort(self, request_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.has_generate_finished = True
                    req.aborted = True
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.has_generate_finished = True
                req.aborted = True
        return

    async def loop_for_fwd(self,):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    token_ratio = (self.running_batch.batch_used_tokens + self.req_queue.pause_req_used_tokens) / self.max_total_token_num
                    print("current batch size:", len(self.running_batch.reqs), "paused req num:", len(self.req_queue.pause_req_dict), "token used ratio:", token_ratio)
                    pass
                self.stats_tool.print_stats()
                
            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms

    async def _step(self):
        """
        事件处理循环
        """
        # 删除所有已经 finished 的 req
        # 当前无运行请求时
        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch, self.max_prefill_batch_size)
            if new_batch is not None:
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch
                if "base" in self.mode:
                    prefill_start = time.time()
                    await self._prefill_batch(self.running_batch)
                    prefill_end = time.time()
                    logging.critical("[STEP] Prefill {} reqs. Total token length: {}. Lat: {}".format( len(self.running_batch.reqs), self.running_batch.input_tokens(), 1000*(prefill_end-prefill_start)))

                if "aaas-gpu" in self.mode or "aaas-cached" in self.mode: 
                    prepare_gpu_start = time.time()
                    if "aaas-gpu" in self.mode:
                        rets = [self.model_rpcs[tp_rank].load_lora_prefill( len(self.running_batch.reqs) ) for tp_rank in range(self.world_size)]
                        ans = await asyncio.gather(*rets)
                    elif "aaas-cached" in self.mode:
                        logging.critical("Use cached LoRA for prefill (new batch)")
                    prepare_gpu_end = time.time()
                    logging.critical("[STEP] Prepare GPU LoRA for {} reqs. Lat: {}".format( len(self.running_batch.reqs), 1000*(prepare_gpu_end-prepare_gpu_start)))

                    prefill_start = time.time()
                    await self._prefill_batch(self.running_batch, aaas_mode="gpu_lora_prefill")
                    prefill_end = time.time()
                    logging.critical("[STEP] Prefill {} reqs. Total token length: {}. Lat: {}".format( len(self.running_batch.reqs), self.running_batch.input_tokens(), 1000*(prefill_end-prefill_start)))

                
                elif "aaas" in self.mode:
                    prepare_gpu_start = time.time()
                    for tp_rank in range(self.world_size):
                        for j in range(min(len(self.running_batch.reqs), self.max_prefill_batch_size)):
                            # set 1 to start swap
                            self.progress[tp_rank, j] = 1
                    prepare_gpu_end = time.time()
                    logging.critical("[STEP] Prepare GPU LoRA for {} reqs. Lat: {}.".format( len(self.running_batch.reqs), \
                                1000*(prepare_gpu_end-prepare_gpu_start)))

                    prefill_start = time.time()
                    await self._prefill_batch(self.running_batch, aaas_mode="cpu_lora")
                    prefill_end = time.time()
                    logging.critical("[STEP] Prefill {} reqs. Total token length: {}. Lat: {}".format( len(self.running_batch.reqs), self.running_batch.input_tokens(), 1000*(prefill_end-prefill_start)))

                self._filter_runing_batch()
                self.has_wait_tokens = 0
            return

        # 有运行请求，但是已经到了可以调度新的请求合并推理的时机; we require immediate merge by setting self.max_wait_tokens=0
        if self.has_wait_tokens >= self.max_wait_tokens: 
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch, self.max_prefill_batch_size)
            if new_mini_batch is not None:
                self.stats_tool.count_prompt_tokens(new_mini_batch)

                if "base" in self.mode:
                    prefill_start = time.time()
                    await self._prefill_batch(new_mini_batch)
                    prefill_end = time.time()
                    logging.critical("[STEP] Prefill {} reqs. Total token length: {}. Lat: {}".format( len(new_mini_batch.reqs), new_mini_batch.input_tokens(), 1000*(prefill_end-prefill_start)))

                if "aaas-gpu" in self.mode or "aaas-cached" in self.mode: 
                    prepare_gpu_start = time.time()

                    if "aaas-gpu" in self.mode:
                        loaded_lora_ids = [ item.lora_id for item in self.running_batch.reqs ]
                        lora_loaded = True
                        for item in new_mini_batch.reqs:
                            if item.lora_id in loaded_lora_ids:
                                continue
                            else:
                                lora_loaded = False
                        logging.critical("[STEP] LoRA_loaded: {}. Required: {}. Have: {}".format(lora_loaded, \
                            [ item.lora_id for item in new_mini_batch.reqs ], loaded_lora_ids))

                        if not lora_loaded:
                            rets = [self.model_rpcs[tp_rank].load_lora_prefill( len(new_mini_batch.reqs) ) for tp_rank in range(self.world_size)]
                            ans = await asyncio.gather(*rets)

                    elif "aaas-cached" in self.mode:
                        logging.critical("Use cached LoRA for prefill (preemption)")


                    prepare_gpu_end = time.time()
                    logging.critical("[STEP] Prepare GPU LoRA for {} reqs. Lat: {}".format( len(new_mini_batch.reqs), 1000*(prepare_gpu_end-prepare_gpu_start)))

                    prefill_start = time.time()
                    await self._prefill_batch(new_mini_batch, aaas_mode="gpu_lora_prefill")
                    prefill_end = time.time()
                    logging.critical("[STEP] Prefill {} reqs. Total token length: {}. Lat: {}".format( len(new_mini_batch.reqs), new_mini_batch.input_tokens(), 1000*(prefill_end-prefill_start)))

                elif "aaas" in self.mode:
                    prepare_gpu_start = time.time()
                    loaded_lora_ids = [ item.lora_id for item in self.running_batch.reqs ]
                    lora_loaded = True
                    for item in new_mini_batch.reqs:
                        if item.lora_id in loaded_lora_ids:
                            continue
                        else:
                            lora_loaded = False

                    logging.critical("[STEP] LoRA_loaded: {}. Required: {}. Have: {}".format(lora_loaded, \
                            [ item.lora_id for item in new_mini_batch.reqs ], loaded_lora_ids))
                    if lora_loaded:
                        prefill_start = time.time()
                        await self._prefill_batch(new_mini_batch, aaas_mode="gpu_lora_prefill")
                        prefill_end = time.time()

                        logging.critical("[STEP] Prefill {} reqs. Lat: {}".format( len(new_mini_batch.reqs), 1000*(prefill_end-prefill_start)))
                    else: # need to swap
                        for tp_rank in range(self.world_size):
                            for j in range(min(len(new_mini_batch.reqs), self.max_prefill_batch_size)):
                                # set 1 to start swap
                                self.progress[tp_rank, j] = 1
                        prepare_gpu_end = time.time()
                        logging.critical("[STEP] Prepare GPU LoRA for {} reqs. Lat: {}.".format( len(new_mini_batch.reqs), \
                                    1000*(prepare_gpu_end-prepare_gpu_start)))

                        prefill_start = time.time()
                        await self._prefill_batch(new_mini_batch, aaas_mode="cpu_lora")
                        prefill_end = time.time()
                        logging.critical("[STEP] Prefill {} reqs. Lat: {}".format( len(new_mini_batch.reqs), 1000*(prefill_end-prefill_start)))

                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                self.has_wait_tokens = 0
                return

        # 正常 decode 阶段， 如果可以直接decode就直接decode，否则通过暂停策略暂停一些请求
        # 释放一些管理的 token
        if self._can_decode(self.running_batch):
            self.stats_tool.count_output_tokens(self.running_batch)
            # check GPU LoRA loading here
            decode_start = time.time()
            # assert np.sum(self.progress) == 0, "not swapped"
            if (not hasattr(self, "progress")) or np.sum(self.progress) == 0:
                await self._decode_batch(self.running_batch, aaas_mode="gpu_lora_decode")
            elif hasattr(self, "progress"):
                while np.sum(self.progress) != 0:
                    pass
                await self._decode_batch(self.running_batch, aaas_mode="gpu_lora_decode")
            # await self._decode_batch(self.running_batch, aaas_mode="cpu_lora")
            decode_end = time.time()
            logging.critical("[STEP] Decode {} reqs. Lat: {}".format( len(self.running_batch.reqs), 1000*(decode_end-decode_start)))
            self._filter_runing_batch()
            self.has_wait_tokens += 1
            return
        else:
            # pause strategy
            paused_reqs = select_paused_reqs(self.running_batch, self.pause_strategy, self.req_queue, self.max_total_token_num)
            await self._pause_reqs(self.running_batch, paused_reqs)
            print("pasued req num:", len(self.req_queue.pause_req_dict))
            self.has_wait_tokens = 0
            return
        return

    async def _init_batch(self, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _prefill_batch(self, batch:Batch, aaas_mode:str=None):
        await self._init_batch(batch)
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id, aaas_mode=aaas_mode) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        finished_reqs, unfinished_reqs = batch.mark_and_get_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        batch.filter_out_finished_req(finished_reqs, unfinished_reqs)
        await self._handle_finish_req(batch, finished_reqs)
        # prefill 完成以后，将batch中所有保留的请求的状态标记为 RUNNING， 并重新更新其token消耗量
        # 必须先调用状态更新，后进行token使用量的计算
        batch.update_req_status_to_running()
        # batch.recalcu_batch_used_tokens()

        if "aaas-gpu" in self.mode or "aaas" in self.mode:
            await self.merge_lora_prefill( len(batch.reqs) )
        return

    async def merge_lora_prefill(self, num_lora_to_merge):
        merge_start = time.time()
        for tp_rank in range(self.world_size):
            await self.model_rpcs[tp_rank].merge_lora_prefill( num_lora_to_merge )
        merge_end = time.time()
        logging.critical("[STEP] Merge {} GPU LoRAs. Lat: {}".format(num_lora_to_merge, 1000*(merge_end-merge_start)))
        return
    
    async def _decode_batch(self, batch:Batch, aaas_mode:str=None):
        old_req_num = len(batch.reqs)
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id, aaas_mode=aaas_mode) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        finished_reqs, unfinished_reqs = batch.mark_and_get_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        batch.filter_out_finished_req(finished_reqs, unfinished_reqs)
        await self._handle_finish_req(batch, finished_reqs)

        # decode 之后, 更新 batch 的 token 使用量的统计
        if len(finished_reqs) != 0:
            batch.batch_used_tokens += old_req_num - sum([req.input_len + len(req.output_ids) - 1 for req in finished_reqs])
            
        else:
            batch.batch_used_tokens += old_req_num
        return

    async def _filter_batch(self, batch: Batch, finished_req_ids: List):
        req_id_list = [r.request_id for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].filter_batch(batch.batch_id, req_id_list, finished_req_ids) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch):
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return
    
    async def _pause_reqs(self, batch: Batch, pasue_reqs):
        pasue_reqs_info = [(r.request_id, r.req_status, r.offload_kv_len) for r in pasue_reqs]
        rets = [self.model_rpcs[tp_rank].pause_reqs(batch.batch_id, pasue_reqs_info) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, batch: Batch, finished_reqs):
        if len(finished_reqs) != 0:
            finished_req_ids = [req.request_id for req in finished_reqs]
            if batch.is_clear():
                await self._remove_batch(batch)
            else:
                await self._filter_batch(batch, finished_req_ids)
        return

    def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            self.running_batch = None
            return
    
    def _add_token_id_to_req(self, batch: Batch, req_ans):
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            req.output_ids.append(new_token_id)
            req.output_metadata_list.append(new_gen_metadata)
        return
        
    def _can_decode(self, batch: Batch):
        remaining_tokens = self.max_total_token_num - batch.batch_used_tokens - self.req_queue.pause_req_used_tokens
        return len(batch.reqs) <= remaining_tokens
        
    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        batch_out = BatchTokenIdOut()
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.has_generate_finished, req.aborted))
    
        self.send_to_detokenization.send_pyobj(batch_out)
        return

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 4:
                # if we use lora serving
                prompt_ids, sampling_params, request_id, lora_id = recv_req
                self.add_req(prompt_ids, sampling_params, request_id, lora_id)
            elif isinstance(recv_req, tuple) and len(recv_req) == 3:
                # base model serving, no lora here, 0 means dummy lora
                prompt_ids, sampling_params, request_id = recv_req
                self.add_req(prompt_ids, sampling_params, request_id, 0)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                await self.abort(request_id)
                self.send_to_detokenization.send_pyobj(abort_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return

def start_router_process(args, router_port, detokenization_port, model_rpc_ports, pipe_writer):
    try:
        router = RouterManager(
            args,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports)
    
        asyncio.run(router.wait_to_model_ready())
    except Exception as e:
        import traceback
        err_str = '\n'.join(traceback.format_exception(e))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send('init ok')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
