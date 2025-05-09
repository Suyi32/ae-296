import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from collections import Counter
from .instance import AaaSInstance, Req
from .cost_model import SLoRACostModel, PunicaCostModel


class InstanceManager(object):
    # if an instance is empty more than this, it will be closed
    MAX_EMPTY_DURATION = 5
    # max #instances
    CLUSTER_LIMIT = 80
    AVG_SEQ_LEN = 60

    def __init__(self, lora_type='slora', method='None'):
        self.instances: Dict[int, AaaSInstance] = {}
        self.mapping: Dict[str, List[Tuple[int, int]]] = {}
        self.stop_counts = Counter()
        self.instance_id = 100
        self.lora_type = lora_type
        self.method = method
        self.max_bsz = 32
        self.cur_t = 0
        self.req_buffer = []
        if lora_type == 'slora':
            self.cost_model = SLoRACostModel("./slora_cost_model")
            self.cost_func = lambda reqs: np.mean([i.lora_rank for i in reqs]) if len(reqs) != 0 else 0
            self.rank_func = lambda ranks: np.mean(ranks) if len(ranks) != 0 else 0
        else:
            self.cost_model = PunicaCostModel("./punica_cost_model")
            self.cost_func = lambda reqs: np.max([i.lora_rank for i in reqs]) if len(reqs) != 0 else 0
            self.rank_func = lambda ranks: np.max(ranks) if len(ranks) != 0 else 0
        while len(self.instances) < self.CLUSTER_LIMIT:
            self.add_instance(0, 64, self.max_bsz)

    def add_instance(self, model_name, lora_rank, max_batch_size):
        instance_id = self.instance_id
        self.instances[instance_id] =\
            AaaSInstance(model_name, lora_rank, max_batch_size, self.cost_model, self.cost_func, self.lora_type, self.method)
        if model_name not in self.mapping:
            self.mapping[model_name] = [(instance_id, lora_rank)]
        else:
            self.mapping[model_name].append((instance_id, lora_rank))
            # sort the instances by lora_rank, increasing order
            self.mapping[model_name].sort(key=lambda x: x[1])
        # forward to next id
        self.instance_id += 1
        return instance_id

    def search(self, req: Req, req_t, delay):
        # update time stamp & buffer
        self.req_buffer.append((req_t, req))
        self.cur_t = req_t
        # find candidates
        candidates = self.mapping[req.model]
        impacts = []
        full_list = []
        delay_reqs = [i[1] for i in self.req_buffer]
        delay_time = [i[0] for i in self.req_buffer]
        delay_ranks = [i.lora_rank for i in delay_reqs]

        if len(delay_reqs) <= delay:
            # No schedule if should delay
            return

        for (inst_id, rank) in candidates:
            inst = self.instances[inst_id]
            run_ranks, queue_ranks, _, _ = inst.get_status()
            already_on_machine = run_ranks + queue_ranks
            all_ranks = already_on_machine + delay_ranks
            if len(all_ranks) > self.max_bsz:
                full_list.append([len(all_ranks), inst_id])
                continue
            # if queue is empty, this req will interrupt the existing decoding, hence prefill
            # interference is its prefill latency
            if len(queue_ranks) == 0:
                prefill_t =  1.5 * self.cost_model.get_latency(len(delay_ranks), self.rank_func(delay_ranks))
            # otherwise, the queue is not empty, hence the next step is already a prefill, hence
            # this prefill only introduce (prefill_new - prefill_old) interference
            else:
                prefill_t = 1.5 * (
                    self.cost_model.get_latency(len(delay_ranks) + len(queue_ranks), self.rank_func(delay_ranks + queue_ranks)) -\
                    self.cost_model.get_latency(len(queue_ranks), self.rank_func(queue_ranks))
                )
            decode_before = self.cost_model.get_latency(len(already_on_machine), self.rank_func(already_on_machine))
            decode_after = self.cost_model.get_latency(len(all_ranks), self.rank_func(all_ranks))
            burden = len(already_on_machine) * (prefill_t / self.AVG_SEQ_LEN + (decode_after - decode_before))
            impacts.append([burden, inst_id])

        if len(impacts) > 0:
            best_inst_id = min(impacts)[-1]
        elif len(full_list) == self.CLUSTER_LIMIT:
            best_inst_id = min(full_list)[-1]
        for _, req in self.req_buffer:
            self.instances[best_inst_id].add_request(req)
        self.req_buffer = []


    def rand_assign(self, req: Req):
        return random.choice(self.mapping[req.model])[0]
    
    def most_idle_assign(self, req: Req):
        most_idle_val, most_idle_id = float("inf"), float("inf")
        for (inst_id, rank) in self.mapping[req.model]:
            inst = self.instances[inst_id]
            run_ranks, queue_ranks, _, _ = inst.get_status()
            all_ranks = run_ranks + queue_ranks
            if len(all_ranks) < most_idle_val:
                most_idle_val = len(all_ranks)
                most_idle_id = inst_id
        return most_idle_id
    
    def dlora_assign(self, req: Req):
        most_idle_val, most_idle_id = float("inf"), float("inf")
        for (inst_id, rank) in self.mapping[req.model]:
            inst = self.instances[inst_id]
            run_ranks, queue_ranks, run_tokens, queue_tokens = inst.get_status()
            all_tokens = run_tokens + queue_tokens
            if all_tokens < most_idle_val:
                most_idle_val = all_tokens
                most_idle_id = inst_id
        return most_idle_id

    def first_fit_assign(self, req: Req):
        for (inst_id, rank) in self.mapping[req.model]:
            inst = self.instances[inst_id]
            run_ranks, queue_ranks, _, _ = inst.get_status()
            all_ranks = run_ranks + queue_ranks
            if len(all_ranks) < self.max_bsz:
                return inst_id
        return self.mapping[req.model][0][0]

    def remove_instance(self, instance_id):
        # remove it from instance set
        self.instances.pop(instance_id)
        # remove it from mapping
        for k in self.mapping:
            for j in range(len(self.mapping[k])):
                machine_id, _ = self.mapping[k][j]
                if machine_id == instance_id:
                    self.mapping[k].pop(j)
                    break

    def add_request(self, instance_id: int, req: Req):
        self.instances[instance_id].add_request(req)

    def run_until(self, cur_time):
        for i in list(self.instances.keys()):
            self.instances[i].run_until(cur_time)


class BaseScheduler(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def route_request(self, req: Req):
        pass

    @abstractmethod
    def get_instances(self):
        pass

    @abstractmethod
    def dump_logs(self):
        pass

    def get_num_instances(self):
        return len(self.get_instances())

    def calc_utilization(self):
        util = 0
        count = 0
        for i in self.get_instances():
            if i.empty():
                continue
            util += i.calc_utilization()
            count += 1
        return util / count if count != 0 else 0

    def done(self):
        for i in self.get_instances():
            if not i.empty():
                return False
        return True


class Scheduler(BaseScheduler):
    def __init__(self, instance_max_batch_size, threshold, lora_type='slora', method="None"):
        super().__init__()
        self.method = method
        self.lora_type = lora_type
        self.manager = InstanceManager(self.lora_type, self.method)
        self.utilization_thresh = threshold
        self.max_batch_size = instance_max_batch_size
        self.method = None

    def get_instances(self):
        return self.manager.instances.values()

    def run_until(self, t):
        self.manager.run_until(t)

    def route_request(self, req: Req, req_t: float, method: str='default', delay: int=10):
        # match method:
        #     case 'Toppings':
        #         # Toppings schedules it in "search" function
        #         instance_id = self.manager.search(req, req_t, delay)
        #         return
        #     case 'Random':
        #         instance_id = self.manager.rand_assign(req)
        #     case 'LoadBalance':
        #         instance_id = self.manager.most_idle_assign(req)
        #     case 'FirstFit':
        #         instance_id = self.manager.first_fit_assign(req)
        #     case 'dLoRA':
        #         instance_id = self.manager.dlora_assign(req)
        if method == 'Toppings':
            self.manager.search(req, req_t, delay)
            return
        elif method == 'Random':
            instance_id = self.manager.rand_assign(req)
        elif method == 'LoadBalance':
            instance_id = self.manager.most_idle_assign(req)
        elif method == 'FirstFit':
            instance_id = self.manager.first_fit_assign(req)
        elif method == 'dLoRA':
            instance_id = self.manager.dlora_assign(req)
        self.manager.add_request(instance_id, req)
    

    def dump_logs(self):
        del self.manager.instances
