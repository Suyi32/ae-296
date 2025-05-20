import numpy as np


class Req(object):
    def __init__(self, model, lora_rank, num_tokens):
        self.is_prefilled = False
        self.model = model
        self.lora_rank = lora_rank
        self.num_tokens_left = num_tokens
        self.max_tokens = num_tokens
        self.exec_time = 0
        self.first_token_lat = 0
        self.gid = None

    def prefill(self):
        self.is_prefilled = True
        self.first_token_lat = self.exec_time

    def decode(self):
        assert self.is_prefilled, "Can't decode: req not prefilled"
        self.num_tokens_left -= 1

    def done(self):
        return self.num_tokens_left == 0

    def __str__(self):
        return f"<Req: {self.num_tokens_left}tokens, rank={self.lora_rank}, model={self.model}>"

    def __repr__(self):
        return str(self)


class Batch(object):
    def __init__(self):
        self.reqs = []
        self.cur_batch_size = 0

    def add(self, req):
        self.reqs.append(req)
        self.cur_batch_size += 1

    def merge(self, other):
        self.cur_batch_size += other.cur_batch_size
        self.reqs += other.reqs

    def empty(self):
        return self.cur_batch_size == 0

    def __str__(self):
        return f"<Batch: batch_size={self.cur_batch_size}, reqs={self.reqs}>"

    def __repr__(self):
        return str(self)


class ReqQueue(object):
    def __init__(self, max_batch_size):
        self.reqs = []
        self.max_batch_size = max_batch_size

    def push(self, new_req: Req):
        self.reqs.append(new_req)

    def generate_new_batch(self, bsz_needed):
        batch_size = min(bsz_needed, min(len(self.reqs), self.max_batch_size))
        new_batch = Batch()
        new_batch.reqs = self.reqs[:batch_size]
        new_batch.cur_batch_size = batch_size
        self.reqs = self.reqs[batch_size:]
        return new_batch

    def __str__(self):
        return f"<ReqQueue: num_reqs_in_queue={len(self.reqs)}, reqs={self.reqs}>"

    def __repr__(self):
        return str(self)

    def migrate_to(self, other_inst):
        # print(f"!!migrate: {sum([i.num_tokens_left for i in self.reqs])}, {sum([i.num_tokens_left for i in other_inst.reqs])}")
        idxmap = []
        for i, r in enumerate(self.reqs):
            idxmap.append([r.num_tokens_left, i])
        idxmap.sort(reverse=True)
        nummove = (len(self.reqs) + len(other_inst.reqs)) // 2
        mig_ids = [idxmap[i][1] for i in range(nummove)]
        tmp_reqs = []
        for i, r in enumerate(self.reqs):
            if i not in mig_ids:
                tmp_reqs.append(r)
            else:
                other_inst.reqs.append(r)
        self.reqs = tmp_reqs

class AaaSInstance(object):
    def __init__(self, model_name, lora_rank, max_batch_size, cost_model, cost_func, lora_type, method):
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.max_batch_size = max_batch_size
        self.running_batch = Batch()
        self.req_queue = ReqQueue(self.max_batch_size)
        self.metric_buffer = []
        self.cost_model = cost_model
        self.cost_func = cost_func
        self.machine_time = 0
        self.lora_type = lora_type
        self.method = method

    def __del__(self):
        # destructor: log all request metrics
        self.dump_logs(self.method, self.lora_type)

    def dump_logs(self, method, lora_type):
        with open(f"metrics_{method}_{lora_type}.log", 'a+') as f:
            f.write(f"{self.machine_time}\n")
            for (first, total, len) in self.metric_buffer:
                f.write(f"{first}, {total}, {len}\n")
        self.metric_buffer = []

    def migrate_to(self, other_inst):
        self.req_queue.migrate_to(other_inst.req_queue)

    def add_request(self, req: Req):
        self.req_queue.push(req)

    def prefill(self, batch: Batch):
        for i in range(len(batch.reqs)):
            batch.reqs[i].prefill()
        return batch

    def decode(self, batch: Batch):
        new_batch = Batch()
        for i in range(len(batch.reqs)):
            batch.reqs[i].decode()
            # filter out done reqs
            if not batch.reqs[i].done():
                new_batch.add(batch.reqs[i])
            else:
                self.metric_buffer.append(
                    (batch.reqs[i].first_token_lat, batch.reqs[i].exec_time, batch.reqs[i].max_tokens))
        return new_batch
    
    def run_until(self, cur_time):
        if self.machine_time >= cur_time:
            return
        while self.machine_time < cur_time:
            res = self.step()
            if not res:
                self.machine_time = cur_time
                break

    def add_exec_time(self, t):
        self.machine_time += (t / 1000)
        # add exec time for all reqs
        for i in range(len(self.running_batch.reqs)):
            self.running_batch.reqs[i].exec_time += t
        for i in range(len(self.req_queue.reqs)):
            self.req_queue.reqs[i].exec_time += t

    def step(self):
        # 当前无运行请求时
        if self.running_batch.empty():
            new_batch = self.req_queue.generate_new_batch(self.max_batch_size)
            if not new_batch.empty():
                new_batch = self.prefill(new_batch)
                self.running_batch = new_batch
                self.add_exec_time(
                    1.5 * self.cost_model.get_latency(
                        len(new_batch.reqs), self.cost_func(new_batch.reqs)
                    )
                )
                return True
        # 有运行请求，但是已经到了可以调度新的请求合并推理的时机
        if self.running_batch.cur_batch_size < self.max_batch_size:
            new_batch = self.req_queue.generate_new_batch(self.max_batch_size - self.running_batch.cur_batch_size)
            if not new_batch.empty():
                new_batch = self.prefill(new_batch)
                self.running_batch.merge(new_batch)
                self.add_exec_time(
                    1.5 * self.cost_model.get_latency(
                        len(new_batch.reqs), self.cost_func(new_batch.reqs)
                    )
                )
                return True

        # 正常 decode 阶段， 如果可以直接decode就直接decode
        if not self.running_batch.empty():
            self.running_batch = self.decode(self.running_batch)
            self.add_exec_time(
                self.cost_model.get_latency(
                    len(self.running_batch.reqs), self.cost_func(self.running_batch.reqs)
                )
            )
            return True
        return False

    def get_status(self):
        running_tokens = sum((i.max_tokens - i.num_tokens_left) for i in self.running_batch.reqs)
        queuing_tokens = sum(i.num_tokens_left for i in self.req_queue.reqs)
        running_ranks = [i.lora_rank for i in self.running_batch.reqs]
        queuing_ranks = [i.lora_rank for i in self.req_queue.reqs]
        return running_ranks, queuing_ranks, running_tokens, queuing_tokens

    def calc_utilization(self):
        '''
        In LoRA serving, instance with large LoRA rank is able to serve lower-rank LoRA requests
        e.g., if the instance's rank is 64, it can serve requests with rank<=64. However, serving
        lower-ranked requests is sub-optimal, because:
        (1) the request's LoRA will be padded to the higher rank, causing longer serving latency
        (2) the instance's GPU memory & computation is wasted, causing resource waste
        Hence, we define utilization = sum_i( req_i.lora_rank ) / (batch_size * instance.lora_rank)
        '''
        if self.empty():
            return 0
        used_rank = sum([r.lora_rank for r in (self.running_batch.reqs + self.req_queue.reqs)])
        all_rank = self.lora_rank * (self.running_batch.cur_batch_size + len(self.req_queue.reqs))
        return used_rank / all_rank

    def empty(self):
        return self.running_batch.empty() and len(self.req_queue.reqs) == 0
