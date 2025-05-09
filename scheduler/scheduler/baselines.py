import random
import numpy as np
from typing import List, Tuple
from multiprocessing import Pool
from math import ceil
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, value
from .manager import BaseScheduler
from .instance import Req, AaaSInstance


def calc_group_spec(traces, max_batch_size, step_time):
    t_start, t_end = traces[0][0], traces[-1][0]
    max_rank = max(req.lora_rank for (_, req) in traces)
    instance_cap = max_batch_size / step_time  # tokens/s
    required_rps = {}
    for _, req in traces:
        if req.model not in required_rps:
            required_rps[req.model] = req.max_tokens
        else:
            required_rps[req.model] += req.max_tokens
    total_insts = 0
    for model_rps in required_rps.values():
        rps = model_rps / (t_end - t_start)
        total_insts += ceil(rps / instance_cap)
    return total_insts, max_rank


def ilp_solve(trace: List[List[Tuple[float, Req]]], start_gid: int, step_time: float,
              period_time: float, group_size: int, max_batch_size: int):
    num_users = len(trace)
    max_groups = ceil(num_users / group_size)
    sum_tkn_len = sum(
        sum(t[1].max_tokens for t in usr_trace)
        for usr_trace in trace
    )
    avg_tkn_len = sum_tkn_len / sum(len(usr_trace) for usr_trace in trace)
    rps = np.array([len(i) / period_time for i in trace])
    model_rps = max_batch_size / step_time
    inv_n = 1 / (rps / model_rps)
    cluster_size = ceil(3 * np.sum(rps) * avg_tkn_len / model_rps)
    per_group_max_workers = ceil(cluster_size / max_groups)

    problem = LpProblem("AaaSGrouping", LpMaximize)
    min_bt = LpVariable("min_bt", lowBound=0, cat='Continuous')
    # big-M method for linearization
    M = 1000
    # Whether assign user_i to group_j
    x = LpVariable.dicts(
        "x",
        [(i, j) for i in range(num_users) for j in range(max_groups)],
        cat='Binary'
    )
    z = LpVariable.dicts(
        "z",
        [(i, j) for i in range(num_users) for j in range(max_groups)],
        cat='Continuous'
    )
    n_workers_in_group = LpVariable.dicts(
        "B",
        range(max_groups),
        cat="Integer",
        lowBound=0
    )

    # each model has a group
    for i in range(num_users):
        problem += lpSum([x[i, j] for j in range(max_groups)]) == 1

    # users_in_each_group <= group_size && sum(workers_in_each_group) <= cluster_size
    for j in range(max_groups):
        problem += (lpSum(x[i, j] for i in range(num_users)) <= group_size)
        problem += (n_workers_in_group[j] <= per_group_max_workers)

    # z[i, j] = x[i, j] * n_workers_in_group[j]
    for i in range(num_users):
        for j in range(max_groups):
            problem += (z[i, j] <= M * x[i, j])
            problem += (z[i, j] >= n_workers_in_group[j] - M * (1 - x[i, j]))
            problem += (z[i, j] <= n_workers_in_group[j])
            problem += (z[i, j] >= 0)

    # min_bt <= min_i(bt_i), where bt_i = sum(z_ij) / rps_i
    for i in range(num_users):
        problem += (min_bt <= lpSum(z[i, j] for j in range(max_groups)) * inv_n[i])
    problem += min_bt

    problem.solve()
    grp_stat = [0] * max_groups
    for i in range(num_users):
        for j in range(max_groups):
            if value(x[i, j]) != 0:
                for k in range(len(trace[i])):
                    trace[i][k][1].gid = j + start_gid
                grp_stat[j] += 1
    group_specs = []
    for j in range(max_groups):
        spec = (start_gid + j, int(value(n_workers_in_group[j])), 64)
        group_specs.append(spec)
    return trace, group_specs


class GroupScheduler(BaseScheduler):
    def __init__(self, instance_max_batch_size, group_specs) -> None:
        super().__init__()
        self.instances = {}
        self.max_batch_size = instance_max_batch_size
        for (gid, num_instances, instance_rank) in group_specs:
            self.instances[gid] =\
                [AaaSInstance("Dummy", instance_rank, self.max_batch_size)
                 for _ in range(num_instances)]

    def route_request(self, req: Req):
        assert req.gid is not None
        num_avail_insts = len(self.instances[req.gid])
        inst_id = random.randint(0, num_avail_insts - 1)
        self.instances[req.gid][inst_id].add_request(req)

    def get_instances(self):
        insts = []
        for g_insts in self.instances.values():
            insts += g_insts
        return insts

    def step(self):
        for i in self.instances:
            for j in range(len(self.instances[i])):
                self.instances[i][j].step()

    def dump_logs(self):
        for i in self.get_instances():
            i.dump_logs()


class ShepherdScheduler(GroupScheduler):
    def __init__(self, instance_max_batch_size, group_specs) -> None:
        super().__init__(instance_max_batch_size, group_specs)

    @classmethod
    def merge_traces(cls, traces_by_user):
        traces = []
        for model_name in traces_by_user:
            for user_trace in traces_by_user[model_name]:
                traces += user_trace
        traces.sort(key=lambda x: x[0])
        return traces

    @classmethod
    def solve_groupping(cls, traces_by_user, group_size, period_time, max_batch_size, step_time):
        proc_pool = Pool(len(traces_by_user) + 1)
        g_start = 0
        all_specs = []
        results = []
        for model in traces_by_user:
            results.append((
                model,
                proc_pool.apply_async(
                    ilp_solve,
                    (traces_by_user[model], g_start, step_time,
                     period_time, group_size, max_batch_size))
            ))
            g_start += 100
        proc_pool.close()
        proc_pool.join()
        for model, res in results:
            res_trace, g_spec = res.get()
            traces_by_user[model] = res_trace
            all_specs += g_spec
        return all_specs, traces_by_user
