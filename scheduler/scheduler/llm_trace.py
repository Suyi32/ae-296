import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .instance import Req


MERGE_N = 16  # merge N user into one to reduce ILP complexity


class SimTrace(object):
    def __init__(self, invocation_path, llm_data_path, lora_ranks,
                 num_base_models=10, count_output=False, plot=True):
        self.num_base_models = num_base_models
        self.lora_ranks = lora_ranks
        with open(invocation_path, 'r') as f_inv:
            invocations = pd.read_csv(f_inv)
            invocations = invocations.drop(['HashOwner', 'HashApp', 'HashFunction', 'Trigger'], axis=1)
            self.invocations = invocations.to_numpy(dtype=np.int32)
            # filter out high RPM requests
            self.invocations[np.where(self.invocations > 10)] = 0
            # sample every 30 minutes -> 48 points in total
            self.invocations = self.invocations[:, ::30]
            if plot:
                plt.plot(np.arange(self.invocations.shape[1]), np.sum(self.invocations, axis=0))
                plt.xlabel("Time/min")
                plt.ylabel("RPM")
                plt.tight_layout()
                plt.savefig("request_per_min.png")

        with open(llm_data_path, 'r') as f_llm:
            json_data = json.load(f_llm)
            len_data = np.zeros(len(json_data))
            for i in range(len(json_data)):
                # simply use split as the tokenizer (x
                inst_tokens = json_data[i]['instruction'].split(" ")
                input_tokens = json_data[i]['input'].split(" ")
                len_data[i] = len(inst_tokens) + len(input_tokens)
                if count_output:
                    output_tokens = json_data[i]['output'].split(" ")
                    len_data[i] += len(output_tokens)
            prob_hist, _ = np.histogram(
                len_data, bins=np.arange(np.min(len_data), np.max(len_data)), density=True)
            print("length (#tokens): mean={}, min={}, max={}".format(
                np.mean(len_data), np.min(len_data), np.max(len_data))
            )
            if plot:
                plt.clf()
                plt.bar(np.arange(len(prob_hist)), prob_hist)
                plt.xlabel("#tokens")
                plt.ylabel("Density")
                plt.savefig("token_len.png")
            self.len_dist = prob_hist
            self.all_lengths = np.arange(np.min(len_data) + 1, np.max(len_data))

    def gen_trace(self, t_start=0, t_end=1440, usr_indices=None):
        # invocations by time (per minute)
        if usr_indices is not None:
            invocations = np.sum(self.invocations[usr_indices, t_start: t_end], axis=0)
        else:
            invocations = np.sum(self.invocations[:, t_start: t_end], axis=0)
        # total invocations number
        total = np.sum(invocations)
        # print(f"Total: {total} requests, t_start={t_start}, t_end={t_end}")
        start_time_sec = 60 * np.random.random(size=(total,))
        lengths = np.random.choice(self.all_lengths, total, p=self.len_dist).astype(np.int32)
        ranks = np.random.choice(self.lora_ranks, total)
        models = np.random.choice(np.arange(self.num_base_models), total)
        requests = []
        i = 0
        for t in range(0, t_end - t_start):
            for _ in range(invocations[t]):
                r = Req(models[i], ranks[i], lengths[i])
                req_t = (t + t_start) * 60 + start_time_sec[i]
                requests.append((req_t, r))
                i += 1
        # sort by request time
        requests.sort(key=lambda x: x[0])
        return requests

    def gen_trace_by_user(self, t_start=0, t_end=1440, usr_indices=None):
        if usr_indices is not None:
            invocations = self.invocations[usr_indices, t_start: t_end]
        else:
            invocations = self.invocations[:, t_start: t_end]
        # total invocations number
        total = np.sum(invocations)
        requests = {}
        start_time_sec = 60 * np.random.random(size=(total,))
        lengths = np.random.choice(self.all_lengths, total, p=self.len_dist).astype(np.int32)
        i = 0
        for u in range(0, len(invocations), MERGE_N):
            usr_reqs = []
            usr_rank = np.random.choice(self.lora_ranks, 1)[0]
            usr_model = np.random.choice(np.arange(self.num_base_models), 1)[0]
            tmp_invocation = np.sum(invocations[u: u + MERGE_N], axis=0)
            for t in range(0, t_end - t_start):
                for _ in range(tmp_invocation[t]):
                    r = Req(usr_model, usr_rank, lengths[i])
                    req_t = (t + t_start) * 60 + start_time_sec[i]
                    usr_reqs.append((req_t, r))
                    i += 1
            if usr_reqs == []:
                continue
            usr_reqs.sort(key=lambda x: x[0])
            if usr_model not in requests:
                requests[usr_model] = [usr_reqs]
            else:
                requests[usr_model].append(usr_reqs)
        return requests

    def gen_group_trace(self, t_start, t_end, group_size, num_users=None):
        group_traces = []
        if num_users is None:
            num_users = self.invocations.shape[0]
        for i in range(0, num_users, group_size):
            end = min(i + group_size, num_users)
            trace = self.gen_trace(t_start, t_end, np.arange(i, end))
            for j in range(len(trace)):
                trace[j][1].gid = (i // group_size)
            group_traces.append(trace)
        return group_traces
