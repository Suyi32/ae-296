import random
import os
import argparse
import logging
import matplotlib
from matplotlib.legend_handler import HandlerTuple
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .manager import Scheduler
from .llm_trace import SimTrace


random.seed(42)
np.random.seed(42)

def run_simulation(scheduler: Scheduler, trace, method='default', delay=10):
    fn = f"metrics_{method}_{scheduler.lora_type}.log"
    print(fn)
    if not os.path.exists(fn):
        for i in tqdm(range(len(trace))):
            t = trace[i][0]
            scheduler.run_until(t)
            scheduler.route_request(trace[i][1], trace[i][0], method, delay)
        scheduler.dump_logs()

    with open(fn, 'r') as f:
        total_lat_acc, per_tok_len_acc = [], []
        content = f.read().split("\n")
        machine_time = 0
        for line in content:
            if len(line) <= 1:
                continue
            if ',' not in line:
                machine_time += float(line)
                continue
            first_lat, total_lat, length = line.split(',')
            total_lat_acc.append(float(total_lat))
            per_tok_len_acc.append(float(total_lat) / float(length))
    return total_lat_acc, per_tok_len_acc


def histogram(data, end_bin):
    hist, bin_edges = np.histogram(data, bins=np.arange(0, end_bin, 0.001))
    hist = hist / np.sum(hist).astype(np.float64)
    return bin_edges[:-1], np.cumsum(hist)


def run_aaas(generator, t_start_min, t_end_min, thresh, max_bsz, method, lora_type, delay):
    trace = generator.gen_trace(t_start_min, t_end_min)
    duration = 60 * (t_end_min - t_start_min)
    logging.critical(f"RPS: {len(trace) / duration}")
    scheduler = Scheduler(instance_max_batch_size=max_bsz, threshold=thresh, lora_type=lora_type, method=method)
    req_time, token_lat = run_simulation(scheduler, trace, method, delay)
    avg_f2 = np.mean(req_time)
    avg_t2 = np.mean(token_lat)
    logging.critical(f"{method}, Avg. request lat: {avg_f2}, lat/token: {avg_t2}")
    return req_time, token_lat 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--invocation_path", type=str, default="./invocations_per_function_md.anon.d01.csv",
        help='path to Azure function trace csv file')
    parser.add_argument(
        "--llm_data_path", type=str, default="./input_output_data.json",
        help='path to trace json file')
    parser.add_argument("--t_start", type=int, default=0, help='simulation start time (minute)')
    parser.add_argument("--t_end", type=int, default=1, help='simulation end time (minute)')
    parser.add_argument("--num_base_models", type=int, default=1, help='# of base models')
    parser.add_argument("--num_loras", type=int, default=4, help='# of different lora ranks')
    parser.add_argument("--max_bsz", type=int, default=16, help='max batch size of each base model')
    parser.add_argument("--thresh", type=float, default=0.7, help='rank utilization threshold for AaaS scheduler')
    parser.add_argument("--slo", type=float, default=0.04, help='SLO of scheduler')
    parser.add_argument("--regroup_period", type=int, default=4, help='re-grouping period for shepherd')
    args = parser.parse_args()

    invocation_path = args.invocation_path
    llm_data_path = args.llm_data_path
    t_start_min, t_end_min = args.t_start, args.t_end
    assert args.num_loras <= 6, "Too many LoRA ranks."
    ranks = [8, 16, 32, 64, 128, 256][:args.num_loras]
    num_base_models = args.num_base_models
    max_bsz = args.max_bsz
    thresh = args.thresh
    slo = args.slo
    generator = SimTrace(invocation_path, llm_data_path, ranks, num_base_models, plot=False, count_output=True)
    configs = [('Random', None), ('FirstFit', None), ('dLoRA', None), ('Toppings', 0)]
    colors = ["C4", "C1", "C2", "C3"]
    fig1, ax1 = plt.subplots(1, 2, figsize=(21, 4.5))
    fig2, ax2 = plt.subplots(1, 2, figsize=(21, 4.5))
    axs = [ax1, ax2]
    bar_width = 0.66

    for j, lora_type_ in enumerate(["slora",'punica']):
        bar_legend = []
        line_legend = []
        req_lats = []
        token_lats = []
        for method, delay in configs:
            if f"{method}_{lora_type_}.log" in os.listdir():
                # load from log
                req_lat, token_lat = [], []
                print(f"{method}_{lora_type_}.log")
                with open(f"{method}_{lora_type_}.log", 'r') as f:
                    content = f.readlines()
                    for line in content:
                        line = line.strip('\n').split(", ")
                        req_lat.append(float(line[0]))
                        token_lat.append(float(line[1]))
                    token_lats.append(token_lat)
                    req_lats.append(req_lat)
            else:
                print(f"{method}_{lora_type_}.log not found")
                req_lat, token_lat = run_aaas(generator, t_start_min, t_end_min, thresh, max_bsz, method, lora_type_, delay)
                req_lats.append(req_lat)
                token_lats.append(token_lat)
                with open(f"{method}_{lora_type_}.log", 'w') as f:
                    for i in range(len(req_lat)):
                        f.write(f"{req_lat[i]}, {token_lat[i]}\n")

        styles = ['dashdot', 'dotted', "dashed", 'solid']
        hatchs = ['/', '\\', 'x', '']
        pos = [-0.4, 0.6, 1.6, 2.6]
        

        for i, lat_i in enumerate(token_lats):
            color = colors[i]
            att = len(np.where(np.array(lat_i)/1000 < slo)[0]) / len(lat_i) - 0.01
            m, _ = configs[i]
            bar_legend.append(axs[j][0].bar([i - bar_width/2.0] , [att], label=m, hatch=hatchs[i], color=color, width=bar_width))
            axs[j][0].text(pos[i] + 0.05, att, "{}".format(round(att, 2)), fontsize=32, ha='center', va='bottom')

        bound = 0.15
        for i, lat_i in enumerate(token_lats):
            color = colors[i]
            hist = histogram(np.array(lat_i)/1000, bound)
            m, _ = configs[i]
            print(m, np.mean(lat_i))
            tmp, = axs[j][1].plot(hist[0], hist[1], label=m, linewidth=8, linestyle=styles[i],color=color)
            line_legend.append(tmp)

        axs[j][1].set_xlim(0.025, bound)
        axs[j][0].set_ylim(0, 1.18)
        axs[j][0].grid()
        axs[j][1].grid()
        axs[j][0].tick_params(labelsize=40)
        axs[j][1].tick_params(labelsize=40)
        axs[j][1].set_ylabel("CDF", fontsize=40)
        axs[j][0].get_xaxis().set_ticks([])
    axs[0][0].set_ylabel("SLO", fontsize=40)
    axs[1][0].set_ylabel("SLO", fontsize=40)
    axs[1][0].set_xlabel("\nMethod", fontsize=40)
    axs[1][1].set_xlabel("Time Per Token", fontsize=40)


    legend = [(x,y) for x,y in zip(bar_legend, line_legend)]
    label = ["Random", "FirstFit", "dLoRA", "Toppings"]
    axs[0][1].set_xlim(0.025, 0.075)
    axs[1][1].set_xlim(0.025, 0.075)
    
    fig1.tight_layout()
    fig2.tight_layout()
    p = fig1.legend(legend, label, loc='lower left', borderaxespad=0.006, bbox_to_anchor=(0.06, 0.93, 2.0, 0.05), ncol=4, labelspacing=0.0,
                        handlelength = 5.5,
                        frameon=False,columnspacing=0.2,handletextpad=0.1, fontsize=36, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)})
    p2 = fig2.legend(legend, label, loc='lower left', borderaxespad=0.006, bbox_to_anchor=(0.06, 0.93, 2.0, 0.05), ncol=4, labelspacing=0.0,
                        handlelength = 5.5,
                        frameon=False,columnspacing=0.2,handletextpad=0.1, fontsize=36, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)})

    fig1.savefig(f"schduler_results_1.pdf", bbox_inches='tight', bbox_extra_artists=(p,), pad_inches=0.03)
    fig2.savefig(f"schduler_results_2.pdf", bbox_inches='tight', bbox_extra_artists=(p2,), pad_inches=0.03)



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.CRITICAL)
    main()
