import os 
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import json

foldername = "./rank_perf_model"
def get_decoding_performance(filename):
    fn_items = filename[:-4].split("_")
    bsz = int(fn_items[2])
    ranks = []

    rank_start = False
    for item in fn_items:
        if not rank_start and "rank" in item:
            rank_start = True
        elif rank_start:
            ranks.append( int(item) )
    print(filename, bsz, ranks)
    
    total_ranks = 0
    rank_num = len(ranks)
    for rank in ranks:
        total_ranks += rank * (bsz // rank_num)
    print("total_ranks:", total_ranks)

    decoding_lats = []
    with open(os.path.join(foldername, filename), "r") as fr:
        for line in fr:
            if "[STEP]: Decode" in line:
                items = line.strip().split(" ")
                if int(items[-4]) == bsz:
                    decoding_time = float(items[-1])
                    decoding_lats.append( decoding_time )

    return bsz, total_ranks, np.sort(decoding_lats)[:-20]


filenames = [ item for item in os.listdir("./{}".format(foldername)) if "server" in item ]

bsz_rank_lats = {}
for filename in filenames:
    bsz, total_ranks, decoding_lat = get_decoding_performance(filename)
    if bsz not in bsz_rank_lats:
        bsz_rank_lats[bsz] = {}
    if total_ranks not in bsz_rank_lats[bsz]:
        bsz_rank_lats[bsz][total_ranks] = []
    bsz_rank_lats[bsz][total_ranks].extend(decoding_lat)
print(bsz_rank_lats)

fig, ax = plt.subplots(1, 1, figsize=(14,7))


for bsz in sorted(bsz_rank_lats.keys()):
    plot_x = []
    plot_y = []
    err_y = []
    for rank in sorted(bsz_rank_lats[bsz].keys()):
        print(len(bsz_rank_lats[bsz][rank]))
        plot_x.append(rank)
        plot_y.append(np.mean(bsz_rank_lats[bsz][rank]))
        err_y.append(np.std(bsz_rank_lats[bsz][rank]))

    plot_x = np.array(plot_x)
    plot_y = np.array(plot_y)
    err_y = np.array(err_y)

    # ax.plot(plot_x, plot_y, marker="o", label="SLORACostModel")
    ax.errorbar(plot_x, plot_y, yerr=err_y, marker="o", label="bsz={}".format(bsz))


fontsize = 20
ax.set_xlabel("Number of Total Ranks", fontsize=fontsize)
ax.set_ylabel("Decoding Latency (ms)", fontsize=fontsize)
ax.tick_params(axis='x', labelsize=fontsize)
ax.tick_params(axis='y', labelsize=fontsize)
ax.legend(fontsize=fontsize)
ax.grid()

fig.tight_layout()
plt.show()