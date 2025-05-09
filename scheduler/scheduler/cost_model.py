import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from mpl_toolkits.mplot3d import Axes3D
from abc import abstractmethod
from glob import glob


class BaseCostModel(object):
    def __init__(self, logdir) -> None:
        self.logdir = logdir
        self.costs = self.parse_logs()
        self.points, self.values = self.cost_to_grid(self.costs)
        self.interp = LinearNDInterpolator(self.points, self.values, fill_value=np.nan,
                                  rescale=False)

    @abstractmethod
    def parse_logs(self):
        pass

    def cost_to_grid(self, costs):
        points, values = [], []
        for bsz in costs:
            for rank in costs[bsz]:
                points.append((bsz, rank))
                values.append(costs[bsz][rank])
        return np.array(points), np.array(values)
    
    def get_latency(self, bsz, rank):
        if rank == 0:
            return 0
        # print("cost model get:", bsz, rank, self.interp((bsz, rank)))
        return self.interp((bsz, rank))

    def show_heatmap(self, metric_name, ax):
        heatmap_data = []
        for key, value in self.costs.items():
            for subkey, subvalue in value.items():
                # if key in [4, 8, 12, 16, 20, 24, 28, 32]:
                heatmap_data.append([key, subkey, subvalue])

        df = pd.DataFrame(heatmap_data, columns=['BatchSize', metric_name, 'Time Per Token (ms)'])
        pivot_df = df.pivot(index='BatchSize', columns=metric_name, values='Time Per Token (ms)')
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", ax=ax)
    
    def show_surface(self, metric_name, ax):
        x = sorted(set([k for d in self.costs.values() for k in d.keys() ]))
        y = sorted(self.costs.keys())
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self.costs[yi][xi] if xi in self.costs[yi] else np.nan for xi in x] for yi in y])
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.tick_params(labelsize=14)
        ax.set_xlabel(metric_name, fontsize=16)
        ax.set_ylabel('BatchSize', fontsize=16)
        ax.set_zlabel('Time Per Token (ms)', fontsize=16)


class SLoRACostModel(BaseCostModel):
    def __init__(self, logdir) -> None:
        super().__init__(logdir)

    def parse_logs(self):
        costs = {}
        for fn in glob(self.logdir + "/*.log"):
            bsz, *ranks = [int(i) for i in re.findall(r"_(\d+)", fn)]
            if bsz not in costs:
                costs[bsz] = {}
            avg_rank = np.mean(ranks)
            with open(fn, 'r') as f:
                content = f.read()
            lats = [float(i) for i in re.findall(r"reqs. Latency: (\d+\.\d+)", content)]
            avg_lat = np.percentile(lats, 50)
            costs[bsz][avg_rank] = avg_lat
        # boundary of batch size 0, assume it is linear
        costs[0] = {}
        for avg_rank in costs[4]:
            costs[0][avg_rank] = costs[4][avg_rank] - (costs[8][avg_rank] - costs[4][avg_rank])
        return costs


class PunicaCostModel(BaseCostModel):
    def __init__(self, logdir) -> None:
        super().__init__(logdir)

    def parse_logs(self):
        costs = {}
        for fn in glob(self.logdir + "/*.log"):
            rank = int(re.findall(r"launch_server_(\d+)", fn)[0])
            with open(fn, 'r') as f:
                content = f.read()
            ret = map(lambda t: (int(t[0]), float(t[1])), re.findall(r"Decode (\d+) reqs. Lat: (\d+\.\d+)", content))
            tmp_lat = {}
            for bsz, lat in ret:
                if bsz not in tmp_lat:
                    tmp_lat[bsz] = []
                tmp_lat[bsz].append(lat)
            for bsz in tmp_lat:
                if (bsz % 4 != 0 or bsz == 0):
                    continue
                if bsz not in costs:
                    costs[bsz] = {}
                costs[bsz][rank] = np.percentile(tmp_lat[bsz], 50)
        costs[0] = {}
        for rank in costs[4]:
            costs[0][rank] = costs[4][rank] - (costs[8][rank] - costs[4][rank])
        return costs


def show_heatmap():
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)
    m = PunicaCostModel("./punica_cost_model")
    m.show_heatmap("Max. ranks in batch", ax1)
    plt.title("Punica Cost Model")

    ax2 = fig.add_subplot(122)
    m = SLoRACostModel("./slora_cost_model")
    m.show_heatmap("Avg. ranks in batch", ax2)
    plt.title("SLoRA Cost Model")
    plt.tight_layout()
    plt.show()


def show_3dplot():
    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(111, projection='3d')
    m = PunicaCostModel("./punica_cost_model")
    m.show_surface("Max. ranks in batch", ax1)
    # plt.title("Punica Cost Model")
    plt.tight_layout()
    plt.savefig("punicaCostModel.pdf", bbox_inches='tight', pad_inches=0.4)
    plt.clf()

    fig = plt.figure(figsize=(4, 4))
    ax2 = fig.add_subplot(111, projection='3d')
    m = SLoRACostModel("./slora_cost_model")
    m.show_surface("Avg. ranks in batch", ax2)
    # plt.title("SLoRA Cost Model")
    plt.tight_layout()
    plt.savefig("sloraCostModel.pdf", bbox_inches='tight', pad_inches=0.4)


if __name__ == "__main__":
    show_3dplot()
