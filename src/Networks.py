import networkx as nx
import numpy as np
import os
from src import config
from src import Standard
from matplotlib import pyplot as plt

class Coherence:
    def __init__(self, labels=config.channel_names):

        self.labels = labels
        # add each channel as fully-connected node
        self.G = nx.complete_graph(self.labels)
        self.edgelist = [edge for edge in self.G.edges()]
        for edge in self.edgelist:
            self.G.edges[edge]['n'] = 0
            self.G.edges[edge]['mean'] = 0
            self.G.edges[edge]['sum'] = 0
            self.G.edges[edge]['sum2'] = 0
            self.G.edges[edge]['std'] = 0

        # print("Edge list:", self.edgelist)

        self.coherences = []

        self.f = None

    def write(self, path):
        nx.write_gpickle(self.G, path)

    def load_network(self, path):
        self.G = nx.read_gpickle(path)

    def load_data(self, path):
        for fname in os.listdir(path):
            arr = np.genfromtxt(path+"/"+fname, delimiter=",")
            coh = Standard.CoherenceMap()
            coh.subject = fname.split('_')[0]
            coh.task = fname.split('_')[1]
            coh.map = arr[1:]
            coh.f = arr[0]
            if self.f is None:
                self.f = coh.f
            self.coherences.append(coh)

    def score(self, band="alpha"):
        self.band = band
        min, max =\
        config.frequency_bands[self.band][0],\
        config.frequency_bands[self.band][1]

        Cxy_range = np.where((self.f >= min) & (self.f <= max))[0]
        Cxy_min, Cxy_max = Cxy_range[0], Cxy_range[-1]

        for coh_map in self.coherences:
            for Cxy, edge in zip(coh_map.map, self.edgelist):
                coh_map.coherence_value = np.mean(Cxy[Cxy_min:Cxy_max])
                self.G.edges[edge]['n'] += 1
                self.G.edges[edge]['mean'] =\
                    (self.G.edges[edge]['mean']*(self.G.edges[edge]['n'] - 1)\
                    + coh_map.coherence_value) / self.G.edges[edge]['n']
                self.G.edges[edge]['sum'] += coh_map.coherence_value
                self.G.edges[edge]['sum2'] +=\
                    (coh_map.coherence_value - self.G.edges[edge]['mean'])**2
                self.G.edges[edge]['std'] = np.sqrt(
                    self.G.edges[edge]['sum2'] / (self.G.edges[edge]['n']))
        self.edgelist = [edge for edge in self.G.edges()]

    def draw(self, weighting=True, threshold=False):

        for node, pos in zip(
            [node for node in self.G.nodes()], config.networkx_positions):
            self.G.nodes[node]['pos'] = pos

        if (weighting is True) and (threshold is False):
            edges, weights = zip(*nx.get_edge_attributes(self.G,'mean').items())
        if (weighting is True) and (threshold is True):
            edges, weights = zip(*nx.get_edge_attributes(self.G, 'z-score').items())
        else:
            weights = [1 for edge in self.G.edges()]
            edges = [edge for edge in self.G.edges()]

        pos = nx.get_node_attributes(self.G, 'pos')

        cmap=plt.cm.RdBu
        vmin = min(weights)
        vmax = max(weights)

        nx.draw(
            self.G,
            pos,
            with_labels=True,
            node_color='b',
            edgelist=edges,
            edge_color=weights if weighting is True else None,
            width=2.0,
            edge_cmap=cmap,
            vmin=vmin,
            vmax=max)

        if weighting is True:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
                vmin = vmin, vmax=vmax))
            sm._A = []
            plt.colorbar(sm)

        plt.show()

    def threshold(self, reference=None, z_score=2):
        if reference is None:
            reference = self.G
        means = [val[1] for val in nx.get_edge_attributes(reference,'mean').items()]
        stds = [val[1] for val in nx.get_edge_attributes(reference,'std').items()]

        for i, edge in enumerate([edge for edge in self.G.edges()]):
            z = (self.G.edges[edge]['mean'] - means[i]) / stds[i]
            print(z)
            if np.abs(z) < z_score:
                self.G.remove_edge(edge[0], edge[1])
                print("Removing:", edge)
            else:
                self.G.edges[edge]['z-score'] = z
