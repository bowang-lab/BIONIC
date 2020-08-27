from pathlib import Path
from functools import reduce

from tqdm import tqdm
import typer
import torch
import numpy as np
import networkx as nx
from sklearn.decomposition import TruncatedSVD

from .common import cyan, magenta

from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import from_networkx, add_remaining_self_loops, is_undirected

"""
Preprocesses input networks
"""


class Preprocessor:
    def __init__(self, file_names, delimiter=" ", save_weights=False, svd_dim=0):

        self.names = file_names
        self.save_weights = save_weights
        self.svd_dim = svd_dim
        self.graphs = self._load(delimiter)
        self.union = self._get_union()

    def _load(self, delimiter, scale=True):
        """
        """

        typer.echo("Preprocessing...")

        graphs = [
            nx.read_weighted_edgelist(name, delimiter=delimiter).to_undirected()
            for name in self.names
        ]

        # Add weights of 1.0 if weights are missing
        for G in graphs:
            if not nx.is_weighted(G):
                G.add_weighted_edges_from([(a, b, 1.0) for (a, b) in G.edges])

        return graphs

    def _get_union(self):
        """
        """

        union = reduce(np.union1d, [G.nodes() for G in self.graphs])
        return union

    def _create_masks(self):
        """
        """

        masks = torch.FloatTensor([np.isin(self.union, G.nodes()) for G in self.graphs])
        masks = torch.t(masks)
        return masks

    def _create_weights(self):
        """
        """

        # TODO in the future alternative network weighting schemes can be implemented here
        weights = torch.FloatTensor([1.0 for G in self.graphs])
        return weights

    def _create_features(self):
        """
        """

        if bool(self.svd_dim):

            all_edges = [e for G in self.graphs for e in list(G.edges())]
            G_max = nx.Graph()
            G_max.add_edges_from(all_edges)
            for e in G_max.edges():
                weights = []
                for G in self.graphs:
                    if e in G.edges():
                        if "weight" not in G.edges()[e]:
                            weights.append(1.0)
                        else:
                            weights.append(G.edges()[e]["weight"])
                max_weight = max(weights)
                G_max.edges()[e]["weight"] = max_weight

            svd = TruncatedSVD(n_components=self.svd_dim)
            feat = torch.tensor(svd.fit_transform(nx.normalized_laplacian_matrix(G_max)))

        else:
            # Create feature matrix (identity).
            idx = np.arange(len(self.union))
            i = torch.LongTensor([idx, idx])
            v = torch.FloatTensor(np.ones(len(self.union)))
            feat = torch.sparse.FloatTensor(i, v)

        return feat

    def _create_pyg_graphs(self):
        """
        """

        # Extend all graphs with nodes in `self.union` and add self-loops
        # to all nodes.
        new_graphs = [nx.Graph() for _ in self.graphs]
        for G, nG in zip(self.graphs, new_graphs):
            nG.add_nodes_from(self.union)
            nG.add_weighted_edges_from(
                [(s, t, weight["weight"]) for (s, t, weight) in G.edges(data=True)]
            )
            nG.add_weighted_edges_from([(n, n, 1.0) for n in nG.nodes()])
        self.graphs = new_graphs

        # print((np.array(new_graphs[0].nodes()) == np.array(new_graphs[1].nodes())).all())

        pyg_graphs = [from_networkx(G) for G in self.graphs]
        for G in pyg_graphs:
            G.edge_weight = G.weight
            del G.weight

        to_sparse_tensor = ToSparseTensor(remove_edge_index=False)
        for G in pyg_graphs:
            to_sparse_tensor(G)

        return pyg_graphs

    def process(self, cuda=False):
        """
        """

        masks = self._create_masks()
        weights = self._create_weights()
        features = self._create_features()
        pyg_graphs = self._create_pyg_graphs()

        if cuda:
            device = torch.device("cuda")
            masks = masks.to(device)
            weights = weights.to(device)
            if isinstance(features, list):
                features = [feature.to(device) for feature in features]
            else:
                features = features.to(device)
            pyg_graphs = [t.to(device) for t in pyg_graphs]

        typer.echo(f"Preprocessing finished: {magenta(f'{len(self.union)}')} total nodes.")

        return self.union, masks, weights, features, pyg_graphs
