from functools import reduce

from tqdm import tqdm
import torch
import numpy as np
import networkx as nx
from sklearn.decomposition import TruncatedSVD

from torch_geometric.utils import add_remaining_self_loops, is_undirected

"""
Preprocesses input networks
"""


class Preprocessor:
    def __init__(
        self,
        file_names,
        config_name,
        delimiter=" ",
        weight_type="equal",
        save_weights=False,
        svd_dim=0,
    ):

        self.names = file_names
        self.config_name = config_name
        self.weight_type = weight_type
        self.save_weights = save_weights
        self.svd_dim = svd_dim
        self.graphs = self._load(delimiter)
        self.union = self._get_union()

    def _load(self, delimiter, scale=True):
        """
        """

        graphs = [
            nx.read_weighted_edgelist(
                f"inputs/{name}", delimiter=delimiter
            ).to_undirected()
            for name in tqdm(self.names, desc="Loading networks")
        ]
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

        weights = torch.FloatTensor([1.0 for G in self.graphs])
        return weights

    def _create_features(self):
        """
        """

        if bool(self.svd_dim):

            all_edges = [e for net in self.graphs for e in list(net.edges())]
            G_max = nx.Graph()
            G_max.add_edges_from(all_edges)
            for e in G_max.edges():
                weights = []
                for net in self.graphs:
                    if e in net.edges():
                        if "weight" not in net.edges()[e]:
                            weights.append(1.0)
                        else:
                            weights.append(net.edges()[e]["weight"])
                max_weight = max(weights)
                G_max.edges()[e]["weight"] = max_weight

            svd = TruncatedSVD(n_components=self.svd_dim)
            feat = torch.tensor(
                svd.fit_transform(nx.normalized_laplacian_matrix(G_max))
            )

        else:
            # Create feature matrix (identity).
            idx = np.arange(len(self.union))
            i = torch.LongTensor([idx, idx])
            v = torch.FloatTensor(np.ones(len(self.union)))
            feat = torch.sparse.FloatTensor(i, v)

        return feat

    def _create_sparse_graphs(self):
        """
        """

        # Extend all graphs with nodes in `self.union` and add self-loops
        # to all nodes.
        for G in tqdm(self.graphs, desc="Extending networks"):
            G.add_nodes_from(self.union)
            G.add_weighted_edges_from([(n, n, 1.0) for n in G.nodes()])

        # Create sparse matrices from graphs.
        coo_mats = [
            np.abs(nx.to_scipy_sparse_matrix(G, nodelist=self.union, format="coo"))
            for G in tqdm(self.graphs, desc="Creating sparse COO matrices")
        ]

        # Map COOrdinate sparse matrices to tensors.
        coo_tensors = []
        for mat in tqdm(coo_mats, desc="Creating sparse tensors"):
            idx = torch.LongTensor(np.vstack((mat.row, mat.col)))
            val = torch.FloatTensor(mat.data)
            # Ensure each node has a self-loop.
            idx, val = add_remaining_self_loops(idx, val)
            assert is_undirected(idx)

            coo_tensor = torch.sparse.FloatTensor(idx, val).coalesce()
            coo_tensors.append(coo_tensor)

        return coo_tensors

    def process(self, cuda=False):
        """
        """

        masks = self._create_masks()
        weights = self._create_weights()
        coo_tensors = self._create_sparse_graphs()
        features = self._create_features()

        if cuda:
            device = torch.device("cuda")

            masks = masks.to(device)
            weights = weights.to(device)
            if isinstance(features, list):
                features = [feature.to(device) for feature in features]
            else:
                features = features.to(device)
            coo_tensors = [t.to(device) for t in coo_tensors]

        print(f"Preprocessing finished! {len(self.union)} total nodes.")

        return self.union, masks, weights, features, coo_tensors
