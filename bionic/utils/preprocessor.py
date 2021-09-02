import json
from typing import List, Optional
from pathlib import Path
from functools import reduce

import typer
import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer

from .common import magenta, Device

from torch import Tensor
from torch_geometric.transforms import ToSparseTensor
from torch_sparse import SparseTensor
from torch_geometric.utils import from_networkx, is_undirected


class Preprocessor:
    def __init__(
        self,
        net_names: List[Path],
        label_names: Optional[List[Path]] = None,
        delimiter: str = " ",
        svd_dim: int = 0,
    ):
        """Preprocesses input networks.

        Args:
            net_names (List[Path]): Paths to input networks.
            label_names(Optional[List[Path]], optional): Paths to gene/protein labels if provided,
                otherwise `None`.
            delimiter (str, optional): Delimiter used in input network files. Defaults to " ".
            svd_dim (int, optional): Dimension of input node feature SVD approximation.
                0 implies no approximation. Defaults to 0.
        """

        self.net_names = net_names
        self.label_names = label_names
        self.svd_dim = svd_dim
        self.graphs, self.labels = self._load(delimiter)
        self.union = self._get_union()

    def _load(self, delimiter):

        # Import networks
        graphs = [
            nx.read_weighted_edgelist(name, delimiter=delimiter).to_undirected()
            for name in self.net_names
        ]

        # Add weights of 1.0 if weights are missing
        for G in graphs:
            if not nx.is_weighted(G):
                G.add_weighted_edges_from([(a, b, 1.0) for (a, b) in G.edges])

        # Import label data if applicable
        labels = None
        if self.label_names is not None:
            labels = []
            for label_name in self.label_names:
                with label_name.open("r") as f:
                    labels.append(json.load(f))

        return graphs, labels

    def _get_union(self):

        union = reduce(np.union1d, [G.nodes() for G in self.graphs])
        return union

    def _create_masks(self):

        masks = torch.FloatTensor([np.isin(self.union, G.nodes()) for G in self.graphs])
        masks = torch.t(masks)
        masks = masks.to(Device())
        return masks

    def _create_weights(self):

        # NOTE: In the future alternative network weighting schemes can be implemented here
        weights = torch.FloatTensor([1.0 for G in self.graphs])
        weights = weights.to(Device())
        return weights

    def _create_features(self):

        if bool(self.svd_dim):

            all_edges = [e for G in self.graphs for e in list(G.edges())]
            G_max = nx.Graph()
            G_max.add_nodes_from(self.union)  # Ensures nodes are correctly ordered
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

        if isinstance(feat, list):
            feat = [feature.to(Device()) for feature in feat]
        else:
            feat = feat.to(Device())

        return feat

    def _create_pyg_graphs(self):

        typer.echo("Preprocessing input networks...")

        # Extend all graphs with nodes in `self.union` and add self-loops
        # to all nodes.
        new_graphs = [nx.Graph() for _ in self.graphs]
        for G, nG in zip(self.graphs, new_graphs):
            nG.add_nodes_from(self.union)
            nG.add_weighted_edges_from(
                [(s, t, weight["weight"]) for (s, t, weight) in G.edges(data=True)]
            )
            nG.remove_edges_from(nx.selfloop_edges(nG))  # remove existing selfloops first
            nG.add_weighted_edges_from([(n, n, 1.0) for n in nG.nodes()])
        self.graphs = new_graphs

        pyg_graphs = [from_networkx(G) for G in self.graphs]
        for G in pyg_graphs:
            G.edge_weight = G.weight
            del G.weight

        to_sparse_tensor = ToSparseTensor(remove_edge_index=False)
        for G in pyg_graphs:
            to_sparse_tensor(G)

        pyg_graphs = [t.to(Device()) for t in pyg_graphs]

        return pyg_graphs

    def _create_labels(self):

        if self.labels is None:
            return (None, None, None)

        typer.echo("Preprocessing labels...")

        final_labels = []
        final_masks = []
        final_label_names = []

        for curr_labels in self.labels:

            # Remove genes from labels not in `self.union`
            labels = {gene: labels_ for gene, labels_ in curr_labels.items() if gene in self.union}

            # Create multi-hot encoding
            mlb = MultiLabelBinarizer()
            labels_mh = mlb.fit_transform(labels.values())
            labels_mh = pd.DataFrame(labels_mh, index=labels.keys())

            # Reindex `labels_mh` to include any missing genes in `self.union` and create tensor
            labels_mh = labels_mh.reindex(self.union).fillna(0)
            labels_mh = torch.FloatTensor(labels_mh.values)
            labels_mh = labels_mh.to(Device())

            # Create mask tensor to indicate missing genes
            labels_genes = set(labels.keys())
            mask = torch.FloatTensor([gene in labels_genes for gene in self.union])
            mask = torch.t(mask)
            mask = mask.to(Device())

            final_labels.append(labels_mh)
            final_masks.append(mask)
            final_label_names.append(mlb.classes_)

        return final_labels, final_masks, final_label_names

    def process(self):
        """Calls relevant preprocessing functions.

        Returns:
            np.ndarray: Array of all nodes present across input networks (union of nodes).
            Tensor: 2D binary mask tensor indicating nodes (rows) present in each network (columns).
            Tensor: 1D network weight tensor.
            Tensor: 2D node feature tensor. One-hot encoding or SVD union network approximation.
            List[SparseTensor]: Processed networks in Pytorch Geometric `SparseTensor` format.
        """

        masks: Tensor = self._create_masks()
        weights: Tensor = self._create_weights()
        features: Tensor = self._create_features()
        pyg_graphs: List[SparseTensor] = self._create_pyg_graphs()

        labels: Optional[List[Tensor]]
        label_masks: Optional[List[Tensor]]
        label_names: Optional[List[list]]
        labels, label_masks, label_names = self._create_labels()

        typer.echo(f"Preprocessing finished: {magenta(f'{len(self.union)}')} total nodes.")

        return self.union, masks, weights, features, pyg_graphs, labels, label_masks, label_names
