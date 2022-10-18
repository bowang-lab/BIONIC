import json
from typing import List, Optional
from pathlib import Path
from functools import reduce

import typer
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from .common import magenta, Device

from torch import Tensor
from torch_geometric.transforms import ToSparseTensor
from torch_sparse import SparseTensor
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_geometric.data import Data


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
            label_names(Optional[List[Path]], optional): Paths to node labels if provided,
                otherwise `None`.
            delimiter (str, optional): Delimiter used in input network files. Defaults to " ".
            svd_dim (int, optional): Deprecated and safely ignored.
        """

        self.net_names = net_names
        self.label_names = label_names
        self.svd_dim = svd_dim
        self.graphs, self.labels = self._load(delimiter)
        self.node_sets, self.union = self._get_union()

    def _load(self, delimiter):

        # Import networks
        graphs = [pd.read_csv(name, delimiter=delimiter, header=None) for name in self.net_names]

        # Add weights of 1.0 if weights are missing
        for G in graphs:
            if G.shape[1] < 3:
                G[2] = pd.Series([1.0] * len(G))

        # Import label data if applicable
        labels = None
        if self.label_names is not None:
            labels = []
            for label_name in self.label_names:
                with label_name.open("r") as f:
                    labels.append(json.load(f))

        return graphs, labels

    def _get_union(self):

        node_sets = [np.union1d(G[0].values, G[1].values) for G in self.graphs]
        union = reduce(np.union1d, node_sets)
        return node_sets, union

    def _create_masks(self):

        masks = torch.FloatTensor([np.isin(self.union, nodes) for nodes in self.node_sets])
        masks = torch.t(masks)
        masks = masks.to(Device())
        return masks

    def _create_weights(self):

        # NOTE: In the future alternative network weighting schemes can be implemented here
        weights = torch.FloatTensor([1.0 for G in self.graphs])
        weights = weights.to(Device())
        return weights

    def _create_pyg_graphs(self):

        typer.echo("Preprocessing input networks...")

        # Uniquely map node names to integers
        mapper = {name: idx for idx, name in enumerate(self.union)}

        # Transform networks to PyG graphs
        pyg_graphs = []
        for G in self.graphs:

            # Map node names to integers given by `mapper`
            G[[0, 1]] = G[[0, 1]].applymap(lambda node: mapper[node])

            # Extract weights and edges from `G` and convert to tensors
            weights = torch.FloatTensor(G[2].values)
            edge_index = torch.LongTensor(G[[0, 1]].values.T)

            # Remove existing self loops and add self loops from `union` nodes,
            # updating `weights` accordingly
            edge_index, weights = remove_self_loops(edge_index, edge_attr=weights)
            edge_index, weights = to_undirected(edge_index, edge_attr=weights)
            union_idxs = list(range(len(self.union)))
            self_loops = torch.LongTensor([union_idxs, union_idxs])
            edge_index = torch.cat([edge_index, self_loops], dim=1)
            weights = torch.cat([weights, torch.Tensor([1.0] * len(self.union))])

            # Create PyG `Data` object
            pyg_graph = Data(edge_index=edge_index)
            pyg_graph.edge_weight = weights
            pyg_graph.num_nodes = len(self.union)
            pyg_graph = ToSparseTensor(remove_edge_index=True)(pyg_graph)
            pyg_graph = pyg_graph.to(Device())

            pyg_graphs.append(pyg_graph)

        return pyg_graphs

    def _create_labels(self):

        if self.labels is None:
            return (None, None, None)

        typer.echo("Preprocessing labels...")

        final_labels = []
        final_masks = []
        final_label_names = []

        for curr_labels in self.labels:

            # Remove nodes from labels not in `self.union`
            labels = {node: labels_ for node, labels_ in curr_labels.items() if node in self.union}

            # Create multi-hot encoding
            mlb = MultiLabelBinarizer()
            labels_mh = mlb.fit_transform(labels.values())
            labels_mh = pd.DataFrame(labels_mh, index=labels.keys())

            # Reindex `labels_mh` to include any missing nodes in `self.union` and create tensor
            labels_mh = labels_mh.reindex(self.union).fillna(0)
            labels_mh = torch.FloatTensor(labels_mh.values)
            labels_mh = labels_mh.to(Device())

            # Create mask tensor to indicate missing nodes
            labels_nodes = set(labels.keys())
            mask = torch.FloatTensor([node in labels_nodes for node in self.union])
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
            List[SparseTensor]: Processed networks in Pytorch Geometric `SparseTensor` format.
            List[Tensor]: Multi-label tensors where each element in the list is a different
                label set (i.e. standard).
            List[Tensor]: Label masks corresponding to `labels`. Masks out classification loss for
                nodes with no labels present in the label set.
            List[str]: Label set names.
        """

        masks: Tensor = self._create_masks()
        weights: Tensor = self._create_weights()
        pyg_graphs: List[SparseTensor] = self._create_pyg_graphs()

        labels: Optional[List[Tensor]]
        label_masks: Optional[List[Tensor]]
        label_names: Optional[List[list]]
        labels, label_masks, label_names = self._create_labels()

        typer.echo(f"Preprocessing finished: {magenta(f'{len(self.union)}')} total nodes.")

        return self.union, masks, weights, pyg_graphs, labels, label_masks, label_names
