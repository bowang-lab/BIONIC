import json
from pathlib import Path
from functools import reduce
import numpy as np
import networkx as nx

import torch
from torch_geometric.transforms import ToSparseTensor
from torch_sparse import SparseTensor
from torch_geometric.utils import from_networkx, remove_self_loops
from bionic.utils.preprocessor import Preprocessor

import numpy as np

config_path = Path(__file__).resolve().parents[1] / "config" / "mock_config.json"
with config_path.open() as f:
    mock_config = json.load(f)

# Update `mock_config` "names" parameter to take this file's path into account
mock_config["names"] = [
    Path(__file__).resolve().parents[1] / "inputs" / Path(name).name
    for name in mock_config["net_names"]
]


class TestPreprocessor:
    def test_final_networks_have_same_nodes_and_node_ordering_and_correct_weights(self):

        # Get PyG graphs from `Preprocessor`
        p = Preprocessor(mock_config["net_names"])
        _, _, _, output_pyg_graphs, _, _, _ = p.process()

        # Independently process networks
        graphs = [nx.read_weighted_edgelist(name) for name in mock_config["net_names"]]

        union = reduce(np.union1d, [G.nodes() for G in graphs])

        # Add weights of 1.0 if weights are missing
        for G in graphs:
            if not nx.is_weighted(G):
                G.add_weighted_edges_from([(a, b, 1.0) for (a, b) in G.edges])

        # Create PyG graphs
        new_graphs = []
        for G in graphs:
            nG = nx.Graph()
            nG.add_nodes_from(union)
            nG.add_weighted_edges_from(
                [(s, t, weight["weight"]) for (s, t, weight) in G.edges(data=True)]
            )
            nG.remove_edges_from(nx.selfloop_edges(nG))  # remove existing selfloops first
            nG.add_weighted_edges_from([(n, n, 1.0) for n in nG.nodes()])
            new_graphs.append(nG)
        graphs = new_graphs

        target_pyg_graphs = [from_networkx(G) for G in graphs]
        for G in target_pyg_graphs:
            G.edge_weight = G.weight
            del G.weight

        to_sparse_tensor = ToSparseTensor(remove_edge_index=False)
        for G in target_pyg_graphs:
            to_sparse_tensor(G)

        def get_edge_index_from_sparse_tensor(tensor_):
            """Returns the correctly ordered edge indices and weights from
            PyG SparseTensor, `tensor_`.
            """
            row, col, weights = tensor_.adj_t.t().coo()
            edge_index = torch.stack([row, col], dim=0)
            return edge_index, weights

        # Compare output graphs to target graphs
        for output_G, target_G in zip(output_pyg_graphs, target_pyg_graphs):
            output_G, target_G = output_G.to("cpu"), target_G.to("cpu")

            output_ei, output_weights = get_edge_index_from_sparse_tensor(output_G)
            target_ei, target_weights = get_edge_index_from_sparse_tensor(target_G)

            assert (output_ei == target_ei).all()
            assert (output_weights == target_weights).all()
