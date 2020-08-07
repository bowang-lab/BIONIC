import networkx as nx

import torch
from torch.utils.data import Sampler
from torch_sparse import SparseTensor
from torch_geometric.data import Data, NeighborSampler
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import ToSparseTensor

from utils.preprocessor import Preprocessor

class StatefulSampler(Sampler):
    """A random sampler that ensures instances share the same permutation.

    Instances are passed to PyTorch Geometric `NeighborSampler`. Each instance
    returns an iterable of the class variable `perm`, ensuring each instance
    has the same random ordering. Calling `step` will create a new current
    random permutation. `step` should be called each epoch.
    """

    perm = None  # replaced with a new random permutation on `step` call

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(StatefulSampler.perm.tolist())

    def __len__(self):
        return len(self.data_source)
    
    @classmethod
    def step(cls, n_samples=None):
        if n_samples is None and cls.perm is None:
            raise Exception("`n_samples` must be passed on first call to `step`.")
        elif n_samples is None:
            cls.perm = torch.randperm(len(cls.perm))
        else:
            cls.perm = torch.randperm(n_samples)
        
# pp = Preprocessor(["Gavin.txt", "Krogan.txt"])
# union, _, _, feat, tensors = pp.process()

# row, col, edge_attr = tensors[0].adj_t.t().coo()
# edge_index1 = torch.stack([row, col])

# row, col, edge_attr = tensors[0].adj_t.t().coo()
# edge_index2 = torch.stack([row, col])

# n_nodes = len(union)
# StatefulSampler.step(n_nodes)
# smp1 = StatefulSampler(torch.arange(n_nodes))
# smp2 = StatefulSampler(torch.arange(n_nodes))

# x = torch.arange(n_nodes)

# ns1 = NeighborSampler(edge_index1, sizes=[5, 2], shuffle=False, batch_size=1024, sampler=smp1)
# ns2 = NeighborSampler(edge_index2, sizes=[5, 2], shuffle=False, batch_size=1024, sampler=smp2)
# for i, (batch_size, n_id, adjs) in enumerate(ns1):
#     x = torch.arange(n_nodes)
#     x = x[n_id]
#     for j, (edge_index, e_id, size) in enumerate(adjs):
#         x = x[:size[1]]
#         if j == 1:
#             print((x == StatefulSampler.perm[:1024]).all())


# for i, (batch_size, n_id, adjs) in enumerate(ns2):
#     x = torch.arange(n_nodes)
#     x = x[n_id]
#     for j, (edge_index, e_id, size) in enumerate(adjs):
#         x = x[:size[1]]
#         if j == 1:
#             print((x == StatefulSampler.perm[:1024]).all())
