import torch
from torch.utils.data import Sampler
from torch_geometric.data import NeighborSampler

from typing import List, Optional, Tuple, NamedTuple


class StatefulSampler(Sampler):
    """A random sampler that ensures instances share the same permutation.

    Instances are passed to PyTorch Geometric `NeighborSampler`. Each instance
    returns an iterable of the class variable `perm`, ensuring each instance
    has the same random ordering. Calling `step` will create a new current
    random permutation. `step` should be called each epoch. NOTE: This is
    unlikely to work if multiple threads are used in `torch.DataLoader` due
    to GIL.
    """

    perm = None  # replaced with a new random permutation on `step` call

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(StatefulSampler.perm.tolist())

    def __len__(self):
        return len(self.data_source)

    @classmethod
    def step(cls, n_samples=None, random=True):
        if n_samples is None and cls.perm is None:
            raise Exception("`n_samples` must be passed on first call to `step`.")
        elif n_samples is None:
            cls.perm = torch.randperm(len(cls.perm))
        else:
            cls.perm = torch.randperm(n_samples)

        if not random:
            cls.perm = torch.arange(len(cls.perm))
        return cls.perm


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    weights: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(
            self.edge_index.to(*args, **kwargs),
            self.e_id.to(*args, **kwargs),
            self.weights.to(*args, **kwargs),
            self.size,
        )


class NeighborSamplerWithWeights(NeighborSampler):
    """Allows neighbor sampling with weighted networks."""

    def __init__(self, data, *args, **kwargs):
        data = data.to("cpu")
        row, col, edge_attr = data.adj_t.t().coo()
        edge_index = torch.stack([row, col], dim=0)
        self.weights = data.edge_weight
        super().__init__(edge_index, *args, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        adjs: List[Adj] = []

        n_id = batch
        for size in self.sizes:
            adj, n_id = self.adj.sample_adj(n_id, size, replace=False)
            if self.flow == "source_to_target":
                adj = adj.t()
            row, col, e_id = adj.coo()
            size = adj.sparse_sizes()
            edge_index = torch.stack([row, col], dim=0)
            weights = self.weights[e_id]

            adjs.append(Adj(edge_index, e_id, weights, size))

        if len(adjs) > 1:
            return batch_size, n_id, adjs[::-1]
        else:
            return batch_size, n_id, adjs[0]
