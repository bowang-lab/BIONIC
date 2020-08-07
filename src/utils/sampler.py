import torch
from torch.utils.data import Sampler

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
    def step(cls, n_samples=None):
        if n_samples is None and cls.perm is None:
            raise Exception("`n_samples` must be passed on first call to `step`.")
        elif n_samples is None:
            cls.perm = torch.randperm(len(cls.perm))
        else:
            cls.perm = torch.randperm(n_samples)