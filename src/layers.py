import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import GATConv

from typing import Optional, Union
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size


def weighted_softmax(
    src: Tensor,
    index: Tensor,
    edge_weights: Tensor,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    if ptr is None:
        N = maybe_num_nodes(index, num_nodes)
        out = src - scatter(src, index, dim=0, dim_size=N, reduce="max")[index]
        out = edge_weights.unsqueeze(-1) * out.exp()  # multiply softmax by `edge_weights`
        out_sum = scatter(out, index, dim=0, dim_size=N, reduce="sum")[index]
        return out / (out_sum + 1e-16)
    else:
        raise NotImplementedError("Using `ptr` with `weighted_softmax` has not been implemented.")


class WGATConv(GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, edge_weights=None, **kwargs):
        """Extends the `GATConv` forward function to include edge weights.

        The `edge_weights` variable is made an instance attribute so it can
        easily be accessed in the `message` method and then passed to 
        `weighted_softmax`.
        """

        self.edge_weights = edge_weights
        return super().forward(*args, **kwargs)

    def message(
        self,
        x_j: Tensor,
        alpha_j: Tensor,
        alpha_i: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = weighted_softmax(alpha, index, self.edge_weights, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class Interp(nn.Module):
    def __init__(self, n_modalities, cuda=True):
        super(Interp, self).__init__()

        self.cuda = cuda
        self.scales = nn.Parameter(
            (
                torch.FloatTensor([1.0 for _ in range(n_modalities)]) / n_modalities
            ).reshape((1, -1))
        )

    def forward(self, mask, idxs, evaluate=False):

        scales = F.softmax(self.scales, dim=-1)
        scales = scales[:, idxs]

        if evaluate:
            random_mask = torch.IntTensor(mask.shape).random_(1, 2).float().cuda()
        else:
            random_mask = torch.IntTensor(mask.shape).random_(0, 2).float().cuda()

        if self.cuda:
            random_mask = random_mask.cuda()

        mask_sum = 1 / (1 + torch.sum(random_mask, dim=-1)) ** 20
        random_mask += mask_sum.reshape((-1, 1))
        random_mask += 1 / (torch.sum(mask, dim=-1) ** 20).reshape((-1, 1))
        random_mask = random_mask.int().float()
        random_mask = random_mask / (random_mask + 1e-10)

        mask = mask * random_mask
        mask = F.softmax(mask + ((1 - mask) * -1e10), dim=-1)

        return scales, mask