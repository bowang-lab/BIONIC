import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import GATConv

from typing import Optional, Union, Tuple
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size


def weighted_softmax(
    src: Tensor,
    index: Tensor,
    edge_weights: Tensor,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
    """Extends the PyTorch Geometric `softmax` functionality to
    incorporate edge weights.
    """

    if ptr is None:
        N = maybe_num_nodes(index, num_nodes)
        out = src - scatter(src, index, dim=0, dim_size=N, reduce="max")[index]
        out = (
            edge_weights.unsqueeze(-1) * out.exp()
        )  # multiply softmax by `edge_weights`
        out_sum = scatter(out, index, dim=0, dim_size=N, reduce="sum")[index]
        return out / (out_sum + 1e-16)
    else:
        raise NotImplementedError(
            "Using `ptr` with `weighted_softmax` has not been implemented."
        )


class WGATConv(GATConv):
    """Weighted version of the Graph Attention Network (`GATConv`).
    """

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
    """Stochastic summation layer.

    Performs random node feature dropout and feature scaling.
    NOTE: This is not very nice, but it is differentiable. Future work
    should involve rewriting this.
    """

    def __init__(self, n_modalities, cuda=True):
        super(Interp, self).__init__()

        self.cuda = cuda
        self.scales = nn.Parameter(
            (
                torch.FloatTensor([1.0 for _ in range(n_modalities)]) / n_modalities
            ).reshape((1, -1))
        )

    def forward(
        self, mask: Tensor, idxs: Tensor, evaluate: bool = False
    ) -> Tuple[Tensor, Tensor]:

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
