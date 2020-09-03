import torch
from torch import Tensor

from ..utils.common import Device


def masked_scaled_mse(
    output: Tensor,
    target: Tensor,
    scale: Tensor,
    node_ids: Tensor,
    mask: Tensor,
    cuda: bool = False,
) -> Tensor:
    """Masked and scaled MSE loss.
    """

    # Subset `target` to current batch and make dense
    target = target.to(Device())
    target = target.adj_t[node_ids, node_ids].to_dense()

    loss = scale * torch.mean(
        mask.reshape((-1, 1)) * torch.mean((output - target) ** 2, dim=-1) * mask
    )
    return loss
