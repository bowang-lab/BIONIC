import torch
from torch import Tensor

from bionic.train import Trainer


def masked_scaled_mse(
    output: Tensor, target: Tensor, scale: Tensor, node_ids: Tensor, mask: Tensor
) -> Tensor:
    """Masked and scaled MSE loss.
    """

    # Subset `target` to current batch and make dense
    if Trainer.cuda:
        target = target.cuda()
    target = target.adj_t[node_ids, node_ids].to_dense()

    loss = scale * torch.mean(
        mask.reshape((-1, 1)) * torch.mean((output - target) ** 2, dim=-1) * mask
    )
    return loss
