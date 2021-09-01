import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor

from ..utils.sampler import Adj
from ..utils.common import Device

from typing import Dict, List, Tuple, Optional

from .layers import WGATConv, Interp


class Bionic(nn.Module):
    def __init__(
        self,
        in_size: int,
        gat_shapes: Dict[str, int],
        emb_size: int,
        n_modalities: int,
        alpha: float = 0.1,
        svd_dim: int = 0,
        shared_encoder: bool = False,
        n_classes: Optional[List[int]] = None,
    ):
        """The BIONIC model.

        Args:
            in_size (int): Number of nodes in input networks.
            gat_shapes (Dict[str, int]): Graph attention layer hyperparameters.
            emb_size (int): Dimension of learned node features.
            n_modalities (int): Number of input networks.
            alpha (float, optional): LeakyReLU negative slope. Defaults to 0.1.
            svd_dim (int, optional): Dimension of input node feature SVD approximation.
                Defaults to 0.
            shared_encoder (bool, optional): Whether to use the same encoder (pre-GAT
                + GAT) for all networks.
            n_classes (list of int, optional): Number of classes per supervised
                standard, if supervised standards are provided.
        """

        super(Bionic, self).__init__()

        self.in_size = in_size
        self.emb_size = emb_size
        self.alpha = alpha
        self.n_modalities = n_modalities
        self.svd_dim = svd_dim
        self.shared_encoder = shared_encoder
        self.n_classes = n_classes

        self.adj_dense_layers = []
        self.pre_gat_layers = []
        self.gat_layers = []
        self.post_gat_layers = []  # Dense transform after each GAT encoder.

        self.dimension: int = gat_shapes["dimension"]
        self.n_heads: int = gat_shapes["n_heads"]
        self.n_layers: int = gat_shapes["n_layers"]

        # GAT
        for i in range(self.n_modalities):
            if bool(self.svd_dim):
                self.pre_gat_layers.append(nn.Linear(self.svd_dim, self.dimension * self.n_heads))
            else:
                self.pre_gat_layers.append(nn.Linear(in_size, self.dimension * self.n_heads))
            self.gat_layers.append(
                WGATConv(
                    (self.dimension * self.n_heads, self.dimension * self.n_heads),
                    self.dimension,
                    heads=self.n_heads,
                    dropout=0,
                    negative_slope=self.alpha,
                    add_self_loops=True,
                )
            )

            if self.shared_encoder:
                break

        for g, gat_layer in enumerate(self.gat_layers):
            self.add_module(f"GAT_{g}", gat_layer)

        for d, dense_layer in enumerate(self.pre_gat_layers):
            self.add_module(f"Pre_GAT_Dense_{d}", dense_layer)

        self.integration_size = self.dimension * self.n_heads
        self.interp = Interp(self.n_modalities)

        # Embedding
        self.emb = nn.Linear(self.integration_size, self.emb_size)

        # Supervised classification head
        if self.n_classes:
            self.cls_heads = [nn.Linear(self.emb_size, n_classes_) for n_classes_ in self.n_classes]
            for h, cls_head in enumerate(self.cls_heads):
                self.add_module(f"Classification_Head_{h}", cls_head)
        else:
            self.cls_heads = None

    def forward(
        self,
        datasets: List[SparseTensor],
        data_flows: List[Tuple[int, Tensor, List[Adj]]],
        features: Tensor,
        masks: Tensor,
        evaluate: bool = False,
        rand_net_idxs: Optional[np.ndarray] = None,
    ):
        """Forward pass logic.

        Args:
            datasets (List[SparseTensor]): Input networks.
            data_flows (List[Tuple[int, Tensor, List[Adj]]]): Sampled bi-partite data flows.
                See PyTorch Geometric documentation for more details.
            features (Tensor): 2D node features tensor.
            masks (Tensor): 2D masks indicating which nodes (rows) are in which networks (columns)
            evaluate (bool, optional): Used to turn off random sampling in forward pass.
                Defaults to False.
            rand_net_idxs (np.ndarray, optional): Indices of networks if networks are being
                sampled. Defaults to None.

        Returns:
            Tensor: 2D tensor of final reconstruction to be used in loss function.
            Tensor: 2D tensor of node features. Each row is a node, each column is a feature.
            List[Tensor]: Pre-integration network-specific node feature tensors. Not currently
                implemented.
            Tensor: Learned network scaling coefficients.
        """

        if rand_net_idxs is not None:
            idxs = rand_net_idxs
        else:
            idxs = list(range(self.n_modalities))

        net_scales, interp_masks = self.interp(masks, idxs, evaluate)

        # Define encoder logic.
        out_pre_cat_layers = []  # Final layers before concatenation, not currently used

        batch_size = data_flows[0][0]
        x_store_modality = torch.zeros(
            (batch_size, self.integration_size), device=Device()
        )  # Tensor to store results from each modality.

        # Iterate over input networks
        for i, data_flow in enumerate(data_flows):
            if self.shared_encoder:
                net_idx = 0
            else:
                net_idx = idxs[i]

            _, n_id, adjs = data_flow
            if isinstance(adjs, list):
                adjs = [adj.to(Device()) for adj in adjs]
            else:
                adjs = [adjs.to(Device())]

            x_store_layer = []
            # Iterate over flow (pass data through GAT)
            for j, (edge_index, e_id, weights, size) in enumerate(adjs):

                # Initial `x` is feature matrix
                if j == 0:
                    if bool(self.svd_dim):
                        x = features[n_id].float()

                    else:
                        x = torch.zeros(len(n_id), self.in_size, device=Device())
                        x[np.arange(len(n_id)), n_id] = 1.0

                    x = self.pre_gat_layers[net_idx](x)

                if j != 0:
                    x_store_layer = [x_s[: size[1]] for x_s in x_store_layer]
                    x_pre = x[: size[1]]
                    x_store_layer.append(x_pre)

                x = self.gat_layers[net_idx](
                    (x, x[: size[1]]), edge_index, size=size, edge_weights=weights
                )

            x = sum(x_store_layer) + x  # Compute tensor with residuals
            x = net_scales[:, i] * interp_masks[:, i].reshape((-1, 1)) * x
            x_store_modality += x

        # Embedding
        emb = self.emb(x_store_modality)

        # Dot product (network reconstruction)
        dot = torch.mm(emb, torch.t(emb))

        # Classification (if standards are provided)
        if self.cls_heads:
            classes = [head(emb) for head in self.cls_heads]
        else:
            classes = None

        return dot, emb, out_pre_cat_layers, net_scales, classes
