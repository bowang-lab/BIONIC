import warnings
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor

from ..utils.sampler import Adj
from ..utils.common import Device

from typing import Dict, List, Tuple, Optional

from .layers import WGATConv, Interp

from .classification_heads import *
from .attention import AttentiveIntegration


class BionicEncoder(nn.Module):
    def __init__(self, in_size: int, gat_shapes: Dict[str, int], alpha: float = 0.1):
        """BIONIC network encoder module.

        Args:
            in_size (int): Number of nodes in input networks.
            gat_shapes (Dict[str, int]): Graph attention layer hyperparameters.
            alpha (float, optional): LeakyReLU negative slope. Defaults to 0.1.

        Returns:
            Tensor: 2D tensor of node features. Each row is a node, each column is a feature.
        """
        super(BionicEncoder, self).__init__()
        self.in_size = in_size
        self.dimension: int = gat_shapes["dimension"]
        self.n_heads: int = gat_shapes["n_heads"]
        self.n_layers: int = gat_shapes["n_layers"]
        self.alpha = alpha

        self.pre_gat = nn.Linear(self.in_size, self.dimension * self.n_heads)
        self.gat = WGATConv(
            (self.dimension * self.n_heads,) * 2,
            self.dimension,
            heads=self.n_heads,
            dropout=0,
            negative_slope=self.alpha,
            add_self_loops=True,
        )

    def forward(self, data_flow, device=None):
        _, n_id, adjs = data_flow

        if device is None:
            device = Device()

        if isinstance(adjs, list):
            adjs = [adj.to(device) for adj in adjs]
        else:
            adjs = [adjs.to(device)]

        x_store_layer = []
        # Iterate over flow (pass data through GAT)
        for j, (edge_index, e_id, weights, size) in enumerate(adjs):

            # Initial `x` is feature matrix
            if j == 0:
                x = torch.t(self.pre_gat.weight)[n_id] + self.pre_gat.bias

            if j != 0:
                x_store_layer = [x_s[: size[1]] for x_s in x_store_layer]
                x_pre = x[: size[1]]
                x_store_layer.append(x_pre)

            x = self.gat((x, x[: size[1]]), edge_index, size=size, edge_weights=weights)

        x = sum(x_store_layer) + x  # Compute tensor with residuals

        return x


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
        head_type: int = 0,
        attention: bool = False,
    ):
        """The BIONIC model.

        Args:
            in_size (int): Number of nodes in input networks.
            gat_shapes (Dict[str, int]): Graph attention layer hyperparameters.
            emb_size (int): Dimension of learned node features.
            n_modalities (int): Number of input networks.
            alpha (float, optional): LeakyReLU negative slope. Defaults to 0.1.
            svd_dim (int, optional): Dimension of input node feature SVD approximation.
                Defaults to 0. No longer required and is safely ignored.
            shared_encoder (bool, optional): Whether to use the same encoder (pre-GAT
                + GAT) for all networks.
            n_classes (list of int, optional): Number of classes per supervised
                standard, if supervised standards are provided.
            head_type (int, optional): type of classification head.
                Defaults to 0 (corresponds to defalt classification head)
        """

        super(Bionic, self).__init__()

        self.in_size = in_size
        self.emb_size = emb_size
        self.alpha = alpha
        self.n_modalities = n_modalities
        self.svd_dim = svd_dim
        self.shared_encoder = shared_encoder
        self.n_classes = n_classes
        self.gat_shapes = gat_shapes
        self.head_type = head_type

        self.attention = attention

        self.dimension: int = self.gat_shapes["dimension"]
        self.n_heads: int = self.gat_shapes["n_heads"]

        self.encoders = []

        if bool(self.svd_dim):
            warnings.warn(
                "SVD approximation is no longer required for large networks and will be ignored."
            )

        # GAT
        for i in range(self.n_modalities):
            self.encoders.append(BionicEncoder(self.in_size, self.gat_shapes, self.alpha))

            if self.shared_encoder:
                break

        for i, enc_module in enumerate(self.encoders):
            self.add_module(f"Encoder_{i}", enc_module)

        self.integration_size = self.dimension * self.n_heads
        self.interp = Interp(self.n_modalities)

        # Embedding
        self.emb = nn.Linear(self.integration_size, self.emb_size)

        # Attentive integration
        if self.attention:
            self.attentive_integration_layer = AttentiveIntegration(
                embedding_dim=self.integration_size, n_head=1,
            )

        # Supervised classification head
        if self.n_classes:
            if self.head_type == 0:
                self.cls_heads = [
                    nn.Sequential(
                        nn.Linear(self.emb_size, self.emb_size),  # improves optimization
                        nn.Linear(self.emb_size, n_classes_),
                    )
                    for n_classes_ in self.n_classes
                ]

            elif self.head_type == 1:
                self.cls_heads = [
                    BasicLeakyReLUHead(self.emb_size, n_classes_, slope=0.01) for n_classes_ in self.n_classes
                ]

            elif self.head_type == 2:
                self.cls_heads = [
                    LeakyReLUDropoutHead(self.emb_size, n_classes_, slope=0.01, dropout_rate=0.25)
                    for n_classes_ in self.n_classes
                ]

            elif self.head_type == 3:
                self.cls_heads = [
                    MultilayerLeakyReLUDropoutHead(self.emb_size, n_classes_, slope=0.01, dropout_rate=0.25, n_layers=3)
                    for n_classes_ in self.n_classes
                ]

            elif self.head_type == 4:
                self.cls_heads = [
                    MultilayerLeakyReLUDropoutHead(self.emb_size, n_classes_, slope=0.01, dropout_rate=0.25, n_layers=4)
                    for n_classes_ in self.n_classes
                ]

            elif self.head_type == 5:
                self.cls_heads = [
                    MultilayerLeakyReLUDropoutHead(self.emb_size, n_classes_, slope=0.01, dropout_rate=0.25, n_layers=5)
                    for n_classes_ in self.n_classes
                ]

            elif self.head_type == 6:
                self.cls_heads = [
                    MultilayerSkipConnectionHead(self.emb_size, n_classes_, slope=0.01, dropout_rate=0.25, n_blocks=3)
                    for n_classes_ in self.n_classes
                ]

            elif self.head_type == 7:
                self.cls_heads = [
                    MultilayerSkipConnectionHead(self.emb_size, n_classes_, slope=0.01, dropout_rate=0.25, n_blocks=4)
                    for n_classes_ in self.n_classes
                ]

            elif self.head_type == 8:
                self.cls_heads = [
                    MultilayerSkipConnectionHead(self.emb_size, n_classes_, slope=0.01, dropout_rate=0.25, n_blocks=5)
                    for n_classes_ in self.n_classes
                ]

            for h, cls_head in enumerate(self.cls_heads):
                self.add_module(f"Classification_Head_{h}", cls_head)
        else:
            self.cls_heads = None

    def forward(
        self,
        data_flows: List[Tuple[int, Tensor, List[Adj]]],
        masks: Tensor,
        evaluate: bool = False,
        rand_net_idxs: Optional[np.ndarray] = None,
    ):
        """Forward pass logic.

        Args:
            data_flows (List[Tuple[int, Tensor, List[Adj]]]): Sampled bi-partite data flows.
                See PyTorch Geometric documentation for more details.
            masks (Tensor): 2D masks indicating which nodes (rows) are in which networks (columns)
            evaluate (bool, optional): Used to turn off random sampling in forward pass.
                Defaults to False.
            rand_net_idxs (np.ndarray, optional): Indices of networks if networks are being
                sampled. Defaults to None.

        Returns:
            Tensor: 2D tensor of final reconstruction to be used in loss function.
            Tensor: 2D tensor of integrated node features. Each row is a node, each column is a feature.
            List[Tensor]: Pre-integration network-specific node feature tensors. Not currently
                implemented.
            Tensor: Learned network scaling coefficients.
            Tensor or None: 2D tensor of label predictions if using supervision.
        """

        if rand_net_idxs is not None:
            idxs = rand_net_idxs
        else:
            idxs = list(range(self.n_modalities))

        if not self.attention:
            net_scales, interp_masks = self.interp(masks, idxs, evaluate)

        # Define encoder logic.
        out_pre_cat_layers = []  # Final layers before concatenation, not currently used

        batch_size = data_flows[0][0]
        embs_container = []
        x_store_modality = torch.zeros(
            (batch_size, self.integration_size), device=Device()
        )  # Tensor to store results from each modality.

        # Iterate over input networks
        for i, data_flow in enumerate(data_flows):
            if self.shared_encoder:
                net_idx = 0
            else:
                net_idx = idxs[i]

            x = self.encoders[net_idx](data_flow)

            if self.attention:
                embs_container.append(x)
            else:
                x = net_scales[:, i] * interp_masks[:, i].reshape((-1, 1)) * x
            x_store_modality += x

        # Embedding
        if self.attention:
            embs = torch.stack(embs_container, dim=1)

            emb = self.attentive_integration_layer(
                embeddings=embs, attention_mask=masks, return_att_scores=False
            )

            emb = emb.mean(dim=1)
            emb = self.emb(emb)
        else:
            emb = self.emb(x_store_modality)

        # Dot product (network reconstruction)
        dot = torch.mm(emb, torch.t(emb))

        # Classification (if standards are provided)
        if self.cls_heads:
            classes = [head(emb) for head in self.cls_heads]
        else:
            classes = None

        return dot, emb, out_pre_cat_layers, net_scales, classes


class BionicParallel(Bionic):
    def __init__(self, *args, **kwargs):
        """A GPU parallelized version of `Bionic`. See `Bionic` for arguments. 
        """
        super(BionicParallel, self).__init__(*args, **kwargs)

        self.cuda_count = torch.cuda.device_count()

        # split network indices into `cuda_count` chunks
        self.net_idx_splits = torch.tensor_split(torch.arange(len(self.encoders)), self.cuda_count)

        # create a dictionary mapping from network idx to cuda device idx
        self.net_to_cuda_mapper = {}

        # distribute encoders across GPUs
        encoders = []
        for cuda_idx, split in enumerate(self.net_idx_splits):
            split_encoders = [self.encoders[idx].to(f"cuda:{cuda_idx}") for idx in split]
            encoders += split_encoders
            for idx in split:
                self.net_to_cuda_mapper[idx.item()] = cuda_idx
        self.encoders = encoders

        for i, enc_module in enumerate(self.encoders):
            self.add_module(f"Encoder_{i}", enc_module)

        # put remaining tensors on first GPU
        self.emb = self.emb.to("cuda:0")
        self.interp = self.interp.to("cuda:0")

        if self.cls_heads is not None:
            self.cls_heads = [head.to("cuda:0") for head in self.cls_heads]

            for h, cls_head in enumerate(self.cls_heads):
                self.add_module(f"Classification_Head_{h}", cls_head)

    def forward(
        self,
        data_flows: List[Tuple[int, Tensor, List[Adj]]],
        masks: Tensor,
        evaluate: bool = False,
        rand_net_idxs: Optional[np.ndarray] = None,
    ):
        """See `Bionic` forward methods for argument details.
        """
        if rand_net_idxs is not None:
            raise NotImplementedError("Network sampling is not used with model parallelism.")

        idxs = list(range(self.n_modalities))
        net_scales, interp_masks = self.interp(masks, idxs, evaluate, device="cuda:0")

        # Define encoder logic.
        out_pre_cat_layers = []  # Final layers before concatenation, not currently used

        batch_size = data_flows[0][0]
        x_store_modality = torch.zeros(
            (batch_size, self.integration_size), device="cuda:0"
        )  # Tensor to store results from each modality.

        # Iterate over input networks
        for i, data_flow in enumerate(data_flows):
            if self.shared_encoder:
                net_idx = 0
            else:
                net_idx = idxs[i]
            device = f"cuda:{self.net_to_cuda_mapper[net_idx]}"

            x = self.encoders[net_idx](data_flow, device).to("cuda:0")
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
