import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class Interp(nn.Module):
    def __init__(self, n_modalities, cuda=True):
        super(Interp, self).__init__()

        self.cuda = cuda
        self.weights = nn.Parameter(
            (
                torch.FloatTensor([1.0 for _ in range(n_modalities)]) / n_modalities
            ).reshape((1, -1))
        )

    def forward(self, mask, idxs, evaluate=False):

        weights = F.softmax(self.weights, dim=-1)
        weights = weights[:, idxs]

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

        return weights, mask


class Bionic(nn.Module):
    def __init__(
        self,
        in_size,
        gat_shapes,
        emb_size,
        n_modalities,
        alpha=0.1,
        dropout=0.0,
        svd_dim=0,
    ):
        """
        The BIONIC model.

            in_size: int, the size of the input networks (assumed to be the same
                size, missing observations should extend the input networks with
                disconnected nodes).

            gat_shapes: list of list of tuple, each tuple contains the layer
                size and number of attention heads, each list contains these
                shapes for each modality.
                i.e. [[(128, 4), (128, 4)], [(64, 2), (64, 2), (64, 1)], ...]

            emb_size: int, dimension of the shared embedding (bottleneck)

            alpha: float, LeakyReLU negative component slope.
        """

        super(Bionic, self).__init__()

        self.in_size = in_size
        self.emb_size = emb_size
        self.alpha = alpha
        self.dropout = dropout
        self.n_modalities = n_modalities
        self.svd_dim = svd_dim
        self.adj_dense_layers = []
        self.pre_gat_layers = []
        self.gat_layers = []
        self.post_gat_layers = []  # Dense transform after each GAT encoder.

        self.dimension = gat_shapes["dimension"]
        self.n_heads = gat_shapes["n_heads"]
        self.n_layers = gat_shapes["n_layers"]

        # GAT
        for i in range(self.n_modalities):
            if bool(self.svd_dim):
                self.pre_gat_layers.append(
                    nn.Linear(self.svd_dim, self.dimension * self.n_heads)
                )
            else:
                self.pre_gat_layers.append(
                    nn.Linear(in_size, self.dimension * self.n_heads)
                )
            self.gat_layers.append(
                GATConv(
                    self.dimension * self.n_heads,
                    self.dimension,
                    heads=self.n_heads,
                    dropout=self.dropout,
                )
            )

        for g, gat_layer in enumerate(self.gat_layers):
            self.add_module("GAT_{}".format(g), gat_layer)

        for d, dense_layer in enumerate(self.pre_gat_layers):
            self.add_module("Pre_GAT_Dense_{}".format(d), dense_layer)

        self.integration_size = self.dimension * self.n_heads
        self.interp = Interp(self.n_modalities)

        # Embedding.
        self.emb = nn.Linear(self.integration_size, emb_size)

    def forward(
        self, datasets, data_flows, features, masks, evaluate=False, rand_net_idxs=None
    ):
        """
        Forward pass logic.
        """

        if rand_net_idxs is not None:
            idxs = rand_net_idxs
        else:
            idxs = list(range(self.n_modalities))

        weights, interp_masks = self.interp(masks, idxs, evaluate)

        # Define encoder logic.
        pre_cat_layers = []
        out_pre_cat_layers = (
            []
        )  # Final layers before concatenation (no skip connections)

        batch_size = data_flows[0].blocks[0].size[1]
        x_store_modality = torch.zeros(
            (batch_size, self.integration_size)
        ).cuda()  # Tensor to store results from each modality.

        for i, (data_flow, dataset) in enumerate(zip(data_flows, datasets)):
            idx = idxs[i]

            data_flow = data_flow.to("cuda")

            x_store_layer = []
            for j, data in enumerate(data_flow):

                # Get edge weights.
                vals = dataset.edge_attr[data.e_id]

                # Initial `x` is feature matrix.
                if j == 0:
                    if bool(self.svd_dim):
                        x = features[data.n_id].float()

                    else:
                        x = torch.zeros(len(data.n_id), self.in_size).cuda()
                        x[np.arange(len(data.n_id)), data.n_id] = 1.0

                    x = self.pre_gat_layers[idx](x)

                if j != 0:
                    x_store_layer = [x_s[data.res_n_id] for x_s in x_store_layer]
                    x_pre = x[data.res_n_id]
                    x_store_layer.append(x_pre)

                x = self.gat_layers[idx]((x, None), data.edge_index, vals, data.size)

            x = sum(x_store_layer) + x  # Compute tensor with residuals
            x = weights[:, i] * interp_masks[:, i].reshape((-1, 1)) * x
            x_store_modality += x

        # Embedding
        emb = self.emb(x_store_modality)

        # Dot product.
        dot = torch.mm(emb, torch.t(emb))

        return dot, emb, out_pre_cat_layers, weights
