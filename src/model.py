import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch_geometric.utils import subgraph
from torch_geometric.nn import GATConv, SAGEConv, GCNConv, RGCNConv

import random

class Interp(nn.Module):

    def __init__(self,
                 n_modalities,
                 cuda=True):
        super(Interp, self).__init__()

        self.cuda = cuda
        self.weights = nn.Parameter((torch.FloatTensor([1.0 for _ in range(n_modalities)]) / n_modalities).reshape((1, -1)))

    def forward(self, mask, idxs, evaluate=False):

        weights = F.softmax(self.weights, dim=-1)
        weights = weights[:, idxs]

        if evaluate:
            random_mask = torch.IntTensor(mask.shape).random_(1, 2).float().cuda()
        else:
            random_mask = torch.IntTensor(mask.shape).random_(0, 2).float().cuda()

        if self.cuda:
            random_mask = random_mask.cuda()

        mask_sum = 1 / (1 + torch.sum(random_mask, dim=-1))**20
        random_mask += mask_sum.reshape((-1, 1))
        random_mask += (1 / (torch.sum(mask, dim=-1)**20).reshape((-1, 1)))
        random_mask = random_mask.int().float()
        random_mask = random_mask / (random_mask + 1e-10)

        mask = mask * random_mask
        mask = F.softmax(mask + ((1 - mask) * -1e10), dim=-1)

        return weights, mask


class DeepSNF(nn.Module):
    def __init__(self, 
                 in_size, 
                 gat_shapes, 
                 emb_size,
                 n_modalities,
                 alpha=0.1, 
                 dropout=0.0,
                 use_SVD=False,
                 SVD_dim=2048):
        """
        The DeepSNF model.

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
        
        super(DeepSNF, self).__init__()

        self.in_size = in_size
        self.emb_size = emb_size
        self.alpha = alpha
        self.dropout = dropout
        # self.n_modalities = len(gat_shapes)
        self.n_modalities = n_modalities
        self.use_SVD = use_SVD
        self.SVD_dim = SVD_dim
        # self.gat_layers = [[] for _ in range(self.n_modalities)]
        self.adj_dense_layers = []
        self.pre_gat_layers = []
        self.gat_layers = []
        self.post_gat_layers = []  # Dense transform after each GAT encoder.

        self.dimension = gat_shapes['dimension']
        self.n_heads = gat_shapes['n_heads']
        self.n_layers = gat_shapes['n_layers']

        # self.identity_encoder = nn.Linear(in_size, self.dimension * self.n_heads)

        # GAT
        for i in range(self.n_modalities):
            if self.use_SVD:
                self.pre_gat_layers.append(nn.Linear(self.SVD_dim, self.dimension * self.n_heads))
            else:
                self.pre_gat_layers.append(nn.Linear(in_size, self.dimension * self.n_heads))
            # self.pre_gat_layers.append(nn.Linear(2048, self.dimension * self.n_heads))
            # self.pre_gat_layers.append(nn.Linear(1, self.dimension * self.n_heads))
            # self.pre_gat_layers.append(GATConv(1,
            #                                self.dimension,
            #                                heads=self.n_heads,
            #                                dropout=self.dropout))
            self.gat_layers.append(GATConv(self.dimension * self.n_heads,
                    self.dimension,
                    heads=self.n_heads,
                    dropout=self.dropout))

        for g, gat_layer in enumerate(self.gat_layers):
            self.add_module('GAT_{}'.format(g), gat_layer)

        for d, dense_layer in enumerate(self.pre_gat_layers):
            self.add_module('Pre_GAT_Dense_{}'.format(d), dense_layer)


        # # RGCN
        # self.identity_emb = nn.Linear(in_size, self.dimension)
        # self.rgcn = RGCNConv(self.dimension, self.dimension, self.n_modalities, 1)



        self.cat_size = self.dimension * self.n_heads
        # self.cat_size = self.dimension
        self.interp = Interp(self.n_modalities)

        # Embedding.
        self.emb = nn.Linear(self.cat_size, emb_size)


    def forward(self, datasets, data_flows, features, masks, evaluate=False, rand_net_idxs=None):
        """
        Forward pass logic.
        """ 

        # print(rand_net_idxs)

        if rand_net_idxs is not None:
            idxs = rand_net_idxs
            # pre_gat_layers = [self.pre_gat_layers[i] for i in rand_net_idxs]
            # gat_layers = [self.gat_layers[i] for i in rand_net_idxs]
        else:
            idxs = list(range(self.n_modalities))
            # pre_gat_layers = self.pre_gat_layers
            # gat_layers = self.gat_layers
        
        weights, interp_masks = self.interp(masks, idxs, evaluate)

        # Define encoder logic.
        pre_cat_layers = []
        out_pre_cat_layers = []  # Final layers before concatenation (no skip connections)

        batch_size = data_flows[0].blocks[0].size[1]
        x_store_modality = torch.zeros((batch_size, self.cat_size)).cuda()  # Tensor to store results from each modality.
        # x_store = []
        # for i, (data_flow, dataset, layer, pre_gat_layer) in enumerate(zip(data_flows, datasets, gat_layers, pre_gat_layers)):
        for i, (data_flow, dataset) in enumerate(zip(data_flows, datasets)):
            idx = idxs[i]

            data_flow = data_flow.to('cuda')

            x_store_layer = []
            for j, data in enumerate(data_flow):

                # Get edge weights.
                vals = dataset.edge_attr[data.e_id]

                # Initial `x` is feature matrix.
                if j == 0:
                    # if feature is not None:
                    if self.use_SVD:
                        x = features[data.n_id].float()

                    else:
                        x = torch.zeros(len(data.n_id), self.in_size).cuda()
                        x[np.arange(len(data.n_id)), data.n_id] = 1.0

                    # print(max(data.n_id), len(data.n_id))
                    # print(max(data.res_n_id), len(data.res_n_id))
                    # print('\n')

                    # x = data.n_id.float().reshape((-1, 1)).cuda() / self.in_size

                    # x = torch.randn(len(data.n_id)).cuda().reshape((-1, 1))
                    # x = torch.ones(len(data.n_id)).cuda().reshape((-1, 1))
                    # print(data.n_id)

                    # x_i = torch.LongTensor([np.arange(len(data.n_id)), data.n_id.numpy()])
                    # x_v = torch.ones(len(data.n_id))
                    # x = torch.sparse.FloatTensor(x_i, x_v, torch.Size([len(data.n_id), self.in_size])).cuda()
                    
                    # print(x)
                    # x = pre_gat_layer((x, None), data.edge_index, vals, data.size)
                    
                    # x = pre_gat_layer(x)
                    x = self.pre_gat_layers[idx](x)
                    
                    # x = F.leaky_relu(x, self.alpha)
                    
                    # x = self.identity_encoder(x)
                    # print(x.shape)
                    # curr_node_weights = node_weights[data.n_id, i].reshape((-1, 1))
                    # curr_node_weights = curr_node_weights[data.res_n_id]
                    # curr_node_idxs = data.n_id[data.res_n_id]
 
                if j != 0:
                    x_store_layer = [x_s[data.res_n_id] for x_s in x_store_layer]
                    x_pre = x[data.res_n_id]
                    x_store_layer.append(x_pre)

                    # curr_node_weights = curr_node_weights[data.res_n_id]
                    # curr_node_idxs = curr_node_idxs[data.res_n_id]

                # x = layer((x, None), data.edge_index, vals, data.size)
                # node_weights[data.n_id, idx].reshape((-1, 1))
                # print(curr_node_weights[data.res_n_id].shape)
                # x = node_weights[curr_node_idxs, i].view((-1, 1)) * self.gat_layers[idx]((x, None), data.edge_index, vals, data.size)
                x = self.gat_layers[idx]((x, None), data.edge_index, vals, data.size)
                
                # x = F.leaky_relu(x, self.alpha)

            x = sum(x_store_layer) + x  # Compute tensor with residuals
            x = weights[:, i] * interp_masks[:, i].reshape((-1, 1)) * x
            
            # x.register_hook(set_grad(x))
            
            # x = self.interp([x], masks[:, i].reshape((-1, 1)), evaluate)
            x_store_modality += x


        # x, weights = self.interp(x_store, masks, idxs, evaluate)

        # # Define encoder logic.
        # pre_cat_layers = []
        # out_pre_cat_layers = []  # Final layers before concatenation (no skip connections)

        # dense_layer = self.pre_gat_layers[0]
        # layer = self.gat_layers[0]

        # batch_size = data_flows[0].blocks[0].size[1]
        # x_store_modality = torch.zeros((batch_size, self.cat_size)).cuda()  # Tensor to store results from each modality.
        # for i, (data_flow, dataset) in enumerate(zip(data_flows, datasets)):

        #     x_store_layer = []
        #     for j, data in enumerate(data_flow):

        #         # Initial `x` is feature matrix.
        #         if j == 0:
        #             x = features[data.n_id]
        #             x = dense_layer(x)
 
        #         # Get edge weights.
        #         vals = dataset.edge_attr[data.e_id]

        #         if j != 0:
        #             x_store_layer = [x_s[data.res_n_id] for x_s in x_store_layer]
        #             x_pre = x[data.res_n_id]
        #             x_store_layer.append(x_pre)

        #         x = layer((x, None), data.edge_index, vals, data.size)
        #         x = F.leaky_relu(x, self.alpha)

        #     x = sum(x_store_layer) + x  # Compute tensor with residuals
        #     x = self.interp([x], masks[:, i].reshape((-1, 1)), evaluate)
        #     x_store_modality += x





        # x = self.interp(pre_cat_layers, masks, evaluate)
        # x = F.leaky_relu(x, self.alpha)

        # x = torch.mean(torch.stack(pre_cat_layers), dim=0)
        # x = torch.mul(pre_cat_layers[0], pre_cat_layers[1])

        # x = self.pre_emb(x)
        # x = F.leaky_relu(x, self.alpha)

        # Embedding.
        # x = self.bn(x)
        # x = F.leaky_relu(x_store_modality, 0.01)
        emb = self.emb(x_store_modality)
        # emb = self.emb(x)
        # emb = self.bn(emb)
        # emb = F.leaky_relu(emb, self.alpha)
        # emb = torch.tanh(emb)

        # post_emb = self.post_emb(emb)
        # x = F.leaky_relu(post_emb, self.alpha)


        # Dot product.
        dot = torch.mm(emb, torch.t(emb)) # <--- Uncomment this!
        # for i in range(100):
        #     for j in range(100):
        #         dot[i, j].backward(retain_graph=True)
        #         print(i, j)
        #         print(torch.norm(self.pre_gat_layers[0].weight.grad))
        #         print(torch.norm(self.pre_gat_layers[1].weight.grad))
        #         print('\n')
        
        # dot = F.sigmoid(dot)
        # dot = F.leaky_relu(dot, 0.001)
        # dot = self.pdist(emb, torch.t(emb))
        # norms = torch.mm(torch.diag(torch.mm(x, torch.t(x))).reshape(-1, 1), torch.ones((1, emb.shape[0])).cuda())#.reshape(1, -1))
        # print(norms)
        # euc_dist = norms + torch.t(norms) - 2*torch.mm(x, torch.t(x))
        # print(euc_dist)
        # emb_norm = emb / (emb.norm(dim=1)[:, None] + 1e-10)
        # dot = torch.mm(emb_norm, torch.t(emb_norm))
        # dot = 1 / (1 + euc_dist)
        # x = F.leaky_relu(dot, self.alpha)
        
        # Remove these later
        # dot = F.leaky_relu(self.dense1(emb), self.alpha)
        # dot = F.leaky_relu(self.dense2(dot), self.alpha)
        # dot = torch.mm(dot, torch.t(dot))

        # Define decoder logic.
        # out_layers = []
        # for dec_layer in self.dec_layers:
        #     x = dec_layer(emb)
        #     x = torch.mm(x, torch.t(x))
        #     # x = torch.sigmoid(x)
        #     out_layers.append(x)

        # return out_layers, emb, out_pre_cat_layers
        # return [dot for _ in range(self.n_modalities)], emb
        
        # return dot, dot, out_pre_cat_layers # <--- comment this
        return dot, emb, out_pre_cat_layers, weights # <--- uncomment this
        
