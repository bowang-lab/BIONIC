import torch
import torch.nn as nn
import numpy as np


class AttentiveIntegration(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(AttentiveIntegration, self).__init__()
        assert embedding_dim % n_head == 0
        self.n_head = n_head
        self.head_dim = int(embedding_dim / n_head)
        self.intermediate_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, self.intermediate_dim)
        self.key = nn.Linear(embedding_dim, self.intermediate_dim)
        self.value = nn.Linear(embedding_dim, self.intermediate_dim)
        self.output = nn.Linear(self.intermediate_dim, embedding_dim)

    def forward(self, embeddings, attention_mask=None, return_att_scores=False):
        Q = self.query(embeddings)
        K = self.key(embeddings)
        V = self.value(embeddings)
        Q = Q.view(Q.size()[:-1] + (self.n_head, self.head_dim)).permute(0, 2, 1, 3)
        K = K.view(K.size()[:-1] + (self.n_head, self.head_dim)).permute(0, 2, 1, 3)
        V = V.view(V.size()[:-1] + (self.n_head, self.head_dim)).permute(0, 2, 1, 3)

        A = torch.matmul(Q, K.transpose(-1, -2))  # Q*t(K)
        A = A / np.sqrt(self.head_dim)

        # MASK should maskout entities not in specific input modality
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(dim=-1)
            M_shape = attention_mask.shape[:-1] + (attention_mask.shape[-2],)
            M = attention_mask.expand(M_shape)
            M = M.unsqueeze(dim=1)
            M = M.permute(0, 1, 3, 2)
            M = (1.0 - M) * - 1e9
            A = A + M

        A = nn.Softmax(dim=-1)(A)
        H = torch.matmul(A, V)
        H = H.permute(0, 2, 1, 3).contiguous()
        H = H.view(H.size()[:-2] + (self.intermediate_dim,))

        H = self.output(H)

        if return_att_scores:
            return H, A
        else:
            return H
