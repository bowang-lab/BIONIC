import torch
import torch.nn as nn
#
# nn.Linear(self.emb_size, self.emb_size),  # improves optimization
# nn.Linear(self.emb_size, n_classes_),


class BasicLeakyReLUHead(torch.nn.Module):

    def __init__(self, emb_size, n_classes, slope=0.01):
        super(BasicLeakyReLUHead, self).__init__()

        self.linear1 = nn.Linear(emb_size, emb_size)
        self.activation = nn.LeakyReLU(negative_slope=slope)
        self.linear2 = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class LeakyReLUDropoutHead(torch.nn.Module):

    def __init__(self, emb_size, n_classes, slope=0.01, dropout_rate=0.25):
        super(LeakyReLUDropoutHead, self).__init__()

        self.linear1 = nn.Linear(emb_size, emb_size)
        self.activation = nn.LeakyReLU(negative_slope=slope)
        self.linear2 = nn.Linear(emb_size, n_classes)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
