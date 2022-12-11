import torch
import torch.nn as nn


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


class MultilayerLeakyReLUDropoutHead(torch.nn.Module):

    def __init__(self, emb_size, n_classes, slope=0.01, dropout_rate=0.25, n_layers=3):
        super(MultilayerLeakyReLUDropoutHead, self).__init__()

        self.linear_layers = []

        for j in range(n_layers):
            self.linear_layers.append(nn.Linear(emb_size // (2 ** j), emb_size // (2 ** (j + 1))).to("cuda:0"))

        self.output_layer = nn.Linear(emb_size // (2 ** n_layers), n_classes)

        self.activation = nn.LeakyReLU(negative_slope=slope)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        for linear_layer in self.linear_layers:
            x = linear_layer(x)
            x = self.dropout(x)
            x = self.activation(x)

        x = self.output_layer(x)
        return x


class ResidualConnectionBlock(nn.Module):
    def __init__(self, input_size, slope=0.01):
        super().__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.shortcut = nn.Sequential()

        self.linear2 = nn.Linear(input_size, input_size)

        self.activation = nn.LeakyReLU(negative_slope=slope)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = x + shortcut
        x = self.activation(x)
        return x


class MultilayerSkipConnectionHead(torch.nn.Module):

    def __init__(self, emb_size, n_classes, slope=0.01, dropout_rate=0.25, n_blocks=3):
        super(MultilayerSkipConnectionHead, self).__init__()

        self.n_blocks = n_blocks

        self.skip_connection_blocks = []
        self.linear_layers = []

        for j in range(n_blocks):
            self.skip_connection_blocks.append(
                nn.Sequential(
                    ResidualConnectionBlock(emb_size // (2 ** j), slope=slope)
                ).to("cuda:0")
            )
            self.linear_layers.append(nn.Linear(emb_size // (2 ** j), emb_size // (2 ** (j + 1))).to("cuda:0"))

        self.output_layer = nn.Linear(emb_size // (2 ** n_blocks), n_classes)

        self.activation = nn.LeakyReLU(negative_slope=slope)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        for j in range(self.n_blocks):
            x = self.skip_connection_blocks[j](x)
            x = self.linear_layers[j](x)
            x = self.dropout(x)
            x = self.activation(x)

        x = self.output_layer(x)
        return x
