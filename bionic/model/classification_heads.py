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

        model = []
        for j in range(n_layers):
            model += [self.get_layers(emb_size // (2 ** j), emb_size // (2 ** (j + 1)))]

        model += [nn.Linear(emb_size // (2 ** n_layers), n_classes)]

        self.model = nn.Sequential(*model)

    def get_layers(self, in_size, out_size, dropout_rate=0.25, slope=0.01):
        layers = [nn.Linear(in_size, out_size)]
        layers += [nn.Dropout(dropout_rate)]
        layers += [nn.LeakyReLU(negative_slope=slope)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


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

        model = []

        for j in range(n_blocks):
            model += [nn.Sequential(
                    ResidualConnectionBlock(emb_size // (2 ** j), slope=slope)
                )]

            model += [nn.Linear(emb_size // (2 ** j), emb_size // (2 ** (j + 1)))]
            model += [nn.Dropout(dropout_rate)]
            model += [nn.LeakyReLU(negative_slope=slope)]

        model += [nn.Linear(emb_size // (2 ** n_blocks), n_classes)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
