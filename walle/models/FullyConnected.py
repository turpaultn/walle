import torch
from torch import nn as nn

from models.GLU import GLU


class FullyConnected(nn.Module):

    def __init__(self, input_dim, dimensions, out_dim, batch_norm=False, activation="relu", final_activation="sigmoid",
                 dropout=0):
        super(FullyConnected, self).__init__()

        network = nn.Sequential()

        def linear(dim_in, dim_out, num_layer, batch_normalization=False, dropout_val=None, activ="relu"):
            network.add_module('linear{0}'.format(num_layer),
                               nn.Linear(dim_in, dim_out, bias=True))
            if batch_normalization:
                network.add_module('batchnorm{0}'.format(num_layer), nn.BatchNorm1d(dim_out, eps=0.001, momentum=0.99))
            if activ is not None:
                if activ.lower() == "leakyrelu":
                    network.add_module('leakyrelu{0}'.format(num_layer),
                                       nn.LeakyReLU(0.2))
                elif activ.lower() == "relu":
                    network.add_module('relu{0}'.format(num_layer), nn.ReLU())
                elif activ.lower() == "tanh":
                    network.add_module('tanh{0}'.format(num_layer), nn.Tanh())
                elif activ.lower() == "glu":
                    network.add_module('glu{0}'.format(num_layer), GLU(dim_out))
                elif activ.lower() == "sigmoid":
                    network.add_module('sigmoid{0}'.format(num_layer), nn.Sigmoid())
                elif activ.lower() == "softmax":
                    network.add_module('softmax{0}'.format(num_layer), nn.Softmax())
                elif activ.lower() == "log_softmax":
                    network.add_module('log_softmax{0}'.format(num_layer), nn.LogSoftmax())
            if dropout_val is not None:
                network.add_module('dropout{0}'.format(num_layer), nn.Dropout(dropout_val))

        for j in range(len(dimensions)):
            d_in = input_dim if j == 0 else dimensions[j - 1]
            d_out = dimensions[j]
            linear(d_in, d_out, j, batch_norm, dropout, activ=activation)

        linear(dimensions[-1], out_dim, len(dimensions), activ=final_activation)

        self.network = network

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.network.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.network.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.network.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def forward(self, x):
        x = self.network(x)
        return x
