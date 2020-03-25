import torch.nn as nn
import torch

from models.GLU import GLU


class CNN(nn.Module):

    def __init__(self, n_in_channel, activation="Relu", conv_dropout=0, batch_norm=True,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)], linear_dims=None, linear_activation="relu", linear_dropout=0.,
                 aggregation="mean", norm_out=False, frames=None
                 ):
        super(CNN, self).__init__()
        self.agg = aggregation
        self.norm_out = norm_out
        print(f"will aggregate for inputs > {frames} frames")
        self.frames = frames

        cnn = nn.Sequential()

        def conv(ind, batch_normalisation=False, dropout=None, activ="relu"):
            dim_in = n_in_channel if ind == 0 else nb_filters[ind - 1]
            dim_out = nb_filters[ind]
            cnn.add_module('conv{0}'.format(ind),
                           nn.Conv2d(dim_in, dim_out, kernel_size[ind], stride[ind], padding[ind]))
            if batch_normalisation:
                cnn.add_module('batchnorm{0}'.format(ind), nn.BatchNorm2d(dim_out))  #, eps=0.001, momentum=0.99))
            if activ.lower() == "leakyrelu":
                cnn.add_module('leakyrelu{0}'.format(ind),
                               nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module('relu{0}'.format(ind), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module('glu{0}'.format(ind), GLU(dim_out))
            if dropout is not None:
                cnn.add_module('dropout{0}'.format(ind),
                               nn.Dropout(dropout))

        # 128x862x64
        for i in range(len(nb_filters)):
            conv(i, batch_norm, conv_dropout, activ=activation)
            cnn.add_module('pooling{0}'.format(i), nn.MaxPool2d(pooling[i]))  # bs x tframe x mels
        self.cnn = cnn

        if linear_dims is not None:
            if type(linear_dims) is not list:
                linear_dims = [linear_dims]
            self.linear_in = linear_dims[0][0]
            linear = nn.Sequential()
            for i, layer_dim in enumerate(linear_dims):
                linear_in, linear_out = layer_dim
                suffix = str(i)
                linear.add_module('linear' + suffix, nn.Linear(linear_in, linear_out))
                if linear_activation.lower() == "leakyrelu":
                    linear.add_module('leakyrelu' + suffix, nn.LeakyReLU(0.2))
                elif linear_activation.lower() == "relu":
                    linear.add_module('relu' + suffix, nn.ReLU())
                elif linear_activation.lower() == "glu":
                    linear.add_module('glu' + suffix, GLU(linear_out))
                if linear_dropout is not None:
                    linear.add_module('linear_dropout' + suffix,
                                      nn.Dropout(linear_dropout))
            self.linear = linear
        else:
            self.linear = None

    def get_embedding(self, x):
        x = self.forward(x)
        return x

    def load(self, filename=None, parameters=None):
        if self.linear is None:
            if filename is not None:
                self.cnn.load_state_dict(torch.load(filename))
            elif parameters is not None:
                self.cnn.load_state_dict(parameters)
            else:
                raise NotImplementedError("load is a filename or a list of parameters (state_dict)")
        else:
            if filename is not None:
                dic = torch.load(filename)
                self.cnn.load_state_dict(dic["cnn"])
                self.linear.load_state_dict(dic["linear"])
            elif parameters is not None:
                self.cnn.load_state_dict(parameters["cnn"])
                self.linear.load_state_dict(parameters["linear"])
            else:
                raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if self.linear is None:
            dic = self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        else:
            dic = {"cnn": self.cnn.state_dict(),
                   "linear": self.linear.state_dict()}
        return dic

    def save(self, filename):
        if self.linear is None:
            dic = self.cnn.state_dict()
        else:
            dic = {"cnn": self.cnn.state_dict(),
                   "linear": self.linear.state_dict()}
        torch.save(dic, filename)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        bs, feat, fr, mel = x.shape
        if self.frames is not None and fr > self.frames:
            mul = round(fr // self.frames)
            x = x.view(bs*mul, feat, self.frames, mel)
        x = self.cnn(x)

        if self.linear is not None:
            x = x.view(x.shape[0], -1, self.linear_in)
            x = self.linear(x)
        else:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
                x = x.permute(0, 2, 1).contiguous()
            else:
                x = x.permute(0, 2, 3, 1).contiguous()
                x = x.view(x.shape[0], x.shape[1], -1)

        if self.norm_out:
            x = x / torch.norm(x, dim=-1, keepdim=True)
        if len(x.shape) > 2:
            if self.agg == "mean":
                x = x.mean(-2)
            if self.agg == "attention":
                x = (nn.functional.softmax(x, dim=-2) * x).sum(-2)

        if self.frames is not None and fr != self.frames:
            x = x.view((bs, mul) + x.shape[1:])
            x = x.mean(1)
        return x
