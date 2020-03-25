import torch
from torch import nn


class CombineModel(nn.Module):
    """ Class to combine multiple models. Similar to Sequential except the save and load."""

    def __init__(self, *args):
        super(CombineModel, self).__init__()
        self.models = nn.Sequential(*args)

    def forward(self, inputs):
        return self.models(inputs)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            parameters = torch.load(filename)
        if parameters is None:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

        for model in self.models:
            model.load(parameters[model.__class__.__name__])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {}
        for model in self.models:
            state_dict[model.__class__.__name__] = model.state_dict(destination=destination, prefix=prefix,
                                                                    keep_vars=keep_vars)
        return state_dict

    def save(self, filename):
        parameters = self.state_dict()
        torch.save(parameters, filename)
