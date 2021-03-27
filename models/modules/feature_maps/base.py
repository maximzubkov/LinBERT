#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Create the feature map interface and some commonly used feature maps.
All attention implementations that expect a feature map shall receive a factory
function that returns a feature map instance when called with the query
dimensions.
"""

import torch
from torch.nn import Module


class FeatureMap(Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.
        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


def exp_feature_map(x):
    return torch.exp(x)


def dpfp_feature_map(x, nu=1):
    x_ = torch.cat([
      torch.nn.functional.relu(x), torch.nn.functional.relu(-x)
    ], dim=-1)

    x_rolled = torch.cat([
        x_.roll(shifts=j, dims=-1) for j in range(1, nu + 1)
    ], dim=-1)

    x_repeat = torch.cat([x_] * nu, dim=-1)

    return x_repeat * x_rolled
