import torch
from .base import dpfp_feature_map, elu_feature_map, exp_feature_map  # noqa
from .fourier_features import Favor  # noqa


def fm_name2func(name: str, q_dim: int, device: torch.device = None):
    if name == "elu":
        return elu_feature_map
    elif name == "exp":
        return exp_feature_map
    elif name == "dpfp":
        return dpfp_feature_map
    elif name == "favor":
        favor = Favor(q_dim)
        favor.new_feature_map(device)
        return Favor(q_dim)

