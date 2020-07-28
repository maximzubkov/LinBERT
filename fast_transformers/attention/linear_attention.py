#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement causally masked linear attention."""

import torch
from torch.nn import Module
from typing import Union

from fast_transformers.attention.causal_product import causal_dot_product


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, feature_map=elu_feature_map, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = feature_map
        self.eps = eps

    def forward(self, queries, keys, values, mask: Union[str, torch.Tensor] = "causal"):

        q = self.feature_map(queries)
        k = self.feature_map(keys)

        if mask == "causal":
            z = torch.einsum("nlhi,nlhi->nlh", q, k.cumsum(1)) + self.eps
            v = self.causal_linear(q, k, values)
            return v / z.unsqueeze(-1)
        else:
            k = k * mask.view(mask.size(0), mask.size(1), 1, 1)

            kv = torch.einsum("nshd,nshm->nhmd", k, values)
            z = torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + self.eps
            return torch.einsum("nlhd,nhmd,nlh->nlhm", q, kv, 1 / z)

    @staticmethod
    def causal_linear(q, k, v):
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        V_new = causal_dot_product(q, k, v)
        return V_new.permute(0, 2, 1, 3).contiguous()


if __name__ == "__main__":
    torch.manual_seed(100)
    att = LinearAttention()
    x = torch.randn(1, 3, 2, 10)
    y = torch.randn(1, 4, 2, 10)
    z = torch.randn(1, 4, 2, 10)

    print(att(x, y, z, torch.tensor([[1.0, 1, 1, 0]])).mean(-1))
    print(att(x, y[:, :3], z[:, :3], torch.tensor([[1.0, 1, 1]])).mean(-1))
