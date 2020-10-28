#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

from typing import Optional

import torch
from torch.nn import Module

from fast_transformers.causal_product import causal_dot_product


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, feature_map=elu_feature_map, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = feature_map
        self.eps = eps

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):

        q = self.feature_map(q)
        k = self.feature_map(k)

        if mask is None:  # causal attention
            z = torch.einsum("nlhi,nlhi->nlh", q, k.cumsum(1)) + self.eps
            v = self.causal_linear(q, k, v)
            return v / z.unsqueeze(-1)
        else:
            torch.jit._unwrap_optional(mask)
            k = k * mask.view(mask.size(0), mask.size(1), 1, 1)

            # [batch_size, n_heads, p_s, p_s]
            kv = torch.einsum("nshd,nshm->nhmd", k, v)
            # [batch_size, target_seq_len, n_heads]
            z = torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + self.eps
            # [batch_size, target_seq_len, n_heads, p_s]
            return torch.einsum("nlhd,nhmd,nlh->nlhm", q, kv, 1 / z)

    def recurrent(self, q, k, v, memory=None):
        q = self.feature_map(q)
        k = self.feature_map(k)

        s_i, z_i = memory
        if z_i.requires_grad or s_i.requires_grad:
            z_i = z_i + k
            s_i = s_i + torch.einsum("nhd,nhm->nhdm", k, v)
        else:
            z_i += k
            s_i += torch.einsum("nhd,nhm->nhdm", k, v)

        z = torch.einsum("nhd,nhd->nh", q, z_i) + self.eps
        v = torch.einsum("nhd,nhdm,nh->nhm", q, s_i, 1 / z)

        return v, (s_i, z_i)

    @staticmethod
    def causal_linear(q, k, v):
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        V_new = causal_dot_product(q, k, v)
        return V_new.permute(0, 2, 1, 3).contiguous()
