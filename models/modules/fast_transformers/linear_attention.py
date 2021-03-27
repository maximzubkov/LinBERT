#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Module

from models.modules.feature_maps import (
    elu_feature_map,
    exp_feature_map,
    dpfp_feature_map,
    Favor
)
from models.modules.fast_transformers.causal_product import causal_dot_product
from models.modules.positional_attention import PositionalAttention
from models.modules.positional_bias import PositionalBias


class LinearAttention(Module):
    def __init__(self, config, pos_attention: PositionalAttention = None, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.pos_attention = pos_attention
        self.feature_map_name = config.feature_map

        attn_head_size = config.hidden_size // config.num_attention_heads
        max_seq_len = config.max_position_embeddings

        if config.feature_map == "elu":
            self.feature_map = elu_feature_map
        elif config.feature_map == "exp":
            self.feature_map = exp_feature_map
        elif config.feature_map == "dpfp":
            self.feature_map = dpfp_feature_map
        elif config.feature_map == "favor":
            self.feature_map = Favor(attn_head_size)
        else:
            raise ValueError("Invalid feature map specified")
        self.eps = eps

        self.bn_k = nn.LayerNorm([max_seq_len, attn_head_size]) if config.has_batch_norm else None
        self.bn_q = nn.LayerNorm([max_seq_len, attn_head_size]) if config.has_batch_norm else None
        self.pos_bias = PositionalBias(config) if config.pos_bias_type is not None else None

    def forward(self, q, k, v, attention_mask: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None):
        if self.feature_map_name == "exp":
            offset = q.mean(-3).mean(-1) + k.mean(-3).mean(-1)
            offset = offset.unsqueeze(-1).unsqueeze(-3)
            q = q - offset
            k = k - offset
        if self.bn_q is not None:
            q = self.bn_q(q.transpose(2, 1)).transpose(2, 1)
        # [batch_size, q_seq_len, n_heads, p_s]
        q = self.feature_map(q)

        if self.bn_k is not None:
            k = self.bn_k(k.transpose(2, 1)).transpose(2, 1)
        # [batch_size, k_seq_len, n_heads, p_s]
        k = self.feature_map(k)

        # y equals to numerator value after applying attention
        # [batch_size, target_seq_len, n_heads, p_s]
        y = None

        if attention_mask is None:  # causal attention
            z = torch.einsum("nlhi,nlhi->nlh", q, k.cumsum(1)) + self.eps
            v_ = self.causal_linear(q, k, v)

            if self.pos_bias is not None:
                pbv, z_pb = self.pos_bias(v)
                y = pbv
            if head_mask is not None:
                v_ = v_ * head_mask.view(1, 1, *head_mask.shape, 1)

            output = v_ / z.unsqueeze(-1)
        else:
            torch.jit._unwrap_optional(attention_mask)
            k = k * attention_mask.view(*attention_mask.shape, 1, 1)
            # [batch_size, n_heads, p_s, p_s]
            kv = torch.einsum("nshd,nshm->nhmd", k, v)
            if head_mask is not None:
                kv = kv * head_mask.view(1, *head_mask.shape, 1, 1)
            # z equals to denominator value after applying attention
            # [batch_size, target_seq_len, n_heads]
            z = torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + self.eps

            if self.pos_bias is not None:
                pbv, z_pb = self.pos_bias(v)
                y = pbv

            if self.pos_attention is not None:
                ppv, z_pp = self.pos_attention(q, v, attention_mask, head_mask)
                y = y + ppv if y is not None else ppv

            inv_z = 1 / z
            output = torch.einsum("nlhd,nhmd,nlh->nlhm", q, kv, inv_z)
        if y is not None:
            output = output + y
        return output

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
