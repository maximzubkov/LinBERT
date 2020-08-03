from typing import Optional

import torch
import torch.nn as nn

from fast_transformers.linear_attention import LinearAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, h_s, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.h_s = h_s
        self.p_s = h_s // n_heads

        proj_s = self.p_s * self.n_heads
        self.q = nn.Sequential(nn.LayerNorm(h_s), nn.Linear(h_s, proj_s))
        self.k = nn.Sequential(nn.LayerNorm(h_s), nn.Linear(h_s, proj_s))
        self.v = nn.Sequential(nn.LayerNorm(h_s), nn.Linear(h_s, proj_s))

        self.attention = LinearAttention()

        self.fc = nn.Linear(self.n_heads * self.p_s, h_s)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
        residual = q

        q = self.split_heads(self.q(q))
        k = self.split_heads(self.k(k))
        v = self.split_heads(self.v(v))

        result = self.attention(q, k, v, mask)
        result = self.fc(result.flatten(2))

        return residual + self.dropout(result)

    def recurrent(self, input, memory=None):
        q = self.split_heads(self.q(input)).squeeze(1)
        k = self.split_heads(self.k(input)).squeeze(1)
        v = self.split_heads(self.v(input)).squeeze(1)

        if memory is None:
            b_s = q.size(0)
            s_i = q.new_zeros((b_s, self.n_heads, self.p_s, self.p_s))
            z_i = q.new_zeros((b_s, self.n_heads, self.p_s))

            memory = (s_i, z_i)

        result, memory = self.attention.recurrent(q, k, v, memory)
        result = self.fc(result.unsqueeze(1).flatten(2))

        return input + self.dropout(result), memory

    def split_heads(self, input):
        b_s, s_l, _ = input.size()
        return input.view(b_s, s_l, self.n_heads, self.p_s)
