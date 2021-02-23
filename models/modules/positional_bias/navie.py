import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


class NaiveBiasBase(nn.Module):
    def __init__(self, config):
        super(NaiveBiasBase, self).__init__()
        self.bias_base_type = config.bias_base_type
        self.feature_map = config.feature_map
        self.type_ = config.pos_bias_type

    def _construct_bias(self, w_: torch.Tensor, seq_len: int, offset: torch.Tensor):
        if offset is not None:
            w_ = w_ - offset.unsqueeze(-1)

        if self.feature_map == "exp":
            w_ = torch.exp(w_)

        if self.bias_base_type == "full":
            p = w_
            bias = torch.cat([
                p[..., self.shape - i - seq_len: self.shape - i].unsqueeze(-1)
                for i in range(seq_len)
            ], -1)
        elif self.bias_base_type == "symmetric":
            p = torch.cat([torch.flip(w_[..., 1:], dims=[-1]), w_], dim=-1)
            bias = torch.cat([
                p[..., seq_len - i - 1: 2 * seq_len - i - 1].unsqueeze(-1)
                for i in range(seq_len)
            ], -1)
        else:
            raise ValueError("Unknown bias base type")

        return bias


class NaiveBias(NaiveBiasBase):
    def __init__(self, config):
        super(NaiveBias, self).__init__(config)
        n_heads = config.num_attention_heads
        full_seq_len = config.max_position_embeddings
        seq_len_without_special = full_seq_len - 2
        if self.bias_base_type == "full":
            self.shape = 2 * seq_len_without_special - 1
        elif self.bias_base_type == "symmetric":
            self.shape = seq_len_without_special
        else:
            raise ValueError("Unknown bias base type")

        self.w = torch.nn.Parameter(
            torch.randn(1, n_heads, self.shape),
            requires_grad=True
        )
        self.w.data.uniform_(-0.1, 0.1)

    def forward(self, v, offset):
        # [batch_size, seq_len, seq_len]
        v_ = v[:, 1:-1, :, :]
        batch_size, seq_len, n_heads, emb_dim = v_.shape

        if self.bias_base_type == "full":
            w_ = self.w
        elif self.bias_base_type == "symmetric":
            w_ = self.w[..., :seq_len]
        else:
            raise ValueError("Unknown bias base type")

        bias = self._construct_bias(w_, seq_len, offset)
        bias = F.pad(input=bias, pad=[1, 1, 1, 1], mode='constant', value=0)
        if (len(bias.shape) == 4) and (bias.shape[0] == 1):
            bias = bias.squeeze()
            bias = repeat(bias, "h l j -> n h l j", n=v.shape[0])
        z_pb = bias.sum(-1).transpose(-2, -1).unsqueeze(0)
        pbv = torch.einsum("nlhd,nhlj->njhd", v, bias)
        return pbv, z_pb


class NaiveBias2d(NaiveBiasBase):
    def __init__(self, config):
        super(NaiveBias2d, self).__init__(config)
        n_heads = config.num_attention_heads
        full_seq_len = config.max_position_embeddings
        seq_len_without_special = full_seq_len - 2

        self.shape = int(seq_len_without_special ** 0.5)

        if self.bias_base_type == "full":
            self.w_shape = 2 * self.shape - 1
        elif self.bias_base_type == "symmetric":
            self.w_shape = self.shape
        else:
            raise ValueError("Unknown bias base type")

        self.w = torch.nn.Parameter(
            torch.randn(1, n_heads, self.w_shape),
            requires_grad=True
        )
        self.w.data.uniform_(-0.1, 0.1)

    def forward(self, v, offset):
        # [batch_size, seq_len, seq_len]
        batch_size, seq_len, n_heads, emb_dim = v.shape
        w_ = self.w
        bias = self._construct_bias(w_, self.shape, offset)
        x_ = bias.unsqueeze(-3).unsqueeze(-2)
        y_ = bias.unsqueeze(-2).unsqueeze(-1)
        w_ = x_ + y_
        w_ = w_.reshape(n_heads, self.shape, self.shape, -1)
        w_ = w_.reshape(n_heads, -1, self.shape ** 2)
        w_ = F.pad(input=w_, pad=[1, 1, 1, 1], mode='constant', value=0)
        w_ = repeat(w_, 'h l j -> n h l j', n=v.shape[0])
        z_pb = w_.sum(-1).transpose(-2, -1).unsqueeze(0)
        pbv = torch.einsum("nlhd,nhlj->njhd", v, w_)
        return pbv, z_pb
