import torch
import torch.nn.functional as F
from einops import repeat

from .common import BiasBase


class NaiveBiasBase(BiasBase):
    def __init__(self, config):
        super(NaiveBiasBase, self).__init__(config)

    def _process(self, w_: torch.Tensor, batch_size: int):
        if self.has_specials:
            w_ = F.pad(input=w_, pad=[1, 1, 1, 1], mode='constant', value=0)
        if self.lm:
            w_ = w_ * self.mask
        if (len(w_.shape) == 4) and (w_.shape[0] == 1):
            w_ = w_.squeeze()
            w_ = repeat(w_, "h l j -> n h l j", n=batch_size)
        return w_

    def _construct_bias(self, w_: torch.Tensor, seq_len: int):
        if self.bias_base_type == "full":
            w_ = torch.cat([
                w_[..., -1].unsqueeze(-1),  # w_{N-1}
                torch.flip(w_[..., 1:], dims=[-1]),  # w_{N-1}, w_{N-2}, ..., w_{1}
                w_[..., :-1]  # w_{0}, w_{1}, ..., w_{N-2}
            ], dim=-1)
        elif self.bias_base_type == "symmetric":
            w_ = torch.cat([
                torch.flip(w_[..., 1:], dims=[-1]),
                w_
            ], dim=-1)
        else:
            raise ValueError("Unknown bias base type")
        bias = torch.cat([
            w_[..., seq_len - i - 1: 2 * seq_len - i - 1].unsqueeze(-1)
            for i in range(seq_len)
        ], dim=-1)

        return bias


class NaiveBias(NaiveBiasBase):
    def __init__(self, config):
        super(NaiveBias, self).__init__(config)
        self.shape = self.full_seq_len
        self._init_bias()

    def forward(self, v):
        # [batch_size, seq_len, seq_len]
        batch_size, seq_len, n_heads, emb_dim = v.shape
        if self.has_specials:
            seq_len -= 2

        if self.bias_base_type == "full":
            w_ = self.w[..., self.shape - seq_len: self.shape + seq_len - 1]
        elif self.bias_base_type == "symmetric":
            w_ = self.w[..., :seq_len]
        else:
            raise ValueError("Unknown bias base type")

        bias = self._construct_bias(w_, seq_len)
        bias = self._process(bias, batch_size)
        z_pb = bias.sum(-1).transpose(-2, -1)
        pbv = torch.einsum("nlhd,nhlj->njhd", v, bias.transpose(-2, -1))
        return pbv, z_pb


class NaiveBias2d(NaiveBiasBase):
    def __init__(self, config):
        super(NaiveBias2d, self).__init__(config)
        self.shape = int(self.full_seq_len ** 0.5)
        self._init_bias()

    def forward(self, v):
        # [batch_size, seq_len, seq_len]
        batch_size, seq_len, n_heads, emb_dim = v.shape
        bias = self._construct_bias(self.w, self.shape)
        x_ = bias.unsqueeze(-3).unsqueeze(-2)
        y_ = bias.unsqueeze(-2).unsqueeze(-1)
        w_ = x_ + y_
        w_batch_shape, *_ = w_.shape
        w_ = w_.reshape(w_batch_shape, n_heads, self.shape, self.shape, -1)
        w_ = w_.reshape(w_batch_shape, n_heads, -1, self.shape ** 2)
        w_ = self._process(w_, batch_size)
        z_pb = w_.sum(-1).transpose(-2, -1)
        pbv = torch.einsum("nlhd,nhlj->njhd", v, w_.transpose(-2, -1))
        return pbv, z_pb
