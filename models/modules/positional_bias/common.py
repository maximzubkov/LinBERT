import torch
import torch.nn as nn


class BiasBase(nn.Module):
    def __init__(self, config):
        super(BiasBase, self).__init__()
        self.bias_base_type = config.bias_base_type
        self.feature_map = config.feature_map
        self.type_ = config.pos_bias_type
        self.lm = config.lm
        self.has_specials = config.has_specials
        self.n_heads = config.num_attention_heads
        self.full_seq_len = config.max_position_embeddings
        if self.has_specials:
            self.full_seq_len = self.full_seq_len - 2

    def _init_bias(self):
        if self.bias_base_type == "full":
            self.w_shape = 2 * self.shape - 1
        elif self.bias_base_type == "symmetric":
            self.w_shape = self.shape
        else:
            raise ValueError("Unknown bias base type")

        w_ = torch.arange(self.w_shape).unsqueeze(0)
        w_ = w_ * 0.00001 * torch.rand(self.n_heads, 1) + 0.000001 * torch.randn(self.n_heads, 1)
        self.w = torch.nn.Parameter(
            torch.cat([torch.flip(w_[..., 1:], dims=[-1]), w_], dim=1).unsqueeze(0),
            requires_grad=True
        )
        self.w.data.uniform_(-0.1, 0.1)
