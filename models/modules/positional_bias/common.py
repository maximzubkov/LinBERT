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

        self.alpha = config.alpha
        self.beta = config.beta

        if self.has_specials:
            self.full_seq_len = self.full_seq_len - 2

        if self.lm:
            ones = torch.ones(self.full_seq_len, self.full_seq_len)
            self.mask = nn.Parameter(
                torch.tril(ones).unsqueeze(0),
                requires_grad=False
            )

    def _init_bias(self):
        w_ = torch.arange(self.shape).unsqueeze(0)
        w_ = w_ * self.alpha + self.beta
        w_ = w_ * torch.ones(self.n_heads, 1)

        if self.bias_base_type == "full":
            self.w_shape = 2 * self.shape - 1
            w_ = torch.cat([
                w_[..., -1].unsqueeze(-1),  # w_{N-1}
                torch.flip(w_[..., 1:], dims=[-1]),  # w_{N-1}, w_{N-2}, ..., w_{1}
                w_[..., :-1]  # w_{0}, w_{1}, ..., w_{N-2}
            ], dim=-1).unsqueeze(0)
        elif self.bias_base_type == "symmetric":
            self.w_shape = self.shape
            w_ = w_.unsqueeze(0)
        else:
            raise ValueError("Unknown bias base type")

        self.w = torch.nn.Parameter(w_, requires_grad=True)
