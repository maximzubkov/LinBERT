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

        self.w = torch.nn.Parameter(
            torch.randn(1, self.n_heads, self.w_shape),
            requires_grad=True
        )

        if self.lm:
            ones = torch.ones(self.full_seq_len, self.full_seq_len)
            self.mask = nn.Parameter(
                torch.tril(ones).unsqueeze(0).unsqueeze(0),
                requires_grad=False
            )
