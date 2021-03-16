import torch.nn as nn

from .fft import FFTBias, FFTBias2d
from .navie import NaiveBias, NaiveBias2d


class PositionalBias(nn.Module):
    def __init__(self, config):
        super(PositionalBias, self).__init__()
        if config.pos_bias_type == "naive":
            self.bias = NaiveBias(config=config)
        elif config.pos_bias_type == "naive_2d":
            self.bias = NaiveBias2d(config=config)
        elif config.pos_bias_type == "fft":
            self.bias = FFTBias(config=config)
        elif config.pos_bias_type == "fft_2d":
            self.bias = FFTBias2d(config=config)

    def forward(self, v):
        return self.bias(v)
