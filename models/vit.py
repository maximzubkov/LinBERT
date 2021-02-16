import torch
import torch.nn as nn
from einops import rearrange
from transformers import BertConfig
from vit_pytorch import ViT
import torch.nn.functional as F

from models.modules.fast_transformers import LinearAttention


class ViTLinBertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size

        self.scale = self.attention_head_size ** -0.5

        self.attention = LinearAttention(config, None)

        self.to_qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask=None):
        x_ = F.pad(x, [0, 0, 0, 1])
        qkv = self.to_qkv(x_).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.num_attention_heads), qkv)

        mask = torch.ones(1)
        out = self.attention(q, k, v, mask)

        out = rearrange(out, 'b n h d -> b n (h d)')
        out = self.to_out(out)[:, :-1, :]
        return out


class ViTModel(ViT):
    def __init__(self, config: BertConfig):
        self.config = config
        super().__init__(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            num_classes=self.config.num_labels,
            channels=self.config.channels,
            dim=self.config.hidden_size,
            depth=self.config.num_hidden_layers,
            dim_head=self.config.hidden_size // self.config.num_attention_heads,
            heads=self.config.num_attention_heads,
            mlp_dim=self.config.intermediate_size,
            dropout=self.config.hidden_dropout_prob,
            emb_dropout=self.config.hidden_dropout_prob
        )

        for i, _ in enumerate(self.transformer.layers):
            if self.config.is_linear:
                self.transformer.layers[i][0].fn.fn = ViTLinBertSelfAttention(config)
