import torch
from einops import repeat
from transformers import BertConfig

from models.modules import PositionalBias

seq_len = 28 * 28
num_heads = 10
batch_size = 16
embed_dim = 64

config = BertConfig(
    vocab_size=15,
    max_position_embeddings=seq_len,
    hidden_size=128,
    num_attention_heads=num_heads,
    num_hidden_layers=2,
    type_vocab_size=1,
    has_pos_attention=True,
    has_batch_norm=False,
    pos_bias_type="naive",
    bias_base_type="full",
    lm=True,
    has_specials=False,
    feature_map="elu"
)

v = torch.rand(batch_size, seq_len, num_heads, embed_dim)


@torch.no_grad()
def test_lm_naive_2d():
    cumsum = 2 * torch.cumsum(v, dim=-3)
    config.pos_bias_type = "naive_2d"
    pos_bias = PositionalBias(config)
    pos_bias.eval()
    pos_bias.bias.w.data = repeat(torch.ones(2 * int(seq_len ** 0.5) - 1), 's -> h s', h=num_heads).unsqueeze(0)

    ppb, z_pb = pos_bias(v)
    assert torch.allclose(ppb, cumsum, atol=1e-3), "Cumsum and new v are not equal"


@torch.no_grad()
def test_lm_naive():
    cumsum = torch.cumsum(v, dim=-3)
    config.pos_bias_type = "naive"
    pos_bias = PositionalBias(config)
    pos_bias.eval()
    pos_bias.bias.w.data = repeat(torch.ones(2 * seq_len - 1), 's -> h s', h=num_heads).unsqueeze(0)

    ppb, z_pb = pos_bias(v)
    assert torch.allclose(ppb, cumsum, atol=1e-3), "Cumsum and new v are not equal"