import torch
from transformers import BertConfig

from models.modules import PositionalBias

config = BertConfig(
    vocab_size=15,
    max_position_embeddings=4,
    hidden_size=4,
    num_attention_heads=2,
    num_hidden_layers=2,
    type_vocab_size=1,
    has_pos_attention=True,
    has_pos_bias=False,
    has_batch_norm=False,
    feature_map="elu"
)


@torch.no_grad()
def test_pos_bias():
    pos_bias = PositionalBias(config)
    pos_bias.eval()

    batch_size = 4
    seq_len = config.max_position_embeddings
    num_heads = config.num_attention_heads
    embed_dim = int(config.hidden_size / config.num_attention_heads)
    v = torch.rand(batch_size, seq_len, num_heads, embed_dim)

    ppb_fft, z_pb_fft = pos_bias(v)
    w = pos_bias.w

    z = torch.cat([torch.flip(w[1:], dims=[0]), w], dim=0).unsqueeze(0)
    pb = torch.cat([
        z[:, seq_len - i - 1: 2 * seq_len - i - 1] for i in range(seq_len)
    ], 0)

    z_pb_orig = pb.sum(-1).view(1, pb.shape[0], 1)
    ppb_orig = torch.einsum("nlhd,lj->njhd", v, pb)

    assert torch.allclose(z_pb_orig, z_pb_fft), "Z not equal"
    assert torch.allclose(ppb_orig, ppb_fft), "PPB not equal"
