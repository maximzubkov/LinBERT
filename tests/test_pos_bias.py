import torch
from transformers import BertConfig

from models.modules import PositionalBias

fft_config = BertConfig(
    vocab_size=15,
    max_position_embeddings=4,
    hidden_size=4,
    num_attention_heads=2,
    num_hidden_layers=2,
    type_vocab_size=1,
    has_pos_attention=True,
    has_batch_norm=False,
    pos_bias_type="fft",
    feature_map="elu"
)

naive_config = BertConfig(
    vocab_size=15,
    max_position_embeddings=4,
    hidden_size=4,
    num_attention_heads=2,
    num_hidden_layers=2,
    type_vocab_size=1,
    has_pos_attention=True,
    has_batch_norm=False,
    pos_bias_type="naive",
    feature_map="elu"
)


@torch.no_grad()
def test_pos_bias():
    fft_pos_bias = PositionalBias(fft_config)
    fft_pos_bias.eval()

    naive_pos_bias = PositionalBias(naive_config)
    naive_pos_bias.eval()

    naive_pos_bias.w.data = fft_pos_bias.w.data

    batch_size = 4
    seq_len = fft_config.max_position_embeddings
    num_heads = fft_config.num_attention_heads
    embed_dim = int(fft_config.hidden_size / fft_config.num_attention_heads)
    v = torch.rand(batch_size, seq_len, num_heads, embed_dim)

    ppb_fft, z_pb_fft = fft_pos_bias(v)

    ppb_orig, z_pb_orig = naive_pos_bias(v)
    print(ppb_orig - ppb_fft)
    assert torch.allclose(z_pb_orig, z_pb_fft, atol=1e-4), "Z not equal"
    assert torch.allclose(ppb_orig, ppb_fft, atol=1e-4), "PPB not equal"
