import torch
import torch.nn as nn
from transformers import BertConfig
from models.modules import PositionalAttention, LinPositionalAttention

config = BertConfig(
    vocab_size=15,
    max_position_embeddings=15,
    hidden_size=15,
    num_attention_heads=3,
    num_hidden_layers=2,
    type_vocab_size=1,
    has_pos_attention=True,
    has_batch_norm=False
)


@torch.no_grad()
def test_pos_attn():
    embed = nn.Embedding(num_embeddings=15, embedding_dim=15)
    del embed.weight
    embed.weight = torch.eye(15, 15)
    pos_attn = PositionalAttention(embed)
    pos_attn.eval()

    assert torch.all(torch.eq(pos_attn(15), pos_attn.pos_linear.weight))


@torch.no_grad()
def test_lin_pos_attn():
    embed = nn.Embedding(num_embeddings=15, embedding_dim=15)
    del embed.weight
    embed.weight = torch.eye(15, 15)
    lin_pos_attn = LinPositionalAttention(config, embed, feature_map=nn.functional.relu)
    lin_pos_attn.eval()
    batch_size = 3
    seq_len = 15
    num_heads = config.num_attention_heads
    embed_dim = int(15 / config.num_attention_heads)
    q = torch.rand(batch_size, seq_len, num_heads, embed_dim)
    v = torch.rand(batch_size, seq_len, num_heads, embed_dim)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.uint8)
    head_mask = torch.ones((num_heads,), dtype=torch.uint8)
    ppv, z_pp = lin_pos_attn(q, v, head_mask=head_mask, attention_mask=attention_mask)
    # positional information is independent to batch in case of full sequences
    for i in range(batch_size - 1):
        assert torch.all(torch.eq(z_pp[i, :, :], z_pp[i + 1, :, :]))

    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.uint8)
    ppv, z_pp = lin_pos_attn(q, v, head_mask=head_mask, attention_mask=attention_mask)
    assert torch.equal(ppv, torch.zeros_like(ppv))
