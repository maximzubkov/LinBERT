import torch
from transformers import BertConfig

from models.modules.fast_transformers import LinearAttention

config = BertConfig(
    has_pos_attention=False,
    has_pos_bias=False,
    has_batch_norm=False
)


@torch.no_grad()
def test_lin_head_mask():
    attn = LinearAttention(config)
    attn.eval()

    seq_len = 3
    batch_size = 2
    embed_dim = 4
    num_heads = 5
    q = torch.rand(batch_size, seq_len, num_heads, embed_dim)
    v = torch.rand(batch_size, seq_len, num_heads, embed_dim)
    k = torch.rand(batch_size, seq_len, num_heads, embed_dim)

    mask = torch.zeros((num_heads,), dtype=torch.uint8)  # or dtype=torch.ByteTensor

    res = attn(q, k, v, head_mask=mask)
    assert torch.all(torch.eq(res, torch.zeros_like(v)))

    head_mask = torch.zeros((num_heads,), dtype=torch.uint8)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.uint8)
    attn(q, k, v, head_mask=head_mask, attention_mask=attention_mask)
    assert True


@torch.no_grad()
def test_lin_attention_mask():
    attn = LinearAttention(config)
    attn.eval()

    seq_len = 3
    batch_size = 2
    embed_dim = 4
    num_heads = 1
    q = torch.rand(batch_size, seq_len, num_heads, embed_dim)
    v = torch.rand(batch_size, seq_len, num_heads, embed_dim)

    k1 = torch.rand(batch_size, seq_len, num_heads, embed_dim)
    k2 = k1.clone().detach()
    k2[:, 1:, :, :] = torch.rand(batch_size, seq_len - 1, num_heads, embed_dim)

    mask = torch.zeros((batch_size, seq_len), dtype=torch.uint8)  # or dtype=torch.ByteTensor
    mask[:, 0] = 1

    res1 = attn(q, k1, v, attention_mask=mask)
    res2 = attn(q, k2, v, attention_mask=mask)

    assert torch.all(torch.eq(res1, res2))
