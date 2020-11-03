import torch

from fast_transformers import LinearAttention


@torch.no_grad()
def test_lin_attention_mask():
    attn = LinearAttention()
    attn.eval()

    seq_len = 3
    batch_size = 2
    embed_dim = 4
    num_heads = 1
    q = torch.rand(seq_len, batch_size, num_heads, embed_dim)
    v = torch.rand(seq_len, batch_size, num_heads, embed_dim)

    k1 = torch.rand(seq_len, batch_size, 1, embed_dim)
    k2 = k1.clone().detach()
    k2[:, 1:, :, :] = torch.rand(seq_len, batch_size - 1, num_heads, embed_dim)

    mask = torch.zeros((seq_len, batch_size), dtype=torch.uint8)  # or dtype=torch.ByteTensor
    mask[:, 0] = 1

    res1 = attn(q, k1, v, mask)
    res2 = attn(q, k2, v, mask)

    assert torch.all(torch.eq(res1, res2))
