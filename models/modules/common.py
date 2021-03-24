import torch


def transpose_for_scores(x, num_attention_heads, attention_head_size):
    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
    # [batch_size, seq_len,  n_heads, p_s]
    return x.view(*new_x_shape)


def compute_mask(x: torch.Tensor):
    attn_mask = torch.full(
        (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
    )
    return torch.triu(attn_mask, diagonal=1)
