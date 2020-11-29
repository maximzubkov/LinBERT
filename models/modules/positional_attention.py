import torch
import torch.nn as nn

from .common import elu_feature_map, transpose_for_scores


class PositionalAttention(nn.Module):
    def __init__(self, pos_embedding_layer: nn.Embedding):
        super().__init__()
        assert pos_embedding_layer is not None, "No embedding layer provided"
        self.pos_embedding_layer = pos_embedding_layer
        _, emb_dim = self.pos_embedding_layer.weight.data.shape
        self.pos_linear = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, seq_len: int):
        positional_embeddings = self.pos_embedding_layer.weight.data[:seq_len, :]
        return torch.matmul(positional_embeddings, self.pos_linear(positional_embeddings).transpose(1, 0))


class PositionalBias(nn.Module):
    def __init__(self, max_length: int):
        super().__init__()
        self.N = max_length
        self.w = torch.nn.Parameter(torch.randn(self.N), requires_grad=True)
        self.z = torch.cat([torch.flip(self.w[1:], dims=[0]), self.w], dim=0)

    def forward(self):
        # [batch_size, seq_len, seq_len]
        return torch.cat([self.z[self.N - i - 1: 2 * self.N - i - 1].unsqueeze(0) for i in range(self.N)], 0)


class LinPositionalAttention(nn.Module):
    def __init__(self, config,
                 pos_embedding_layer: nn.Embedding,
                 feature_map=elu_feature_map,
                 eps=1e-6):
        super().__init__()

        self.feature_map = feature_map
        self.eps = eps

        assert pos_embedding_layer is not None, "No embedding layer provided"
        self.pos_embedding_layer = pos_embedding_layer

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def forward(self, q: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor, head_mask: torch.Tensor):
        _, seq_len, _, _ = q.shape
        p = transpose_for_scores(
            self.pos_embedding_layer.weight.data[:seq_len, :],
            self.num_attention_heads,
            self.attention_head_size
        )

        p = self.feature_map(p)

        torch.jit._unwrap_optional(attention_mask)

        p = p.reshape(1, *p.shape) * attention_mask.view(*attention_mask.shape, 1, 1)
        # [batch_size, n_heads, p_s, p_s]
        pv = torch.einsum("nshd,nshm->nhmd", p, v)
        if head_mask is not None:
            pv = pv * head_mask.view(1, *head_mask.shape, 1, 1)
        ppv = torch.einsum("nlhd,nhmd->nlhm", p, pv)
        # [batch_size, target_seq_len, n_heads]
        z_pp = torch.einsum("nlhd,nhd->nlh", p, p.sum(dim=1)) + self.eps
        # [batch_size, target_seq_len, n_heads, p_s]
        return ppv, z_pp
