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
    def __init__(self, config):
        super(PositionalBias, self).__init__()
        self.type_ = config.pos_bias_type
        self.seq_len = config.max_position_embeddings
        if self.type_ in ["fft_2d", "naive_2d"]:
            self.n = int(self.seq_len ** 0.5)
            self.w = torch.nn.Parameter(
                torch.sort(torch.randn(self.n))[0],
                requires_grad=True
            )
        else:
            self.w = torch.nn.Parameter(
                torch.sort(torch.randn(self.seq_len), descending=True)[0],
                requires_grad=True
            )
        self.w.data.uniform_(-0.1, 0.1)
        if self.type_ == "fft":
            self.o_ = torch.ones(config.num_attention_heads, self.seq_len)
            self.o_ = nn.functional.pad(self.o_, [self.seq_len - 1, 0])
            self.o_fft = torch.nn.Parameter(torch.rfft(self.o_, 2), requires_grad=False)

    def forward(self, v):
        if self.type_ == "naive":
            return self._naive(v)
        elif self.type_ == "naive_2d":
            return self._naive_2d(v)
        elif self.type_ == "fft":
            return self._fft(v)
        elif self.type_ == "fft_2d":
            return self._fft_2d(v)
        elif self.type_ == "orig":
            return self._construnct_bias()
        else:
            raise ValueError("Unknown positional bias type")

    def _construnct_bias(self):
        p = torch.cat([torch.flip(self.w[1:], dims=[0]), self.w], dim=0)
        shape = self.w.shape[0]
        bias = torch.cat([
            p[shape - i - 1: 2 * shape - i - 1].unsqueeze(0)
            for i in range(shape)
        ], 0)
        return bias

    def _naive(self, v):
        # [batch_size, seq_len, seq_len]
        bias = self._construnct_bias()
        z_pb = bias.sum(-1).view(1, bias.shape[0], 1)
        pbv = torch.einsum("nlhd,lj->njhd", v, bias)
        return pbv, z_pb

    def _naive_2d(self, v):
        # [batch_size, seq_len, seq_len]
        bias = self._construnct_bias()
        x_ = bias.unsqueeze(0).unsqueeze(2)
        y_ = bias.unsqueeze(1).unsqueeze(3)
        w_ = x_ + y_
        w_ = w_.reshape(self.n, self.n, -1)
        w_ = w_.reshape(-1, self.n ** 2)
        z_pb = w_.sum(-1).view(1, w_.shape[0], 1)
        pbv = torch.einsum("nlhd,lj->njhd", v, w_)
        return pbv, z_pb

    @staticmethod
    def _complex_mul(x, y):
        assert x.shape[-1] == 2 and y.shape[-1] == 2, 'Last dimension must be 2'
        return torch.stack(
            (x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1],
             x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]),
            dim=-1)

    def _fft(self, v):
        # [batch_size, seq_len, seq_len]
        z = torch.cat([
            self.w[-1].unsqueeze(0),  # w_{N-1}
            torch.flip(self.w[1:], dims=[0]),  # w_{N-1}, w_{N-2}, ..., w_{1}
            self.w[:-1]  # w_{0}, w_{1}, ..., w_{N-2}
        ], dim=0)

        z_fft = torch.rfft(z, 1)
        batch_size, seq_len, n_heads, emb_dim = v.shape

        v_ = v.permute(0, 2, 3, 1).reshape(batch_size * n_heads * emb_dim, seq_len)

        v_ = nn.functional.pad(v_, [seq_len - 1, 0])
        v_fft = torch.rfft(v_, 2)

        pbv = torch.irfft(self._complex_mul(v_fft, z_fft), 2, signal_sizes=v_.shape)
        pbv = pbv[:, :seq_len]
        pbv = pbv.reshape(batch_size, n_heads, emb_dim, seq_len).permute(0, 3, 1, 2)

        z_pb = torch.irfft(self._complex_mul(z_fft, self.o_fft), 2, signal_sizes=self.o_.shape)
        z_pb = z_pb[:, :seq_len]
        z_pb = z_pb.transpose(1, 0).unsqueeze(0)

        return pbv, z_pb


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
