import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.num_heads = config.num_attention_heads
        if self.type_ in ["fft_2d", "naive_2d"]:
            self.seq_len = config.max_position_embeddings - 2
            self.n = int(self.seq_len ** 0.5)
            self.w = torch.nn.Parameter(
                torch.sort(torch.randn(self.num_heads, self.n), descending=True, dim=-1)[0],
                requires_grad=True
            )
        else:
            self.seq_len = config.max_position_embeddings
            self.w = torch.nn.Parameter(
                torch.sort(torch.randn(self.num_heads, self.seq_len), dim=-1)[0],
                requires_grad=True
            )
        if self.type_ == "fft":
            self.o_ = torch.nn.Parameter(torch.ones(self.seq_len), requires_grad=False)
        elif self.type_ == "fft_2d":
            self.o_ = torch.ones(self.n)
            self.o_ = nn.functional.pad(self.o_, [self.n - 1, 0])
            self.o_fft = torch.nn.Parameter(torch.rfft(self.o_, 1), requires_grad=False)

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

    def _construct_bias(self, seq_len: int):
        w_ = self.w[:, :seq_len]
        p = torch.cat([torch.flip(w_[:, 1:], dims=[1]), w_], dim=1)
        bias = torch.cat([
            p[:, seq_len - i - 1: 2 * seq_len - i - 1].unsqueeze(-1)
            for i in range(seq_len)
        ], -1)
        return bias

    def _naive(self, v):
        # [batch_size, seq_len, seq_len]
        v_ = v[:, 1:-1, :, :]
        _, seq_len, *_ = v_.shape
        bias = self._construct_bias(seq_len)
        bias = F.pad(input=bias, pad=[1, 1, 1, 1], mode='constant', value=0)
        z_pb = bias.sum(-1).transpose(1, 0).unsqueeze(0)
        pbv = torch.einsum("nlhd,hlj->njhd", v, bias)
        return pbv, z_pb

    def _naive_2d(self, v):
        # [batch_size, seq_len, seq_len]
        bias = self._construct_bias(self.n)
        x_ = bias.unsqueeze(1).unsqueeze(3)
        y_ = bias.unsqueeze(2).unsqueeze(-1)
        w_ = x_ + y_
        w_ = w_.reshape(self.num_heads, self.n, self.n, -1)
        w_ = w_.reshape(self.num_heads, -1, self.n ** 2)
        w_ = F.pad(input=w_, pad=[1, 1, 1, 1], mode='constant', value=0)
        z_pb = w_.sum(-1).transpose(1, 0).unsqueeze(0)
        pbv = torch.einsum("nlhd,hlj->njhd", v, w_)
        return pbv, z_pb

    @staticmethod
    def _complex_mul(x, y):
        assert x.shape[-1] == 2 and y.shape[-1] == 2, 'Last dimension must be 2'
        return torch.stack(
            (x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1],
             x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]),
            dim=-1)

    def _compute_z_fft(self, seq_len: int):
        w_ = self.w[:, :seq_len]
        z = torch.cat([
            w_[:, -1].reshape(-1, 1),  # w_{N-1}
            torch.flip(w_[:, 1:], dims=[1]),  # w_{N-1}, w_{N-2}, ..., w_{1}
            w_[:, :-1]  # w_{0}, w_{1}, ..., w_{N-2}
        ], dim=1)
        # z_fft has shape [num_heads, seq_len * 2 - 1, 2]
        return torch.rfft(z, 1)

    def _fft(self, v):
        # [batch_size, seq_len, seq_len]
        batch_size, seq_len, n_heads, emb_dim = v.shape
        seq_len -= 2
        z_fft = self._compute_z_fft(seq_len)

        v_ = v[:, 1:-1, :, :].permute(0, 3, 2, 1).reshape(batch_size * emb_dim, n_heads, seq_len)

        # Since z has shape [num_heads, seq_len * 2 - 1] we need to pad
        # values with zeros
        v_ = nn.functional.pad(v_, [seq_len - 1, 0])
        v_fft = torch.rfft(v_, 1)

        pbv = torch.irfft(self._complex_mul(v_fft, z_fft), 1)
        pbv = pbv[:, :, :seq_len]
        pbv = pbv.reshape(batch_size, emb_dim, n_heads, seq_len)
        pbv = F.pad(input=pbv, pad=[1, 1], mode='constant', value=0)
        pbv = pbv.permute(0, 3, 2, 1)

        o_ = nn.functional.pad(self.o_[:seq_len], [seq_len - 1, 0])
        o_fft = torch.rfft(o_, 1)

        z_pb = torch.irfft(self._complex_mul(z_fft, o_fft), 1)
        z_pb = z_pb[:, :seq_len]
        z_pb = F.pad(input=z_pb, pad=[1, 1], mode='constant', value=0)
        z_pb = z_pb.transpose(1, 0).unsqueeze(0)

        return pbv, z_pb

    def _fft_2d(self, v):
        # [batch_size, seq_len, seq_len]
        batch_size, seq_len, n_heads, emb_dim = v.shape
        seq_len -= 2
        shape = self.w.shape[1]

        z_fft = self._compute_z_fft(self.n)

        v_ = v[:, 1:-1, :, :].transpose(-3, -1).reshape(-1, n_heads, shape, shape).transpose(-3, -2)

        v_m = nn.functional.pad(v_.sum(-3), [shape - 1, 0])
        v_m_fft = torch.rfft(v_m, 1)

        u_m = nn.functional.pad(v_.transpose(-3, -1).sum(-3), [shape - 1, 0])
        u_m_fft = torch.rfft(u_m, 1)

        RxV_m = torch.irfft(self._complex_mul(v_m_fft, z_fft), 1)
        RxV_m = RxV_m[..., :shape]
        RxU_m = torch.irfft(self._complex_mul(u_m_fft, z_fft), 1)
        RxU_m = RxU_m[..., :shape]

        pbv = RxV_m.unsqueeze(-2) + RxU_m.unsqueeze(-1)
        pbv = pbv.reshape(batch_size, emb_dim, n_heads, seq_len)
        pbv = F.pad(input=pbv, pad=[1, 1], mode='constant', value=0)
        pbv = pbv.permute(0, 3, 2, 1)

        z_pb = torch.irfft(self._complex_mul(self.o_fft, z_fft), 1)
        z_pb = z_pb[..., :shape] * shape

        z_pb = z_pb.unsqueeze(-2) + z_pb.unsqueeze(-1)
        z_pb = z_pb.reshape(-1, shape * shape)
        z_pb = F.pad(input=z_pb, pad=[1, 1], mode='constant', value=0)
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
