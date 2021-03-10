import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTBiasBase(nn.Module):
    def __init__(self, config):
        super(FFTBiasBase, self).__init__()
        self.bias_base_type = config.bias_base_type
        self.feature_map = config.feature_map
        self.type_ = config.pos_bias_type

    @staticmethod
    def _complex_mul(x, y):
        assert x.shape[-1] == 2 and y.shape[-1] == 2, 'Last dimension must be 2'
        return torch.stack((
            x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1],
            x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
        ), dim=-1)

    def _compute_z_fft(self, seq_len: int, offset: torch.Tensor):
        # [num_heads, seq_len]
        if self.bias_base_type == "full":
            z = self.w
        elif self.bias_base_type == "symmetric":
            w_ = self.w[..., :seq_len]
            z = torch.cat([
                w_[..., -1].unsqueeze(-1),  # w_{N-1}
                torch.flip(w_[..., 1:], dims=[-1]),  # w_{N-1}, w_{N-2}, ..., w_{1}
                w_[..., :-1]  # w_{0}, w_{1}, ..., w_{N-2}
            ], dim=-1)
        else:
            raise ValueError("Unknown bias base type")

        if offset is not None:
            z = z - offset.unsqueeze(-1)

        if self.feature_map == "exp":
            z = torch.exp(z)

        # z_fft has shape [num_heads, seq_len * 2 - 1, 2], the last two dims belongs to real and img parts
        return torch.rfft(z, 1)


class FFTBias(FFTBiasBase):
    def __init__(self, config):
        super(FFTBias, self).__init__(config)
        n_heads = config.num_attention_heads
        full_seq_len = config.max_position_embeddings
        seq_len_without_special = full_seq_len - 2

        if self.bias_base_type == "full":
            self.shape = 2 * seq_len_without_special - 1
        elif self.bias_base_type == "symmetric":
            self.shape = seq_len_without_special
        else:
            raise ValueError("Unknown bias base type")

        self.w_shape = self.shape
        self.w = torch.nn.Parameter(
            torch.randn(1, n_heads, self.w_shape),
            requires_grad=True
        )
        self.w.data.uniform_(-0.1, 0.1)
        self.o_ = torch.nn.Parameter(torch.ones(seq_len_without_special), requires_grad=False)

    def forward(self, v, offset):
        # [batch_size, [bos] + [...] x seq_len + [eos], seq_len]
        v_ = v[:, 1:-1, :, :]
        batch_size, seq_len, n_heads, emb_dim = v_.shape
        z_fft = self._compute_z_fft(seq_len, offset)

        v_ = v_.permute(0, 3, 2, 1).reshape(batch_size, emb_dim, n_heads, seq_len)

        if self.bias_base_type == "full":
            pad_size = self.w_shape - seq_len
        elif self.bias_base_type == "symmetric":
            pad_size = seq_len - 1
        else:
            raise ValueError("Unknown bias base type")

        v_ = nn.functional.pad(v_, [pad_size, 0])
        v_fft = torch.rfft(v_, 1)

        pbv = torch.irfft(self._complex_mul(v_fft, z_fft), 1)
        pbv = pbv[..., :seq_len]
        pbv = pbv.reshape(batch_size, emb_dim, n_heads, seq_len)
        pbv = F.pad(input=pbv, pad=[1, 1], mode='constant', value=0)
        pbv = pbv.permute(0, 3, 2, 1)

        o_ = nn.functional.pad(self.o_[:seq_len], [pad_size, 0])
        o_fft = torch.rfft(o_, 1)

        z_pb = torch.irfft(self._complex_mul(z_fft, o_fft), 1)
        z_pb = z_pb[..., :seq_len]
        z_pb = F.pad(input=z_pb, pad=[1, 1], mode='constant', value=0)
        z_pb = z_pb.transpose(-2, -1)
        return pbv, z_pb


class FFTBias2d(FFTBiasBase):
    def __init__(self, config):
        super(FFTBias2d, self).__init__(config)
        n_heads = config.num_attention_heads
        full_seq_len = config.max_position_embeddings
        seq_len_without_special = full_seq_len - 2

        self.shape = int(seq_len_without_special ** 0.5)

        if self.bias_base_type == "full":
            self.w_shape = 2 * self.shape - 1
        elif self.bias_base_type == "symmetric":
            self.w_shape = self.shape
        else:
            raise ValueError("Unknown bias base type")

        self.w = torch.nn.Parameter(
            torch.randn(1, n_heads, self.w_shape),
            requires_grad=True
        )
        self.w.data.uniform_(-0.1, 0.1)

        self.o_ = torch.ones(self.shape)
        self.o_ = nn.functional.pad(self.o_, [self.shape - 1, 0])
        self.o_fft = torch.nn.Parameter(torch.rfft(self.o_, 1), requires_grad=False)

    def forward(self, v, offset):
        # [batch_size, [bos] + [...] x seq_len + [eos], seq_len]
        v_ = v[:, 1:-1, :, :]
        batch_size, seq_len, n_heads, emb_dim = v_.shape
        z_fft = self._compute_z_fft(self.w_shape, offset)

        v_ = v_.transpose(-3, -1).reshape(batch_size, emb_dim, n_heads, self.shape, self.shape).transpose(-3, -2)

        v_m = nn.functional.pad(v_.sum(-3), [self.shape - 1, 0])
        v_m_fft = torch.rfft(v_m, 1)

        u_m = nn.functional.pad(v_.transpose(-3, -1).sum(-3), [self.shape - 1, 0])
        u_m_fft = torch.rfft(u_m, 1)

        RxV_m = torch.irfft(self._complex_mul(v_m_fft, z_fft.unsqueeze(1)), 1)
        RxV_m = RxV_m[..., :self.shape]
        RxU_m = torch.irfft(self._complex_mul(u_m_fft, z_fft.unsqueeze(1)), 1)
        RxU_m = RxU_m[..., :self.shape]

        pbv = RxV_m.unsqueeze(-2) + RxU_m.unsqueeze(-1)
        pbv = pbv.reshape(batch_size, emb_dim, n_heads, seq_len)
        pbv = F.pad(input=pbv, pad=[1, 1], mode='constant', value=0)
        pbv = pbv.permute(0, 3, 2, 1)

        z_pb = torch.irfft(self._complex_mul(self.o_fft, z_fft), 1)
        z_pb = z_pb[..., :self.shape] * self.shape

        z_pb = z_pb.unsqueeze(-2) + z_pb.unsqueeze(-1)
        z_pb = z_pb.reshape(batch_size, n_heads, self.shape * self.shape)
        z_pb = F.pad(input=z_pb, pad=[1, 1], mode='constant', value=0)
        z_pb = z_pb.transpose(-2, -1)

        return pbv, z_pb
