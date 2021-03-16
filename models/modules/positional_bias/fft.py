import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BiasBase


class FFTBiasBase(BiasBase):
    def __init__(self, config):
        super(FFTBiasBase, self).__init__(config)

    @staticmethod
    def _complex_mul(x, y):
        assert x.shape[-1] == 2 and y.shape[-1] == 2, 'Last dimension must be 2'
        return torch.stack((
            x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1],
            x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
        ), dim=-1)

    def _process(self, x: torch.Tensor):
        if self.has_specials:
            x = F.pad(input=x, pad=[1, 1], mode='constant', value=0)
        return x

    def _compute_z_fft(self, seq_len: int):
        # [num_heads, seq_len]
        if self.bias_base_type == "full":
            z = self.w[..., self.shape - seq_len: self.shape + seq_len - 1]
        elif self.bias_base_type == "symmetric":
            w_ = self.w[..., :seq_len]
            z = torch.cat([
                w_[..., -1].unsqueeze(-1),  # w_{N-1}
                torch.flip(w_[..., 1:], dims=[-1]),  # w_{N-1}, w_{N-2}, ..., w_{1}
                w_[..., :-1]  # w_{0}, w_{1}, ..., w_{N-2}
            ], dim=-1)
        else:
            raise ValueError("Unknown bias base type")

        if self.lm:
            mask = torch.ones_like(z)
            *_, shape = z.shape
            mask_len = (shape + 1) // 2
            mask[mask_len:] = 0
            z = z * mask
        # z_fft has shape [num_heads, seq_len * 2 - 1, 2], the last two dims belongs to real and img parts
        return torch.rfft(z, 1)

    def _init_bias(self):
        super()._init_bias()
        if self.bias_base_type == "full":
            w_ = torch.arange(self.shape).unsqueeze(0)
            w_ = w_ * 0.00001 * torch.rand(self.n_heads, 1) + 0.000001 * torch.randn(self.n_heads, 1)
            w_ = torch.cat([
                w_[..., -1].unsqueeze(-1),  # w_{N-1}
                torch.flip(w_[..., 1:], dims=[-1]),  # w_{N-1}, w_{N-2}, ..., w_{1}
                w_[..., :-1]  # w_{0}, w_{1}, ..., w_{N-2}
            ], dim=-1).unsqueeze(0)
            self.w = torch.nn.Parameter(w_, requires_grad=True)


class FFTBias(FFTBiasBase):
    def __init__(self, config):
        super(FFTBias, self).__init__(config)
        self.shape = self.full_seq_len
        self._init_bias()
        self.o_ = torch.nn.Parameter(torch.ones(self.shape), requires_grad=False)

    def forward(self, v):
        # [batch_size, [bos] + [...] x seq_len + [eos], n_heads, emb_dim]
        v_ = v[:, 1:-1, :, :] if self.has_specials else v
        batch_size, seq_len, n_heads, emb_dim = v_.shape
        z_fft = self._compute_z_fft(seq_len)

        v_ = v_.permute(0, 3, 2, 1)

        pad_size = seq_len - 1

        v_ = nn.functional.pad(v_, [pad_size, 0])
        v_fft = torch.rfft(v_, 1)
        pbv = torch.irfft(self._complex_mul(v_fft, z_fft.unsqueeze(1)), 1)
        pbv = pbv[..., :seq_len]
        pbv = self._process(pbv)
        pbv = pbv.permute(0, 3, 2, 1)

        o_ = nn.functional.pad(self.o_[:seq_len], [pad_size, 0])
        o_fft = torch.rfft(o_, 1)

        z_pb = torch.irfft(self._complex_mul(z_fft, o_fft), 1)
        z_pb = z_pb[..., :seq_len]
        z_pb = self._process(z_pb)
        z_pb = z_pb.transpose(-2, -1)
        return pbv, z_pb


class FFTBias2d(FFTBiasBase):
    def __init__(self, config):
        super(FFTBias2d, self).__init__(config)
        self.shape = int(self.full_seq_len ** 0.5)
        self._init_bias()

        self.o_ = torch.ones(self.shape)
        self.o_ = nn.functional.pad(self.o_, [self.shape - 1, 0])
        self.o_fft = torch.nn.Parameter(torch.rfft(self.o_, 1), requires_grad=False)

    def forward(self, v):
        # [batch_size, [bos] + [...] x seq_len + [eos], seq_len]
        v_ = v[:, 1:-1, :, :] if self.has_specials else v
        batch_size, seq_len, n_heads, emb_dim = v_.shape
        z_fft = self._compute_z_fft(self.shape)

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
        pbv = self._process(pbv)
        pbv = pbv.permute(0, 3, 2, 1)

        z_pb = torch.irfft(self._complex_mul(self.o_fft, z_fft), 1)
        z_pb = z_pb[..., :self.shape] * self.shape

        z_pb = z_pb.unsqueeze(-2) + z_pb.unsqueeze(-1)
        z_pb = z_pb.reshape(-1, n_heads, self.shape * self.shape)
        z_pb = self._process(z_pb)
        z_pb = z_pb.transpose(-2, -1)

        return pbv, z_pb
