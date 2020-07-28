import torch.nn as nn

from fast_transformers.attention.linear_attention import LinearAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, h_s, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.h_s = h_s
        self.p_s = h_s // n_heads

        proj_s = self.p_s * self.n_heads
        self.q = nn.Sequential(nn.LayerNorm(h_s), nn.Linear(h_s, proj_s))
        self.k = nn.Sequential(nn.LayerNorm(h_s), nn.Linear(h_s, proj_s))
        self.v = nn.Sequential(nn.LayerNorm(h_s), nn.Linear(h_s, proj_s))

        self.attention = LinearAttention()

        self.fc = nn.Linear(self.n_heads * self.p_s, h_s)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        residual = q

        q = self.split_heads(self.q(q))
        k = self.split_heads(self.k(k))
        v = self.split_heads(self.v(v))

        result = self.attention(q, k, v, mask)
        result = self.fc(result.flatten(2))
        print(result.size())

        return residual + self.dropout(result)

    def split_heads(self, input):
        b_s, s_l, _ = input.size()
        return input.view(b_s, s_l, self.n_heads, self.p_s)


# if __name__ == "__main__":
#     import torch
#
#     att = MultiHeadAttention(2, 4, dropout=0.0)
#     # x = torch.randn(2, 5, 4)
#     # print(att(x, x, x, "causal"))
#     # print("__")
#     # print(att(x[:, :4], x[:, :4], x[:, :4], "causal"))
#     # print("__")
#     # print(att(x[:, :2], x[:, :2], x[:, :2], "causal"))
#     # print("__")
#
#     x = torch.randn(2, 5, 4)
#     print(att(x, x, x, mask=torch.LongTensor([[1, 1, 1, 1, 0], [1, 0, 0, 0, 0]])))
#     print("__")
#     print(att(x, x, x, mask=torch.LongTensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])))
#     print("__")
#     print(att(x[:, :4], x[:, :4], x[:, :4], mask=torch.LongTensor([[1, 1, 1, 0], [1, 0, 0, 0]])))
#     print("__")
#     print(att(x[:, :3], x[:, :3], x[:, :3], mask=torch.LongTensor([[1, 1, 1], [1, 0, 0]])))
#     print("__")
#     print(att(x[:, :2], x[:, :2], x[:, :2], mask=torch.LongTensor([[1, 1], [1, 0]])))
#     print("__")
