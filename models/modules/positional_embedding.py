import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEmbeddings


class Embedding2D(nn.Module):
    def __init__(self, x_shape: int, y_shape: int, hidden_size: int):
        super().__init__()
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.x_embeddings = nn.Embedding(self.x_shape, hidden_size)
        self.y_embeddings = nn.Embedding(self.y_shape, hidden_size)
        self.spec_embeddings = nn.Embedding(2, hidden_size)

    def forward(self, position_ids):
        img_pos_ids = position_ids[:, 1:-1] - 1
        img_emb = self.y_embeddings(img_pos_ids // self.x_shape) + self.y_embeddings(img_pos_ids % self.y_shape)
        return torch.cat([
            self.spec_embeddings.weight[0].unsqueeze(0),
            img_emb.squeeze(),
            self.spec_embeddings.weight[1].unsqueeze(0)
        ], dim=0).unsqueeze(0)


class Bert2DEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.position_embeddings = Embedding2D(config.x_shape, config.y_shape, config.hidden_size)
