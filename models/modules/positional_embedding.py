import torch.nn as nn
from transformers.modeling_bert import BertEmbeddings


class Embedding2D(nn.Module):
    def __init__(self, x_shape: int, y_shape: int, hidden_size: int):
        super().__init__()
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.x_embeddings = nn.Embedding(self.x_shape, hidden_size)
        self.y_embeddings = nn.Embedding(self.y_shape, hidden_size)

    def forward(self, position_ids):
        return self.y_embeddings(position_ids // self.x_shape) + self.y_embeddings(position_ids % self.y_shape)


class Bert2DEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.position_embeddings = Embedding2D(config.x_shape, config.y_shape, config.hidden_size)
