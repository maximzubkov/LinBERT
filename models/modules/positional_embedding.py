import torch
import torch.nn as nn
from transformers.modeling_bert import BertEmbeddings


class Bert2DEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.x_shape = config.x_shape
        self.y_shape = config.y_shape
        self.position_x_embeddings = nn.Embedding(self.x_shape, config.hidden_size)
        self.position_y_embeddings = nn.Embedding(self.y_shape, config.hidden_size)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_x_embeddings = self.position_x_embeddings(position_ids % self.x_shape)
            position_y_embeddings = self.position_y_embeddings(position_ids // self.y_shape)
            embeddings += position_x_embeddings + position_y_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
