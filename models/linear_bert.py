import torch.nn as nn
from transformers import BertModel, BertForMaskedLM

from models.modules import LinPositionalAttention
from models.modules.common import transpose_for_scores
from models.modules.fast_transformers import LinearAttention


class LinBertSelfAttention(nn.Module):
    def __init__(self, config, pos_attention: nn.Module = None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.max_position_embeddings = config.max_position_embeddings

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attention = LinearAttention(config, pos_attention)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = transpose_for_scores(
            mixed_query_layer,
            self.num_attention_heads,
            self.attention_head_size
        )
        key_layer = transpose_for_scores(
            mixed_key_layer,
            self.num_attention_heads,
            self.attention_head_size
        )
        value_layer = transpose_for_scores(
            mixed_value_layer,
            self.num_attention_heads,
            self.attention_head_size
        )

        if attention_mask is not None:
            attention_mask = attention_mask.permute(0, 3, 1, 2).squeeze()

        context_layer = self.attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            head_mask
        )
        context_layer = self.dropout(context_layer)

        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, None) if output_attentions else (context_layer,)
        return outputs


class LinBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.pos_attention = \
            LinPositionalAttention(config, self.embeddings.position_embeddings) if config.has_pos_attention else None

        for i, _ in enumerate(self.encoder.layer):
            self.encoder.layer[i].attention.self = LinBertSelfAttention(config, self.pos_attention)


class LinBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bert = LinBertModel(config)
