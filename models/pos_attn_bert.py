import math

import torch
import torch.nn as nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention

from models.modules import PositionalAttention
from models.modules.positional_bias import PositionalBias
from models.modules.positional_embedding import Bert2DEmbeddings


class PosAttnBertSelfAttention(BertSelfAttention):
    def __init__(self, config, pos_attention: nn.Module = None):
        super().__init__(config)
        self.pos_attention = pos_attention
        self.pos_bias = PositionalBias(config) if config.pos_bias_type is not None else None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        past_key_value=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.pos_attention is not None:
            _, _, seq_len, _ = attention_scores.shape
            attention_scores += self.pos_attention(seq_len)
            scaling_factor = math.sqrt(2 * self.attention_head_size)
        else:
            scaling_factor = math.sqrt(self.attention_head_size)
        attention_scores = attention_scores / scaling_factor
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if self.pos_bias is not None:
            attention_probs = attention_probs + self.pos_bias(value_layer)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class PosAttnBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.pos_attention = \
            PositionalAttention(self.embeddings.position_embeddings) if config.has_pos_attention else None

        if config.has_pos_embed_2d:
            self.embeddings = Bert2DEmbeddings(config)

        for i, _ in enumerate(self.encoder.layer):
            self.encoder.layer[i].attention.self = PosAttnBertSelfAttention(config, self.pos_attention)
