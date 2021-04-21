import math

import torch
import torch.nn as nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention

from positional_bias.pytorch import PositionalBias


class PosBiasBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        if config.pos_bias_type is not None:
            self.pos_bias = PositionalBias(
                bias_base_type=config.bias_base_type,
                pos_bias_type=config.pos_bias_type,
                num_attention_heads=config.num_attention_heads,
                max_seq_len=config.max_position_embeddings,
                lm=config.lm,
                has_specials=config.has_specials
            )
        else:
            self.pos_bias = None

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
        scaling_factor = math.sqrt(self.attention_head_size)
        attention_scores = attention_scores / scaling_factor
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        if self.pos_bias is not None:
            pbv, z_pb = self.pos_bias(value_layer.transpose(-2, -3))
            context_layer = context_layer + pbv

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class OrigBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)

        for i, _ in enumerate(self.encoder.layer):
            self.encoder.layer[i].attention.self = PosBiasBertSelfAttention(config)
