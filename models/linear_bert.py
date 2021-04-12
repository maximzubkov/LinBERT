import torch
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention

from models.common import transpose_for_scores
from models.modules.fast_transformers import LinearAttention


class LinBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)

        self.attention = LinearAttention(config)

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
            attention_mask = attention_mask.squeeze()
            attention_mask = torch.where(attention_mask == 0,
                                         torch.ones_like(attention_mask),
                                         torch.zeros_like(attention_mask)
                                         )

        context_layer = self.attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            head_mask
        )

        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, None) if output_attentions else (context_layer,)
        return outputs


class LinBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)

        for i, _ in enumerate(self.encoder.layer):
            self.encoder.layer[i].attention.self = LinBertSelfAttention(config)
