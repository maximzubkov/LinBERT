from transformers import BertForSequenceClassification

from .linear_bert import LinBertModel  # noqa
from .pos_attn_bert import PosAttnBertModel  # noqa


class Classifier(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        if config.is_linear:
            self.bert = LinBertModel(config)
        else:
            self.bert = PosAttnBertModel(config)
