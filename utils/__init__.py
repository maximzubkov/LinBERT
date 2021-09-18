import random
from argparse import ArgumentParser
from copy import deepcopy
from os.path import join

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction, BertModel, BertConfig, BertTokenizerFast
from transformers.trainer_utils import set_seed

from configs import ModelConfig, mnist_config
from models import Classifier
from models.linear_bert import LinBertSelfAttention
from models.orig_bert import PosBiasBertSelfAttention


def set_seed_(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    accuracy = accuracy_score(labels, preds)
    return {
        "f1": f1,
        "recall": recall,
        "accuracy": accuracy,
        "precision": precision,
    }


def parse_model_config(arg_parser: ArgumentParser) -> ModelConfig:
    arg_parser.add_argument("--is_linear", action='store_true')
    arg_parser.add_argument("--feature_map", choices=["elu", "exp", "dpfp", "favor"], default="elu")
    arg_parser.add_argument("--pos_bias_type", choices=["fft", "naive", "fft_2d", "naive_2d"], default=None)
    arg_parser.add_argument("--bias_base_type", choices=["full", "symmetric"], default=None)
    args = arg_parser.parse_args()

    return ModelConfig(
        is_linear=args.is_linear,
        feature_map=args.feature_map,
        pos_bias_type=args.pos_bias_type,
        bias_base_type=args.bias_base_type
    )


def _copy_weights(input_, target):
    input_.weight.data = target.weight.data
    input_.bias.data = target.bias.data


def _copy_self_attn(self_attn, target_self_attn):
    _copy_weights(
        input_=self_attn.query,
        target=target_self_attn.query
    )
    _copy_weights(
        input_=self_attn.value,
        target=target_self_attn.value
    )
    _copy_weights(
        input_=self_attn.key,
        target=target_self_attn.key
    )


def make_attn_linear(model: BertModel, config: BertConfig):
    tmp_model = deepcopy(model)

    for i, _ in enumerate(tmp_model.bert.encoder.layer):
        model.bert.encoder.layer[i].attention.self = LinBertSelfAttention(config)

        _copy_self_attn(
            self_attn=model.bert.encoder.layer[i].attention.self,
            target_self_attn=tmp_model.bert.encoder.layer[i].attention.self
        )
    return model


def add_pos_bias(model: BertModel, config: BertConfig):
    tmp_model = deepcopy(model)

    for i, _ in enumerate(tmp_model.bert.encoder.layer):
        model.bert.encoder.layer[i].attention.self = PosBiasBertSelfAttention(config)

        _copy_self_attn(
            self_attn=model.bert.encoder.layer[i].attention.self,
            target_self_attn=tmp_model.bert.encoder.layer[i].attention.self
        )
    return model


def freeze_weights(model: BertModel):
    for param in model.parameters():
        param.requires_grad = False
    for i, _ in enumerate(model.bert.encoder.layer):
        for param in model.bert.encoder.layer[i].attention.parameters():
            param.requires_grad = True
    return model


def construct_model(
    is_linear: bool = False,
    feature_map: str = "elu",
    pos_bias_type: str = None,
    shape: int = None
):
    if pos_bias_type is not None:
        model_config = ModelConfig(
            is_linear=is_linear,
            feature_map=feature_map,
            pos_bias_type=pos_bias_type,
            bias_base_type="full"
        )
    else:
        model_config = ModelConfig(
            is_linear=is_linear,
            feature_map=feature_map,
        )
    data_path = "data"
    dataset_name = "mnist"
    seed = 9
    set_seed_(seed)

    output_path = join(data_path, dataset_name)
    vocab_path = join(data_path, dataset_name)
    tokenizer = BertTokenizerFast.from_pretrained(vocab_path)
    vocab_size = tokenizer.vocab_size
    config, args = mnist_config(
        dataset_name=dataset_name,
        output_path=output_path,
        seed=seed,
        is_test=False,
        x_shape=shape,
        y_shape=shape,
        n_channels=1,
        vocab_size=vocab_size,
        model_config=model_config
    )

    config.hidden_size = 256
    config.num_layers = 6
    model = Classifier(config=config)
    inputs = torch.LongTensor(6, config.max_position_embeddings).random_(0, vocab_size)
    return model, inputs
