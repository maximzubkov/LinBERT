import random
from argparse import ArgumentParser

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction
from transformers.trainer_utils import set_seed

from configs import ModelConfig
from .img_gpt import unquantize, quantize, squared_euclidean_distance  # noqa


def set_seed_(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(preds, labels)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    accuracy = accuracy_score(labels, preds)
    return {
        "f1": f1,
        "recall": recall,
        "accuracy": accuracy,
        "precision": precision,
    }


def parse_model_config(arg_parser: ArgumentParser) -> ModelConfig:
    arg_parser.add_argument("--has_pos_embed_2d", action='store_true')
    arg_parser.add_argument("--is_linear", action='store_true')
    arg_parser.add_argument("--has_batch_norm", action='store_true')
    arg_parser.add_argument("--has_pos_attention", action='store_true')
    arg_parser.add_argument("--feature_map", choices=["elu", "relu", "exp", "favor", "dpfp"], default="elu")
    arg_parser.add_argument("--pos_bias_type", choices=["fft", "naive", "orig", "fft_2d", "naive_2d"], default=None)
    arg_parser.add_argument("--bias_base_type", choices=["full", "symmetric"], default=None)
    args = arg_parser.parse_args()

    return ModelConfig(
        is_linear=args.is_linear,
        has_batch_norm=args.has_batch_norm,
        has_pos_attention=args.has_pos_attention,
        has_pos_embed_2d=args.has_pos_embed_2d,
        feature_map=args.feature_map,
        pos_bias_type=args.pos_bias_type,
        bias_base_type=args.bias_base_type
    )
