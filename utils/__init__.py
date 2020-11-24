import random

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction
from transformers.trainer_utils import set_seed

data_path = "data"


def set_seed_(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
