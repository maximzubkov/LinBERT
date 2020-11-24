import random
from os.path import join

import numpy as np
from transformers import BertTokenizerFast
from transformers import EvalPrediction
from transformers.trainer_utils import set_seed
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import load_dataset, datasets

data_path = "data"


def get_dataset(
        dataset_name: str,
        experiment_name: str,
        type: str,
        tokenizer: BertTokenizerFast,
        seed: int
):
    return load_dataset(
        experiment_name=experiment_name,
        dataset_config=datasets[dataset_name],
        dataset_path=join(data_path, dataset_name, type + ".csv"),
        tokenizer=tokenizer,
        seed=seed
    )


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
