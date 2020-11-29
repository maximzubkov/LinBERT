import random
from os.path import join

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction
from transformers import PreTrainedTokenizerFast
from transformers.trainer_utils import set_seed

data_path = "data"

dataset_config = {
    "yelp_full": {
        "num_labels": 5,
        "labels": [1, 2, 3, 4, 5]
    },
    "yelp_polarity": {
        "num_labels": 2,
        "labels": [1, 2]
    }
}


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


def get_classification_dataset(
        name: str,
        split: str,
        max_length: int,
        tokenizer: PreTrainedTokenizerFast,
        is_test: bool
):
    path = join(data_path, name, f"{split}.csv")
    dataset = load_dataset("csv", data_files=[path])["train"]
    dataset = dataset.map(
        lambda e: tokenizer(e["text"],  max_length=max_length, truncation=True, padding="max_length"),
        batched=True
    )
    label2idx = {
        label: idx for idx, label in enumerate(dataset_config[name]["labels"])
    }

    def _update(e):
        e.update({"label": [label2idx[label] for label in e["label"]]})
        return e

    dataset = dataset.map(
        _update,
        batched=True
    )
    dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    return label2idx, dataset
