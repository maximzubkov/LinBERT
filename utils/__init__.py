import random
from os.path import join

import numpy as np
from datasets import load_dataset, list_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction
from transformers import PreTrainedTokenizerFast
from transformers.trainer_utils import set_seed

data_path = "data"

num_classes = {
    "yelp_full": 5,
    "yelp_polarity": 2,
}


def set_seed_(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)
    return {
        "f1": f1,
        "recall": recall,
        "accuracy": accuracy,
        "precision": precision,
    }


def get_classification_dataset(name: str, split: str, max_length: int, tokenizer: PreTrainedTokenizerFast):
    if name in list_datasets():
        dataset = load_dataset(name, split=split)
    else:
        path = join(data_path, name, f"{split}.csv")
        dataset = load_dataset("csv", data_files=[path])["train"]
    dataset = dataset.map(
        lambda e: tokenizer(e["text"],  max_length=max_length, truncation=True, padding="max_length"),
        batched=True
    )
    dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    return dataset
