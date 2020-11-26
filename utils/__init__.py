import random
from os.path import join

import numpy as np
from datasets import load_dataset, list_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction
from transformers import PreTrainedTokenizerFast
from transformers.trainer_utils import set_seed

data_path = "data"

dataset_config = {
    "yelp_full": {
        "columns": {"type": "yelp_full", "feature": "text", "target": "class"},
        "names": ["class", "text"],
        "num_labels": 5,
        "max_length": 3072,
    },
    "yelp_polarity": {
        "columns": {"type": "yelp_full", "feature": "text", "target": "class"},
        "names": ["class", "text"],
        "num_labels": 2,
        "max_length": 3072,
    },
}


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


def get_classification_dataset(name: str, split: str, tokenizer: PreTrainedTokenizerFast):
    if name in list_datasets():
        dataset = load_dataset(name, split=split)
    else:
        path = join(data_path, name, f"{split}.csv")
        dataset = load_dataset("csv", data_files=[path], autogenerate_column_names=True)["train"]
    dataset = dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, padding='max_length'),
        batched=True
    )
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    return dataset
