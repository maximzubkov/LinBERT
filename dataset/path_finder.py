from os.path import join
from os import listdir
import numpy as np

from datasets import load_dataset
from .common import dataset_config
from transformers import BertTokenizerFast


data_path = "data"


def get_path_finder_dataset(
        name: str,
        split: str,
        cache_dir: str,
        is_test: bool,
        tokenizer: BertTokenizerFast
):
    split = f"{split}_small" if is_test else split
    path = join(data_path, name, split, "csv")
    files = [join(path, file) for file in listdir(path)]

    dataset = load_dataset("csv", data_files=files, cache_dir=cache_dir)["train"]
    dataset = dataset.map(
        lambda e: tokenizer(e["text"], max_length=300 * 300, truncation=True, padding="max_length"),
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
