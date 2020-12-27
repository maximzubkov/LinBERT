from os.path import join

from datasets import load_dataset
from .common import dataset_config
from transformers import PreTrainedTokenizerFast


data_path = "data"


def get_yelp_dataset(
        name: str,
        split: str,
        max_length: int,
        tokenizer: PreTrainedTokenizerFast,
        cache_dir: str,
        is_test: bool
):
    split = f"{split}_small" if is_test else split
    path = join(data_path, name, f"{split}.csv")
    dataset = load_dataset("csv", data_files=[path], cache_dir=cache_dir)["train"]
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
