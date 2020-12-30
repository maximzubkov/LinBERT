from os.path import join
from os import listdir

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

data_path = "data"

dataset_config = {
    "yelp_full": {
        "num_labels": 5,
        "labels": [1, 2, 3, 4, 5]
    },
    "yelp_polarity": {
        "num_labels": 2,
        "labels": [1, 2]
    },
    "pf_6_full": {
        "num_labels": 2,
        "labels": [1, 2],
        "im_size": [300, 300]
    }
}


def get_dataset(
        name: str,
        split: str,
        max_length: int,
        cache_dir: str,
        is_test: bool,
        tokenizer: PreTrainedTokenizerFast,
):
    if name in ["yelp_full", "yelp_polarity"]:
        split = f"{split}_small" if is_test else split
        paths = [join(data_path, name, f"{split}.csv")]
        batch_size = 1000
    elif name in ["pf_6_full"]:
        name = f"{name}_small" if is_test else name
        path = join(data_path, name, split, "csv")
        paths = [join(path, file) for file in listdir(path)]
        batch_size = 100
        max_length = 300 * 300
    else:
        raise ValueError("Unknown Dataset")

    return _hf_dataset(
        name=name,
        paths=paths,
        max_length=max_length,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        batch_size=batch_size
    )


def _hf_dataset(
        name: str,
        paths: list,
        max_length: int,
        tokenizer: PreTrainedTokenizerFast,
        cache_dir: str,
        batch_size: int = 1000
):
    dataset = load_dataset("csv", data_files=paths, cache_dir=cache_dir)["train"]
    dataset = dataset.map(
        lambda e: tokenizer(e["text"], max_length=max_length, truncation=True, padding="max_length"),
        batched=True, batch_size=batch_size
    )

    label2idx = {label: idx for idx, label in enumerate(dataset_config[name]["labels"])}

    def _update(e):
        e.update({"label": [label2idx[label] for label in e["label"]]})
        return e

    dataset = dataset.map(
        _update,
        batched=True
    )
    dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    return label2idx, dataset
