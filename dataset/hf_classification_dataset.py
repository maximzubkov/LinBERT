from os.path import join

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, BertConfig

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
    "mnist": {
        "num_labels": 10,
        "labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    "mnist_small": {
        "num_labels": 10,
        "labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
}


def get_dataset(
        name: str,
        split: str,
        cache_dir: str,
        seed: int,
        is_test: bool,
        config: BertConfig,
        tokenizer: PreTrainedTokenizerFast,
):
    if name in ["yelp_full", "yelp_polarity"]:
        split = f"{split}_small" if is_test else split
        paths = [join(data_path, name, f"{split}.csv")]
        batch_size = 1000
    elif name in ["mnist"]:
        name = f"{name}_small" if is_test else name
        paths = [join(data_path, name, f"{split}.csv")]
        batch_size = 100
    else:
        raise ValueError("Unknown Dataset")

    return _hf_dataset(
        name=name,
        paths=paths,
        cache_dir=cache_dir,
        seed=seed,
        tokenizer=tokenizer,
        batch_size=batch_size,
        return_attention_mask=config.return_attention_mask,
        return_token_type_ids=config.return_token_type_ids,
        max_length=config.max_position_embeddings,
    )


def _hf_dataset(
        name: str,
        paths: list,
        max_length: int,
        tokenizer: PreTrainedTokenizerFast,
        cache_dir: str,
        seed: int,
        batch_size: int = 1000,
        return_attention_mask: bool = False,
        return_token_type_ids: bool = False,
):
    dataset = load_dataset("csv", data_files=paths, cache_dir=cache_dir)["train"]

    label2idx = {label: idx for idx, label in enumerate(dataset_config[name]["labels"])}

    def _update(e):
        e_ = tokenizer(
            e["text"],
            max_length=max_length,
            truncation=True,
            padding=False,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids
        )
        e_.update({"label": [label2idx[label] for label in e["label"]]})
        return e_

    dataset = dataset.map(
        _update,
        batched=True,
        batch_size=batch_size,
        writer_batch_size=100000
    )
    columns = ["input_ids", "label"]

    if return_attention_mask:
        columns.append("attention_mask")
    if return_token_type_ids:
        columns.append("token_type_ids")

    dataset.set_format(type="torch", columns=columns)
    dataset = dataset.shuffle(seed=seed)
    return label2idx, dataset
