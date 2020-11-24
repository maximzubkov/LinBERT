import os
import pickle
import typing as tp

from transformers import BertTokenizer

from .classification_dataset import ClassificationDataset

config_type = tp.Optional[tp.Dict[str, tp.Dict[str, tp.List[str]]]]


def load_dataset(
    experiment_name: str,
    dataset_config: config_type,
    dataset_path: str,
    tokenizer: BertTokenizer,
    seed: int,
):
    dataset = None
    dataset_file_path = (
        dataset_path.rsplit(".")[0] + "_"
        + experiment_name
        + type(tokenizer).__name__
        + f"_{dataset_config['max_length']}_"
        + ".ds"
    )
    if os.path.exists(dataset_file_path):
        with open(dataset_file_path, "rb") as file:
            dataset = pickle.load(file)
    else:
        data = ClassificationDataset(
            dataset_path,
            dataset_config["columns"],
            tokenizer,
            seed=seed,
            names=dataset_config["names"],
            max_length=dataset_config["max_length"],
        )
        with open(dataset_file_path, "wb") as file:
            pickle.dump(data, file)
        dataset = data

    return dataset
