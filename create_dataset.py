from argparse import ArgumentParser
from os.path import join

from datasets import datasets, ClassificationDataset
from transformers import BertTokenizerFast

from configs import configure_bert_training
from utils import set_seed_

data_path = "data"


def create_dataset(dataset_name: str, seed: int, is_test: bool):
    set_seed_(seed)

    output_path = join(data_path, dataset_name)
    config, _ = configure_bert_training(
        output_path,
        seed=seed,
        is_test=is_test,
        has_batch_norm=False,
        has_pos_attention=False
    )

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    dataset_config = datasets[dataset_name]

    ClassificationDataset(
        join(data_path, dataset_name, "train_small.csv" if is_test else "train.csv"),
        dataset_config["columns"],
        tokenizer=tokenizer,
        seed=seed,
        names=dataset_config["names"],
        max_length=dataset_config["max_length"],
    )

    ClassificationDataset(
        join(data_path, dataset_name, "test_small.csv" if is_test else "test.csv"),
        dataset_config["columns"],
        tokenizer=tokenizer,
        seed=seed,
        names=dataset_config["names"],
        max_length=dataset_config["max_length"],
    )


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset", choices=["yelp_polarity", "yelp_full"])
    arg_parser.add_argument("--test", action="store_true")
    arg_parser.add_argument("--seed", type=int, default=9)
    args = arg_parser.parse_args()
    create_dataset(args.dataset, args.seed, args.test)
