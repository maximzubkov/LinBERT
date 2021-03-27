from argparse import ArgumentParser
from os.path import join

from transformers import BertTokenizerFast
from transformers import Trainer, DataCollatorWithPadding
from transformers import EarlyStoppingCallback

from configs import yelp_config, pf_config, ModelConfig
from dataset import get_dataset
from models import Classifier
from utils import set_seed_, compute_metrics, parse_model_config

data_path = "data"

img_datasets = ["pf_6_full", "pf_9_full", "pf_14_full", "mnist"]
text_datasets = ["yelp_polarity", "yelp_full"]


def train(
        dataset_name: str,
        seed: int,
        is_test: bool,
        model_config: ModelConfig,
        max_seq_len: int = None,
        x_shape: int = None,
        y_shape: int = None,
        lr: float = 1e-4
):
    set_seed_(seed)

    output_path = join(data_path, dataset_name)

    if dataset_name in text_datasets:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        vocab_size = 52_000
        config, training_args = yelp_config(
            dataset_name=dataset_name,
            output_path=output_path,
            seed=seed,
            is_test=is_test,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            lr=lr,
            model_config=model_config
        )
    elif dataset_name in img_datasets:
        vocab_path = join(data_path, dataset_name) + ("_small" if is_test else "")
        tokenizer = BertTokenizerFast.from_pretrained(vocab_path)
        vocab_size = tokenizer.vocab_size
        config, training_args = pf_config(
            dataset_name=dataset_name,
            output_path=output_path,
            seed=seed,
            is_test=is_test,
            x_shape=x_shape,
            y_shape=y_shape,
            vocab_size=vocab_size,
            lr=lr,
            model_config=model_config
        )
    else:
        raise ValueError("Unknown dataset")

    model = Classifier(config=config)

    _, train_dataset = get_dataset(
        dataset_name,
        split="train",
        is_test=is_test,
        seed=seed,
        cache_dir=data_path,
        config=config,
        tokenizer=tokenizer,
    )

    _, eval_dataset = get_dataset(
        dataset_name,
        split="test",
        is_test=is_test,
        seed=seed,
        cache_dir=data_path,
        config=config,
        tokenizer=tokenizer,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback()
        ],
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            max_length=max_seq_len
        )
    )

    trainer.train()
    trainer.save_model(output_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--dataset", choices=["pf_6_full", "pf_9_full", "pf_14_full"] + ["yelp_polarity", "yelp_full"] + ["mnist"]
    )
    arg_parser.add_argument("--test", action="store_true")
    arg_parser.add_argument("--seed", type=int, default=9)
    arg_parser.add_argument("--resume", type=str, default=None)
    arg_parser.add_argument("--max_seq_len", type=int, default=None)
    arg_parser.add_argument("--x_shape", type=int, default=None)
    arg_parser.add_argument("--y_shape", type=int, default=None)
    arg_parser.add_argument("--learning_rate", type=float, default=1e-4)

    model_config = parse_model_config(arg_parser)
    args = arg_parser.parse_args()

    train(
        dataset_name=args.dataset,
        seed=args.seed,
        is_test=args.test,
        max_seq_len=args.max_seq_len,
        x_shape=args.x_shape,
        y_shape=args.y_shape,
        lr=args.learning_rate,
        model_config=model_config,
    )
