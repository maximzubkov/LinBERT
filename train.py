from argparse import ArgumentParser
from os.path import join

import wandb
from transformers import BertTokenizerFast
from transformers import Trainer

from configs import configure_bert_training
from models import LinBertForSequenceClassification, PosAttnBertForSequenceClassification
from utils import get_dataset, set_seed_, compute_metrics

data_path = "data"


def train(
        project_name: str,
        dataset_name: str,
        seed: int,
        is_test: bool,
        is_linear: bool,
        has_batch_norm: bool,
        has_pos_attention: bool
):
    set_seed_(seed)

    wandb.init(project=project_name)
    output_path = join(data_path, dataset_name)
    config, training_args = configure_bert_training(
        output_path,
        seed=seed,
        is_test=is_test,
        has_batch_norm=has_batch_norm,
        has_pos_attention=has_pos_attention
    )

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    if is_linear:
        model = LinBertForSequenceClassification(config=config)
    else:
        model = PosAttnBertForSequenceClassification(config=config)

    experiment_name = dataset_name + "_" + \
        ("lin_" if is_linear else "") + \
        ("pos_" if has_pos_attention else "") + \
        ("bn_" if has_batch_norm else "")

    train_dataset = get_dataset(
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        type="train_small" if is_test else "train",
        tokenizer=tokenizer,
        seed=seed
    )

    eval_dataset = get_dataset(
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        type="test_small" if is_test else "test",
        tokenizer=tokenizer,
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset", choices=["yelp_polarity", "yelp_full"])
    arg_parser.add_argument("--project_name", type=str)
    arg_parser.add_argument("--test", action="store_true")
    arg_parser.add_argument("--seed", type=int, default=9)
    arg_parser.add_argument("--resume", type=str, default=None)
    arg_parser.add_argument("--is_linear", action='store_true')
    arg_parser.add_argument("--has_batch_norm", action='store_true')
    arg_parser.add_argument("--has_pos_attention", action='store_true')
    args = arg_parser.parse_args()
    train(
        args.project_name,
        args.dataset,
        args.seed,
        args.test,
        args.is_linear,
        args.has_batch_norm,
        args.has_pos_attention
    )
