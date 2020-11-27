from argparse import ArgumentParser
from os.path import join

from transformers import BertTokenizerFast
from transformers import Trainer

from configs import configure_bert_training
from models import LinBertForSequenceClassification, PosAttnBertForSequenceClassification
from utils import set_seed_, compute_metrics, get_classification_dataset, num_classes

data_path = "data"


def train(
        run_name: str,
        dataset_name: str,
        seed: int,
        is_test: bool,
        is_linear: bool,
        has_batch_norm: bool,
        has_pos_attention: bool
):
    set_seed_(seed)

    output_path = join(data_path, dataset_name)
    config, training_args = configure_bert_training(
        output_path,
        seed=seed,
        is_test=is_test,
        run_name=run_name,
        has_batch_norm=has_batch_norm,
        has_pos_attention=has_pos_attention,
        num_labels=num_classes[dataset_name]
    )

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    if is_linear:
        model = LinBertForSequenceClassification(config=config)
    else:
        model = PosAttnBertForSequenceClassification(config=config)

    train_dataset = get_classification_dataset(
        dataset_name,
        split="train_small" if is_test else "train",
        max_length=config.max_position_embeddings,
        tokenizer=tokenizer
    )

    eval_dataset = get_classification_dataset(
        dataset_name,
        split="test_small" if is_test else "test",
        max_length=config.max_position_embeddings,
        tokenizer=tokenizer
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
    arg_parser.add_argument("--run_name", type=str)
    arg_parser.add_argument("--test", action="store_true")
    arg_parser.add_argument("--seed", type=int, default=9)
    arg_parser.add_argument("--resume", type=str, default=None)
    arg_parser.add_argument("--is_linear", action='store_true')
    arg_parser.add_argument("--has_batch_norm", action='store_true')
    arg_parser.add_argument("--has_pos_attention", action='store_true')
    args = arg_parser.parse_args()
    train(
        args.run_name,
        args.dataset,
        args.seed,
        args.test,
        args.is_linear,
        args.has_batch_norm,
        args.has_pos_attention
    )
