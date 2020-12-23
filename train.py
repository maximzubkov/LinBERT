from argparse import ArgumentParser
from os.path import join

from transformers import BertTokenizerFast
from transformers import Trainer

from configs import configure_bert_training
from models import LinBertForSequenceClassification, PosAttnBertForSequenceClassification
from utils import set_seed_, compute_metrics, get_classification_dataset, dataset_config

data_path = "data"


def train(
        run_name: str,
        dataset_name: str,
        seed: int,
        is_test: bool,
        max_seq_len: int,
        is_linear: bool,
        has_batch_norm: bool,
        has_pos_attention: bool,
        feature_map: str,
        pos_bias_type: str = None
):
    set_seed_(seed)

    output_path = join(data_path, dataset_name)
    config, training_args = configure_bert_training(
        output_path,
        seed=seed,
        is_test=is_test,
        max_seq_len=max_seq_len,
        run_name=run_name,
        has_batch_norm=has_batch_norm,
        has_pos_attention=has_pos_attention,
        feature_map=feature_map,
        num_labels=dataset_config[dataset_name]["num_labels"],
        pos_bias_type=pos_bias_type
    )

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    if is_linear:
        model = LinBertForSequenceClassification(config=config)
    else:
        model = PosAttnBertForSequenceClassification(config=config)

    _, train_dataset = get_classification_dataset(
        dataset_name,
        split="train_small" if is_test else "train",
        max_length=config.max_position_embeddings,
        tokenizer=tokenizer,
        is_test=is_test,
        cache_dir=data_path
    )

    _, eval_dataset = get_classification_dataset(
        dataset_name,
        split="test_small" if is_test else "test",
        max_length=config.max_position_embeddings,
        tokenizer=tokenizer,
        is_test=is_test,
        cache_dir=data_path
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
    arg_parser.add_argument("--max_seq_len", type=int, default=1024)
    arg_parser.add_argument("--is_linear", action='store_true')
    arg_parser.add_argument("--has_batch_norm", action='store_true')
    arg_parser.add_argument("--has_pos_attention", action='store_true')
    arg_parser.add_argument("--feature_map", choices=["elu", "relu"], default="elu")
    arg_parser.add_argument("--pos_bias_type", choices=["fft", "naive"], default=None)
    args = arg_parser.parse_args()
    train(
        run_name=args.run_name,
        dataset_name=args.dataset,
        seed=args.seed,
        is_test=args.test,
        max_seq_len=args.max_seq_len,
        is_linear=args.is_linear,
        has_batch_norm=args.has_batch_norm,
        has_pos_attention=args.has_pos_attention,
        feature_map=args.feature_map,
        pos_bias_type=args.pos_bias_type
    )
