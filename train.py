from argparse import ArgumentParser
from os.path import join

from transformers import BertTokenizerFast
from transformers import Trainer

from configs import yelp_config, pf_config, ModelConfig
from dataset import get_dataset
from models import Classifier
from utils import set_seed_, compute_metrics

data_path = "data"


def train(
        dataset_name: str,
        seed: int,
        is_test: bool,
        model_config: ModelConfig,
        max_seq_len: int = None,
        x_shape: int = None,
        y_shape: int = None,
):
    set_seed_(seed)

    output_path = join(data_path, dataset_name)

    if dataset_name in ["yelp_polarity", "yelp_full"]:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        vocab_size = 52_000
        config, training_args = yelp_config(
            dataset_name=dataset_name,
            output_path=output_path,
            seed=seed,
            is_test=is_test,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            model_config=model_config
        )
    elif dataset_name in ["pf_6_full", "pf_9_full", "pf_14_full", "mnist"]:
        vocab_path = join(data_path, dataset_name) + ("_small" if is_test else "")
        tokenizer = BertTokenizerFast.from_pretrained(vocab_path)
        vocab_size = tokenizer.vocab_size
        if dataset_name == "mnist":
            x_shape = 28
            y_shape = 28
        config, training_args = pf_config(
            dataset_name=dataset_name,
            output_path=output_path,
            seed=seed,
            is_test=is_test,
            x_shape=x_shape,
            y_shape=y_shape,
            vocab_size=vocab_size,
            model_config=model_config
        )
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
    arg_parser.add_argument("--has_pos_embed_2d", action='store_true')
    arg_parser.add_argument("--is_linear", action='store_true')
    arg_parser.add_argument("--has_batch_norm", action='store_true')
    arg_parser.add_argument("--has_pos_attention", action='store_true')
    arg_parser.add_argument("--feature_map", choices=["elu", "relu"], default="elu")
    arg_parser.add_argument("--pos_bias_type", choices=["fft", "naive", "orig"], default=None)
    args = arg_parser.parse_args()

    model_config = ModelConfig(
        is_linear=args.is_linear,
        has_batch_norm=args.has_batch_norm,
        has_pos_attention=args.has_pos_attention,
        has_pos_embed_2d=args.has_pos_embed_2d,
        feature_map=args.feature_map,
        pos_bias_type=args.pos_bias_type,
    )

    train(
        dataset_name=args.dataset,
        seed=args.seed,
        is_test=args.test,
        max_seq_len=args.max_seq_len,
        x_shape=args.x_shape,
        y_shape=args.y_shape,
        model_config=model_config
    )
