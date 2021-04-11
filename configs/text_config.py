from typing import Tuple

from transformers import BertConfig, TrainingArguments

from dataset import dataset_config
from .model_config import ModelConfig


def yelp_config(
        dataset_name: str,
        output_path: str,
        seed: int,
        max_seq_len: int,
        is_test: bool,
        model_config: ModelConfig,
        vocab_size: int,
        lr: float = 1e-4,
        run_name: str = "default_yelp",
) -> Tuple[BertConfig, TrainingArguments]:
    if is_test:
        training_args = TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=5,
            per_device_eval_batch_size=5,
            load_best_model_at_end=True,
            evaluation_strategy="steps",
            save_steps=10_000,
            seed=seed,
            do_train=True,
            do_eval=True,
            eval_steps=50,
            logging_steps=50,
            learning_rate=lr,
            save_total_limit=2,
            run_name=run_name
        )

        config = BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_seq_len,
            num_attention_heads=2,
            num_hidden_layers=2,
            hidden_size=12,
            type_vocab_size=1,
            num_labels=dataset_config[dataset_name]["num_labels"],
            return_attention_mask=True,
            return_token_type_ids=False,
            **model_config.__dict__
        )
    else:
        training_args = TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            load_best_model_at_end=True,
            evaluation_strategy="steps",
            save_steps=10_000,
            seed=seed,
            do_train=True,
            do_eval=True,
            eval_steps=200,
            logging_steps=50,
            learning_rate=lr,
            save_total_limit=2,
            run_name=run_name
        )

        config = BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_seq_len,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            type_vocab_size=1,
            num_labels=dataset_config[dataset_name]["num_labels"],
            return_attention_mask=True,
            return_token_type_ids=False,
            **model_config.__dict__
        )

    return config, training_args
