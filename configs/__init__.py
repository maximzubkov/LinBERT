from typing import Tuple

from transformers import BertConfig, TrainingArguments

data_path = "data"
models_path = "models"


def configure_bert_training(
        output_path: str,
        run_name: str,
        seed: int,
        is_test: bool,
        has_pos_attention: bool,
        has_batch_norm: bool,
) -> Tuple[BertConfig, TrainingArguments]:
    if is_test:
        training_args = TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=5,
            per_device_eval_batch_size=5,
            evaluation_strategy="steps",
            save_steps=10_000,
            seed=seed,
            do_train=True,
            do_eval=True,
            eval_steps=50,
            logging_steps=50,
            save_total_limit=2,
            run_name=run_name
        )

        config = BertConfig(
            vocab_size=52_000,
            max_position_embeddings=3072,
            num_attention_heads=2,
            num_hidden_layers=2,
            hidden_size=12,
            type_vocab_size=1,
            has_pos_attention=has_pos_attention,
            has_batch_norm=has_batch_norm
        )
    else:
        training_args = TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            evaluation_strategy="steps",
            save_steps=10_000,
            seed=seed,
            do_train=True,
            do_eval=True,
            eval_steps=200,
            logging_steps=50,
            save_total_limit=2,
            run_name=run_name
        )

        config = BertConfig(
            vocab_size=52_000,
            max_position_embeddings=3072,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            type_vocab_size=1,
            has_pos_attention=has_pos_attention,
            has_batch_norm=has_batch_norm,
        )

    return config, training_args
