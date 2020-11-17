from typing import Tuple

from transformers import BertConfig, TrainingArguments

data_path = "data"
models_path = "models"

SEED = 9


def configure_bert_training(output_path: str, is_test: bool,
                            has_pos_attention: bool,
                            has_batch_norm: bool) -> Tuple[BertConfig, TrainingArguments]:
    if is_test:
        training_args = TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=5,
            per_device_eval_batch_size=5,
            evaluation_strategy="steps",
            save_steps=10_000,
            seed=SEED,
            do_train=True,
            do_eval=True,
            eval_steps=50,
            logging_steps=50,
            save_total_limit=2
        )

        config = BertConfig(
            vocab_size=52_000,
            max_position_embeddings=1024,
            num_attention_heads=2,
            num_hidden_layers=2,
            type_vocab_size=1,
            has_pos_attention=has_pos_attention,
            has_batch_norm=has_batch_norm
        )
    else:
        training_args = TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            evaluation_strategy="steps",
            save_steps=10_000,
            seed=SEED,
            do_train=True,
            do_eval=True,
            eval_steps=50,
            logging_steps=50,
            save_total_limit=2,
        )

        config = BertConfig(
            vocab_size=52_000,
            max_position_embeddings=1024,
            num_attention_heads=6,
            num_hidden_layers=2,
            type_vocab_size=1,
            has_pos_attention=has_pos_attention,
            has_batch_norm=has_batch_norm
        )

    return config, training_args
