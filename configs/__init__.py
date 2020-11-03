from typing import Tuple
from transformers import BertConfig, TrainingArguments

data_path = "data"
models_path = "models"

SEED = 9


def configure_bert_training(output_path: str, is_test: bool) -> Tuple[BertConfig, TrainingArguments]:
    if is_test:
        training_args = TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=True,
            num_train_epochs=10,
            per_device_train_batch_size=5,
            save_steps=10_000,
            seed=SEED,
            save_total_limit=2
        )

        config = BertConfig(
            vocab_size=2_000,
            max_position_embeddings=512,
            num_attention_heads=2,
            num_hidden_layers=2,
            type_vocab_size=1,
        )
    else:
        training_args = TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=64,
            save_steps=10_000,
            seed=SEED,
            save_total_limit=2,
        )

        config = BertConfig(
            vocab_size=52_000,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )

    return config, training_args
