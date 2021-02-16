from typing import Tuple

from transformers import BertConfig, TrainingArguments

from dataset import dataset_config
from .model_config import ModelConfig


def vit_config(
        dataset_name: str,
        output_path: str,
        seed: int,
        is_test: bool,
        model_config: ModelConfig,
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
            learning_rate=0.003,
            save_total_limit=2,
        )

        config = BertConfig(
            image_size=28,
            patch_size=7,
            max_position_embeddings=17 + 2,
            num_attention_heads=2,
            intermediate_size=32,
            num_hidden_layers=2,
            hidden_size=16,
            type_vocab_size=1,
            num_labels=dataset_config[dataset_name]["num_labels"],
            channels=1,
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
            evaluation_strategy="steps",
            save_steps=10_000,
            seed=seed,
            do_train=True,
            do_eval=True,
            eval_steps=200,
            logging_steps=50,
            learning_rate=0.003,
            save_total_limit=2,
        )

        config = BertConfig(
            image_size=28,
            patch_size=7,
            max_position_embeddings=17 + 2,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=4,
            intermediate_size=128,
            type_vocab_size=1,
            num_labels=dataset_config[dataset_name]["num_labels"],
            channels=1,
            return_attention_mask=True,
            return_token_type_ids=False,
            **model_config.__dict__
        )
    return config, training_args
