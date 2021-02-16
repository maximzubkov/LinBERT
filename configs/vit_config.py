from dataclasses import dataclass
from typing import Tuple

from transformers import BertConfig

from dataset import dataset_config
from .model_config import ModelConfig


@dataclass
class TrainingArguments:
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    seed: int
    num_train_epochs: int
    val_every_epoch: int
    save_every_epoch: int


def vit_config(
        dataset_name: str,
        seed: int,
        is_test: bool,
        model_config: ModelConfig,
) -> Tuple[BertConfig, TrainingArguments]:
    if is_test:
        training_args = TrainingArguments(
            num_train_epochs=2,
            train_batch_size=5,
            eval_batch_size=5,
            seed=seed,
            learning_rate=0.003,
            val_every_epoch=1,
            save_every_epoch=1,
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
            num_train_epochs=2,
            train_batch_size=32,
            eval_batch_size=32,
            seed=seed,
            learning_rate=0.003,
            save_every_epoch=1,
            val_every_epoch=1
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
