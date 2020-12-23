from typing import Tuple

from transformers import BertConfig, TrainingArguments

data_path = "data"
models_path = "models"


def configure_bert_training(
        output_path: str,
        num_labels: int,
        seed: int,
        max_seq_len: int,
        is_test: bool,
        has_pos_attention: bool,
        has_batch_norm: bool,
        run_name: str = "default_yelp",
        feature_map: str = "elu",
        pos_bias_type: str = None
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
            max_position_embeddings=max_seq_len,
            num_attention_heads=2,
            num_hidden_layers=2,
            hidden_size=12,
            type_vocab_size=1,
            num_labels=num_labels,
            has_pos_attention=has_pos_attention,
            has_batch_norm=has_batch_norm,
            feature_map=feature_map,
            pos_bias_type=pos_bias_type
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
            max_position_embeddings=max_seq_len,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            type_vocab_size=1,
            num_labels=num_labels,
            has_pos_attention=has_pos_attention,
            has_batch_norm=has_batch_norm,
            feature_map=feature_map,
            pos_bias_type=pos_bias_type
        )

    return config, training_args
