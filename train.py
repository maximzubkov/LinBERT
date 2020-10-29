from os import mkdir
from os.path import join, exists

import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import BertConfig
from transformers import RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from fast_transformers import LinBertForMaskedLM

data_path = "data"
models_path = "models"

SEED = 9


def build_tokenizer(paths: list, output_path: str):
    vocab_path = join(output_path, "vocab.json")
    merges_path = join(output_path, "merges.txt")
    if not exists(vocab_path) or not exists(merges_path):
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=paths, vocab_size=2_000, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])

        mkdir(output_path)
        tokenizer.save_model(output_path)
    return RobertaTokenizerFast.from_pretrained(output_path, max_len=32)


def train():
    if not exists(models_path):
        mkdir(models_path)

    output_path = join(models_path, "EsperBERTo")
    file_path = join(data_path, "oscar_small.eo.txt")
    paths = [file_path]
    tokenizer = build_tokenizer(paths=paths, output_path=output_path)

    torch.cuda.is_available()

    config = BertConfig(
        vocab_size=2_000,
        max_position_embeddings=512,
        num_attention_heads=2,
        num_hidden_layers=2,
        type_vocab_size=1,
    )

    model = LinBertForMaskedLM(config=config)

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_gpu_train_batch_size=5,
        save_steps=10_000,
        seed=SEED,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )

    trainer.train()
    trainer.save_model(output_path)


if __name__ == "__main__":
    train()
