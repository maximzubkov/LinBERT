from os import mkdir
from argparse import ArgumentParser
from os.path import join, exists
from configs import configure_bert_training

from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import RobertaTokenizer
from transformers import Trainer
from fast_transformers import LinBertForMaskedLM

data_path = "data"
models_path = "models"

SEED = 9


def build_tokenizer(paths: list, output_path: str):
    vocab_path = join(output_path, "vocab.json")
    merges_path = join(output_path, "merges.txt")
    if not exists(vocab_path) or not exists(merges_path):
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])

        mkdir(output_path)
        tokenizer.save_model(output_path)
    return RobertaTokenizer.from_pretrained(output_path, max_len=512)


def train(is_test: bool):
    if not exists(models_path):
        mkdir(models_path)

    output_path = join(models_path, "EsperBERTo")
    file_path = join(data_path, "oscar_small.eo.txt" if is_test else "oscar.eo.txt")
    paths = [file_path]
    tokenizer = build_tokenizer(paths=paths, output_path=output_path)

    config, training_args = configure_bert_training(output_path, is_test)

    model = LinBertForMaskedLM(config=config)

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--test", action="store_true")
    arg_parser.add_argument("--resume", type=str, default=None)
    args = arg_parser.parse_args()
    train(args.test)
