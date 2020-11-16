from argparse import ArgumentParser
from os import mkdir
from os.path import join, exists

from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import RobertaTokenizer
from transformers import Trainer

from configs import configure_bert_training
from models import LinBertForMaskedLM, PosAttnBertForMaskedLM

data_path = "data"
save_path = "data"

SEED = 9


def build_tokenizer(paths: list, output_path: str, vocab_size: int):
    vocab_path = join(output_path, "vocab.json")
    merges_path = join(output_path, "merges.txt")
    if not exists(vocab_path) or not exists(merges_path):
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])

        mkdir(output_path)
        tokenizer.save_model(output_path)
    return RobertaTokenizer.from_pretrained(output_path, max_len=128)


def train(is_test: bool, is_linear: bool, has_batch_norm: bool, has_pos_attention: bool):
    if not exists(save_path):
        mkdir(save_path)

    output_path = join(save_path, "EsperBERTo")
    config, training_args = configure_bert_training(
        output_path,
        is_test=is_test,
        has_batch_norm=has_batch_norm,
        has_pos_attention=has_pos_attention
    )

    file_path = join(data_path, "oscar_small.eo.txt" if is_test else "oscar.eo.txt")
    paths = [file_path]
    tokenizer = build_tokenizer(paths=paths, output_path=output_path, vocab_size=config.vocab_size)

    if is_linear:
        model = LinBertForMaskedLM(config=config)
    else:
        model = PosAttnBertForMaskedLM(config=config)

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
    arg_parser.add_argument("--is_linear", action='store_true')
    arg_parser.add_argument("--has_batch_norm", action='store_true')
    arg_parser.add_argument("--has_pos_attention", action='store_true')
    args = arg_parser.parse_args()
    train(args.test, args.is_linear, args.has_batch_norm, args.has_pos_attention)
