import argparse

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from configs import ModelConfig
from dataset import dataloaders
from models import ImageGPT
from utils import parse_model_config


def train(args, model_config: ModelConfig):
    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)

    # experiment name
    project = f"{config['name']}_{args.dataset}"
    is_test = config['name'].startswith("test")

    if args.pretrained is not None:
        model = ImageGPT.load_from_checkpoint(args.pretrained)
        # potentially modify model for finetuning
        model.learning_rate = config["learning_rate"]
        model.classify = config["classify"]
    else:
        model = ImageGPT(centroids=args.centroids, model_config=model_config, **config)

    train_dl, valid_dl, test_dl = dataloaders(args.dataset, config["batch_size"])

    # define logger
    wandb_logger = WandbLogger(project=project, log_model=False, offline=is_test)
    # define early stopping callback
    early_stopping_callback = EarlyStopping(patience=10, monitor="val_loss", verbose=True, mode="min")
    # define learning rate logger
    lr_logger = LearningRateMonitor("step")
    gpu = -1 if torch.cuda.is_available() else None
    accelerator = "ddp" if torch.cuda.device_count() else None

    if config["classify"]:
        # classification
        # stop early for best validation accuracy for finetuning
        checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_acc")
        trainer = pl.Trainer(
            max_steps=config["steps"],
            gpus=gpu,
            logger=wandb_logger,
            accelerator=accelerator,
            progress_bar_refresh_rate=1,
            check_val_every_n_epoch=1,
            val_check_interval=1000,
            callbacks=[
                lr_logger,
                early_stopping_callback,
                checkpoint
            ],
        )

    else:
        # pretraining
        checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss")
        trainer = pl.Trainer(
            max_steps=config["steps"],
            gpus=gpu,
            logger=wandb_logger,
            accelerator=accelerator,
            progress_bar_refresh_rate=1,
            check_val_every_n_epoch=1,
            val_check_interval=1000,
            callbacks=[
                lr_logger,
                early_stopping_callback,
                checkpoint
            ],
        )

    trainer.fit(model, train_dl, valid_dl)


def test(args):
    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)
    trainer = pl.Trainer(gpus=config["gpus"])
    _, _, test_dl = dataloaders(args.dataset, config["batch_size"])
    model = ImageGPT.load_from_checkpoint(args.checkpoint)
    trainer.test(model, test_dataloaders=test_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="mnist")

    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--pretrained", type=str, default=None)
    parser_train.add_argument("config", type=str)
    parser_train.set_defaults(func=train)

    parser_test = subparsers.add_parser("test")
    parser_test.add_argument("checkpoint", type=str)
    parser_test.add_argument("config", type=str)
    parser_test.set_defaults(func=test)

    model_config_ = parse_model_config(parser)
    args = parser.parse_args()
    args.centroids = f"data/{args.dataset}_centroids.npy"

    args.func(args, model_config_)
