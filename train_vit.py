from argparse import ArgumentParser

import torch
import vit_pytorch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from configs import ModelConfig, vit_config
from dataset import ViTDataModule
from models.vit import ViTModel
from utils import set_seed_, parse_model_config

vit_pytorch.vit_pytorch.MIN_NUM_PATCHES = 7

data_path = "data"

img_datasets = ["mnist"]

MIN_NUM_PATCHES = 7


def train(
    dataset_name: str,
    seed: int,
    is_test: bool,
    model_config: ModelConfig,
):
    set_seed_(seed)

    if dataset_name == "mnist":
        config, training_args = vit_config(
            dataset_name=dataset_name,
            seed=seed,
            is_test=is_test,
            model_config=model_config
        )
    else:
        raise ValueError("Unknown dataset")

    model = ViTModel(config=config, training_args=training_args)
    dm = ViTDataModule(dataset_name=dataset_name, training_args=training_args, is_test=is_test)

    # define logger
    project = "vit_test" if is_test else "vit"
    wandb_logger = WandbLogger(
        project=project, log_model=False, offline=is_test
    )
    # define early stopping callback
    early_stopping_callback = EarlyStopping(patience=10, monitor="val_loss", verbose=True, mode="min")
    # define learning rate logger
    lr_logger = LearningRateMonitor("step")
    gpu = -1 if torch.cuda.is_available() else None
    accelerator = "ddp" if torch.cuda.device_count() else None
    trainer = Trainer(
        max_epochs=training_args.num_train_epochs,
        check_val_every_n_epoch=training_args.val_every_epoch,
        accelerator=accelerator,
        logger=wandb_logger,
        gpus=gpu,
        progress_bar_refresh_rate=20,
        callbacks=[
            lr_logger,
            early_stopping_callback,
        ],
    )

    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset", choices=["mnist"])
    arg_parser.add_argument("--test", action="store_true")
    arg_parser.add_argument("--seed", type=int, default=9)
    arg_parser.add_argument("--resume", type=str, default=None)

    model_config = parse_model_config(arg_parser)
    args = arg_parser.parse_args()

    train(
        dataset_name=args.dataset,
        seed=args.seed,
        is_test=args.test,
        model_config=model_config
    )
