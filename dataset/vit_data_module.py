from os.path import join
from typing import Any

import torch
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from transformers import TrainingArguments
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

text_datasets = ["poj_104"]

data_path = "data"


class ViTDataModule(LightningDataModule):
    def __init__(self, dataset_name: str, training_args: TrainingArguments, is_test: bool = False):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_path = join("data", dataset_name)
        self.training_args = training_args
        self.is_test = is_test

    def prepare_data(self):
        if self.dataset_name == "mnist":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
            self.train_dataset = MNIST(root=self.dataset_path, train=True, download=True, transform=transform)
            self.eval_dataset = MNIST(root=self.dataset_path, train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_args.train_batch_size,
            collate_fn=self._collate,
            num_workers=10
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.training_args.eval_batch_size,
            collate_fn=self._collate,
            num_workers=10
        )

    def test_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.training_args.eval_batch_size,
            collate_fn=self._collate,
            num_workers=10
        )

    @staticmethod
    def _collate(batch):
        # batch contains a list of tuples of structure (sequence, target)
        imgs = torch.stack([item[0] for item in batch])
        labels = torch.LongTensor([item[1] for item in batch])
        return imgs, labels

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        return imgs, labels
