import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class ClassificationDataset(Dataset):
    def __init__(
        self,
        path: str,
        columns: dict,
        tokenizer: BertTokenizer,
        seed: int,
        names: list = None,
        max_length: int = 128,
    ):
        self.data = pd.read_csv(path, names=names)
        self.length = max_length
        self.tokenizer = tokenizer

        random.seed(seed)
        indices = list(range(len(self.data[columns["feature"]])))
        random.shuffle(indices)

        self.texts = self.data[columns["feature"]][indices].tolist()
        targets = self.data[columns["target"]][indices]

        self.target_map = {c: i for i, c in enumerate(set(targets))}
        self.labels = targets.apply(lambda x: self.target_map[x]).values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        buffer = self.tokenizer(
            self.texts[idx],
            max_length=self.length,
            truncation=True,
            padding="max_length",
        )
        item = {key: torch.tensor(val) for key, val in buffer.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
