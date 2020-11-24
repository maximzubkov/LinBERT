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
        names=None,
        max_length=128,
    ):
        self.data = pd.read_csv(path, names=names)
        self.length = max_length

        random.seed(seed)
        indices = list(range(len(self.data[columns["feature"]])))
        random.shuffle(indices)

        texts = self.data[columns["feature"]][indices].tolist()
        targets = self.data[columns["target"]][indices]

        target_map = {c: i for i, c in enumerate(set(targets))}
        self.labels = targets.apply(lambda x: target_map[x]).values

        self.feature = tokenizer(
            texts,
            max_length=self.length,
            truncation=True,
            padding=True,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.feature.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
