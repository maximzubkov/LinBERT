import random
from os.path import exists
from tqdm import tqdm

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
        buffer_size=16384,
    ):
        full_path = path.rsplit(".")[0]
        save_path = f"{full_path}_{type(tokenizer).__name__}_{max_length}_"
        labels_path = save_path + "labels.ds"

        self.data = pd.read_csv(path, names=names)
        self.length = max_length
        self.buffer_size = buffer_size

        if exists(labels_path):
            self.labels = torch.load(labels_path)
            idx = 0
            self.buffers = []
            while exists(save_path + f"buffer_{idx}.ds"):
                self.buffers.append(torch.load(save_path + f"buffer_{idx}.ds"))
                idx += 1
        else:
            random.seed(seed)
            indices = list(range(len(self.data[columns["feature"]])))
            random.shuffle(indices)

            texts = self.data[columns["feature"]][indices].tolist()
            targets = self.data[columns["target"]][indices]

            target_map = {c: i for i, c in enumerate(set(targets))}
            self.labels = targets.apply(lambda x: target_map[x]).values

            torch.save(self.labels, labels_path)

            self.buffers = []

            for buffer_idx, idx in tqdm(enumerate(range(0, len(texts), buffer_size)), total=len(texts) // buffer_size):
                buffer = tokenizer(
                    texts[idx: min(idx + buffer_size, len(texts))],
                    max_length=self.length,
                    truncation=True,
                    padding=True,
                )
                torch.save(buffer, save_path + f"buffer_{buffer_idx}.ds")
                self.buffers.append(buffer)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        buffer_idx, txt_idx_in_buffer = idx // self.buffer_size, idx % self.buffer_size
        item = {key: torch.tensor(val[txt_idx_in_buffer]) for key, val in self.buffers[buffer_idx].items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
