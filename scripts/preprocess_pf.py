from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import join, isfile, exists

import numpy as np
import pandas as pd
from tqdm import tqdm

data_path = "data"


def create_vocab(dataset_path: str):
    with open(join(dataset_path, "vocab.txt"), "w") as f:
        tokens = [str(i) for i in range(256)] + ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        f.write("\n".join(tokens))


def preprocess(dataset: str):
    dataset_path = join(data_path, dataset)
    create_vocab(dataset_path)
    for split_ in ["train", "test"]:
        path = join(dataset_path, split_)
        csv_path = join(path, "csv")
        if not exists(csv_path):
            mkdir(csv_path)

        files = [file for file in listdir(path) if isfile(join(path, file)) and (file[-3:] == "npz")]
        for file in tqdm(files):
            file_path = join(path, file)
            if isfile(file_path) and (file_path[-3:] == "npz"):
                loaded = np.load(file_path)
                batch = loaded["images"].reshape(-1, 300 * 300).astype("int")
                texts = [" ".join([str(e) for e in img]) for img in batch]
                df = pd.DataFrame()
                df["text"] = texts
                df["label"] = loaded["labels"].reshape(-1)
                df.to_csv(join(csv_path, f"""{file.split(".")[0]}.csv"""), index=False)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset")
    args = arg_parser.parse_args()
    preprocess(dataset=args.dataset)
