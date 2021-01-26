import csv
from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import join, isfile, exists

import numpy as np
import pandas as pd
from skimage.transform import resize
from tqdm import tqdm

data_path = "data"


def preprocess_pf(dataset: str, shape: tuple = (100, 100)):
    dataset_path = join(data_path, dataset)
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
                df = pd.DataFrame()
                df["text"] = [
                    "\"" + " ".join([
                        str(e) for e in resize(img, shape, preserve_range=True).reshape(-1).astype(int)
                    ]) + "\"" for img in loaded["images"]
                ]
                df["label"] = ["\"" + str(label) + "\"" for label in loaded["labels"].reshape(-1)]
                df.to_csv(join(csv_path, f"""{file.split(".")[0]}.csv"""), index=False, quoting=csv.QUOTE_NONE)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset")
    arg_parser.add_argument("--x_shape", type=int)
    arg_parser.add_argument("--y_shape", type=int)
    args = arg_parser.parse_args()
    preprocess_pf(dataset=args.dataset, shape=(args.x_shape, args.y_shape))
    with open(join(data_path, args.dataset, "vocab.txt"), "w") as f:
        tokens = [str(i) for i in range(256)] + ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        f.write("\n".join(tokens))
