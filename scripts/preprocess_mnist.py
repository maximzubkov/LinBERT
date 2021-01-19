from argparse import ArgumentParser
from os.path import join
import numpy as np

from tqdm import tqdm


def _construct_line(
        img: list,
        label: int,
):
    img_line = " ".join(img)
    line = f"""\"{label}\",\"{img_line}\"\n"""
    return line


def preprocess_mnist(
        imgs_path: str,
        labels_path: str,
        output_path: str,
        n: int,
        rotate: bool
):
    with open(imgs_path, "rb") as images, open(output_path, "w") as output_, open(labels_path, "rb") as labels:
        images.read(16)
        labels.read(8)
        output_.write("label,text\n")

        for _ in tqdm(range(n)):
            label = ord(labels.read(1))
            img = []
            for j in range(28 * 28):
                img.append(str(ord(images.read(1))))
            if rotate:
                np_img = np.array(img).reshape(28, 28)
                for rotate_times in range(4):
                    rotated_img = np.rot90(np_img, k=rotate_times, axes=(0, 1)).reshape(-1).tolist()
                    output_.write(_construct_line(label=label, img=rotated_img))
            else:
                output_.write(_construct_line(label=label, img=img))


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset_path", type=str)
    arg_parser.add_argument("--rotate", action="store_true")
    args = arg_parser.parse_args()
    dataset_path = args.dataset_path
    preprocess_mnist(
        imgs_path=join(dataset_path, "train_images"),
        labels_path=join(dataset_path, "train_labels"),
        output_path=join(dataset_path, "train.csv"),
        rotate=args.rotate,
        n=60000
    )
    preprocess_mnist(
        imgs_path=join(dataset_path, "test_images"),
        labels_path=join(dataset_path, "test_labels"),
        output_path=join(dataset_path, "test.csv"),
        rotate=args.rotate,
        n=10000
    )

    for path in [dataset_path, f"{dataset_path}_small"]:
        with open(join(path, "vocab.txt"), "w") as f:
            tokens = [str(i) for i in range(256)] + ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            f.write("\n".join(tokens))
