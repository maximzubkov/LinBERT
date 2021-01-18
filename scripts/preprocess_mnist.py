from os.path import join
from argparse import ArgumentParser
from tqdm import tqdm


def _convert(imgs_path, labels_path, output_path, n):
    with open(imgs_path, "rb") as images, open(output_path, "w") as output_, open(labels_path, "rb") as labels:
        images.read(16)
        labels.read(8)
        output_.write("label,text\n")

        for _ in tqdm(range(n)):
            label = ord(labels.read(1))
            img = []
            for j in range(28 * 28):
                img.append(str(ord(images.read(1))))
            img_line = ", ".join(img)
            line = f"""\"{label}\", \"{img_line}\"\n"""
            output_.write(line)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset_path", type=str)
    args = arg_parser.parse_args()
    dataset_path = args.dataset_path
    _convert(
        imgs_path=join(dataset_path, "train_images"),
        labels_path=join(dataset_path, "train_labels"),
        output_path=join(dataset_path, "train.csv"),
        n=60000
    )
    _convert(
        imgs_path=join(dataset_path, "test_images"),
        labels_path=join(dataset_path, "test_labels"),
        output_path=join(dataset_path, "test.csv"),
        n=10000
    )
