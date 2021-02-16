import time
from argparse import ArgumentParser
from os.path import join

import torch
import torch.nn.functional as F
import torchvision
import vit_pytorch
from torch import optim
from torch.utils.data import DataLoader

from configs import ModelConfig, vit_config
from models import ViTModel
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

    output_path = join(data_path, dataset_name)

    if dataset_name == "mnist":
        config, training_args = vit_config(
            dataset_name=dataset_name,
            output_path=output_path,
            seed=seed,
            is_test=is_test,
            model_config=model_config
        )
    else:
        raise ValueError("Unknown dataset")

    vit = ViTModel(config=config)

    transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(
                                                          (0.1307,), (0.3081,)
                                                      )])

    train_ = torchvision.datasets.MNIST("data", train=True, download=True, transform=transform_mnist)
    train_loader = DataLoader(train_, batch_size=training_args.train_batch_size, shuffle=True)

    test_ = torchvision.datasets.MNIST("data", train=False, download=True, transform=transform_mnist)
    test_loader = DataLoader(test_, batch_size=training_args.eval_batch_size, shuffle=True)

    def train_epoch(model, optimizer, data_loader, loss_history):
        total_samples = len(data_loader.dataset)
        model.train()

        for i, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                      ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                      '{:6.4f}'.format(loss.item()))
                loss_history.append(loss.item())

    def evaluate(model, data_loader, loss_history):
        model.eval()

        total_samples = len(data_loader.dataset)
        correct_samples = 0
        total_loss = 0

        with torch.no_grad():
            for data, target in data_loader:
                output = F.log_softmax(model(data), dim=1)
                loss = F.nll_loss(output, target, reduction='sum')
                _, pred = torch.max(output, dim=1)

                total_loss += loss.item()
                correct_samples += pred.eq(target).sum()

        avg_loss = total_loss / total_samples
        loss_history.append(avg_loss)
        print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
              '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
              '{:5}'.format(total_samples) + ' (' +
              '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

    start_time = time.time()

    optimizer = optim.Adam(vit.parameters(), lr=training_args.learning_rate)

    train_loss_history, test_loss_history = [], []
    for epoch in range(1, int(training_args.num_train_epochs) + 1):
        print('Epoch:model', epoch)
        train_epoch(vit, optimizer, train_loader, train_loss_history)
        evaluate(vit, test_loader, test_loss_history)

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')


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
