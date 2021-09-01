from os.path import join

import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import BertTokenizerFast

from build.pytorch_modelsize.pytorch_modelsize import SizeEstimator
from configs import mnist_config, ModelConfig
from models import Classifier
from utils import set_seed_

data_path = "data"


class Estimator(SizeEstimator):
    def __init__(self, model, input_size, bits=32, vocab_size=257):
        self.vocab_size = vocab_size
        super().__init__(model, input_size, bits)

    def get_modes_from_bert(self):
        clf = self.model.classifier.modules()
        bert = self.model.bert
        embs = list(bert.embeddings.modules())
        layer = bert.encoder.layer[0]

    @torch.no_grad()
    def get_output_sizes(self):
        input_ = torch.LongTensor(*self.input_size).random_(0, self.vocab_size)
        mods = self.get_modes_from_bert()
        out_sizes = []
        for i in range(1, len(mods)):
            m = mods[i]
            # print("\n\n\n\n", m)
            out = m(input_ids=input_)
            out_sizes.append(np.array(out.last_hidden_state.size()))
            input_ = out
        self.out_sizes = out_sizes
        return


def estimate_memory():
    model_config = ModelConfig(
        is_linear=True,
        feature_map="exp",
        pos_bias_type="fft_2d",
        bias_base_type="full"
    )
    dataset_name = "mnist"
    seed = 9
    x_shape, y_shape, n_channels = 28, 28, 1
    set_seed_(seed)

    output_path = join(data_path, dataset_name)
    vocab_path = join(data_path, dataset_name)
    tokenizer = BertTokenizerFast.from_pretrained(vocab_path)
    vocab_size = tokenizer.vocab_size
    config, args = mnist_config(
        dataset_name=dataset_name,
        output_path=output_path,
        seed=seed,
        is_test=False,
        x_shape=x_shape,
        y_shape=y_shape,
        n_channels=n_channels,
        vocab_size=vocab_size,
        model_config=model_config
    )

    model = Classifier(config=config)
    inputs = torch.LongTensor(args.train_batch_size, config.max_position_embeddings).random_(0, vocab_size)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            model.bert(inputs)

        print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))


if __name__ == "__main__":
    estimate_memory()
