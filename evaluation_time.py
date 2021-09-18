import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils import construct_model

data_path = "data"


def measure_eval_time(
    is_linear: bool = False,
    feature_map: str = "elu",
    pos_bias_type: str = None,
    shapes: tuple = (16, 24, 32, 40, 48, 50)
):
    # Init logger
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 150
    timings = np.zeros((repetitions, len(shapes)))

    if is_linear:
        output_name = f"Linear Transformer ({feature_map})"
    else:
        output_name = "Transformer"
    if pos_bias_type is not None:
        output_name = output_name + ", FFT"
    print(output_name)

    for i, shape in enumerate(shapes):
        model, inputs = construct_model(is_linear, feature_map, pos_bias_type, shape)
        model, inputs = model.cuda(), inputs.cuda()

        # GPU warm-up
        for _ in range(10):
            _ = model(inputs)

        with torch.no_grad():
            for rep in tqdm(range(repetitions)):
                starter.record()
                _ = model(inputs)
                ender.record()

                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep, i] = curr_time
        print(f"\t{shape * shape}: {timings[:, i].mean()} ± {timings[:, i].std()}")


if __name__ == "__main__":
    measure_eval_time(is_linear=True, feature_map="exp")
    measure_eval_time(is_linear=True, feature_map="elu")
    measure_eval_time(is_linear=True, feature_map="dpfp")
    measure_eval_time(is_linear=True, feature_map="favor")
    measure_eval_time(is_linear=True, feature_map="exp", pos_bias_type="fft_2d")
    measure_eval_time(is_linear=True, feature_map="elu", pos_bias_type="fft_2d")
    measure_eval_time(is_linear=True, feature_map="dpfp", pos_bias_type="fft_2d")
    measure_eval_time(is_linear=True, feature_map="favor", pos_bias_type="fft_2d")
    measure_eval_time(is_linear=False)
    measure_eval_time(is_linear=False, pos_bias_type="fft_2d")