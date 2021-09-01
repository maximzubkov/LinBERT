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
    shapes: tuple = (32, 45, 55, 64)
):
    # Init logger
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings = np.zeros((repetitions, len(shapes)))

    output_name = ""
    if is_linear:
        output_name = f"Linear Transfromer ({feature_map})"
    else:
        output_name = "Transfromer"
    if pos_bias_type is not None:
        output_name = output_name + ", FFT"
    print(output_name)

    for i, max_len in enumerate(shapes):
        model, inputs = construct_model(is_linear, feature_map, pos_bias_type, max_len)

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
        print(f"\t{max_len * max_len}: {timings[:, i].mean()} ± {timings[:, i].std()}")

    # example data
    time = timings.mean(axis=0)
    time_err = timings.std(axis=0)

    plt.errorbar([shape ** 2 for shape in shapes], time, yerr=time_err, label=output_name)


if __name__ == "__main__":
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })

    plt.figure(figsize=(17, 12))
    plt.grid()
    plt.xlabel("Num pixels", fontsize=20)
    plt.ylabel("Inference time", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

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

    plt.legend(fontsize=15)
    plt.savefig("fig.pdf", format="pdf")
