import torch

from utils import construct_model
import numpy as np
import matplotlib.pyplot as plt

data_path = "data"


def measure_eval_time(
    is_linear: bool = False,
    feature_map: str = "elu",
    pos_bias_type: str = None,
    batches: tuple = (16, 32, 64, 128)
):
    # Init logger
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 3000
    timings = np.zeros((repetitions, len(batches)))
    for i, batch_size in enumerate(batches):
        model, inputs = construct_model(is_linear, feature_map, pos_bias_type, batch_size)

        # GPU warm-up
        for _ in range(10):
            _ = model(inputs)

        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(inputs)
                ender.record()

                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep, i] = curr_time

    output_name = ""
    if is_linear:
        output_name = f"Linear Transfromer ({feature_map})"
    else:
        output_name = "Transfromer"
    if pos_bias_type is not None:
        output_name = output_name + ", FFT"

    # example data
    time = timings.mean(axis=0)
    time_err = timings.std(axis=0)

    plt.errorbar(batches, time, yerr=time_err, label=output_name)


if __name__ == "__main__":
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })

    plt.figure(figsize=(17, 12))
    plt.grid()
    plt.xlabel("Batch Size", fontsize=20)
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
