import torch
from torch.profiler import profile, record_function, ProfilerActivity

from utils import construct_model


def estimate_memory(
    is_linear: bool = False,
    feature_map: str = "elu",
    pos_bias_type: str = None,
    shapes: tuple = (16, 24, 32, 40, 48, 50)
):
    if is_linear:
        output_name = f"Linear Transformer ({feature_map})"
    else:
        output_name = "Transformer"
    if pos_bias_type is not None:
        output_name = output_name + ", FFT"
    print(output_name)

    for shape in shapes:
        model, inputs = construct_model(is_linear, feature_map, pos_bias_type, shape)
        model, inputs = model.cuda(), inputs.cuda()

        for _ in range(10):
            model.bert(inputs)

        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                with record_function("model_inference"):
                    model.bert(inputs)

        stats = prof.key_averages().table(sort_by="cuda_time_total", row_limit=1)
        gbs = -float(stats.split("\n")[-5].split()[-3])
        print(f"\t{shape}: {gbs} Gb")


if __name__ == "__main__":
    estimate_memory(is_linear=True, feature_map="exp")
    estimate_memory(is_linear=True, feature_map="elu")
    estimate_memory(is_linear=True, feature_map="dpfp")
    estimate_memory(is_linear=True, feature_map="favor")
    estimate_memory(is_linear=True, feature_map="exp", pos_bias_type="fft_2d")
    estimate_memory(is_linear=True, feature_map="elu", pos_bias_type="fft_2d")
    estimate_memory(is_linear=True, feature_map="dpfp", pos_bias_type="fft_2d")
    estimate_memory(is_linear=True, feature_map="favor", pos_bias_type="fft_2d")
    estimate_memory(is_linear=False)
    estimate_memory(is_linear=False, pos_bias_type="fft_2d")
