from torch.profiler import profile, record_function, ProfilerActivity

from utils import construct_model


def estimate_memory(
    is_linear: bool = False,
    feature_map: str = "elu",
    pos_bias_type: str = None,
    batch_size: int = 32
):
    model, inputs = construct_model(is_linear, feature_map, pos_bias_type, batch_size)
    model, inputs = model.cuda(), inputs.cuda()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            model.bert(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))


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
