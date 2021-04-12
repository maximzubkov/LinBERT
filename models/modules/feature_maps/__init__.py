from .base import dpfp_feature_map, elu_feature_map, exp_feature_map  # noqa
from .fourier_features import Favor  # noqa

fm_name2func = {
    "elu": elu_feature_map,
    "exp": exp_feature_map,
    "dpfp": dpfp_feature_map
}
