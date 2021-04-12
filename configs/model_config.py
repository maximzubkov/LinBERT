from dataclasses import dataclass


@dataclass
class ModelConfig:
    is_linear: bool
    feature_map: str = "elu"
    pos_bias_type: str = None
    bias_base_type: str = None
    lm: bool = False
    has_specials: bool = True
