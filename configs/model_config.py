from dataclasses import dataclass


@dataclass
class ModelConfig:
    is_linear: bool
    has_pos_attention: bool
    has_batch_norm: bool
    has_pos_embed_2d: bool = False
    feature_map: str = "elu"
    pos_bias_type: str = None
    bias_base_type: str = None
    lm: bool = False
    has_specials: bool = True
    alpha: float = 0.00001
    beta: float = 0.000001
