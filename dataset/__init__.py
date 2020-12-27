from transformers import PreTrainedTokenizerFast

from .yelp import get_yelp_dataset
from .path_finder import get_path_finder_dataset
from .common import dataset_config

def get_dataset(
        name: str,
        split: str,
        max_length: int,
        cache_dir: str,
        is_test: bool,
        tokenizer: PreTrainedTokenizerFast = None,
):
    if name in ["yelp_full", "yelp_polarity"]:
        return get_yelp_dataset(
            name=name,
            split=split,
            max_length=max_length,
            cache_dir=cache_dir,
            is_test=is_test,
            tokenizer=tokenizer
        )
    elif name in ["pf_6_full"]:
        return get_path_finder_dataset(
            name=name,
            split=split,
            cache_dir=cache_dir,
            is_test=is_test,
            tokenizer=tokenizer
        )
    else:
        raise ValueError("Unknown Dataset")