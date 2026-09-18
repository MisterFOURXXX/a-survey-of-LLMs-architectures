"""Quantization configuration builders."""
from transformers import BitsAndBytesConfig
import torch


def build_quant_config(cfg: dict | None = None) -> BitsAndBytesConfig | None:
    """Build BitsAndBytesConfig from a dictionary, or return None."""
    if cfg is None:
        return None

    compute_dtype = cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    if compute_dtype == "bfloat16":
        compute_dtype = torch.bfloat16
    elif compute_dtype == "float16":
        compute_dtype = torch.float16

    return BitsAndBytesConfig(
        load_in_4bit=cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
    )