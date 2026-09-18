"""Model and tokenizer loading utilities."""
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = False,
    **kwargs: Any,
) -> AutoTokenizer:
    """Load a tokenizer and ensure a pad token exists."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name: str,
    quantization_config=None,
    torch_dtype: torch.dtype | None = None,
    device_map: str | dict | None = "auto",
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
    low_cpu_mem_usage: bool = True,
    config=None,                 # ← NEW: pass a pre-modified AutoConfig
    offload_folder: str | None = None,  # ← NEW: for MoE/CPU-offload setups
    **kwargs: Any,
):
    """Load a causal language model with optional quantization."""
    dtype = torch_dtype or (
        torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    call_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        **kwargs,
    }

    if device_map is not None:
        call_kwargs["device_map"] = device_map
    if quantization_config is not None:
        call_kwargs["quantization_config"] = quantization_config
    if attn_implementation is not None:
        call_kwargs["attn_implementation"] = attn_implementation
    if config is not None:
        call_kwargs["config"] = config
    if offload_folder is not None:
        call_kwargs["offload_folder"] = offload_folder

    return AutoModelForCausalLM.from_pretrained(model_name, **call_kwargs)