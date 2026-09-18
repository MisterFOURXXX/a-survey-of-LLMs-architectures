"""Text generation strategies."""
import time

import torch


# Only sampling strategies get temperature/top_k/top_p.
STRATEGY_CONFIGS: dict[str, dict] = {
    "greedy": {
        "do_sample": False,
        "num_beams": 1,
    },
    "beam_search": {
        "do_sample": False,
        "num_beams": 4,
        "early_stopping": True,
    },
    "top_k": {
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
    },
    "top_p": {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "temperature_sampling": {
        "do_sample": True,
        "temperature": 0.8,
    },
}


def generate_answer(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int = 100,
    strategy: str = "greedy",
) -> tuple[str, float]:
    """Generate an answer given a pre-built prompt. Returns (answer, elapsed_seconds)."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS["greedy"]))
    gen_kwargs.update({
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    })

    # Strip sampling-only kwargs when doing deterministic decoding.
    if not gen_kwargs.get("do_sample", False):
        for k in ("temperature", "top_k", "top_p"):
            gen_kwargs.pop(k, None)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - start

    # Only decode the newly generated tokens (skip the prompt).
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return answer, elapsed