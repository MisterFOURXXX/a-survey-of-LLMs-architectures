"""Tokenization utilities with model-specific formatting."""
from typing import Callable

import polars as pl
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def get_formatter(model_name: str) -> Callable[[dict], str]:
    """Return a format function based on model family."""
    name_lower = model_name.lower()

    if "llama" in name_lower:
        def fmt(row):
            return (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"Question: {row['question_title']}\n{row['question_body']}"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{row['answer']}<|eot_id|>"
            )
        return fmt

    if "qwen" in name_lower:
        def fmt(row):
            return (
                f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
                f"<|im_start|>user\nQuestion: {row['question_title']}\n"
                f"{row['question_body']}<|im_end|>\n"
                f"<|im_start|>assistant\n{row['answer']}<|im_end|>"
            )
        return fmt

    if "phi" in name_lower and "moe" in name_lower:
        def fmt(row):
            return (
                f"<|system|>\nYou are a helpful AI assistant for question answering.<|end|>\n"
                f"<|user|>\nQuestion: {row['question_title']}\n"
                f"{row['question_body']}<|end|>\n"
                f"<|assistant|>\n{row['answer']}<|end|>"
            )
        return fmt

    # Default format (Pythia, GPT-2, Phi-mini, etc.)
    def fmt(row):
        return (
            f"Question: {row['question_title']}\n{row['question_body']}\n\n"
            f"Answer: {row['answer']}"
        )
    return fmt


def tokenize_dataset(
    df: pl.DataFrame,
    tokenizer: AutoTokenizer,
    model_name: str,
    max_length: int = 128,
) -> Dataset:
    """Tokenize a Polars DataFrame into a HuggingFace Dataset."""
    formatter = get_formatter(model_name)
    texts = [formatter(row) for row in df.iter_rows(named=True)]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,  # ← return Python lists, not torch tensors
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Mask padding positions with -100 so loss ignores them.
    # Uses attention_mask (not equality with pad_token_id) so we do NOT
    # accidentally mask natural EOS tokens.
    labels = [
        [-100 if mask == 0 else tok for tok, mask in zip(ids, mask_)]
        for ids, mask_ in zip(input_ids, attention_mask)
    ]

    return Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    })


def build_dataset_dict(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    tokenizer: AutoTokenizer,
    model_name: str,
    max_length: int = 128,
) -> DatasetDict:
    """Build a DatasetDict with train and validation splits."""
    return DatasetDict({
        "train": tokenize_dataset(train_df, tokenizer, model_name, max_length),
        "validation": tokenize_dataset(val_df, tokenizer, model_name, max_length),
    })