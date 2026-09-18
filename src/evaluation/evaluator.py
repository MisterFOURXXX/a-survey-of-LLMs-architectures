"""Full evaluation pipeline."""
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .generation import STRATEGY_CONFIGS, generate_answer
from .metrics import build_rouge_scorer, compute_bleu, compute_rouge


def _build_prompt(tokenizer, question_title: str, question_body: str) -> str:
    """Prefer the model's native chat template; fall back to a plain format."""
    user_content = f"Question: {question_title}\n{question_body}"
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"{user_content}\n\nAnswer:"


def calculate_perplexity(model, tokenizer, texts, device) -> float:
    """Compute perplexity over a list of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Perplexity"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()
            outputs = model(**inputs)
            if outputs.loss is not None:
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

    if total_tokens == 0:
        return float("inf")
    return float(torch.exp(torch.tensor(total_loss / total_tokens)).item())


def evaluate_model(
    model,
    tokenizer,
    test_data,
    device,
    max_new_tokens: int = 100,
    strategies: list[str] | None = None,
) -> dict:
    """Run full evaluation and return results dictionary."""
    strategies = strategies or list(STRATEGY_CONFIGS.keys())
    scorer = build_rouge_scorer()

    references: list[str] = []
    questions: list[dict] = []
    for row in test_data.iter_rows(named=True):
        references.append(row["answer"].strip())
        questions.append({
            "title": row["question_title"],
            "body": row["question_body"],
        })

    results: dict[str, Any] = {}

    # Perplexity uses the raw Q/A format
    test_texts = [
        f"Question: {q['title']}\n{q['body']}\n\nAnswer: {r}"
        for q, r in zip(questions, references)
    ]
    results["perplexity"] = calculate_perplexity(
        model, tokenizer, test_texts, device
    )

    # Strategy comparison uses the model's native prompt format
    for strategy in strategies:
        r1, r2, rl, bleu, times = [], [], [], [], []
        for ref, q in tqdm(
            zip(references, questions),
            desc=f"Strategy: {strategy}",
            total=len(references),
        ):
            prompt = _build_prompt(tokenizer, q["title"], q["body"])
            answer, elapsed = generate_answer(
                model,
                tokenizer,
                device,
                prompt,
                max_new_tokens=max_new_tokens,
                strategy=strategy,
            )
            times.append(elapsed)
            scores = compute_rouge(scorer, ref, answer)
            r1.append(scores["rouge1"])
            r2.append(scores["rouge2"])
            rl.append(scores["rougeL"])
            bleu.append(compute_bleu(ref, answer))

        results[strategy] = {
            "rouge1": float(np.mean(r1)),
            "rouge2": float(np.mean(r2)),
            "rougeL": float(np.mean(rl)),
            "bleu": float(np.mean(bleu)),
            "avg_time": float(np.mean(times)),
            "total_time": float(sum(times)),
        }

    return results