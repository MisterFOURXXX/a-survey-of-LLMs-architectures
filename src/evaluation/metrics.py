"""Metric computation utilities."""
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def build_rouge_scorer():
    return rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )


def compute_rouge(scorer, reference: str, generated: str) -> dict:
    scores = scorer.score(reference, generated)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def compute_bleu(reference: str, generated: str) -> float:
    ref_tokens = reference.lower().split()
    gen_tokens = generated.lower().split()
    smoothing = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing)