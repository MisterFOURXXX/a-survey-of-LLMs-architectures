"""Evaluation: metrics, generation, and full-pipeline evaluator."""

from .metrics import build_rouge_scorer, compute_rouge, compute_bleu
from .generation import generate_answer, STRATEGY_CONFIGS
from .evaluator import evaluate_model, calculate_perplexity

__all__ = [
    "build_rouge_scorer",
    "compute_rouge",
    "compute_bleu",
    "generate_answer",
    "STRATEGY_CONFIGS",
    "evaluate_model",
    "calculate_perplexity",
]