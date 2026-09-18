"""Data loading utilities for the StackSample dataset."""
import os
from pathlib import Path

import polars as pl
from bs4 import BeautifulSoup


def clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def load_stacksample(
    data_dir: str | None = None,
    score_threshold: int = 5,
    max_questions: int = 100,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load and filter StackSample questions and answers.

    Args:
        data_dir: Directory containing Questions.csv and Answers.csv.
        score_threshold: Minimum score to keep a row.
        max_questions: Maximum number of questions to keep.

    Returns:
        Tuple of (questions, answers) as Polars DataFrames.
    """
    # ── Resolve data path safely (Path object, not string) ──────────────
    data_path = Path(data_dir)
    data_path = data_path.expanduser().resolve()

    questions_csv = data_path / "Questions.csv"
    answers_csv = data_path / "Answers.csv"

    if not questions_csv.exists():
        raise FileNotFoundError(f"Missing file: {questions_csv}")
    if not answers_csv.exists():
        raise FileNotFoundError(f"Missing file: {answers_csv}")

    questions = pl.read_csv(
        questions_csv,
        encoding="utf8-lossy",
        columns=["Id", "Title", "Body", "Score"],
    ).filter(pl.col("Score") > score_threshold)

    answers = pl.read_csv(
        answers_csv,
        encoding="utf8-lossy",
        columns=["Id", "ParentId", "Body", "Score"],
    ).filter(pl.col("Score") > score_threshold)

    questions = questions.sort("Score", descending=True).head(max_questions)

    return questions, answers