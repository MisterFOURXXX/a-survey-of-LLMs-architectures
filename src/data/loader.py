"""Data loading utilities for the StackSample dataset."""
import polars as pl
from pathlib import Path
from bs4 import BeautifulSoup


def clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def load_stacksample(
    data_dir: str = "data",
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
    data_path = Path(data_dir)

    questions = pl.read_csv(
        data_path / "Questions.csv",
        encoding="utf8-lossy",
        columns=["Id", "Title", "Body", "Score"],
    ).filter(pl.col("Score") > score_threshold)

    answers = pl.read_csv(
        data_path / "Answers.csv",
        encoding="utf8-lossy",
        columns=["Id", "ParentId", "Body", "Score"],
    ).filter(pl.col("Score") > score_threshold)

    questions = questions.sort("Score", descending=True).head(max_questions)

    return questions, answers