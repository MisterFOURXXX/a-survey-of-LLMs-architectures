"""Data preprocessing and splitting utilities."""
import polars as pl
from sklearn.model_selection import train_test_split
from .loader import clean_html


def preprocess_qa_pairs(
    questions: pl.DataFrame, answers: pl.DataFrame
) -> pl.DataFrame:
    """Clean HTML and join questions with answers."""
    questions = questions.with_columns([
        pl.col("Body").map_elements(clean_html, return_dtype=pl.Utf8),
        pl.col("Title").str.strip_chars(),
    ])

    answers = answers.with_columns(
        pl.col("Body").map_elements(clean_html, return_dtype=pl.Utf8)
    )

    qa_pairs = answers.join(
        questions,
        left_on="ParentId",
        right_on="Id",
        how="inner",
    ).select([
        pl.col("ParentId").alias("question_id"),
        pl.col("Title").alias("question_title"),
        pl.col("Body_right").alias("question_body"),
        pl.col("Score_right").alias("question_score"),
        pl.col("Id").alias("answer_id"),
        pl.col("Body").alias("answer"),
        pl.col("Score").alias("answer_score"),
    ])

    return qa_pairs


def split_dataset(
    qa_pairs: pl.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split by unique question titles to prevent data leakage."""
    unique_questions = qa_pairs["question_title"].unique().to_list()

    train_q, temp_q = train_test_split(
        unique_questions, test_size=test_size, random_state=random_state
    )
    test_q, val_q = train_test_split(
        temp_q, test_size=val_size, random_state=random_state
    )

    train_data = qa_pairs.filter(
        pl.col("question_title").is_in(train_q)
    )
    test_data = qa_pairs.filter(
        pl.col("question_title").is_in(test_q)
    )
    val_data = qa_pairs.filter(
        pl.col("question_title").is_in(val_q)
    )

    return train_data, test_data, val_data