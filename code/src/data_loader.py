"""Data loading utilities for the crypto sentiment analysis system."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_tweets(path: str | Path, lang_filter: str = "en") -> pd.DataFrame:
    """Load and validate the tweet dataset.

    Args:
        path: Path to the CSV file.
        lang_filter: ISO language code to keep. Defaults to 'en'.

    Returns:
        DataFrame with columns: id, created_at, date, full_text,
        retweet_count, favorite_count.
    """
    df = pd.read_csv(path, parse_dates=["created_at"])
    logger.info("Loaded %d rows from %s", len(df), path)

    if "lang" in df.columns and lang_filter:
        df = df[df["lang"] == lang_filter].copy()
        logger.info("After language filter ('%s'): %d rows", lang_filter, len(df))

    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["created_at"]).dt.date

    required = ["full_text", "date"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from dataset")

    return df.reset_index(drop=True)
