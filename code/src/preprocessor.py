"""Text preprocessing pipeline for tweet data."""
from __future__ import annotations

import html
import re
import logging
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

_NLTK_READY = False


def _ensure_nltk() -> None:
    """Download NLTK data on first use."""
    global _NLTK_READY
    if not _NLTK_READY:
        for resource in ["stopwords", "punkt", "punkt_tab"]:
            nltk.download(resource, quiet=True)
        _NLTK_READY = True


def clean_text(text: str) -> str:
    """Clean a single tweet for FinBERT input.

    Applies the following transformations in order:
    1. Decode HTML entities (&amp; -> &)
    2. Remove URLs
    3. Remove full RT prefix with mention (RT @user:)
    4. Remove @mentions
    5. Strip # from hashtags, keep the word (#BTC -> BTC)
    6. Strip $ from cashtags, keep the word ($BTC -> BTC)
    7. Remove non-ASCII characters (emoji, CJK, etc.)
    8. Normalise whitespace and lowercase

    NOTE: VADER should receive the *original* (uncleaned) text because
    it uses capitalisation, punctuation, and emoji as tonal signals.

    Args:
        text: Raw tweet text.

    Returns:
        Cleaned lowercase string.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"^RT\s+@\w+:?\s*", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\$([A-Za-z]{2,})", r"\1", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenise cleaned text into word tokens.

    Args:
        text: Pre-cleaned text string.
        remove_stopwords: If True, filter English stop-words.

    Returns:
        List of word tokens.
    """
    _ensure_nltk()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    if remove_stopwords:
        stop = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop]
    return tokens


def preprocess_dataframe(df, text_col: str = "full_text"):
    """Add 'cleaned_text' and 'tokens' columns to a tweet DataFrame.

    Args:
        df: Input DataFrame with raw tweets.
        text_col: Column containing raw tweet text.

    Returns:
        DataFrame with additional columns: cleaned_text, tokens.
    """
    logger.info("Preprocessing %d tweets...", len(df))
    df = df.copy()
    df["cleaned_text"] = df[text_col].apply(clean_text)
    df["tokens"] = df["cleaned_text"].apply(tokenize)
    return df
