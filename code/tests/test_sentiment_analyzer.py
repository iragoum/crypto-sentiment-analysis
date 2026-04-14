"""Tests for src.sentiment_analyzer module (VADER only — FinBERT requires GPU)."""
import pandas as pd
import pytest

from src.sentiment_analyzer import score_vader


class TestScoreVader:
    def test_returns_correct_columns(self):
        texts = pd.Series(["Bitcoin is great", "Crypto is crashing"])
        result = score_vader(texts)
        expected_cols = {"vader_compound", "vader_pos", "vader_neg", "vader_neu"}
        assert expected_cols == set(result.columns)

    def test_compound_range(self):
        texts = pd.Series([
            "Bitcoin is amazing and wonderful!",
            "This is terrible, huge crash",
            "BTC traded at 20000 today",
        ])
        result = score_vader(texts)
        assert (result["vader_compound"] >= -1).all()
        assert (result["vader_compound"] <= 1).all()

    def test_positive_text_scores_positive(self):
        texts = pd.Series(["Bitcoin is amazing and wonderful bullish!"])
        result = score_vader(texts)
        assert result["vader_compound"].iloc[0] > 0.05

    def test_negative_text_scores_negative(self):
        texts = pd.Series(["Terrible crash, everything is collapsing badly!"])
        result = score_vader(texts)
        assert result["vader_compound"].iloc[0] < -0.05

    def test_index_preserved(self):
        texts = pd.Series(["hello", "world"], index=[10, 20])
        result = score_vader(texts)
        assert list(result.index) == [10, 20]

    def test_handles_empty_string(self):
        texts = pd.Series(["", "bitcoin"])
        result = score_vader(texts)
        assert len(result) == 2
