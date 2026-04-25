"""Tests for src.sentiment_analyzer module — VADER + FinBERT (mocked)."""
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.sentiment_analyzer import score_vader, score_finbert


# ===========================================================================
# TestScoreVader
# ===========================================================================

class TestScoreVader:
    """Tests for VADER sentiment scoring."""

    def test_returns_correct_columns(self):
        texts = pd.Series(["Bitcoin is great", "Crypto is crashing"])
        result = score_vader(texts)
        expected = {"vader_compound", "vader_pos", "vader_neg", "vader_neu"}
        assert set(result.columns) == expected

    def test_returns_dataframe(self):
        texts = pd.Series(["hello"])
        result = score_vader(texts)
        assert isinstance(result, pd.DataFrame)

    def test_compound_in_range(self):
        texts = pd.Series([
            "Bitcoin is amazing and wonderful!",
            "This is terrible, huge crash",
            "BTC traded at 20000 today",
            "",
            "GREAT AMAZING BEST WONDERFUL!!!!",
        ])
        result = score_vader(texts)
        assert (result["vader_compound"] >= -1).all()
        assert (result["vader_compound"] <= 1).all()

    def test_pos_neg_neu_in_range(self):
        texts = pd.Series(["Some normal tweet about bitcoin"])
        result = score_vader(texts)
        assert 0 <= result["vader_pos"].iloc[0] <= 1
        assert 0 <= result["vader_neg"].iloc[0] <= 1
        assert 0 <= result["vader_neu"].iloc[0] <= 1

    def test_pos_neg_neu_sum_to_one(self):
        texts = pd.Series(["Bitcoin price is rising fast today"])
        result = score_vader(texts)
        total = result["vader_pos"].iloc[0] + result["vader_neg"].iloc[0] + result["vader_neu"].iloc[0]
        assert abs(total - 1.0) < 0.01

    def test_positive_text_scores_positive(self):
        texts = pd.Series(["Bitcoin is amazing and wonderful bullish moon!"])
        result = score_vader(texts)
        assert result["vader_compound"].iloc[0] > 0.05

    def test_negative_text_scores_negative(self):
        texts = pd.Series(["Terrible crash, everything is collapsing badly!"])
        result = score_vader(texts)
        assert result["vader_compound"].iloc[0] < -0.05

    def test_neutral_text_near_zero(self):
        texts = pd.Series(["Bitcoin traded at 20000 today"])
        result = score_vader(texts)
        assert abs(result["vader_compound"].iloc[0]) < 0.5

    def test_index_preserved(self):
        texts = pd.Series(["hello", "world"], index=[10, 20])
        result = score_vader(texts)
        assert list(result.index) == [10, 20]

    def test_index_preserved_noncontiguous(self):
        texts = pd.Series(["a", "b", "c"], index=[5, 100, 999])
        result = score_vader(texts)
        assert list(result.index) == [5, 100, 999]

    def test_handles_empty_string(self):
        texts = pd.Series(["", "bitcoin"])
        result = score_vader(texts)
        assert len(result) == 2
        # Empty string should get compound = 0
        assert result["vader_compound"].iloc[0] == 0.0

    def test_handles_single_text(self):
        texts = pd.Series(["just one tweet"])
        result = score_vader(texts)
        assert len(result) == 1

    def test_handles_many_texts(self):
        texts = pd.Series([f"tweet number {i}" for i in range(100)])
        result = score_vader(texts)
        assert len(result) == 100

    def test_non_string_coerced(self):
        """VADER calls str(text), so numeric values should not crash."""
        texts = pd.Series([123, None, "hello"])
        result = score_vader(texts)
        assert len(result) == 3

    def test_special_characters(self):
        texts = pd.Series(["!!! ??? ... ### $$$"])
        result = score_vader(texts)
        assert len(result) == 1


# ===========================================================================
# TestScoreFinbert (mocked)
# ===========================================================================

class TestScoreFinbert:
    """Tests for FinBERT scoring with mocked pipeline."""

    @pytest.fixture(autouse=True)
    def _mock_finbert(self):
        """Mock the FinBERT pipeline for all tests in this class."""
        # Reset the global _finbert_pipeline before each test
        import src.sentiment_analyzer as sa
        sa._finbert_pipeline = None

        def mock_pipeline_fn(texts):
            """Simulate FinBERT output for a batch of texts."""
            results = []
            for text in texts:
                if not text.strip() or text.strip().lower() == "neutral":
                    results.append({"label": "Neutral", "score": 0.85})
                elif any(w in text.lower() for w in ["great", "good", "amazing", "bullish", "rise"]):
                    results.append({"label": "Positive", "score": 0.92})
                elif any(w in text.lower() for w in ["bad", "crash", "terrible", "bearish", "fall"]):
                    results.append({"label": "Negative", "score": 0.88})
                else:
                    results.append({"label": "Neutral", "score": 0.75})
            return results

        mock_pipe = MagicMock(side_effect=mock_pipeline_fn)

        with patch.object(sa, "_get_finbert", return_value=mock_pipe):
            yield

    def test_returns_correct_columns(self):
        texts = pd.Series(["bitcoin is great", "market is crashing"])
        result = score_finbert(texts)
        assert set(result.columns) == {"finbert_label", "finbert_score"}

    def test_returns_dataframe(self):
        texts = pd.Series(["hello world"])
        result = score_finbert(texts)
        assert isinstance(result, pd.DataFrame)

    def test_label_values(self):
        texts = pd.Series(["great news", "terrible crash", "bitcoin at 20000"])
        result = score_finbert(texts)
        valid_labels = {"positive", "negative", "neutral"}
        for label in result["finbert_label"]:
            assert label in valid_labels

    def test_score_in_range(self):
        texts = pd.Series(["great news", "terrible crash", "neutral info"])
        result = score_finbert(texts)
        assert (result["finbert_score"] >= 0).all()
        assert (result["finbert_score"] <= 1).all()

    def test_positive_text_labeled_positive(self):
        texts = pd.Series(["bitcoin is great and amazing"])
        result = score_finbert(texts)
        assert result["finbert_label"].iloc[0] == "positive"

    def test_negative_text_labeled_negative(self):
        texts = pd.Series(["terrible crash in the market"])
        result = score_finbert(texts)
        assert result["finbert_label"].iloc[0] == "negative"

    def test_index_preserved(self):
        texts = pd.Series(["hello", "world"], index=[10, 20])
        result = score_finbert(texts)
        assert list(result.index) == [10, 20]

    def test_empty_string_handled(self):
        texts = pd.Series(["", "bitcoin is great"])
        result = score_finbert(texts)
        assert len(result) == 2
        # Empty string should be replaced with "neutral" before inference
        assert result["finbert_label"].iloc[0] == "neutral"

    def test_batch_processing(self):
        """Ensure batch_size parameter works without error."""
        texts = pd.Series([f"tweet {i}" for i in range(50)])
        result = score_finbert(texts, batch_size=8)
        assert len(result) == 50

    def test_single_text(self):
        texts = pd.Series(["just one tweet"])
        result = score_finbert(texts, batch_size=16)
        assert len(result) == 1

    def test_labels_are_lowercase(self):
        texts = pd.Series(["great news", "bad news", "no news"])
        result = score_finbert(texts)
        for label in result["finbert_label"]:
            assert label == label.lower()
