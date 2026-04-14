"""Tests for src.preprocessor module."""
import pandas as pd
import pytest

from src.preprocessor import clean_text, tokenize, preprocess_dataframe


class TestCleanText:
    def test_removes_urls(self):
        assert "check this" in clean_text("check this https://t.co/abc123")
        assert "http" not in clean_text("http://example.com is cool")

    def test_removes_mentions(self):
        assert "@elonmusk" not in clean_text("Hey @elonmusk what's up")

    def test_removes_rt_prefix(self):
        assert clean_text("RT @user: some tweet") == "some tweet"

    def test_lowercases(self):
        assert clean_text("BITCOIN TO THE MOON") == "bitcoin to the moon"

    def test_handles_non_string(self):
        assert clean_text(None) == ""
        assert clean_text(123) == ""

    def test_collapses_whitespace(self):
        assert "  " not in clean_text("too   many    spaces")


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = tokenize("bitcoin price is rising fast")
        assert "bitcoin" in tokens
        assert isinstance(tokens, list)

    def test_removes_stopwords(self):
        tokens = tokenize("the price is going up", remove_stopwords=True)
        assert "the" not in tokens
        assert "is" not in tokens

    def test_keeps_stopwords_when_disabled(self):
        tokens = tokenize("the price is up", remove_stopwords=False)
        assert "the" in tokens

    def test_filters_non_alpha(self):
        tokens = tokenize("price123 is $500")
        assert "500" not in tokens


class TestPreprocessDataframe:
    def test_adds_columns(self):
        df = pd.DataFrame({"full_text": [
            "Bitcoin is pumping! https://t.co/xyz",
            "BTC down again @whale_alert",
        ]})
        result = preprocess_dataframe(df)
        assert "cleaned_text" in result.columns
        assert "tokens" in result.columns
        assert len(result) == 2

    def test_preserves_original(self):
        df = pd.DataFrame({"full_text": ["Hello World"]})
        result = preprocess_dataframe(df)
        assert result["full_text"].iloc[0] == "Hello World"
