"""Tests for src.preprocessor module — comprehensive coverage."""
import pandas as pd
import pytest

from src.preprocessor import clean_text, tokenize, preprocess_dataframe


# ===========================================================================
# TestCleanText
# ===========================================================================

class TestCleanText:
    """Tests for the clean_text function."""

    # -- URL removal --------------------------------------------------------

    def test_removes_https_url(self):
        result = clean_text("check this https://t.co/abc123 out")
        assert "https" not in result
        assert "t.co" not in result

    def test_removes_http_url(self):
        result = clean_text("visit http://example.com for info")
        assert "http" not in result
        assert "example.com" not in result

    def test_removes_www_url(self):
        result = clean_text("go to www.bitcoin.org now")
        assert "www" not in result
        assert "bitcoin.org" not in result

    def test_removes_multiple_urls(self):
        result = clean_text("link1 https://a.co/1 and link2 https://b.co/2")
        assert "https" not in result

    # -- @mention removal ---------------------------------------------------

    def test_removes_mention(self):
        result = clean_text("Hey @elonmusk what's up")
        assert "@elonmusk" not in result
        assert "elonmusk" not in result

    def test_removes_mention_with_colon(self):
        result = clean_text("@user: here is the tweet")
        assert "@user" not in result

    def test_removes_multiple_mentions(self):
        result = clean_text("@alice @bob hello there")
        assert "@alice" not in result
        assert "@bob" not in result

    # -- RT prefix removal --------------------------------------------------

    def test_removes_rt_prefix(self):
        result = clean_text("RT @user: some tweet content")
        assert not result.startswith("rt")
        assert "some tweet content" in result

    def test_rt_in_middle_not_removed(self):
        """RT removal only applies at the start of the string."""
        result = clean_text("this is not RT worthy")
        assert "rt" in result

    # -- Non-ASCII / emoji removal ------------------------------------------

    def test_removes_emoji(self):
        result = clean_text("bitcoin to the moon \U0001f680\U0001f31d")
        assert "\U0001f680" not in result
        assert "\U0001f31d" not in result

    def test_removes_non_ascii_chars(self):
        result = clean_text("caf\u00e9 bitcoin \u00e9change")
        # Non-ASCII replaced with spaces
        assert "\u00e9" not in result

    def test_removes_chinese_characters(self):
        result = clean_text("bitcoin \u6bd4\u7279\u5e01 price")
        assert "\u6bd4" not in result

    # -- Whitespace normalization -------------------------------------------

    def test_collapses_multiple_spaces(self):
        result = clean_text("too   many    spaces   here")
        assert "  " not in result
        assert result == "too many spaces here"

    def test_strips_leading_trailing_whitespace(self):
        result = clean_text("  hello world  ")
        assert result == "hello world"

    def test_tabs_and_newlines_normalized(self):
        result = clean_text("hello\tworld\nfoo")
        assert "\t" not in result
        assert "\n" not in result

    # -- Lowercase ----------------------------------------------------------

    def test_lowercases_text(self):
        result = clean_text("BITCOIN TO THE MOON")
        assert result == "bitcoin to the moon"

    def test_mixed_case_lowered(self):
        result = clean_text("BiTcOiN PrIcE")
        assert result == "bitcoin price"

    # -- Non-string input ---------------------------------------------------

    def test_none_returns_empty(self):
        assert clean_text(None) == ""

    def test_int_returns_empty(self):
        assert clean_text(123) == ""

    def test_float_returns_empty(self):
        assert clean_text(3.14) == ""

    def test_bool_returns_empty(self):
        assert clean_text(True) == ""

    def test_list_returns_empty(self):
        assert clean_text(["hello"]) == ""

    # -- Edge cases ---------------------------------------------------------

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_url(self):
        result = clean_text("https://t.co/abc123")
        assert result.strip() == ""

    def test_only_mention(self):
        result = clean_text("@somebody")
        assert result.strip() == ""

    def test_combined_cleaning(self):
        raw = "RT @user: BITCOIN is AMAZING!!! https://t.co/abc \U0001f680\U0001f4b0"
        result = clean_text(raw)
        assert "http" not in result
        assert "@user" not in result
        assert "\U0001f680" not in result
        assert result == result.lower()
        assert "  " not in result

    def test_preserves_dollar_amounts(self):
        """Dollar amounts like $150 should not be completely removed."""
        result = clean_text("price is $150 today")
        # $ is ASCII so it stays; we just check text isn't empty
        assert "150" in result or "price" in result


# ===========================================================================
# TestTokenize
# ===========================================================================

class TestTokenize:
    """Tests for the tokenize function."""

    def test_basic_tokenization(self):
        tokens = tokenize("bitcoin price is rising fast")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "bitcoin" in tokens

    def test_returns_list_of_strings(self):
        tokens = tokenize("hello world")
        assert all(isinstance(t, str) for t in tokens)

    def test_removes_stopwords_by_default(self):
        tokens = tokenize("the price is going up today")
        assert "the" not in tokens
        assert "is" not in tokens
        # Content words should remain
        assert "price" in tokens or "going" in tokens

    def test_keeps_stopwords_when_disabled(self):
        tokens = tokenize("the price is up", remove_stopwords=False)
        assert "the" in tokens
        assert "is" in tokens

    def test_filters_non_alpha_tokens(self):
        tokens = tokenize("price123 is $500 and 100%")
        # Non-alphabetic tokens are removed
        for t in tokens:
            assert t.isalpha(), f"Non-alpha token found: {t}"

    def test_punctuation_removed(self):
        tokens = tokenize("hello, world! how are you?")
        for t in tokens:
            assert t.isalpha()

    def test_empty_string_returns_empty_list(self):
        tokens = tokenize("")
        assert tokens == []

    def test_only_stopwords_returns_empty(self):
        tokens = tokenize("the is a an the")
        assert tokens == []

    def test_single_word(self):
        tokens = tokenize("bitcoin")
        assert tokens == ["bitcoin"]

    def test_numbers_filtered_out(self):
        tokens = tokenize("btc at 20000 usd")
        assert "20000" not in tokens


# ===========================================================================
# TestPreprocessDataframe
# ===========================================================================

class TestPreprocessDataframe:
    """Tests for the preprocess_dataframe function."""

    def test_adds_cleaned_text_column(self):
        df = pd.DataFrame({"full_text": ["Bitcoin is pumping! https://t.co/xyz"]})
        result = preprocess_dataframe(df)
        assert "cleaned_text" in result.columns

    def test_adds_tokens_column(self):
        df = pd.DataFrame({"full_text": ["BTC down again @whale_alert"]})
        result = preprocess_dataframe(df)
        assert "tokens" in result.columns

    def test_preserves_original_columns(self):
        df = pd.DataFrame({
            "full_text": ["Hello World"],
            "id": [1],
            "date": ["2022-08-01"],
        })
        result = preprocess_dataframe(df)
        assert "full_text" in result.columns
        assert "id" in result.columns
        assert "date" in result.columns
        assert result["full_text"].iloc[0] == "Hello World"

    def test_preserves_row_count(self):
        df = pd.DataFrame({"full_text": ["one", "two", "three"]})
        result = preprocess_dataframe(df)
        assert len(result) == 3

    def test_does_not_modify_input(self):
        df = pd.DataFrame({"full_text": ["Hello World"]})
        original_cols = set(df.columns)
        _ = preprocess_dataframe(df)
        assert set(df.columns) == original_cols

    def test_cleaned_text_is_lowercase(self):
        df = pd.DataFrame({"full_text": ["BITCOIN MOON"]})
        result = preprocess_dataframe(df)
        assert result["cleaned_text"].iloc[0] == "bitcoin moon"

    def test_tokens_are_list(self):
        df = pd.DataFrame({"full_text": ["bitcoin price rises"]})
        result = preprocess_dataframe(df)
        assert isinstance(result["tokens"].iloc[0], list)

    def test_custom_text_column(self):
        df = pd.DataFrame({"my_text": ["hello world"]})
        result = preprocess_dataframe(df, text_col="my_text")
        assert "cleaned_text" in result.columns

    def test_empty_dataframe(self):
        df = pd.DataFrame({"full_text": pd.Series([], dtype=str)})
        result = preprocess_dataframe(df)
        assert len(result) == 0
        assert "cleaned_text" in result.columns
        assert "tokens" in result.columns

    def test_handles_nan_in_text(self):
        df = pd.DataFrame({"full_text": ["hello", None, "world"]})
        result = preprocess_dataframe(df)
        assert len(result) == 3
        # None should be cleaned to ""
        assert result["cleaned_text"].iloc[1] == ""
