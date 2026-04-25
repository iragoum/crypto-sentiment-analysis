"""Tests for src.data_loader module."""
import datetime
import os
import textwrap

import pandas as pd
import pytest

from src.data_loader import load_tweets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(tmp_path, filename, content):
    """Write CSV content to a file and return the path."""
    path = tmp_path / filename
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# TestLoadTweets — happy path
# ---------------------------------------------------------------------------

class TestLoadTweetsHappyPath:
    """Tests for normal, well-formed CSV input."""

    def test_loads_correct_shape(self, tmp_path):
        csv = _write_csv(tmp_path, "tweets.csv", """\
            created_at,id,full_text,lang,date
            2022-08-01 12:00:00,1,bitcoin is great,en,2022-08-01
            2022-08-01 13:00:00,2,crypto crashing,en,2022-08-01
            2022-08-02 10:00:00,3,btc up today,en,2022-08-02
        """)
        df = load_tweets(csv)
        assert len(df) == 3
        assert "full_text" in df.columns
        assert "date" in df.columns

    def test_date_column_present(self, tmp_path):
        csv = _write_csv(tmp_path, "tweets.csv", """\
            created_at,id,full_text,lang,date
            2022-08-01 12:00:00,1,hello world,en,2022-08-01
        """)
        df = load_tweets(csv)
        assert "date" in df.columns

    def test_created_at_parsed_as_datetime(self, tmp_path):
        csv = _write_csv(tmp_path, "tweets.csv", """\
            created_at,id,full_text,lang,date
            2022-08-01 12:00:00,1,hello,en,2022-08-01
        """)
        df = load_tweets(csv)
        assert pd.api.types.is_datetime64_any_dtype(df["created_at"])

    def test_index_is_reset(self, tmp_path):
        csv = _write_csv(tmp_path, "tweets.csv", """\
            created_at,id,full_text,lang,date
            2022-08-01 12:00:00,1,hello,en,2022-08-01
            2022-08-01 13:00:00,2,world,en,2022-08-01
        """)
        df = load_tweets(csv)
        assert list(df.index) == [0, 1]


# ---------------------------------------------------------------------------
# TestLangFilter
# ---------------------------------------------------------------------------

class TestLangFilter:
    """Tests for the language filter parameter."""

    def test_filters_english_only(self, tmp_path):
        csv = _write_csv(tmp_path, "tweets.csv", """\
            created_at,id,full_text,lang,date
            2022-08-01 12:00:00,1,hello,en,2022-08-01
            2022-08-01 13:00:00,2,hola,es,2022-08-01
            2022-08-01 14:00:00,3,bonjour,fr,2022-08-01
        """)
        df = load_tweets(csv, lang_filter="en")
        assert len(df) == 1
        assert df["full_text"].iloc[0] == "hello"

    def test_filter_other_language(self, tmp_path):
        csv = _write_csv(tmp_path, "tweets.csv", """\
            created_at,id,full_text,lang,date
            2022-08-01 12:00:00,1,hello,en,2022-08-01
            2022-08-01 13:00:00,2,hola,es,2022-08-01
        """)
        df = load_tweets(csv, lang_filter="es")
        assert len(df) == 1
        assert df["full_text"].iloc[0] == "hola"

    def test_no_filter_when_empty_string(self, tmp_path):
        csv = _write_csv(tmp_path, "tweets.csv", """\
            created_at,id,full_text,lang,date
            2022-08-01 12:00:00,1,hello,en,2022-08-01
            2022-08-01 13:00:00,2,hola,es,2022-08-01
        """)
        df = load_tweets(csv, lang_filter="")
        assert len(df) == 2

    def test_no_lang_column_skips_filter(self, tmp_path):
        csv = _write_csv(tmp_path, "tweets.csv", """\
            created_at,id,full_text,date
            2022-08-01 12:00:00,1,hello,2022-08-01
            2022-08-01 13:00:00,2,hola,2022-08-01
        """)
        df = load_tweets(csv, lang_filter="en")
        assert len(df) == 2


# ---------------------------------------------------------------------------
# TestMissingColumns
# ---------------------------------------------------------------------------

class TestMissingColumns:
    """Tests for required column validation."""

    def test_missing_full_text_raises(self, tmp_path):
        csv = _write_csv(tmp_path, "tweets.csv", """\
            created_at,id,lang,date
            2022-08-01 12:00:00,1,en,2022-08-01
        """)
        with pytest.raises(ValueError, match="full_text"):
            load_tweets(csv)

    def test_missing_date_creates_from_created_at(self, tmp_path):
        """If 'date' column is missing, it should be derived from created_at."""
        csv = _write_csv(tmp_path, "tweets.csv", """\
            created_at,id,full_text,lang
            2022-08-01 12:00:00,1,hello,en
            2022-08-02 13:00:00,2,world,en
        """)
        df = load_tweets(csv)
        assert "date" in df.columns
        # The derived date should be a date object
        dates = df["date"].unique()
        assert len(dates) == 2


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for data loading."""

    def test_all_rows_filtered_returns_empty(self, tmp_path):
        csv = _write_csv(tmp_path, "tweets.csv", """\
            created_at,id,full_text,lang,date
            2022-08-01 12:00:00,1,hola,es,2022-08-01
            2022-08-01 13:00:00,2,bonjour,fr,2022-08-01
        """)
        df = load_tweets(csv, lang_filter="en")
        assert len(df) == 0

    def test_single_row(self, tmp_path):
        csv = _write_csv(tmp_path, "tweets.csv", """\
            created_at,id,full_text,lang,date
            2022-08-01 12:00:00,1,single tweet,en,2022-08-01
        """)
        df = load_tweets(csv)
        assert len(df) == 1

    def test_file_not_found_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_tweets("/nonexistent/path/tweets.csv")

    def test_extra_columns_preserved(self, tmp_path):
        csv = _write_csv(tmp_path, "tweets.csv", """\
            created_at,id,full_text,lang,date,retweet_count,favorite_count
            2022-08-01 12:00:00,1,hello,en,2022-08-01,5,10
        """)
        df = load_tweets(csv)
        assert "retweet_count" in df.columns
        assert "favorite_count" in df.columns
