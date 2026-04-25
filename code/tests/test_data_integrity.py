"""Data integrity tests for the per-crypto tweet CSV datasets.

Tests are skipped if the corresponding file is not present on disk.
"""
import re
from pathlib import Path

import pandas as pd
import pytest

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Paths for each crypto file
TWEETS_BTC = RAW_DIR / "tweets_btc.csv"
TWEETS_ETH = RAW_DIR / "tweets_eth.csv"
TWEETS_BNB = RAW_DIR / "tweets_bnb.csv"
TWEETS_ALL = RAW_DIR / "tweets_all.csv"

# Keep backward-compatible path (old pipeline wrote tweets.csv)
TWEETS_LEGACY = RAW_DIR / "tweets.csv"


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["created_at"])


# ── Shared assertions (run on any per-crypto dataframe) ─────────────────────

def _assert_required_columns(df: pd.DataFrame) -> None:
    required = {"created_at", "id", "full_text", "date"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def _assert_no_null_full_text(df: pd.DataFrame) -> None:
    null_count = df["full_text"].isna().sum()
    assert null_count == 0, f"Found {null_count} null full_text values"


def _assert_lowercase(df: pd.DataFrame) -> None:
    sample = df["full_text"].dropna().head(1000)
    ratio = (sample == sample.str.lower()).sum() / len(sample)
    assert ratio > 0.95, f"Only {ratio:.1%} of sampled tweets are lowercase"


def _assert_no_urls(df: pd.DataFrame) -> None:
    sample = df["full_text"].dropna().head(5000)
    url_pct = sample.str.contains(r"https?://", na=False).sum() / len(sample)
    assert url_pct < 0.01, f"{url_pct:.1%} of tweets still contain URLs"


def _assert_date_range(df: pd.DataFrame) -> None:
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
        assert dates.min() >= pd.Timestamp("2022-07-25")
        assert dates.max() <= pd.Timestamp("2022-08-31")


def _assert_no_duplicate_ids(df: pd.DataFrame) -> None:
    if "id" in df.columns:
        dups = df["id"].duplicated().sum()
        assert dups == 0, f"Found {dups} duplicate IDs"


# ── BTC ──────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(not TWEETS_BTC.exists(), reason=f"tweets_btc.csv not found at {TWEETS_BTC}")
class TestBTCDataset:
    @pytest.fixture(scope="class")
    def df(self):
        return _load(TWEETS_BTC)

    def test_row_count_above_300k(self, df):
        assert len(df) > 300_000, f"Expected > 300K tweets, got {len(df)}"

    def test_required_columns(self, df):
        _assert_required_columns(df)

    def test_no_null_full_text(self, df):
        _assert_no_null_full_text(df)

    def test_lowercase(self, df):
        _assert_lowercase(df)

    def test_no_urls(self, df):
        _assert_no_urls(df)

    def test_date_range(self, df):
        _assert_date_range(df)

    def test_no_duplicate_ids(self, df):
        _assert_no_duplicate_ids(df)

    def test_original_text_exists(self, df):
        assert "original_text" in df.columns


# ── ETH ──────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(not TWEETS_ETH.exists(), reason=f"tweets_eth.csv not found at {TWEETS_ETH}")
class TestETHDataset:
    @pytest.fixture(scope="class")
    def df(self):
        return _load(TWEETS_ETH)

    def test_has_rows(self, df):
        assert len(df) > 0, "tweets_eth.csv is empty"

    def test_required_columns(self, df):
        _assert_required_columns(df)

    def test_no_null_full_text(self, df):
        _assert_no_null_full_text(df)

    def test_lowercase(self, df):
        _assert_lowercase(df)

    def test_no_urls(self, df):
        _assert_no_urls(df)

    def test_date_range(self, df):
        _assert_date_range(df)

    def test_no_duplicate_ids(self, df):
        _assert_no_duplicate_ids(df)

    def test_original_text_exists(self, df):
        assert "original_text" in df.columns


# ── BNB ──────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(not TWEETS_BNB.exists(), reason=f"tweets_bnb.csv not found at {TWEETS_BNB}")
class TestBNBDataset:
    @pytest.fixture(scope="class")
    def df(self):
        return _load(TWEETS_BNB)

    def test_has_rows(self, df):
        assert len(df) > 0, "tweets_bnb.csv is empty"

    def test_required_columns(self, df):
        _assert_required_columns(df)

    def test_no_null_full_text(self, df):
        _assert_no_null_full_text(df)

    def test_lowercase(self, df):
        _assert_lowercase(df)

    def test_no_urls(self, df):
        _assert_no_urls(df)

    def test_date_range(self, df):
        _assert_date_range(df)

    def test_no_duplicate_ids(self, df):
        _assert_no_duplicate_ids(df)

    def test_original_text_exists(self, df):
        assert "original_text" in df.columns


# ── Combined (tweets_all.csv) ─────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(not TWEETS_ALL.exists(), reason=f"tweets_all.csv not found at {TWEETS_ALL}")
class TestCombinedDataset:
    @pytest.fixture(scope="class")
    def df(self):
        return _load(TWEETS_ALL)

    def test_crypto_column_exists(self, df):
        assert "crypto" in df.columns

    def test_crypto_values(self, df):
        assert set(df["crypto"].unique()).issubset({"btc", "eth", "bnb"})

    def test_has_multiple_cryptos(self, df):
        assert df["crypto"].nunique() > 1

    def test_required_columns(self, df):
        _assert_required_columns(df)

    def test_no_null_full_text(self, df):
        _assert_no_null_full_text(df)


# ── Legacy backward-compatibility (tweets.csv) ───────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(
    not TWEETS_LEGACY.exists(),
    reason=f"tweets.csv not found at {TWEETS_LEGACY} (legacy path, not required)",
)
class TestLegacyDataset:
    """Tests for the old tweets.csv file produced by earlier pipeline versions."""

    @pytest.fixture(scope="class")
    def df(self):
        return _load(TWEETS_LEGACY)

    def test_row_count_above_300k(self, df):
        en = df[df["lang"] == "en"] if "lang" in df.columns else df
        assert len(en) > 300_000, f"Expected > 300K English tweets, got {len(en)}"

    def test_required_columns(self, df):
        required = {"created_at", "id", "full_text", "date"}
        missing = required - set(df.columns)
        assert not missing, f"Missing columns: {missing}"
