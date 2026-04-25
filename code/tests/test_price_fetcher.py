"""Tests for src.price_fetcher module."""
import os
import textwrap

import numpy as np
import pandas as pd
import pytest

from src.price_fetcher import get_fallback_prices, load_or_fetch_prices


# ===========================================================================
# TestGetFallbackPrices
# ===========================================================================

class TestGetFallbackPrices:
    """Tests for the fallback price data generator."""

    def test_returns_dataframe(self):
        df = get_fallback_prices()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = get_fallback_prices()
        required = {"date", "price", "volume", "market_cap", "price_return", "log_return"}
        assert required.issubset(set(df.columns))

    def test_date_range(self):
        df = get_fallback_prices()
        dates = sorted(df["date"])
        assert str(dates[0]) == "2022-07-26"
        assert str(dates[-1]) == "2022-08-30"

    def test_prices_are_positive(self):
        df = get_fallback_prices()
        assert (df["price"] > 0).all()

    def test_volumes_are_positive(self):
        df = get_fallback_prices()
        assert (df["volume"] > 0).all()

    def test_market_cap_positive(self):
        df = get_fallback_prices()
        assert (df["market_cap"] > 0).all()

    def test_price_return_first_is_nan(self):
        df = get_fallback_prices()
        assert pd.isna(df["price_return"].iloc[0])

    def test_price_return_others_are_float(self):
        df = get_fallback_prices()
        non_first = df["price_return"].iloc[1:]
        assert non_first.notna().all()
        assert all(isinstance(v, (float, np.floating)) for v in non_first)

    def test_log_return_first_is_nan(self):
        df = get_fallback_prices()
        assert pd.isna(df["log_return"].iloc[0])

    def test_log_return_others_are_float(self):
        df = get_fallback_prices()
        non_first = df["log_return"].iloc[1:]
        assert non_first.notna().all()

    def test_row_count(self):
        df = get_fallback_prices()
        assert len(df) == 36  # Jul 26 to Aug 30 = 36 days

    def test_bitcoin_default(self):
        df = get_fallback_prices("bitcoin")
        assert len(df) > 0

    def test_deterministic_prices(self):
        """Fallback prices should be consistent across calls."""
        df1 = get_fallback_prices()
        df2 = get_fallback_prices()
        assert df1["price"].tolist() == df2["price"].tolist()


class TestFallbackEthereum:
    """Fallback prices for Ethereum."""

    def test_returns_dataframe(self):
        df = get_fallback_prices("ethereum")
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = get_fallback_prices("ethereum")
        required = {"date", "price", "volume", "market_cap", "price_return", "log_return"}
        assert required.issubset(set(df.columns))

    def test_row_count(self):
        df = get_fallback_prices("ethereum")
        assert len(df) == 36

    def test_date_range(self):
        df = get_fallback_prices("ethereum")
        dates = sorted(df["date"])
        assert str(dates[0]) == "2022-07-26"
        assert str(dates[-1]) == "2022-08-30"

    def test_prices_positive(self):
        df = get_fallback_prices("ethereum")
        assert (df["price"] > 0).all()

    def test_prices_in_reasonable_range(self):
        """ETH was ~$1400–$2000 in Jul-Aug 2022."""
        df = get_fallback_prices("ethereum")
        assert df["price"].min() > 1000
        assert df["price"].max() < 3000

    def test_price_return_first_nan(self):
        df = get_fallback_prices("ethereum")
        assert pd.isna(df["price_return"].iloc[0])

    def test_deterministic(self):
        df1 = get_fallback_prices("ethereum")
        df2 = get_fallback_prices("ethereum")
        assert df1["price"].tolist() == df2["price"].tolist()


class TestFallbackBinancecoin:
    """Fallback prices for Binance Coin."""

    def test_returns_dataframe(self):
        df = get_fallback_prices("binancecoin")
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = get_fallback_prices("binancecoin")
        required = {"date", "price", "volume", "market_cap", "price_return", "log_return"}
        assert required.issubset(set(df.columns))

    def test_row_count(self):
        df = get_fallback_prices("binancecoin")
        assert len(df) == 36

    def test_date_range(self):
        df = get_fallback_prices("binancecoin")
        dates = sorted(df["date"])
        assert str(dates[0]) == "2022-07-26"
        assert str(dates[-1]) == "2022-08-30"

    def test_prices_positive(self):
        df = get_fallback_prices("binancecoin")
        assert (df["price"] > 0).all()

    def test_prices_in_reasonable_range(self):
        """BNB was ~$250–$340 in Jul-Aug 2022."""
        df = get_fallback_prices("binancecoin")
        assert df["price"].min() > 200
        assert df["price"].max() < 500

    def test_price_return_first_nan(self):
        df = get_fallback_prices("binancecoin")
        assert pd.isna(df["price_return"].iloc[0])

    def test_deterministic(self):
        df1 = get_fallback_prices("binancecoin")
        df2 = get_fallback_prices("binancecoin")
        assert df1["price"].tolist() == df2["price"].tolist()


# ===========================================================================
# TestLoadOrFetchPrices
# ===========================================================================

class TestLoadOrFetchPrices:
    """Tests for the load_or_fetch_prices function."""

    def test_loads_from_cache(self, tmp_path):
        """If cache file exists, load from it without API call."""
        cache_file = tmp_path / "btc_prices.csv"
        # Create a minimal cache CSV
        cache_df = pd.DataFrame({
            "date": ["2022-08-01", "2022-08-02", "2022-08-03"],
            "price": [23000.0, 23200.0, 22800.0],
            "volume": [30e9, 31e9, 29e9],
            "market_cap": [440e9, 443e9, 435e9],
            "price_return": [np.nan, 0.0087, -0.0172],
            "log_return": [np.nan, 0.00866, -0.01735],
        })
        cache_df.to_csv(str(cache_file), index=False)

        result = load_or_fetch_prices(
            "bitcoin", "2022-08-01", "2022-08-03", str(cache_file)
        )
        assert len(result) == 3
        assert "price" in result.columns
        assert "date" in result.columns

    def test_cache_columns_correct(self, tmp_path):
        cache_file = tmp_path / "btc_prices.csv"
        cache_df = pd.DataFrame({
            "date": ["2022-08-01"],
            "price": [23000.0],
            "volume": [30e9],
            "market_cap": [440e9],
            "price_return": [np.nan],
            "log_return": [np.nan],
        })
        cache_df.to_csv(str(cache_file), index=False)

        result = load_or_fetch_prices(
            "bitcoin", "2022-08-01", "2022-08-01", str(cache_file)
        )
        required = {"date", "price", "volume", "market_cap", "price_return", "log_return"}
        assert required.issubset(set(result.columns))

    def test_fallback_when_no_cache_and_no_api(self, tmp_path):
        """When cache doesn't exist and API fails, fallback should kick in."""
        cache_file = tmp_path / "btc_prices_new.csv"
        assert not cache_file.exists()

        result = load_or_fetch_prices(
            "bitcoin", "2022-07-26", "2022-08-30", str(cache_file)
        )
        assert len(result) > 0
        assert "price" in result.columns
        # Fallback should save the cache
        assert cache_file.exists()

    def test_fallback_prices_structure(self, tmp_path):
        cache_file = tmp_path / "btc_fallback.csv"
        result = load_or_fetch_prices(
            "bitcoin", "2022-07-26", "2022-08-30", str(cache_file)
        )
        # market_cap is only present in hardcoded fallback; Binance CSV does not have it
        required = {"date", "price", "volume", "price_return", "log_return"}
        assert required.issubset(set(result.columns))

    def test_date_column_is_date_type_from_cache(self, tmp_path):
        cache_file = tmp_path / "btc_prices.csv"
        cache_df = pd.DataFrame({
            "date": ["2022-08-01", "2022-08-02"],
            "price": [23000.0, 23200.0],
            "volume": [30e9, 31e9],
            "market_cap": [440e9, 443e9],
            "price_return": [np.nan, 0.0087],
            "log_return": [np.nan, 0.00866],
        })
        cache_df.to_csv(str(cache_file), index=False)

        result = load_or_fetch_prices(
            "bitcoin", "2022-08-01", "2022-08-02", str(cache_file)
        )
        import datetime
        assert all(isinstance(d, datetime.date) for d in result["date"])

    def test_fallback_ethereum_no_cache(self, tmp_path):
        """ETH fallback is used when cache is absent and API unavailable."""
        cache_file = tmp_path / "eth_prices.csv"
        result = load_or_fetch_prices(
            "ethereum", "2022-07-26", "2022-08-30", str(cache_file)
        )
        assert len(result) > 0
        assert "price" in result.columns
        assert cache_file.exists()

    def test_fallback_binancecoin_no_cache(self, tmp_path):
        """BNB fallback is used when cache is absent and API unavailable."""
        cache_file = tmp_path / "bnb_prices.csv"
        result = load_or_fetch_prices(
            "binancecoin", "2022-07-26", "2022-08-30", str(cache_file)
        )
        assert len(result) > 0
        assert "price" in result.columns
        assert cache_file.exists()

    def test_ethereum_cache_round_trip(self, tmp_path):
        """ETH prices loaded from cache match what was written."""
        cache_file = tmp_path / "eth_prices.csv"
        cache_df = pd.DataFrame({
            "date": ["2022-08-01", "2022-08-02"],
            "price": [1700.0, 1750.0],
            "volume": [12e9, 13e9],
            "market_cap": [204e9, 210e9],
            "price_return": [np.nan, 0.0294],
            "log_return": [np.nan, 0.0290],
        })
        cache_df.to_csv(str(cache_file), index=False)
        result = load_or_fetch_prices(
            "ethereum", "2022-08-01", "2022-08-02", str(cache_file)
        )
        assert list(result["price"]) == [1700.0, 1750.0]
