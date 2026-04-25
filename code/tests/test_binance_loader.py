"""Tests for src.binance_loader module."""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.binance_loader import load_binance_prices, binance_csv_path_for, COIN_TO_CSV


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_1min_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a minimal Binance 1-minute CSV with the real column layout."""
    df = pd.DataFrame(rows)
    p = tmp_path / "TEST_1m_Binance.csv"
    df.to_csv(p, index=False)
    return p


def _minute_row(dt_str: str, close: float, volume: float) -> dict:
    return {
        "Open time": dt_str,
        "Open": close,
        "High": close,
        "Low": close,
        "Close": close,
        "Volume": volume,
        "Close time": dt_str,
        "Quote asset volume": 0.0,
        "Number of trades": 1,
        "Taker buy base asset volume": 0.0,
        "Taker buy quote asset volume": 0.0,
        "Ignore": 0,
    }


# ── TestLoadBinancePrices ─────────────────────────────────────────────────────

class TestLoadBinancePrices:

    def test_returns_dataframe(self, tmp_path):
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 00:00:00", 21000.0, 10.0),
            _minute_row("2022-07-26 23:59:00", 21500.0, 5.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-26", csv)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self, tmp_path):
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 00:00:00", 21000.0, 10.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-26", csv)
        assert {"date", "price", "volume", "price_return", "log_return"}.issubset(df.columns)

    def test_daily_price_is_last_close(self, tmp_path):
        """price must be the Close of the last 1-min candle in the day."""
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 00:00:00", 21000.0, 10.0),  # first
            _minute_row("2022-07-26 12:00:00", 21300.0, 8.0),   # middle
            _minute_row("2022-07-26 23:59:00", 21500.0, 5.0),   # last ← expected
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-26", csv)
        assert df["price"].iloc[0] == pytest.approx(21500.0)

    def test_daily_price_is_not_first_open(self, tmp_path):
        """Confirm price != first open (which would be 21000)."""
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 00:00:00", 21000.0, 10.0),
            _minute_row("2022-07-26 23:59:00", 21500.0, 5.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-26", csv)
        assert df["price"].iloc[0] != pytest.approx(21000.0)

    def test_daily_price_is_not_mean(self, tmp_path):
        """Confirm price != mean of candles."""
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 00:00:00", 21000.0, 10.0),
            _minute_row("2022-07-26 23:59:00", 21500.0, 5.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-26", csv)
        mean_val = (21000.0 + 21500.0) / 2  # 21250.0
        assert df["price"].iloc[0] != pytest.approx(mean_val)

    def test_daily_volume_is_sum(self, tmp_path):
        """volume must be the sum of all candle volumes in the day."""
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 00:00:00", 21000.0, 10.0),
            _minute_row("2022-07-26 12:00:00", 21300.0, 8.0),
            _minute_row("2022-07-26 23:59:00", 21500.0, 5.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-26", csv)
        assert df["volume"].iloc[0] == pytest.approx(23.0)

    def test_date_filtering_start(self, tmp_path):
        """Rows before start_date must be excluded."""
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-25 23:59:00", 20900.0, 5.0),  # excluded
            _minute_row("2022-07-26 23:59:00", 21500.0, 5.0),  # included
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-27", csv)
        dates = [str(d) for d in df["date"]]
        assert "2022-07-25" not in dates
        assert "2022-07-26" in dates

    def test_date_filtering_end(self, tmp_path):
        """Rows after end_date must be excluded."""
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 23:59:00", 21500.0, 5.0),  # included
            _minute_row("2022-07-27 23:59:00", 21800.0, 5.0),  # excluded
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-26", csv)
        dates = [str(d) for d in df["date"]]
        assert "2022-07-27" not in dates

    def test_end_date_inclusive(self, tmp_path):
        """end_date itself must be included in the result."""
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 23:59:00", 21500.0, 5.0),
            _minute_row("2022-07-27 23:59:00", 21800.0, 5.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-27", csv)
        assert len(df) == 2

    def test_price_return_calculation(self, tmp_path):
        """price_return[1] == (price[1] - price[0]) / price[0]."""
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 23:59:00", 20000.0, 5.0),
            _minute_row("2022-07-27 23:59:00", 21000.0, 5.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-27", csv)
        assert pd.isna(df["price_return"].iloc[0])
        assert df["price_return"].iloc[1] == pytest.approx(0.05)

    def test_log_return_calculation(self, tmp_path):
        """log_return[1] == log(price[1] / price[0])."""
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 23:59:00", 20000.0, 5.0),
            _minute_row("2022-07-27 23:59:00", 21000.0, 5.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-27", csv)
        import math
        assert pd.isna(df["log_return"].iloc[0])
        assert df["log_return"].iloc[1] == pytest.approx(math.log(21000 / 20000))

    def test_price_return_first_row_nan(self, tmp_path):
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 23:59:00", 21000.0, 5.0),
            _minute_row("2022-07-27 23:59:00", 22000.0, 5.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-27", csv)
        assert pd.isna(df["price_return"].iloc[0])

    def test_log_return_first_row_nan(self, tmp_path):
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 23:59:00", 21000.0, 5.0),
            _minute_row("2022-07-27 23:59:00", 22000.0, 5.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-27", csv)
        assert pd.isna(df["log_return"].iloc[0])

    def test_sorted_ascending_by_date(self, tmp_path):
        """Output must be sorted ascending by date."""
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-27 23:59:00", 22000.0, 5.0),
            _minute_row("2022-07-26 23:59:00", 21000.0, 5.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-27", csv)
        dates = list(df["date"])
        assert dates == sorted(dates)

    def test_multi_day_row_count(self, tmp_path):
        """One output row per unique date."""
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 00:00:00", 21000.0, 5.0),
            _minute_row("2022-07-26 23:59:00", 21500.0, 5.0),
            _minute_row("2022-07-27 00:00:00", 21600.0, 5.0),
            _minute_row("2022-07-27 23:59:00", 21800.0, 5.0),
            _minute_row("2022-07-28 23:59:00", 22000.0, 5.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-28", csv)
        assert len(df) == 3

    def test_prices_positive(self, tmp_path):
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 23:59:00", 21000.0, 5.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-26", csv)
        assert (df["price"] > 0).all()

    def test_date_column_type(self, tmp_path):
        """date column must contain datetime.date objects."""
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 23:59:00", 21000.0, 5.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-26", csv)
        assert all(isinstance(d, datetime.date) for d in df["date"])


# ── TestEdgeCases ─────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Binance CSV not found"):
            load_binance_prices(
                "bitcoin", "2022-07-26", "2022-08-30",
                tmp_path / "nonexistent.csv",
            )

    def test_missing_file_message_contains_path(self, tmp_path):
        missing = tmp_path / "ghost.csv"
        with pytest.raises(FileNotFoundError, match=str(missing.name)):
            load_binance_prices("bitcoin", "2022-07-26", "2022-08-30", missing)

    def test_empty_after_filter_raises_value_error(self, tmp_path):
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2020-01-01 23:59:00", 7000.0, 5.0),
        ])
        with pytest.raises(ValueError, match="No Binance data"):
            load_binance_prices("bitcoin", "2022-07-26", "2022-08-30", csv)

    def test_single_day_single_candle(self, tmp_path):
        csv = _make_1min_csv(tmp_path, [
            _minute_row("2022-07-26 12:00:00", 21000.0, 10.0),
        ])
        df = load_binance_prices("bitcoin", "2022-07-26", "2022-07-26", csv)
        assert len(df) == 1
        assert df["price"].iloc[0] == pytest.approx(21000.0)


# ── TestBinanceCsvPathFor ─────────────────────────────────────────────────────

class TestBinanceCsvPathFor:

    def test_bitcoin_maps_correctly(self):
        p = binance_csv_path_for("bitcoin", "/some/dir")
        assert p.name == "BTCUSD_1m_Binance.csv"

    def test_ethereum_maps_correctly(self):
        p = binance_csv_path_for("ethereum", "/some/dir")
        assert p.name == "ETHUSD_1m_Binance.csv"

    def test_binancecoin_maps_correctly(self):
        p = binance_csv_path_for("binancecoin", "/some/dir")
        assert p.name == "BNBUSD_1m_Binance.csv"

    def test_unknown_coin_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown coin_id"):
            binance_csv_path_for("dogecoin", "/some/dir")

    def test_all_known_coins_covered(self):
        for coin in ("bitcoin", "ethereum", "binancecoin"):
            p = binance_csv_path_for(coin, "/d")
            assert p.name in COIN_TO_CSV.values()
