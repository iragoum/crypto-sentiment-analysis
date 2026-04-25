"""Load and aggregate Binance 1-minute OHLCV data to daily prices."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Map CoinGecko coin_id → Binance CSV filename
COIN_TO_CSV: dict[str, str] = {
    "bitcoin":     "BTCUSD_1m_Binance.csv",
    "ethereum":    "ETHUSD_1m_Binance.csv",
    "binancecoin": "BNBUSD_1m_Binance.csv",
}


def load_binance_prices(
    coin_id: str,
    start_date: str,
    end_date: str,
    binance_csv_path: str | Path,
) -> pd.DataFrame:
    """Read Binance 1-minute CSV and aggregate to daily closing prices.

    Daily price = Close of the last 1-minute candle in the day (23:59 UTC).
    Daily volume = sum of Volume (base currency) across all candles in the day.

    Args:
        coin_id: CoinGecko-style coin ID ('bitcoin', 'ethereum', 'binancecoin').
            Used only for logging; the caller provides the exact CSV path.
        start_date: First date to include, 'YYYY-MM-DD'.
        end_date: Last date to include (inclusive), 'YYYY-MM-DD'.
        binance_csv_path: Path to the 1-minute Binance CSV file.

    Returns:
        DataFrame with columns: date (datetime.date), price, volume,
        price_return, log_return. Sorted ascending by date.

    Raises:
        FileNotFoundError: If binance_csv_path does not exist.
        ValueError: If no rows remain after date filtering.
    """
    path = Path(binance_csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Binance CSV not found: {path}. "
            "Place the 1-minute data file at the expected path."
        )

    logger.info("Loading Binance 1-min data for %s from %s", coin_id, path)

    df = pd.read_csv(
        path,
        usecols=["Open time", "Close", "Volume"],
        parse_dates=["Open time"],
    )
    df = df.rename(columns={"Open time": "open_time", "Close": "close", "Volume": "volume"})
    df["open_time"] = pd.to_datetime(df["open_time"])
    df["date"] = df["open_time"].dt.date

    # Filter to requested range
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    if df.empty:
        raise ValueError(
            f"No Binance data for {coin_id} between {start_date} and {end_date}. "
            "Check that the CSV covers the requested period."
        )

    logger.info("  Rows after date filter: %d (%s → %s)", len(df), start, end)

    # Aggregate: last close of the day + sum volume
    daily_close = (
        df.sort_values("open_time")
        .groupby("date", sort=True)
        .agg(price=("close", "last"), volume=("volume", "sum"))
        .reset_index()
    )

    daily_close["price_return"] = daily_close["price"].pct_change()
    daily_close["log_return"] = np.log(
        daily_close["price"] / daily_close["price"].shift(1)
    )

    logger.info("  Daily rows: %d | price range: %.2f – %.2f",
                len(daily_close), daily_close["price"].min(), daily_close["price"].max())

    return daily_close


def binance_csv_path_for(coin_id: str, binance_dir: str | Path) -> Path:
    """Return the expected Binance CSV path for a given coin_id."""
    filename = COIN_TO_CSV.get(coin_id)
    if filename is None:
        raise ValueError(
            f"Unknown coin_id '{coin_id}'. "
            f"Supported: {list(COIN_TO_CSV.keys())}"
        )
    return Path(binance_dir) / filename
