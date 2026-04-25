"""Shared fixtures for the test suite."""
import datetime
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_tweets_df():
    """Minimal tweet DataFrame for unit tests (10 rows, 3 dates)."""
    np.random.seed(42)
    dates = [datetime.date(2022, 8, 1), datetime.date(2022, 8, 2), datetime.date(2022, 8, 3)]
    records = []
    for i, d in enumerate(dates):
        for j in range(10):
            idx = i * 10 + j
            records.append({
                "id": 1000 + idx,
                "created_at": pd.Timestamp(d),
                "date": d,
                "full_text": f"bitcoin is {'great' if j % 2 == 0 else 'terrible'} today number {idx}",
                "original_text": f"Bitcoin is {'GREAT' if j % 2 == 0 else 'TERRIBLE'}! #{idx}",
                "retweet_count": np.random.randint(0, 100),
                "favorite_count": np.random.randint(0, 500),
                "lang": "en",
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_tweets_with_vader(sample_tweets_df):
    """Tweet DataFrame with VADER scores already computed."""
    np.random.seed(42)
    n = len(sample_tweets_df)
    df = sample_tweets_df.copy()
    df["vader_compound"] = np.random.uniform(-1, 1, n)
    df["vader_pos"] = np.random.uniform(0, 1, n)
    df["vader_neg"] = np.random.uniform(0, 1, n)
    df["vader_neu"] = np.random.uniform(0, 1, n)
    return df


@pytest.fixture
def merged_daily_df():
    """Merged daily sentiment + price DataFrame for correlation tests."""
    np.random.seed(42)
    n = 30
    return pd.DataFrame({
        "date": pd.date_range("2022-08-01", periods=n).date,
        "mean_vader": np.random.uniform(-0.3, 0.3, n),
        "std_vader": np.random.uniform(0.1, 0.5, n),
        "price_return": np.random.uniform(-0.05, 0.05, n),
        "log_return": np.random.uniform(-0.05, 0.05, n),
        "price": np.random.uniform(19000, 25000, n),
        "volume": np.random.uniform(20e9, 40e9, n),
        "tweet_count": np.random.randint(50, 500, n),
        "vader_pos_ratio": np.random.uniform(0.3, 0.7, n),
        "vader_neg_ratio": np.random.uniform(0.1, 0.4, n),
    })
