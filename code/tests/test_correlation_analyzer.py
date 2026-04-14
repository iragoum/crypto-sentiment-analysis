"""Tests for src.correlation_analyzer module."""
import numpy as np
import pandas as pd
import pytest

from src.correlation_analyzer import (
    aggregate_daily_sentiment,
    compute_lagged_correlations,
    run_granger_test,
    adf_test,
    correlation_matrix,
)


@pytest.fixture
def sample_tweets():
    """Create a minimal daily tweet DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2022-08-01", periods=30).date
    records = []
    for d in dates:
        for _ in range(10):
            records.append({
                "date": d,
                "vader_compound": np.random.uniform(-1, 1),
                "finbert_label": np.random.choice(["positive", "negative", "neutral"]),
                "finbert_score": np.random.uniform(0.5, 1.0),
            })
    return pd.DataFrame(records)


@pytest.fixture
def merged_daily():
    """Create a minimal merged daily DataFrame."""
    np.random.seed(42)
    n = 30
    return pd.DataFrame({
        "date": pd.date_range("2022-08-01", periods=n).date,
        "mean_vader": np.random.uniform(-0.3, 0.3, n),
        "price_return": np.random.uniform(-0.05, 0.05, n),
        "log_return": np.random.uniform(-0.05, 0.05, n),
        "volume": np.random.uniform(20e9, 40e9, n),
        "tweet_count": np.random.randint(50, 500, n),
        "vader_pos_ratio": np.random.uniform(0.3, 0.7, n),
        "vader_neg_ratio": np.random.uniform(0.1, 0.4, n),
    })


class TestAggregateDailySentiment:
    def test_output_columns(self, sample_tweets):
        result = aggregate_daily_sentiment(sample_tweets)
        assert "mean_vader" in result.columns
        assert "tweet_count" in result.columns
        assert "vader_pos_ratio" in result.columns

    def test_correct_day_count(self, sample_tweets):
        result = aggregate_daily_sentiment(sample_tweets)
        assert len(result) == 30

    def test_tweet_count_per_day(self, sample_tweets):
        result = aggregate_daily_sentiment(sample_tweets)
        assert (result["tweet_count"] == 10).all()


class TestComputeLaggedCorrelations:
    def test_returns_dataframe(self, merged_daily):
        result = compute_lagged_correlations(
            merged_daily, "mean_vader", "price_return", max_lag=3
        )
        assert isinstance(result, pd.DataFrame)
        assert "lag" in result.columns
        assert "pearson_r" in result.columns

    def test_lag_range(self, merged_daily):
        result = compute_lagged_correlations(
            merged_daily, "mean_vader", "price_return", max_lag=5
        )
        assert result["lag"].min() == 0
        assert result["lag"].max() == 5

    def test_correlation_bounds(self, merged_daily):
        result = compute_lagged_correlations(
            merged_daily, "mean_vader", "price_return"
        )
        assert (result["pearson_r"].abs() <= 1.0).all()
        assert (result["spearman_r"].abs() <= 1.0).all()


class TestGrangerTest:
    def test_returns_results(self, merged_daily):
        result = run_granger_test(
            merged_daily, "mean_vader", "price_return", max_lag=3
        )
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "f_stat" in result.columns
            assert "p_value" in result.columns


class TestAdfTest:
    def test_returns_dict(self, merged_daily):
        result = adf_test(merged_daily["mean_vader"], "VADER")
        assert isinstance(result, dict)
        assert "adf_statistic" in result
        assert "p_value" in result
        assert "is_stationary" in result


class TestCorrelationMatrix:
    def test_square_output(self, merged_daily):
        cols = ["mean_vader", "price_return", "volume"]
        result = correlation_matrix(merged_daily, cols)
        assert result.shape == (3, 3)
        assert list(result.columns) == cols

    def test_diagonal_is_one(self, merged_daily):
        cols = ["mean_vader", "price_return"]
        result = correlation_matrix(merged_daily, cols)
        for c in cols:
            assert abs(result.loc[c, c] - 1.0) < 1e-10
