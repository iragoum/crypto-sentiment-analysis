"""Tests for src.correlation_analyzer module — comprehensive coverage."""
import numpy as np
import pandas as pd
import pytest

from src.correlation_analyzer import (
    aggregate_daily_sentiment,
    compute_lagged_correlations,
    run_granger_test,
    run_granger_on_differenced,
    adf_test,
    correlation_matrix,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_tweets():
    """Create a minimal daily tweet DataFrame with VADER scores."""
    np.random.seed(42)
    dates = pd.date_range("2022-08-01", periods=30).date
    records = []
    for d in dates:
        for _ in range(10):
            records.append({
                "date": d,
                "vader_compound": np.random.uniform(-1, 1),
                "vader_pos": np.random.uniform(0, 1),
                "vader_neg": np.random.uniform(0, 1),
                "vader_neu": np.random.uniform(0, 1),
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_tweets_with_finbert(sample_tweets):
    """Tweet DataFrame with both VADER and FinBERT columns."""
    np.random.seed(42)
    n = len(sample_tweets)
    df = sample_tweets.copy()
    df["finbert_label"] = np.random.choice(["positive", "negative", "neutral"], n)
    df["finbert_score"] = np.random.uniform(0.5, 1.0, n)
    return df


@pytest.fixture
def merged_daily():
    """Create a minimal merged daily DataFrame."""
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


# ===========================================================================
# TestAggregateDailySentiment
# ===========================================================================

class TestAggregateDailySentiment:
    """Tests for daily sentiment aggregation."""

    def test_output_columns_vader_only(self, sample_tweets):
        result = aggregate_daily_sentiment(sample_tweets)
        expected = {"date", "tweet_count", "mean_vader", "std_vader",
                    "vader_pos_ratio", "vader_neg_ratio"}
        assert expected.issubset(set(result.columns))

    def test_output_columns_with_finbert(self, sample_tweets_with_finbert):
        result = aggregate_daily_sentiment(sample_tweets_with_finbert)
        assert "finbert_pos_ratio" in result.columns
        assert "finbert_neg_ratio" in result.columns
        assert "finbert_neu_ratio" in result.columns
        assert "mean_finbert_score" in result.columns

    def test_correct_day_count(self, sample_tweets):
        result = aggregate_daily_sentiment(sample_tweets)
        assert len(result) == 30

    def test_one_row_per_date(self, sample_tweets):
        result = aggregate_daily_sentiment(sample_tweets)
        assert result["date"].nunique() == len(result)

    def test_tweet_count_per_day(self, sample_tweets):
        result = aggregate_daily_sentiment(sample_tweets)
        assert (result["tweet_count"] == 10).all()

    def test_tweet_count_positive(self, sample_tweets):
        result = aggregate_daily_sentiment(sample_tweets)
        assert (result["tweet_count"] > 0).all()

    def test_vader_pos_ratio_in_range(self, sample_tweets):
        result = aggregate_daily_sentiment(sample_tweets)
        assert (result["vader_pos_ratio"] >= 0).all()
        assert (result["vader_pos_ratio"] <= 1).all()

    def test_vader_neg_ratio_in_range(self, sample_tweets):
        result = aggregate_daily_sentiment(sample_tweets)
        assert (result["vader_neg_ratio"] >= 0).all()
        assert (result["vader_neg_ratio"] <= 1).all()

    def test_mean_vader_in_range(self, sample_tweets):
        result = aggregate_daily_sentiment(sample_tweets)
        assert (result["mean_vader"] >= -1).all()
        assert (result["mean_vader"] <= 1).all()

    def test_std_vader_non_negative(self, sample_tweets):
        result = aggregate_daily_sentiment(sample_tweets)
        # std can be NaN if only one observation, but here we have 10 per day
        assert (result["std_vader"] >= 0).all()

    def test_finbert_ratios_sum_to_one(self, sample_tweets_with_finbert):
        result = aggregate_daily_sentiment(sample_tweets_with_finbert)
        total = (result["finbert_pos_ratio"]
                 + result["finbert_neg_ratio"]
                 + result["finbert_neu_ratio"])
        np.testing.assert_allclose(total, 1.0, atol=0.01)

    def test_single_day(self):
        df = pd.DataFrame({
            "date": ["2022-08-01"] * 5,
            "vader_compound": [0.5, -0.3, 0.0, 0.8, -0.1],
        })
        result = aggregate_daily_sentiment(df)
        assert len(result) == 1
        assert result["tweet_count"].iloc[0] == 5


# ===========================================================================
# TestComputeLaggedCorrelations
# ===========================================================================

class TestComputeLaggedCorrelations:
    """Tests for lagged correlation computation."""

    def test_returns_dataframe(self, merged_daily):
        result = compute_lagged_correlations(
            merged_daily, "mean_vader", "price_return", max_lag=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, merged_daily):
        result = compute_lagged_correlations(
            merged_daily, "mean_vader", "price_return", max_lag=3
        )
        expected = {"lag", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "n_obs"}
        assert set(result.columns) == expected

    def test_lag_zero_always_present(self, merged_daily):
        result = compute_lagged_correlations(
            merged_daily, "mean_vader", "price_return", max_lag=5
        )
        assert 0 in result["lag"].values

    def test_lag_range(self, merged_daily):
        result = compute_lagged_correlations(
            merged_daily, "mean_vader", "price_return", max_lag=5
        )
        assert result["lag"].min() == 0
        assert result["lag"].max() == 5

    def test_pearson_r_in_bounds(self, merged_daily):
        result = compute_lagged_correlations(
            merged_daily, "mean_vader", "price_return"
        )
        assert (result["pearson_r"] >= -1).all()
        assert (result["pearson_r"] <= 1).all()

    def test_spearman_r_in_bounds(self, merged_daily):
        result = compute_lagged_correlations(
            merged_daily, "mean_vader", "price_return"
        )
        assert (result["spearman_r"] >= -1).all()
        assert (result["spearman_r"] <= 1).all()

    def test_p_values_in_range(self, merged_daily):
        result = compute_lagged_correlations(
            merged_daily, "mean_vader", "price_return"
        )
        assert (result["pearson_p"] >= 0).all()
        assert (result["pearson_p"] <= 1).all()
        assert (result["spearman_p"] >= 0).all()
        assert (result["spearman_p"] <= 1).all()

    def test_n_obs_positive(self, merged_daily):
        result = compute_lagged_correlations(
            merged_daily, "mean_vader", "price_return"
        )
        assert (result["n_obs"] > 0).all()

    def test_n_obs_decreases_with_lag(self, merged_daily):
        result = compute_lagged_correlations(
            merged_daily, "mean_vader", "price_return", max_lag=5
        )
        # n_obs at lag 0 should be >= n_obs at lag 5
        n0 = result[result["lag"] == 0]["n_obs"].iloc[0]
        n5 = result[result["lag"] == 5]["n_obs"].iloc[0]
        assert n0 >= n5

    def test_too_few_observations_empty(self):
        """With fewer than 5 observations per lag, that lag is skipped."""
        tiny = pd.DataFrame({
            "mean_vader": [0.1, 0.2, 0.3],
            "price_return": [0.01, -0.01, 0.02],
        })
        result = compute_lagged_correlations(
            tiny, "mean_vader", "price_return", max_lag=3
        )
        # Lag 0 has 3 obs (< 5), so should be empty
        assert len(result) == 0

    def test_max_lag_zero(self, merged_daily):
        result = compute_lagged_correlations(
            merged_daily, "mean_vader", "price_return", max_lag=0
        )
        assert len(result) == 1
        assert result["lag"].iloc[0] == 0

    def test_nan_handling(self):
        """Rows with NaN should be masked out."""
        data = pd.DataFrame({
            "sent": [0.1, np.nan, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "ret": [0.01, 0.02, np.nan, 0.04, 0.05, 0.06, 0.07, 0.08],
        })
        result = compute_lagged_correlations(data, "sent", "ret", max_lag=0)
        if not result.empty:
            assert result["n_obs"].iloc[0] < 8  # NaN rows excluded


# ===========================================================================
# TestRunGrangerTest
# ===========================================================================

class TestRunGrangerTest:
    """Tests for Granger causality analysis."""

    def test_returns_dataframe(self, merged_daily):
        result = run_granger_test(
            merged_daily, "mean_vader", "price_return", max_lag=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, merged_daily):
        result = run_granger_test(
            merged_daily, "mean_vader", "price_return", max_lag=3
        )
        if not result.empty:
            expected = {"lag", "f_stat", "p_value", "significant_0.05"}
            assert set(result.columns) == expected

    def test_lags_start_at_one(self, merged_daily):
        result = run_granger_test(
            merged_daily, "mean_vader", "price_return", max_lag=3
        )
        if not result.empty:
            assert result["lag"].min() == 1

    def test_p_value_in_range(self, merged_daily):
        result = run_granger_test(
            merged_daily, "mean_vader", "price_return", max_lag=3
        )
        if not result.empty:
            assert (result["p_value"] >= 0).all()
            assert (result["p_value"] <= 1).all()

    def test_f_stat_non_negative(self, merged_daily):
        result = run_granger_test(
            merged_daily, "mean_vader", "price_return", max_lag=3
        )
        if not result.empty:
            assert (result["f_stat"] >= 0).all()

    def test_significant_column_values(self, merged_daily):
        result = run_granger_test(
            merged_daily, "mean_vader", "price_return", max_lag=3
        )
        if not result.empty:
            valid = {"Yes", "No"}
            assert all(v in valid for v in result["significant_0.05"])

    def test_significant_consistent_with_pvalue(self, merged_daily):
        result = run_granger_test(
            merged_daily, "mean_vader", "price_return", max_lag=3
        )
        if not result.empty:
            for _, row in result.iterrows():
                if row["p_value"] < 0.05:
                    assert row["significant_0.05"] == "Yes"
                else:
                    assert row["significant_0.05"] == "No"

    def test_too_few_rows_returns_empty(self):
        tiny = pd.DataFrame({
            "mean_vader": [0.1, 0.2, 0.3],
            "price_return": [0.01, -0.01, 0.02],
        })
        result = run_granger_test(tiny, "mean_vader", "price_return", max_lag=3)
        assert result.empty

    def test_nan_rows_dropped(self, merged_daily):
        df = merged_daily.copy()
        df.loc[0, "mean_vader"] = np.nan
        df.loc[1, "price_return"] = np.nan
        result = run_granger_test(df, "mean_vader", "price_return", max_lag=3)
        # Should still work after dropping NaNs
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# TestAdfTest
# ===========================================================================

class TestAdfTest:
    """Tests for ADF stationarity test."""

    def test_returns_dict(self, merged_daily):
        result = adf_test(merged_daily["mean_vader"], "VADER")
        assert isinstance(result, dict)

    def test_has_all_keys(self, merged_daily):
        result = adf_test(merged_daily["mean_vader"], "VADER")
        expected_keys = {
            "series", "adf_statistic", "p_value",
            "is_stationary", "critical_1pct", "critical_5pct",
        }
        assert set(result.keys()) == expected_keys

    def test_series_name(self, merged_daily):
        result = adf_test(merged_daily["mean_vader"], "My Series")
        assert result["series"] == "My Series"

    def test_p_value_in_range(self, merged_daily):
        result = adf_test(merged_daily["mean_vader"], "VADER")
        assert 0 <= result["p_value"] <= 1

    def test_is_stationary_is_bool(self, merged_daily):
        result = adf_test(merged_daily["mean_vader"], "VADER")
        assert isinstance(result["is_stationary"], (bool, np.bool_))

    def test_critical_values_are_float(self, merged_daily):
        result = adf_test(merged_daily["mean_vader"], "VADER")
        assert isinstance(result["critical_1pct"], float)
        assert isinstance(result["critical_5pct"], float)

    def test_critical_1pct_less_than_5pct(self, merged_daily):
        result = adf_test(merged_daily["mean_vader"], "VADER")
        # 1% critical value is more negative than 5%
        assert result["critical_1pct"] < result["critical_5pct"]

    def test_adf_statistic_is_float(self, merged_daily):
        result = adf_test(merged_daily["mean_vader"], "VADER")
        assert isinstance(result["adf_statistic"], float)

    def test_stationary_series(self):
        """A white noise series should be stationary."""
        np.random.seed(42)
        series = pd.Series(np.random.randn(200))
        result = adf_test(series, "white_noise")
        assert result["is_stationary"] == True  # noqa: E712 (np.bool_ vs bool)
        assert result["p_value"] < 0.05

    def test_nan_handling(self, merged_daily):
        """NaN values should be dropped before running the test."""
        s = merged_daily["mean_vader"].copy()
        s.iloc[0] = np.nan
        s.iloc[5] = np.nan
        result = adf_test(s, "with_nan")
        assert isinstance(result["adf_statistic"], float)

    def test_empty_name(self, merged_daily):
        result = adf_test(merged_daily["mean_vader"], "")
        assert result["series"] == ""


# ===========================================================================
# TestCorrelationMatrix
# ===========================================================================

class TestCorrelationMatrix:
    """Tests for correlation matrix computation."""

    def test_square_output(self, merged_daily):
        cols = ["mean_vader", "price_return", "volume"]
        result = correlation_matrix(merged_daily, cols)
        assert result.shape == (3, 3)

    def test_column_and_index_match(self, merged_daily):
        cols = ["mean_vader", "price_return", "volume"]
        result = correlation_matrix(merged_daily, cols)
        assert list(result.columns) == cols
        assert list(result.index) == cols

    def test_diagonal_is_one(self, merged_daily):
        cols = ["mean_vader", "price_return", "volume"]
        result = correlation_matrix(merged_daily, cols)
        for c in cols:
            assert abs(result.loc[c, c] - 1.0) < 1e-10

    def test_symmetric(self, merged_daily):
        cols = ["mean_vader", "price_return", "volume"]
        result = correlation_matrix(merged_daily, cols)
        for i in cols:
            for j in cols:
                assert abs(result.loc[i, j] - result.loc[j, i]) < 1e-10

    def test_values_in_range(self, merged_daily):
        cols = ["mean_vader", "price_return", "volume"]
        result = correlation_matrix(merged_daily, cols)
        assert (result >= -1).all().all()
        assert (result <= 1).all().all()

    def test_single_column(self, merged_daily):
        cols = ["mean_vader"]
        result = correlation_matrix(merged_daily, cols)
        assert result.shape == (1, 1)
        assert abs(result.iloc[0, 0] - 1.0) < 1e-10

    def test_two_columns(self, merged_daily):
        cols = ["mean_vader", "price_return"]
        result = correlation_matrix(merged_daily, cols)
        assert result.shape == (2, 2)

    def test_spearman_method(self, merged_daily):
        cols = ["mean_vader", "price_return"]
        result = correlation_matrix(merged_daily, cols, method="spearman")
        assert result.shape == (2, 2)
        # Diagonal still 1
        assert abs(result.iloc[0, 0] - 1.0) < 1e-10

    def test_no_nan_in_result(self, merged_daily):
        cols = ["mean_vader", "price_return", "volume"]
        result = correlation_matrix(merged_daily, cols)
        assert not result.isna().any().any()

    def test_perfectly_correlated(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        result = correlation_matrix(df, ["a", "b"])
        assert abs(result.loc["a", "b"] - 1.0) < 1e-10

    def test_perfectly_anticorrelated(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 8.0, 6.0, 4.0, 2.0],
        })
        result = correlation_matrix(df, ["a", "b"])
        assert abs(result.loc["a", "b"] - (-1.0)) < 1e-10


# ===========================================================================
# TestGrangerDifferenced
# ===========================================================================

class TestGrangerDifferenced:
    """Tests for run_granger_on_differenced()."""

    @pytest.fixture
    def stationary_merged(self):
        """Both series stationary (white noise, large n for reliable ADF detection)."""
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            "date": pd.date_range("2022-07-26", periods=n).date,
            "mean_vader": np.random.randn(n),
            "price_return": np.random.randn(n),
        })

    @pytest.fixture
    def nonstationary_merged(self):
        """Sentiment series is a random walk (non-stationary)."""
        np.random.seed(0)
        n = 40
        sentiment = np.cumsum(np.random.randn(n) * 0.1)
        returns = np.random.randn(n) * 0.02
        return pd.DataFrame({
            "date": pd.date_range("2022-07-26", periods=n).date,
            "mean_vader": sentiment,
            "price_return": returns,
        })

    def test_returns_dataframe(self, stationary_merged):
        result = run_granger_on_differenced(
            stationary_merged, "mean_vader", "price_return", max_lag=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self, stationary_merged):
        result = run_granger_on_differenced(
            stationary_merged, "mean_vader", "price_return", max_lag=3
        )
        if not result.empty:
            expected = {
                "lag", "f_stat", "p_value", "significant_0.05",
                "x_differenced", "y_differenced",
            }
            assert expected.issubset(set(result.columns))

    def test_differencing_flags_when_stationary(self, stationary_merged):
        """White noise → no differencing needed."""
        result = run_granger_on_differenced(
            stationary_merged, "mean_vader", "price_return", max_lag=3
        )
        if not result.empty:
            assert result["x_differenced"].iloc[0] == False  # noqa: E712
            assert result["y_differenced"].iloc[0] == False  # noqa: E712

    def test_y_differenced_flag_set_for_random_walk(self, nonstationary_merged):
        """Random-walk sentiment should be differenced."""
        result = run_granger_on_differenced(
            nonstationary_merged, "mean_vader", "price_return", max_lag=3
        )
        if not result.empty:
            assert result["y_differenced"].iloc[0] == True  # noqa: E712

    def test_p_value_in_range(self, stationary_merged):
        result = run_granger_on_differenced(
            stationary_merged, "mean_vader", "price_return", max_lag=3
        )
        if not result.empty:
            assert (result["p_value"] >= 0).all()
            assert (result["p_value"] <= 1).all()

    def test_f_stat_non_negative(self, stationary_merged):
        result = run_granger_on_differenced(
            stationary_merged, "mean_vader", "price_return", max_lag=3
        )
        if not result.empty:
            assert (result["f_stat"] >= 0).all()

    def test_significant_consistent_with_pvalue(self, stationary_merged):
        result = run_granger_on_differenced(
            stationary_merged, "mean_vader", "price_return", max_lag=3
        )
        if not result.empty:
            for _, row in result.iterrows():
                expected_sig = "Yes" if row["p_value"] < 0.05 else "No"
                assert row["significant_0.05"] == expected_sig

    def test_lags_start_at_one(self, stationary_merged):
        result = run_granger_on_differenced(
            stationary_merged, "mean_vader", "price_return", max_lag=4
        )
        if not result.empty:
            assert result["lag"].min() == 1

    def test_too_few_rows_returns_empty(self):
        tiny = pd.DataFrame({
            "mean_vader": [0.1, 0.2, 0.3],
            "price_return": [0.01, -0.01, 0.02],
        })
        result = run_granger_on_differenced(tiny, "mean_vader", "price_return")
        assert result.empty
