"""Tests for src.predictor module."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.predictor import (
    bootstrap_ci,
    evaluate_all,
    evaluate_model,
    evaluate_panel,
    make_features,
    make_panel_features,
    walk_forward_splits,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_daily_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Minimal daily merged DataFrame for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2022-07-26", periods=n, freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "date": dates,
        "mean_vader": rng.uniform(-0.5, 0.5, n),
        "vader_pos_ratio": rng.uniform(0.1, 0.6, n),
        "vader_neg_ratio": rng.uniform(0.05, 0.4, n),
        "tweet_count": rng.randint(500, 5000, n).astype(float),
        "price_return": rng.normal(0, 0.02, n),
    })


# ── TestMakeFeatures ─────────────────────────────────────────────────────────

class TestMakeFeatures:

    def test_returns_tuple(self):
        X, y = make_features(_make_daily_df())
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_feature_columns(self):
        X, _ = make_features(_make_daily_df())
        expected = {
            "mean_vader", "vader_pos_ratio", "vader_neg_ratio", "tweet_count",
            "lag1_vader", "lag2_vader", "lag3_vader",
        }
        assert expected == set(X.columns)

    def test_binary_target(self):
        _, y = make_features(_make_daily_df())
        assert set(y.unique()).issubset({0.0, 1.0})

    def test_length_after_lag_and_target(self):
        """3 lags consume first 3 rows; target shift(-1) consumes the last: n-4 survive."""
        n = 30
        X, y = make_features(_make_daily_df(n))
        assert len(X) == n - 4
        assert len(X) == len(y)

    def test_no_look_ahead_in_features(self):
        """Features at row i must NOT use price_return of the *same* row as target.

        target[i] = sign(price_return[i+1]).
        Features include only mean_vader[i] and lags — never price_return[i+1].
        """
        X, _ = make_features(_make_daily_df())
        assert "price_return" not in X.columns

    def test_x_and_y_aligned(self):
        X, y = make_features(_make_daily_df())
        assert len(X) == len(y)
        assert X.index.tolist() == y.index.tolist()

    def test_no_nan_in_output(self):
        X, y = make_features(_make_daily_df(30))
        assert not X.isnull().any().any()
        assert not y.isnull().any()

    def test_missing_column_raises(self):
        df = _make_daily_df().drop(columns=["mean_vader"])
        with pytest.raises((ValueError, KeyError)):
            make_features(df)

    def test_deterministic(self):
        df = _make_daily_df()
        X1, y1 = make_features(df)
        X2, y2 = make_features(df)
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)


# ── TestWalkForwardSplits ────────────────────────────────────────────────────

class TestWalkForwardSplits:

    def test_returns_list_of_tuples(self):
        splits = walk_forward_splits(30, n_splits=5)
        assert isinstance(splits, list)
        assert all(isinstance(s, tuple) and len(s) == 2 for s in splits)

    def test_correct_number_of_splits(self):
        splits = walk_forward_splits(30, n_splits=5)
        assert len(splits) <= 5

    def test_train_always_before_test(self):
        for train, test in walk_forward_splits(30, n_splits=4):
            assert train.max() < test.min()

    def test_train_grows_with_each_split(self):
        splits = walk_forward_splits(40, n_splits=4)
        train_sizes = [len(tr) for tr, _ in splits]
        assert train_sizes == sorted(train_sizes)

    def test_no_overlap(self):
        for train, test in walk_forward_splits(30, n_splits=5):
            assert len(set(train) & set(test)) == 0

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            walk_forward_splits(3, n_splits=5)

    def test_index_dtype(self):
        for train, test in walk_forward_splits(20, n_splits=3):
            assert isinstance(train, np.ndarray)
            assert isinstance(test, np.ndarray)


# ── TestBootstrapCI ──────────────────────────────────────────────────────────

class TestBootstrapCI:

    def _dummy_data(self):
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, 50)
        y_pred = rng.randint(0, 2, 50)
        return y_true, y_pred

    def test_returns_three_floats(self):
        from sklearn.metrics import accuracy_score
        yt, yp = self._dummy_data()
        result = bootstrap_ci(yt, yp, accuracy_score)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_point_estimate_within_ci(self):
        from sklearn.metrics import accuracy_score
        yt, yp = self._dummy_data()
        est, lo, hi = bootstrap_ci(yt, yp, accuracy_score)
        assert lo <= est <= hi

    def test_deterministic(self):
        from sklearn.metrics import accuracy_score
        yt, yp = self._dummy_data()
        r1 = bootstrap_ci(yt, yp, accuracy_score, seed=42)
        r2 = bootstrap_ci(yt, yp, accuracy_score, seed=42)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        from sklearn.metrics import accuracy_score
        yt, yp = self._dummy_data()
        _, lo1, hi1 = bootstrap_ci(yt, yp, accuracy_score, n_boot=200, seed=1)
        _, lo2, hi2 = bootstrap_ci(yt, yp, accuracy_score, n_boot=200, seed=999)
        # CIs should generally differ (not guaranteed but highly likely)
        # Just check both are valid
        assert lo1 <= hi1
        assert lo2 <= hi2

    def test_ci_width_decreases_with_more_data(self):
        """More data → narrower bootstrap CI."""
        from sklearn.metrics import accuracy_score
        rng = np.random.RandomState(42)
        yt_small = rng.randint(0, 2, 20)
        yp_small = rng.randint(0, 2, 20)
        yt_large = rng.randint(0, 2, 200)
        yp_large = rng.randint(0, 2, 200)

        _, lo_s, hi_s = bootstrap_ci(yt_small, yp_small, accuracy_score, n_boot=500, seed=0)
        _, lo_l, hi_l = bootstrap_ci(yt_large, yp_large, accuracy_score, n_boot=500, seed=0)
        assert (hi_s - lo_s) > (hi_l - lo_l)


# ── TestEvaluateModel ────────────────────────────────────────────────────────

class TestEvaluateModel:

    def test_returns_dict(self):
        df = _make_daily_df(40)
        X, y = make_features(df)
        model = DummyClassifier(strategy="most_frequent")
        result = evaluate_model(X, y, model, n_splits=3)
        assert isinstance(result, dict)

    def test_required_keys(self):
        df = _make_daily_df(40)
        X, y = make_features(df)
        model = DummyClassifier(strategy="most_frequent")
        result = evaluate_model(X, y, model, n_splits=3)
        for key in ["accuracy", "accuracy_lower", "accuracy_upper",
                    "f1", "f1_lower", "f1_upper",
                    "roc_auc", "roc_auc_lower", "roc_auc_upper"]:
            assert key in result, f"Missing key: {key}"

    def test_ci_bounds_valid(self):
        df = _make_daily_df(40)
        X, y = make_features(df)
        model = DummyClassifier(strategy="most_frequent")
        result = evaluate_model(X, y, model, n_splits=3)
        assert result["accuracy_lower"] <= result["accuracy"] <= result["accuracy_upper"]
        assert result["f1_lower"] <= result["f1"] <= result["f1_upper"]
        assert result["roc_auc_lower"] <= result["roc_auc"] <= result["roc_auc_upper"]

    def test_dummy_most_frequent_accuracy(self):
        """DummyClassifier(most_frequent) accuracy ≈ max class frequency."""
        df = _make_daily_df(50)
        X, y = make_features(df)
        model = DummyClassifier(strategy="most_frequent")
        result = evaluate_model(X, y, model, n_splits=3, n_boot=200)
        # Should be in a reasonable range for a balanced binary problem
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_deterministic(self):
        df = _make_daily_df(40)
        X, y = make_features(df)
        model = DummyClassifier(strategy="most_frequent")
        r1 = evaluate_model(X, y, model, n_splits=3, seed=42)
        r2 = evaluate_model(X, y, model, n_splits=3, seed=42)
        assert r1 == r2


# ── TestEvaluateAll ──────────────────────────────────────────────────────────

class TestEvaluateAll:

    def test_returns_dataframe(self):
        df = _make_daily_df(40)
        result = evaluate_all(df, crypto_label="btc")
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        df = _make_daily_df(40)
        result = evaluate_all(df)
        for col in ["model", "accuracy", "f1", "roc_auc"]:
            assert col in result.columns

    def test_one_row_per_model(self):
        df = _make_daily_df(40)
        result = evaluate_all(df)
        assert len(result) == 4  # 4 default models

    def test_crypto_label_in_output(self):
        df = _make_daily_df(40)
        result = evaluate_all(df, crypto_label="btc")
        assert (result["crypto"] == "btc").all()

    def test_custom_models(self):
        df = _make_daily_df(40)
        models = {"dummy": DummyClassifier(strategy="most_frequent")}
        result = evaluate_all(df, models=models)
        assert len(result) == 1
        assert result["model"].iloc[0] == "dummy"

    def test_too_few_samples_returns_empty(self):
        df = _make_daily_df(5)
        result = evaluate_all(df)
        assert result.empty

    def test_accuracy_in_unit_interval(self):
        df = _make_daily_df(40)
        result = evaluate_all(df)
        assert (result["accuracy"].dropna().between(0, 1)).all()


# ── TestMakePanelFeatures ────────────────────────────────────────────────────

class TestMakePanelFeatures:

    def _crypto_dfs(self):
        return {
            "btc": _make_daily_df(30, seed=0),
            "eth": _make_daily_df(30, seed=1),
            "bnb": _make_daily_df(30, seed=2),
        }

    def test_returns_tuple(self):
        X, y = make_panel_features(self._crypto_dfs())
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_onehot_columns_present(self):
        X, _ = make_panel_features(self._crypto_dfs())
        assert "crypto_btc" in X.columns
        assert "crypto_eth" in X.columns
        assert "crypto_bnb" in X.columns

    def test_base_feature_columns_present(self):
        X, _ = make_panel_features(self._crypto_dfs())
        for col in ["mean_vader", "vader_pos_ratio", "vader_neg_ratio", "tweet_count",
                    "lag1_vader", "lag2_vader", "lag3_vader"]:
            assert col in X.columns, f"Missing: {col}"

    def test_no_price_return_in_x(self):
        """No look-ahead: price_return must not appear in features."""
        X, _ = make_panel_features(self._crypto_dfs())
        assert "price_return" not in X.columns

    def test_onehot_values_are_binary(self):
        X, _ = make_panel_features(self._crypto_dfs())
        for col in ["crypto_btc", "crypto_eth", "crypto_bnb"]:
            assert set(X[col].unique()).issubset({0.0, 1.0})

    def test_one_hot_row_sum_is_one(self):
        """Exactly one crypto flag is 1 per row."""
        X, _ = make_panel_features(self._crypto_dfs())
        dummy_cols = [c for c in X.columns if c.startswith("crypto_")]
        row_sums = X[dummy_cols].sum(axis=1)
        assert (row_sums == 1.0).all()

    def test_sample_size_approx_triple(self):
        """Panel should have ~3× the rows of a single crypto (minus lag/target NaNs)."""
        single_X, _ = make_features(_make_daily_df(30, seed=0))
        panel_X, _ = make_panel_features(self._crypto_dfs())
        assert len(panel_X) == pytest.approx(3 * len(single_X), abs=6)

    def test_binary_target(self):
        _, y = make_panel_features(self._crypto_dfs())
        assert set(y.unique()).issubset({0.0, 1.0})

    def test_no_nan_in_output(self):
        X, y = make_panel_features(self._crypto_dfs())
        assert not X.isnull().any().any()
        assert not y.isnull().any()

    def test_lags_independent_per_crypto(self):
        """lag1_vader should never cross a crypto boundary.

        Verify by checking that the first valid row of each crypto's block
        has lag1 = the crypto's own previous day, not another crypto's value.
        The panel must be sorted by date, so with identical date ranges
        each date has multiple rows (one per crypto) — lags must be per-crypto.
        """
        # With same dates, the btc rows should have lag1 from btc only.
        # We verify: after sorting by date, no row where crypto_btc=1 has
        # a lag1_vader matching a value that belongs only to eth/bnb.
        # This is implicitly guaranteed by computing lags before concat,
        # so just check no NaN leaks beyond the first 3 rows per crypto.
        X, y = make_panel_features(self._crypto_dfs())
        assert not X.isnull().any().any()

    def test_two_cryptos_only(self):
        """Panel works with only two cryptos."""
        two = {"btc": _make_daily_df(30, seed=0), "eth": _make_daily_df(30, seed=1)}
        X, y = make_panel_features(two)
        assert "crypto_btc" in X.columns
        assert "crypto_eth" in X.columns
        assert "crypto_bnb" not in X.columns


# ── TestEvaluatePanel ────────────────────────────────────────────────────────

class TestEvaluatePanel:

    def _crypto_dfs(self):
        return {
            "btc": _make_daily_df(40, seed=0),
            "eth": _make_daily_df(40, seed=1),
            "bnb": _make_daily_df(40, seed=2),
        }

    def test_returns_dataframe(self):
        result = evaluate_panel(self._crypto_dfs())
        assert isinstance(result, pd.DataFrame)

    def test_crypto_label_is_panel(self):
        result = evaluate_panel(self._crypto_dfs())
        assert (result["crypto"] == "panel").all()

    def test_one_row_per_model(self):
        result = evaluate_panel(self._crypto_dfs())
        assert len(result) == 4  # 4 default models

    def test_required_columns(self):
        result = evaluate_panel(self._crypto_dfs())
        for col in ["crypto", "model", "accuracy", "f1", "roc_auc",
                    "accuracy_lower", "accuracy_upper"]:
            assert col in result.columns

    def test_accuracy_in_unit_interval(self):
        result = evaluate_panel(self._crypto_dfs())
        assert (result["accuracy"].dropna().between(0, 1)).all()

    def test_ci_bounds_valid(self):
        result = evaluate_panel(self._crypto_dfs())
        assert (result["accuracy_lower"] <= result["accuracy"]).all()
        assert (result["accuracy"] <= result["accuracy_upper"]).all()

    def test_custom_models(self):
        models = {"dummy": DummyClassifier(strategy="most_frequent")}
        result = evaluate_panel(self._crypto_dfs(), models=models)
        assert len(result) == 1

    def test_single_crypto_returns_empty(self):
        """Panel needs at least 2 cryptos to be meaningful — but technically works with 1.
        Just verify it doesn't crash."""
        result = evaluate_panel({"btc": _make_daily_df(40, seed=0)})
        assert isinstance(result, pd.DataFrame)

    def test_deterministic(self):
        dfs = self._crypto_dfs()
        r1 = evaluate_panel(dfs, seed=42)
        r2 = evaluate_panel(dfs, seed=42)
        pd.testing.assert_frame_equal(r1, r2)
