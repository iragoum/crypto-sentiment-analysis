"""Results integrity tests — validate experiment output files.

These tests check that generated CSV tables and PNG figures are
correct and consistent. All tests are skipped if the results
directory is not found or specific files are missing.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Base paths
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def _skip_if_missing(path):
    """Return a pytest skip marker if the path does not exist."""
    return pytest.mark.skipif(
        not path.exists(),
        reason=f"File not found: {path}",
    )


# ===========================================================================
# TestCorrelationMatrixCSV
# ===========================================================================

class TestCorrelationMatrixCSV:
    """Validate correlation_matrix.csv."""

    FILE = TABLES_DIR / "correlation_matrix.csv"
    pytestmark = _skip_if_missing(FILE)

    @pytest.fixture
    def corr_df(self):
        return pd.read_csv(self.FILE, index_col=0)

    def test_is_square(self, corr_df):
        assert corr_df.shape[0] == corr_df.shape[1]

    def test_diagonal_approximately_one(self, corr_df):
        for col in corr_df.columns:
            if col in corr_df.index:
                assert abs(corr_df.loc[col, col] - 1.0) < 0.01, (
                    f"Diagonal for '{col}' is {corr_df.loc[col, col]}, expected ~1.0"
                )

    def test_values_in_range(self, corr_df):
        assert (corr_df >= -1.01).all().all(), "Correlation values below -1"
        assert (corr_df <= 1.01).all().all(), "Correlation values above 1"

    def test_no_nan(self, corr_df):
        assert not corr_df.isna().any().any(), "NaN found in correlation matrix"

    def test_symmetric(self, corr_df):
        for i in corr_df.index:
            for j in corr_df.columns:
                if i in corr_df.columns and j in corr_df.index:
                    assert abs(corr_df.loc[i, j] - corr_df.loc[j, i]) < 0.001


# ===========================================================================
# TestLaggedCorrelationsCSV
# ===========================================================================

class TestLaggedCorrelationsCSV:
    """Validate lagged_correlations.csv."""

    FILE = TABLES_DIR / "lagged_correlations.csv"
    pytestmark = _skip_if_missing(FILE)

    @pytest.fixture
    def lag_df(self):
        return pd.read_csv(self.FILE)

    def test_has_required_columns(self, lag_df):
        required = {"lag", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "n_obs"}
        assert required.issubset(set(lag_df.columns))

    def test_pearson_r_in_range(self, lag_df):
        assert (lag_df["pearson_r"] >= -1).all()
        assert (lag_df["pearson_r"] <= 1).all()

    def test_spearman_r_in_range(self, lag_df):
        assert (lag_df["spearman_r"] >= -1).all()
        assert (lag_df["spearman_r"] <= 1).all()

    def test_p_values_in_range(self, lag_df):
        assert (lag_df["pearson_p"] >= 0).all()
        assert (lag_df["pearson_p"] <= 1).all()
        assert (lag_df["spearman_p"] >= 0).all()
        assert (lag_df["spearman_p"] <= 1).all()

    def test_lag_zero_present(self, lag_df):
        assert 0 in lag_df["lag"].values

    def test_n_obs_positive(self, lag_df):
        assert (lag_df["n_obs"] > 0).all()

    def test_has_multiple_lags(self, lag_df):
        assert len(lag_df) >= 2


# ===========================================================================
# TestGrangerCausalityCSV
# ===========================================================================

class TestGrangerCausalityCSV:
    """Validate granger_causality.csv."""

    FILE = TABLES_DIR / "granger_causality.csv"
    pytestmark = _skip_if_missing(FILE)

    @pytest.fixture
    def granger_df(self):
        return pd.read_csv(self.FILE)

    def test_has_required_columns(self, granger_df):
        required = {"lag", "f_stat", "p_value", "significant_0.05"}
        assert required.issubset(set(granger_df.columns))

    def test_p_value_in_range(self, granger_df):
        assert (granger_df["p_value"] >= 0).all()
        assert (granger_df["p_value"] <= 1).all()

    def test_f_stat_non_negative(self, granger_df):
        assert (granger_df["f_stat"] >= 0).all()

    def test_lags_start_at_one(self, granger_df):
        assert granger_df["lag"].min() >= 1

    def test_significance_column_valid(self, granger_df):
        valid = {"Yes", "No"}
        assert all(v in valid for v in granger_df["significant_0.05"])


# ===========================================================================
# TestAdfStationarityCSV
# ===========================================================================

class TestAdfStationarityCSV:
    """Validate adf_stationarity.csv."""

    FILE = TABLES_DIR / "adf_stationarity.csv"
    pytestmark = _skip_if_missing(FILE)

    @pytest.fixture
    def adf_df(self):
        return pd.read_csv(self.FILE)

    def test_has_required_columns(self, adf_df):
        required = {"series", "adf_statistic", "p_value", "is_stationary"}
        assert required.issubset(set(adf_df.columns))

    def test_p_value_in_range(self, adf_df):
        assert (adf_df["p_value"] >= 0).all()
        assert (adf_df["p_value"] <= 1).all()

    def test_has_rows(self, adf_df):
        assert len(adf_df) >= 1


# ===========================================================================
# TestFigures
# ===========================================================================

class TestFigures:
    """Validate that figure files exist and are not trivially small."""

    EXPECTED_FIGURES = [
        "sentiment_distribution.png",
        "sentiment_vs_price.png",
        "correlation_heatmap.png",
        "lagged_correlation.png",
        "tweet_volume.png",
        "scatter_sentiment_return.png",
    ]

    @pytest.mark.parametrize("filename", EXPECTED_FIGURES)
    def test_figure_exists_and_not_empty(self, filename):
        fig_path = FIGURES_DIR / filename
        if not fig_path.exists():
            pytest.skip(f"Figure not found: {fig_path}")
        size = fig_path.stat().st_size
        assert size > 1024, f"{filename} is too small ({size} bytes), likely empty/corrupt"


# ===========================================================================
# TestTables
# ===========================================================================

class TestTables:
    """Validate that table CSV files have data (not just headers)."""

    EXPECTED_TABLES = [
        "correlation_matrix.csv",
        "lagged_correlations.csv",
        "granger_causality.csv",
        "adf_stationarity.csv",
    ]

    @pytest.mark.parametrize("filename", EXPECTED_TABLES)
    def test_table_has_data(self, filename):
        table_path = TABLES_DIR / filename
        if not table_path.exists():
            pytest.skip(f"Table not found: {table_path}")
        df = pd.read_csv(table_path)
        assert len(df) > 0, f"{filename} has headers but no data rows"
