"""Correlation and causality analysis between sentiment and price returns."""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

logger = logging.getLogger(__name__)


def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tweet-level sentiment scores to daily level.

    Args:
        df: DataFrame with columns: date, vader_compound,
            vader_pos, vader_neg, finbert_label, finbert_score.

    Returns:
        Daily DataFrame with aggregated sentiment metrics.
    """
    agg = df.groupby("date").agg(
        tweet_count=("vader_compound", "count"),
        mean_vader=("vader_compound", "mean"),
        std_vader=("vader_compound", "std"),
        vader_pos_ratio=("vader_compound",
                         lambda x: (x > 0.05).sum() / len(x)),
        vader_neg_ratio=("vader_compound",
                         lambda x: (x < -0.05).sum() / len(x)),
    ).reset_index()

    if "finbert_label" in df.columns:
        fb = df.groupby("date").apply(
            lambda g: pd.Series({
                "finbert_pos_ratio": (g["finbert_label"] == "positive").mean(),
                "finbert_neg_ratio": (g["finbert_label"] == "negative").mean(),
                "finbert_neu_ratio": (g["finbert_label"] == "neutral").mean(),
                "mean_finbert_score": g["finbert_score"].mean(),
            })
        ).reset_index()
        agg = agg.merge(fb, on="date", how="left")

    return agg


def compute_lagged_correlations(
    merged: pd.DataFrame,
    sentiment_col: str,
    price_col: str,
    max_lag: int = 7,
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations at each lag.

    Positive lag k means sentiment at time t is correlated with
    price return at time t+k.

    Args:
        merged: DataFrame with daily sentiment and price columns.
        sentiment_col: Sentiment column name.
        price_col: Price/return column name.
        max_lag: Maximum lag in days.

    Returns:
        DataFrame with columns: lag, pearson_r, pearson_p,
        spearman_r, spearman_p, n_obs.
    """
    records = []
    s = merged[sentiment_col].values
    p = merged[price_col].values

    for lag in range(0, max_lag + 1):
        if lag == 0:
            s_lag, p_lag = s, p
        else:
            s_lag = s[:-lag]
            p_lag = p[lag:]

        mask = ~(np.isnan(s_lag) | np.isnan(p_lag))
        sl, pl = s_lag[mask], p_lag[mask]
        if len(sl) < 5:
            continue

        pr, pp = stats.pearsonr(sl, pl)
        sr, sp = stats.spearmanr(sl, pl)
        records.append({
            "lag": lag,
            "pearson_r": round(pr, 4),
            "pearson_p": round(pp, 4),
            "spearman_r": round(sr, 4),
            "spearman_p": round(sp, 4),
            "n_obs": int(mask.sum()),
        })

    return pd.DataFrame(records)


def run_granger_test(
    merged: pd.DataFrame,
    sentiment_col: str,
    price_col: str,
    max_lag: int = 7,
) -> pd.DataFrame:
    """Run Granger causality test: does sentiment Granger-cause price returns?

    H0: lagged sentiment values do NOT improve price return prediction.
    Rejection (p < 0.05) implies sentiment Granger-causes price.

    Args:
        merged: DataFrame with sentiment and price columns (no NaNs).
        sentiment_col: Sentiment column name.
        price_col: Return column name.
        max_lag: Maximum lag to test.

    Returns:
        DataFrame with columns: lag, f_stat, p_value, significant_0.05.
    """
    data = merged[[price_col, sentiment_col]].dropna()

    if len(data) < max_lag + 10:
        logger.warning("Too few observations for Granger test: %d", len(data))
        return pd.DataFrame()

    try:
        gc_res = grangercausalitytests(data.values, maxlag=max_lag, verbose=False)
        records = []
        for lag in range(1, max_lag + 1):
            f_stat = gc_res[lag][0]["ssr_ftest"][0]
            p_val = gc_res[lag][0]["ssr_ftest"][1]
            records.append({
                "lag": lag,
                "f_stat": round(f_stat, 4),
                "p_value": round(p_val, 4),
                "significant_0.05": "Yes" if p_val < 0.05 else "No",
            })
        return pd.DataFrame(records)
    except Exception as e:
        logger.error("Granger test error: %s", e)
        return pd.DataFrame()


def adf_test(series: pd.Series, name: str = "") -> dict:
    """Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Time series to test.
        name: Name of the series for reporting.

    Returns:
        Dict with: series, adf_statistic, p_value, is_stationary,
        critical_1pct, critical_5pct.
    """
    clean = series.dropna()
    r = adfuller(clean)
    return {
        "series": name,
        "adf_statistic": round(r[0], 4),
        "p_value": round(r[1], 4),
        "is_stationary": r[1] < 0.05,
        "critical_1pct": round(r[4]["1%"], 4),
        "critical_5pct": round(r[4]["5%"], 4),
    }


def correlation_matrix(
    merged: pd.DataFrame,
    cols: List[str],
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute a correlation matrix for given columns.

    Args:
        merged: Input DataFrame.
        cols: Columns to include.
        method: 'pearson' or 'spearman'.

    Returns:
        Square correlation DataFrame.
    """
    return merged[cols].corr(method=method)
