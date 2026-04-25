"""Baseline direction-of-return classifiers with walk-forward CV and bootstrap CI."""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

_MODELS: Dict[str, object] = {
    "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
    "Dummy (stratified)": DummyClassifier(strategy="stratified", random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50, max_depth=3, random_state=42
    ),
}


def make_features(daily_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and target y from daily merged data.

    Features (at time t):
        mean_vader_t, vader_pos_ratio_t, vader_neg_ratio_t, tweet_count_t,
        mean_vader_{t-1}, mean_vader_{t-2}, mean_vader_{t-3}

    Target: sign of next-day price return (1 = up, 0 = down/flat).
    Rows where price_return_{t+1} is NaN are dropped.

    Args:
        daily_df: Daily merged DataFrame (must have mean_vader, vader_pos_ratio,
                  vader_neg_ratio, tweet_count, price_return, sorted by date).

    Returns:
        (X, y) where X is a DataFrame and y is a binary Series.
    """
    df = daily_df.sort_values("date").copy().reset_index(drop=True)

    df["lag1_vader"] = df["mean_vader"].shift(1)
    df["lag2_vader"] = df["mean_vader"].shift(2)
    df["lag3_vader"] = df["mean_vader"].shift(3)

    # Target: direction of *next* day's return (NaN preserved so last row is dropped)
    future_return = df["price_return"].shift(-1)
    df["target"] = np.where(future_return.isna(), np.nan, (future_return > 0).astype(float))

    feature_cols = [
        "mean_vader", "vader_pos_ratio", "vader_neg_ratio", "tweet_count",
        "lag1_vader", "lag2_vader", "lag3_vader",
    ]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in daily_df: {missing}")

    df_feat = df[feature_cols + ["target"]].dropna()
    X = df_feat[feature_cols].reset_index(drop=True)
    y = df_feat["target"].reset_index(drop=True)
    return X, y


def walk_forward_splits(
    n_samples: int, n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window time-series splits (train always precedes test).

    Args:
        n_samples: Total number of samples.
        n_splits: Number of splits.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    if n_samples < n_splits + 1:
        raise ValueError(
            f"n_samples={n_samples} is too small for n_splits={n_splits}"
        )
    fold_size = n_samples // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        test_end = min((i + 1) * fold_size, n_samples)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Percentile bootstrap 95% CI for a binary classification metric.

    Args:
        y_true: True labels.
        y_pred: Predicted labels or scores.
        metric_fn: Function(y_true, y_pred) -> float.
        n_boot: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    boot_scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_scores[i] = metric_fn(y_true[idx], y_pred[idx])
    lo = float(np.percentile(boot_scores, 2.5))
    hi = float(np.percentile(boot_scores, 97.5))
    return float(point), lo, hi


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC that returns 0.5 when only one class is present."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    n_splits: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Walk-forward evaluation of a single model.

    Args:
        X: Feature DataFrame (time-ordered).
        y: Binary target Series (time-ordered).
        model: Scikit-learn estimator.
        n_splits: Number of walk-forward splits.
        n_boot: Bootstrap iterations for CI.
        seed: Random seed.

    Returns:
        Dict with accuracy, f1, roc_auc and their bootstrap CIs.
    """
    X_arr = X.values
    y_arr = y.values.astype(int)

    splits = walk_forward_splits(len(y_arr), n_splits=n_splits)

    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[float] = []

    for train_idx, test_idx in splits:
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        if len(np.unique(y_train)) < 2:
            continue
        m = type(model)(**model.get_params())
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())
        if hasattr(m, "predict_proba"):
            probs = m.predict_proba(X_test)[:, 1]
        else:
            probs = preds.astype(float)
        all_prob.extend(probs.tolist())

    if not all_true:
        return {"accuracy": float("nan"), "f1": float("nan"), "roc_auc": float("nan")}

    yt = np.array(all_true)
    yp = np.array(all_pred)
    yprob = np.array(all_prob)

    acc, acc_lo, acc_hi = bootstrap_ci(yt, yp, accuracy_score, n_boot, seed)
    f1, f1_lo, f1_hi = bootstrap_ci(
        yt, yp, lambda a, b: f1_score(a, b, zero_division=0), n_boot, seed
    )
    roc, roc_lo, roc_hi = bootstrap_ci(
        yt, yprob, _safe_roc_auc, n_boot, seed
    )

    return {
        "accuracy": round(acc, 4),
        "accuracy_lower": round(acc_lo, 4),
        "accuracy_upper": round(acc_hi, 4),
        "f1": round(f1, 4),
        "f1_lower": round(f1_lo, 4),
        "f1_upper": round(f1_hi, 4),
        "roc_auc": round(roc, 4),
        "roc_auc_lower": round(roc_lo, 4),
        "roc_auc_upper": round(roc_hi, 4),
        "n_test_samples": len(yt),
    }


def evaluate_all(
    daily_df: pd.DataFrame,
    models: Optional[Dict[str, object]] = None,
    n_splits: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
    crypto_label: str = "",
) -> pd.DataFrame:
    """Evaluate all baseline models and return a results table.

    Args:
        daily_df: Daily merged DataFrame.
        models: Dict mapping model name → estimator. Defaults to _MODELS.
        n_splits: Walk-forward splits.
        n_boot: Bootstrap iterations.
        seed: Random seed.
        crypto_label: Label for 'crypto' column (e.g. 'btc').

    Returns:
        DataFrame with one row per model and columns:
        crypto, model, accuracy, accuracy_lower, accuracy_upper,
        f1, f1_lower, f1_upper, roc_auc, roc_auc_lower, roc_auc_upper,
        n_test_samples.
    """
    if models is None:
        models = _MODELS

    try:
        X, y = make_features(daily_df)
    except (ValueError, KeyError) as e:
        logger.warning("Could not build features: %s", e)
        return pd.DataFrame()

    if len(X) < n_splits + 2:
        logger.warning(
            "Too few samples (%d) for %d splits — skipping predictor",
            len(X), n_splits,
        )
        return pd.DataFrame()

    records = []
    for name, model in models.items():
        logger.info("  Evaluating: %s", name)
        result = evaluate_model(X, y, model, n_splits=n_splits, n_boot=n_boot, seed=seed)
        records.append({"crypto": crypto_label, "model": name, **result})

    return pd.DataFrame(records)


def make_panel_features(
    crypto_dfs: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build panel feature matrix from multiple cryptos with one-hot crypto encoding.

    Lag features are computed **per-crypto** to avoid cross-crypto contamination.
    One-hot columns (crypto_btc, crypto_eth, crypto_bnb, …) identify the asset.
    Rows are then sorted by date so walk-forward CV respects calendar time.

    Args:
        crypto_dfs: Mapping crypto_label → daily merged DataFrame.

    Returns:
        (X, y) where X includes lag features + one-hot dummies, y is binary.
    """
    base_feature_cols = [
        "mean_vader", "vader_pos_ratio", "vader_neg_ratio", "tweet_count",
        "lag1_vader", "lag2_vader", "lag3_vader",
    ]

    frames = []
    for label, daily_df in crypto_dfs.items():
        df = daily_df.sort_values("date").copy().reset_index(drop=True)

        df["lag1_vader"] = df["mean_vader"].shift(1)
        df["lag2_vader"] = df["mean_vader"].shift(2)
        df["lag3_vader"] = df["mean_vader"].shift(3)

        future_return = df["price_return"].shift(-1)
        df["target"] = np.where(
            future_return.isna(), np.nan, (future_return > 0).astype(float)
        )
        df["_crypto"] = label
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)

    dummies = pd.get_dummies(panel["_crypto"], prefix="crypto", dtype=float)
    dummy_cols = list(dummies.columns)
    panel = pd.concat([panel, dummies], axis=1)

    panel = panel.sort_values("date").reset_index(drop=True)

    feature_cols = base_feature_cols + dummy_cols
    missing = [c for c in base_feature_cols if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing columns in panel: {missing}")

    df_feat = panel[feature_cols + ["target"]].dropna()
    X = df_feat[feature_cols].reset_index(drop=True)
    y = df_feat["target"].reset_index(drop=True)
    return X, y


def evaluate_panel(
    crypto_dfs: Dict[str, pd.DataFrame],
    models: Optional[Dict[str, object]] = None,
    n_splits: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate all models on the combined panel dataset (crypto='panel').

    Args:
        crypto_dfs: Mapping crypto_label → daily merged DataFrame.
        models: Dict mapping model name → estimator. Defaults to _MODELS.
        n_splits: Walk-forward splits.
        n_boot: Bootstrap iterations.
        seed: Random seed.

    Returns:
        DataFrame with crypto='panel', one row per model.
    """
    if models is None:
        models = _MODELS

    try:
        X, y = make_panel_features(crypto_dfs)
    except (ValueError, KeyError) as e:
        logger.warning("Could not build panel features: %s", e)
        return pd.DataFrame()

    if len(X) < n_splits + 2:
        logger.warning(
            "Panel: too few samples (%d) for %d splits — skipping", len(X), n_splits
        )
        return pd.DataFrame()

    logger.info("Panel evaluation: n=%d samples from %d cryptos", len(X), len(crypto_dfs))
    records = []
    for name, model in models.items():
        logger.info("  [panel] Evaluating: %s", name)
        result = evaluate_model(X, y, model, n_splits=n_splits, n_boot=n_boot, seed=seed)
        records.append({"crypto": "panel", "model": name, **result})

    return pd.DataFrame(records)
