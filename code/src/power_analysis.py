"""Post-hoc statistical power analysis for Pearson correlation tests."""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

logger = logging.getLogger(__name__)


def correlation_power(r: float, n: int, alpha: float = 0.05) -> float:
    """Power of a two-sided Pearson correlation test via Fisher z-transformation.

    Formula:
        z_r    = 0.5 * ln((1 + |r|) / (1 - |r|))
        se     = 1 / sqrt(n - 3)
        z_α    = norm.ppf(1 - alpha/2)
        power  = 1 - Φ(z_α - z_r/se) + Φ(-z_α - z_r/se)

    Args:
        r: True population correlation (only |r| matters).
        n: Sample size.
        alpha: Two-sided significance level (default 0.05).

    Returns:
        Power in [0, 1].
    """
    r = abs(float(r))
    if n <= 3:
        return float(alpha)
    if r == 0.0:
        return float(alpha)
    if r >= 1.0:
        return 1.0

    z_r = 0.5 * np.log((1.0 + r) / (1.0 - r))
    se = 1.0 / np.sqrt(n - 3)
    z_alpha = norm.ppf(1.0 - alpha / 2.0)

    power = (
        1.0 - norm.cdf(z_alpha - z_r / se)
        + norm.cdf(-z_alpha - z_r / se)
    )
    return float(np.clip(power, 0.0, 1.0))


def min_detectable_r(n: int, power: float = 0.8, alpha: float = 0.05) -> float:
    """Minimum |r| detectable at the requested power given sample size n.

    Solved numerically via Brent's method.

    Args:
        n: Sample size.
        power: Desired statistical power (default 0.8).
        alpha: Significance level (default 0.05).

    Returns:
        Minimum detectable |r| in (0, 1].
    """
    def objective(r: float) -> float:
        return correlation_power(r, n, alpha) - power

    if objective(0.9999) < 0:
        logger.warning(
            "Cannot reach power=%.2f at n=%d with alpha=%.3f", power, n, alpha
        )
        return 1.0

    return float(brentq(objective, 1e-9, 0.9999, xtol=1e-6))


def power_table(
    n_list: List[int],
    r_list: List[float],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Power table for all (n, r) combinations.

    Args:
        n_list: Sample sizes.
        r_list: Correlation values.
        alpha: Significance level.

    Returns:
        DataFrame with columns: n, r, power.
    """
    records = [
        {"n": n, "r": r, "power": round(correlation_power(r, n, alpha), 4)}
        for n in n_list
        for r in r_list
    ]
    return pd.DataFrame(records)
