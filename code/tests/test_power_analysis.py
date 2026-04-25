"""Tests for src.power_analysis module."""
import numpy as np
import pytest

from src.power_analysis import correlation_power, min_detectable_r, power_table


class TestCorrelationPower:
    """Tests for correlation_power()."""

    def test_returns_float(self):
        assert isinstance(correlation_power(0.3, 50), float)

    def test_power_in_unit_interval(self):
        for r in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
            for n in [10, 20, 35, 100]:
                p = correlation_power(r, n)
                assert 0.0 <= p <= 1.0, f"power={p} for r={r}, n={n}"

    def test_high_power_large_n(self):
        """n=200, r=0.5 → power should be very close to 1."""
        assert correlation_power(0.5, 200) > 0.99

    def test_low_power_small_n(self):
        """n=35, r=0.15 → power should be well below 0.25."""
        assert correlation_power(0.15, 35) < 0.25

    def test_power_increases_with_n(self):
        """Larger sample gives higher power for same r."""
        p_small = correlation_power(0.3, 20)
        p_large = correlation_power(0.3, 100)
        assert p_small < p_large

    def test_power_increases_with_r(self):
        """Larger effect size gives higher power for same n."""
        p_weak = correlation_power(0.1, 50)
        p_strong = correlation_power(0.5, 50)
        assert p_weak < p_strong

    def test_zero_r_returns_alpha(self):
        """r=0 → power equals alpha (size of the test)."""
        assert correlation_power(0.0, 50, alpha=0.05) == pytest.approx(0.05, abs=1e-9)

    def test_r_negative_treated_as_absolute(self):
        """Negative r should give same power as positive r."""
        assert correlation_power(-0.3, 50) == pytest.approx(correlation_power(0.3, 50))

    def test_r_near_one_returns_one(self):
        assert correlation_power(0.9999, 20) == pytest.approx(1.0, abs=1e-4)

    def test_n_le_3_returns_alpha(self):
        """Degenerate case: n ≤ 3."""
        assert correlation_power(0.5, 3, alpha=0.05) == pytest.approx(0.05, abs=1e-9)

    def test_n35_r020_approx(self):
        """At n=35, r=0.20, power should be in [0.18, 0.24] (G*Power reference)."""
        p = correlation_power(0.20, 35)
        assert 0.18 <= p <= 0.24, f"Got power={p:.4f} for n=35, r=0.20"

    def test_n35_r030_approx(self):
        """At n=35, r=0.30, power should be in [0.30, 0.42]."""
        p = correlation_power(0.30, 35)
        assert 0.30 <= p <= 0.42, f"Got power={p:.4f} for n=35, r=0.30"

    def test_custom_alpha(self):
        """alpha=0.01 should give lower power than alpha=0.05."""
        p_strict = correlation_power(0.3, 50, alpha=0.01)
        p_loose = correlation_power(0.3, 50, alpha=0.05)
        assert p_strict < p_loose

    def test_perfect_r_returns_one(self):
        assert correlation_power(1.0, 50) == 1.0

    def test_n100_r05_high_power(self):
        """Textbook case: n=100, r=0.5 should exceed 0.99."""
        assert correlation_power(0.5, 100) > 0.99


class TestMinDetectableR:
    """Tests for min_detectable_r()."""

    def test_returns_float(self):
        assert isinstance(min_detectable_r(50), float)

    def test_in_unit_interval(self):
        r = min_detectable_r(50)
        assert 0.0 < r <= 1.0

    def test_decreases_with_n(self):
        """Larger n → smaller detectable r."""
        r_small = min_detectable_r(30)
        r_large = min_detectable_r(200)
        assert r_small > r_large

    def test_n35_power08_approx(self):
        """n=35, power=0.8 → minimum detectable r should be around 0.40–0.52."""
        r = min_detectable_r(35, power=0.8)
        assert 0.38 <= r <= 0.55, f"Got MDR={r:.4f} for n=35, power=0.8"

    def test_achieved_power_meets_target(self):
        """The returned r should actually achieve the target power."""
        n, target = 50, 0.8
        r = min_detectable_r(n, power=target)
        achieved = correlation_power(r, n)
        assert abs(achieved - target) < 0.01

    def test_higher_power_needs_larger_r(self):
        """Power=0.9 requires a larger r than power=0.8."""
        r80 = min_detectable_r(35, power=0.8)
        r90 = min_detectable_r(35, power=0.9)
        assert r80 < r90


class TestPowerTable:
    """Tests for power_table()."""

    def test_returns_dataframe(self):
        import pandas as pd
        df = power_table([35, 50], [0.2, 0.3])
        assert isinstance(df, pd.DataFrame)

    def test_columns(self):
        df = power_table([35], [0.2])
        assert set(df.columns) == {"n", "r", "power"}

    def test_row_count(self):
        df = power_table([20, 35, 50], [0.1, 0.2, 0.3])
        assert len(df) == 9  # 3 n × 3 r

    def test_power_values_in_unit_interval(self):
        df = power_table([35, 100], [0.1, 0.3, 0.5])
        assert (df["power"] >= 0).all()
        assert (df["power"] <= 1).all()

    def test_power_rounded_to_4dp(self):
        """power_table rounds to 4 decimal places."""
        df = power_table([35], [0.2])
        val = df["power"].iloc[0]
        assert val == round(val, 4)
