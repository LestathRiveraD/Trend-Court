import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from trendcourt import examiner


def random_series(n=240, seed=42):
    """Monthly decimal years from 1981 with a weak trend plus noise."""
    rng = np.random.default_rng(seed)
    t = 1981 + (np.arange(n) + 0.5) / 12
    y = 0.02 * (t - 1981) + rng.normal(scale=0.5, size=n)
    return t, y


def with_nans(t, y):
    y = y.copy()
    t = t.copy()
    y[[3, 50, 51, 199]] = np.nan
    t[[120]] = np.nan
    keep = ~(np.isnan(t) | np.isnan(y))
    return t, y, t[keep], y[keep]


# --- theil_sen ------------------------------------------------------------


def test_theil_sen_matches_scipy_bit_for_bit():
    t, y = random_series()
    res = examiner.theil_sen(t, y)
    ref = stats.theilslopes(y, t, alpha=0.95, method="separate")

    assert res.slope == ref.slope
    assert res.intercept == ref.intercept
    assert res.low_slope == ref.low_slope
    assert res.high_slope == ref.high_slope
    assert res.slope_per_decade == ref.slope * 10
    assert res.low_slope_per_decade == ref.low_slope * 10
    assert res.high_slope_per_decade == ref.high_slope * 10
    assert res.confidence == 0.95
    assert res.method == "separate"
    assert res.n == len(t)
    assert res.t_start == t[0]
    assert res.t_end == t[-1]


def test_theil_sen_alpha_is_confidence_level():
    """scipy's alpha is the confidence level: 0.99 must widen the interval."""
    t, y = random_series()
    r95 = examiner.theil_sen(t, y, confidence=0.95)
    r99 = examiner.theil_sen(t, y, confidence=0.99)
    assert r99.low_slope < r95.low_slope < r95.slope < r95.high_slope < r99.high_slope


def test_theil_sen_records_joint_method():
    t, y = random_series()
    res = examiner.theil_sen(t, y, method="joint")
    ref = stats.theilslopes(y, t, alpha=0.95, method="joint")
    assert res.method == "joint"
    assert res.intercept == ref.intercept


def test_theil_sen_drops_nan_pairs():
    t, y = random_series()
    t_nan, y_nan, t_clean, y_clean = with_nans(t, y)
    res = examiner.theil_sen(t_nan, y_nan)
    ref = stats.theilslopes(y_clean, t_clean, alpha=0.95, method="separate")

    assert res.n == len(t_clean) == len(t) - 5
    assert (res.slope, res.intercept, res.low_slope, res.high_slope) == (
        ref.slope, ref.intercept, ref.low_slope, ref.high_slope)
    assert res == examiner.theil_sen(t_clean, y_clean)


def test_theil_sen_accepts_pandas_series():
    t, y = random_series()
    s = pd.Series(y, index=pd.date_range("1981-01", periods=len(y), freq="MS"))
    assert examiner.theil_sen(t, s) == examiner.theil_sen(t, y)


def test_theil_sen_rejects_bad_confidence():
    t, y = random_series()
    with pytest.raises(ValueError):
        examiner.theil_sen(t, y, confidence=95)


# --- ols ------------------------------------------------------------------


def test_ols_matches_scipy_bit_for_bit():
    t, y = random_series()
    res = examiner.ols(t, y)
    ref = stats.linregress(t, y)

    assert res.slope == ref.slope
    assert res.intercept == ref.intercept
    assert res.r == ref.rvalue
    assert res.p == ref.pvalue
    assert res.stderr == ref.stderr
    assert res.intercept_stderr == ref.intercept_stderr
    assert res.slope_per_decade == ref.slope * 10
    assert res.stderr_per_decade == ref.stderr * 10
    assert res.n == len(t)
    assert res.t_start == t[0]
    assert res.t_end == t[-1]


def test_ols_drops_nan_pairs():
    t, y = random_series()
    t_nan, y_nan, t_clean, y_clean = with_nans(t, y)
    res = examiner.ols(t_nan, y_nan)
    ref = stats.linregress(t_clean, y_clean)

    assert res.n == len(t_clean)
    assert (res.slope, res.intercept, res.r, res.p, res.stderr) == (
        ref.slope, ref.intercept, ref.rvalue, ref.pvalue, ref.stderr)
    assert res == examiner.ols(t_clean, y_clean)


# --- shared behaviour -----------------------------------------------------


def test_per_decade_conversion():
    t = 1981 + (np.arange(240) + 0.5) / 12
    y = 0.03 * t
    for res in (examiner.theil_sen(t, y), examiner.ols(t, y)):
        assert res.slope == pytest.approx(0.03)
        assert res.slope_per_decade == pytest.approx(0.3)


def test_noise_free_line_gives_exact_slope():
    t = np.arange(0.0, 40.0, 1 / 12)
    y = 2 + 0.05 * t
    ts = examiner.theil_sen(t, y)
    lr = examiner.ols(t, y)

    assert ts.slope == pytest.approx(0.05, rel=1e-12)
    assert ts.low_slope == pytest.approx(0.05, rel=1e-12)
    assert ts.high_slope == pytest.approx(0.05, rel=1e-12)
    assert ts.intercept == pytest.approx(2.0, rel=1e-12)
    assert lr.slope == pytest.approx(0.05, rel=1e-12)
    assert lr.intercept == pytest.approx(2.0, rel=1e-12)
    assert lr.r == pytest.approx(1.0)


def test_window_fields_reflect_points_used():
    t, y = random_series()
    y = y.copy()
    y[:12] = np.nan  # first year missing
    for res in (examiner.theil_sen(t, y), examiner.ols(t, y)):
        assert res.n == len(t) - 12
        assert res.t_start == t[12]
        assert res.t_end == t[-1]


def test_results_round_trip_through_json():
    t, y = random_series()
    for res in (examiner.theil_sen(t, y), examiner.ols(t, y)):
        d = res.to_dict()
        assert all(type(v) in (float, int, str) for v in d.values())
        back = json.loads(json.dumps(d))
        assert back == d
        assert type(res)(**back) == res


def test_rejects_mismatched_or_too_short_input():
    with pytest.raises(ValueError):
        examiner.theil_sen([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        examiner.ols([1.0, 2.0, np.nan], [1.0, 2.0, 3.0])
