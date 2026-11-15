import json

import numpy as np
import pandas as pd
import pytest
import pymannkendall as pmk
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
        back = json.loads(json.dumps(d, allow_nan=False))
        assert back == d
        assert type(res)(**back) == res


def test_rejects_mismatched_or_too_short_input():
    with pytest.raises(ValueError):
        examiner.theil_sen([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        examiner.ols([1.0, 2.0, np.nan], [1.0, 2.0, 3.0])


@pytest.mark.parametrize("fn", [examiner.theil_sen, examiner.ols])
@pytest.mark.parametrize("where", ["y", "t"])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_rejects_infinite_values(fn, where, value):
    t, y = random_series()
    t, y = t.copy(), y.copy()
    (y if where == "y" else t)[7] = value
    with pytest.raises(ValueError, match="infinite"):
        fn(t, y)


@pytest.mark.parametrize("fn", [examiner.theil_sen, examiner.ols])
def test_rejects_single_distinct_time(fn):
    with pytest.raises(ValueError, match="distinct"):
        fn([2000.0] * 3, [1.0, 2.0, 3.0])
    # Distinct times only before NaN-dropping still counts as degenerate.
    with pytest.raises(ValueError, match="distinct"):
        fn([2000.0, 2000.0, 2000.0, 2001.0], [1.0, 2.0, 3.0, np.nan])


def test_to_dict_is_strict_json_with_nan_fields():
    t, _ = random_series()
    res = examiner.ols(t, np.ones(len(t)))  # constant y: r, p, stderr are NaN
    assert np.isnan(res.r) and np.isnan(res.p)
    d = res.to_dict()
    assert d["r"] is None and d["p"] is None
    assert d["stderr"] is None and d["intercept_stderr"] is None
    assert d["slope"] == res.slope  # finite fields untouched
    text = json.dumps(d, allow_nan=False)
    assert json.loads(text) == d


# --- Mann–Kendall ---------------------------------------------------------

MK_FIELDS = ("trend", "h", "p", "z", "Tau", "s", "var_s", "slope", "intercept")


def monthly(values, start="1981-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"))


def seasonal_series(n=240, slope=0.0, seed=7):
    """Raw-like monthly series: seasonal cycle + optional trend + noise."""
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    y = 10 + 5 * np.sin(2 * np.pi * k / 12) + slope * k / 12 + rng.normal(scale=0.5, size=n)
    return monthly(y)


def assert_matches_library(res, ref):
    for f in MK_FIELDS:
        assert getattr(res, f) == getattr(ref, f), f


def test_hamed_rao_matches_library_exactly():
    _, y = random_series()
    res = examiner.mk_hamed_rao(y)
    assert_matches_library(res, pmk.hamed_rao_modification_test(y, alpha=0.05, lag=None))
    assert (res.test, res.alpha, res.n, res.lag, res.period) == ("hamed_rao", 0.05, len(y), None, None)


def test_hamed_rao_records_lag_and_alpha():
    _, y = random_series()
    res = examiner.mk_hamed_rao(y, alpha=0.1, lag=3)
    assert_matches_library(res, pmk.hamed_rao_modification_test(y, alpha=0.1, lag=3))
    assert res.lag == 3 and res.alpha == 0.1
    assert res.var_s != examiner.mk_hamed_rao(y, alpha=0.1).var_s  # lag actually used


def test_seasonal_matches_library_exactly():
    y = seasonal_series()
    res = examiner.mk_seasonal(y)
    assert_matches_library(res, pmk.seasonal_test(y.to_numpy(), period=12, alpha=0.05))
    assert (res.test, res.alpha, res.n, res.lag, res.period) == ("seasonal", 0.05, len(y), None, 12)


def test_seasonal_accepts_period_index_and_partial_final_year():
    y = seasonal_series(n=245)  # ends in May
    res = examiner.mk_seasonal(y)
    assert_matches_library(res, pmk.seasonal_test(y.to_numpy(), period=12, alpha=0.05))
    assert res.n == 245
    assert examiner.mk_seasonal(y.to_period("M")) == res


@pytest.mark.parametrize("start", ["1981-02", "1981-12"])
def test_seasonal_rejects_start_not_in_january(start):
    y = seasonal_series()
    y.index = pd.date_range(start, periods=len(y), freq="MS")
    with pytest.raises(ValueError, match="January"):
        examiner.mk_seasonal(y)


def test_seasonal_rejects_gaps_duplicates_and_unindexed_input():
    y = seasonal_series()
    with pytest.raises(ValueError, match="contiguous"):
        examiner.mk_seasonal(y.drop(y.index[30]))  # missing month shifts alignment
    with pytest.raises(ValueError, match="contiguous"):
        examiner.mk_seasonal(pd.concat([y.iloc[:13], y.iloc[12:]]))
    with pytest.raises(TypeError):
        examiner.mk_seasonal(y.to_numpy())
    with pytest.raises(TypeError):
        examiner.mk_seasonal(y.reset_index(drop=True))
    with pytest.raises(ValueError, match="period"):
        examiner.mk_seasonal(y, period=4)


def test_hamed_rao_nan_handling_in_installed_pymannkendall():
    """pymannkendall drops NaN: S, Var(S) before correction, Tau and n are
    those of the NaN-free values; gaps do not raise."""
    _, y = random_series()
    y_nan = y.copy()
    y_nan[[3, 50, 51, 199]] = np.nan
    y_clean = y_nan[~np.isnan(y_nan)]

    res = examiner.mk_hamed_rao(y_nan)
    assert_matches_library(res, pmk.hamed_rao_modification_test(y_nan))
    assert res.n == len(y_clean)
    plain = pmk.original_test(y_clean)
    assert res.s == plain.s and res.Tau == plain.Tau
    assert np.isfinite(res.p)


def test_seasonal_nan_keeps_month_alignment():
    """seasonal_test reshapes to years x 12 before skipping NaN, so S is the
    sum of per-calendar-month S with each month's NaN dropped."""
    y = seasonal_series(slope=0.3)
    y.iloc[[0, 14, 15, 100, 101, 102]] = np.nan

    res = examiner.mk_seasonal(y)
    assert_matches_library(res, pmk.seasonal_test(y.to_numpy(), period=12, alpha=0.05))
    assert res.n == len(y) - 6

    by_month = y.to_numpy().reshape(-1, 12)
    expected_s = sum(pmk.original_test(col[~np.isnan(col)]).s for col in by_month.T)
    assert res.s == expected_s

    # Dropping the NaN before the call would shift months: a different S.
    shifted = pmk.seasonal_test(y.dropna().to_numpy(), period=12)
    assert shifted.s != res.s


@pytest.mark.parametrize("slope, sign, trend", [(0.3, 1, "increasing"), (-0.3, -1, "decreasing")])
def test_strong_trend_detected_with_correct_sign(slope, sign, trend):
    raw = seasonal_series(slope=slope)
    anomalies = raw - raw.groupby(raw.index.month).transform("mean")
    mk, smk = examiner.mk_hamed_rao(anomalies), examiner.mk_seasonal(raw)
    for res in (mk, smk):
        assert res.h is True
        assert res.trend == trend
        assert np.sign(res.s) == np.sign(res.z) == sign
    assert examiner.consistency(mk, smk) == {
        "alpha": 0.05, "mk_sign": sign, "smk_sign": sign, "same_sign": True,
        "mk_significant": True, "smk_significant": True,
        "same_significance": True, "agree": True,
    }


def test_white_noise_fixed_seed_gives_no_trend():
    # Seed 2 of default_rng. Note: with the library default lag=None
    # (all lags) Hamed-Rao rejects white noise ~9% of the time at alpha
    # 0.05 for n=240, so not every seed gives h=False.
    y = np.random.default_rng(2).normal(size=240)
    mk, smk = examiner.mk_hamed_rao(y), examiner.mk_seasonal(monthly(y))
    assert mk.h is False and mk.trend == "no trend"
    assert smk.h is False and smk.trend == "no trend"
    c = examiner.consistency(mk, smk)
    assert c["same_significance"] is True and c["mk_significant"] is False


def test_consistency_reports_disagreement():
    base = examiner.mk_hamed_rao(random_series()[1])
    mk = examiner.MKResult(**{**base.to_dict(), "s": 120.0, "h": True})
    smk = examiner.MKResult(**{**base.to_dict(), "test": "seasonal", "period": 12, "s": -40.0, "h": False})
    c = examiner.consistency(mk, smk)
    assert (c["mk_sign"], c["smk_sign"]) == (1, -1)
    assert c["same_sign"] is False and c["same_significance"] is False and c["agree"] is False
    assert json.loads(json.dumps(c, allow_nan=False)) == c


def test_consistency_rejects_swapped_or_mismatched_inputs():
    mk = examiner.mk_hamed_rao(random_series()[1])
    smk = examiner.mk_seasonal(seasonal_series())
    with pytest.raises(ValueError, match="hamed_rao, seasonal"):
        examiner.consistency(smk, mk)
    with pytest.raises(ValueError, match="alpha"):
        examiner.consistency(mk, examiner.mk_seasonal(seasonal_series(), alpha=0.1))


def test_mk_results_round_trip_through_json():
    mk = examiner.mk_hamed_rao(random_series()[1], lag=3)
    smk = examiner.mk_seasonal(seasonal_series())
    for res in (mk, smk):
        d = res.to_dict()
        assert all(type(v) in (float, int, str, bool, type(None)) for v in d.values())
        assert type(d["h"]) is bool and type(d["s"]) is float and type(d["n"]) is int
        back = json.loads(json.dumps(d, allow_nan=False))
        assert back == d
        assert examiner.MKResult(**back) == res


def test_mk_rejects_bad_input():
    _, y = random_series()
    bad = y.copy()
    bad[5] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        examiner.mk_hamed_rao(bad)
    with pytest.raises(ValueError, match="infinite"):
        examiner.mk_seasonal(monthly(bad))
    with pytest.raises(ValueError, match="at least 3"):
        examiner.mk_hamed_rao([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="alpha"):
        examiner.mk_hamed_rao(y, alpha=5)
    for lag in (0, -1, 2.5, True, len(y)):
        with pytest.raises(ValueError, match="lag"):
            examiner.mk_hamed_rao(y, lag=lag)
