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


def smk_reference(y, alpha=0.05):
    """Direct library call the seasonal wrapper must equal."""
    v = np.asarray(y, dtype=float)
    v = np.pad(v, (0, -len(v) % 12), constant_values=np.nan)
    return pmk.correlated_multivariate_test(v.reshape(-1, 12), alpha=alpha)


def test_hamed_rao_matches_library_exactly():
    _, y = random_series()
    res = examiner.mk_hamed_rao(y)
    assert_matches_library(res, pmk.hamed_rao_modification_test(y, alpha=0.05, lag=3))
    assert (res.test, res.alpha, res.n, res.lag, res.period) == ("hamed_rao", 0.05, len(y), 3, None)


def test_hamed_rao_records_lag_and_alpha():
    _, y = random_series()
    res = examiner.mk_hamed_rao(y, alpha=0.1, lag=12)
    assert_matches_library(res, pmk.hamed_rao_modification_test(y, alpha=0.1, lag=12))
    assert res.lag == 12 and res.alpha == 0.1
    all_lags = examiner.mk_hamed_rao(y, alpha=0.1, lag=None)
    assert_matches_library(all_lags, pmk.hamed_rao_modification_test(y, alpha=0.1, lag=None))
    assert all_lags.lag is None
    assert len({res.var_s, all_lags.var_s, examiner.mk_hamed_rao(y).var_s}) == 3  # lag actually used


def test_seasonal_matches_library_exactly():
    y = seasonal_series()
    res = examiner.mk_seasonal(y)
    assert_matches_library(res, smk_reference(y.to_numpy()))
    assert (res.test, res.alpha, res.n, res.lag, res.period) == ("correlated_seasonal", 0.05, len(y), None, 12)


def test_seasonal_equals_correlated_seasonal_test_at_default_alpha():
    y = seasonal_series(slope=0.05)
    res = examiner.mk_seasonal(y)
    assert_matches_library(res, pmk.correlated_seasonal_test(y.to_numpy(), period=12))


def test_seasonal_forwards_alpha():
    """pymannkendall 1.4.3's correlated_seasonal_test ignores alpha (always
    0.05); the wrapper must not. Pick a series with 0.05 < p < 0.10."""
    for seed in range(200):
        y = seasonal_series(slope=0.02, seed=seed)
        lib = pmk.correlated_seasonal_test(y.to_numpy(), period=12, alpha=0.10)
        if 0.05 < lib.p < 0.10:
            break
    else:
        pytest.fail("no seed with 0.05 < p < 0.10")
    assert not lib.h  # library bug reproduced: alpha=0.10 ignored
    res = examiner.mk_seasonal(y, alpha=0.10)
    assert res.h is True and res.p == lib.p and res.alpha == 0.10
    assert examiner.mk_seasonal(y).h is False


def test_seasonal_accepts_period_index_and_drops_partial_final_year():
    y = seasonal_series(n=245)  # ends in May
    res = examiner.mk_seasonal(y)
    assert_matches_library(res, smk_reference(y.to_numpy()))
    assert res.n == 240  # the partial year is left out by the library
    assert examiner.mk_seasonal(y.to_period("M")) == res
    # The test statistics ignore the partial year; only the library's
    # reference-only intercept still uses its values.
    whole = examiner.mk_seasonal(y.iloc[:240])
    for f in ("trend", "h", "p", "z", "Tau", "s", "var_s", "n"):
        assert getattr(whole, f) == getattr(res, f), f


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
    assert_matches_library(res, pmk.hamed_rao_modification_test(y_nan, lag=3))
    assert res.n == len(y_clean)
    plain = pmk.original_test(y_clean)
    assert res.s == plain.s and res.Tau == plain.Tau
    assert np.isfinite(res.p)


def test_seasonal_nan_drops_whole_years_and_keeps_alignment():
    """The correlated test reshapes to years x 12 and then drops every year
    with a NaN month, so S is the sum of per-month S over complete years."""
    y = seasonal_series(slope=0.3)
    y.iloc[[0, 14, 15, 100, 101, 102]] = np.nan  # years 0, 1 and 8 incomplete

    res = examiner.mk_seasonal(y)
    assert_matches_library(res, smk_reference(y.to_numpy()))
    assert res.n == (20 - 3) * 12

    by_year = y.to_numpy().reshape(-1, 12)
    complete = by_year[~np.isnan(by_year).any(axis=1)]
    expected_s = sum(pmk.original_test(col).s for col in complete.T)
    assert res.s == expected_s
    assert examiner.mk_seasonal(monthly(complete.ravel())).s == res.s

    # Dropping the NaN before the call would shift months: a different S.
    shifted = smk_reference(y.dropna().to_numpy())
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
    y = np.random.default_rng(2).normal(size=240)
    mk, smk = examiner.mk_hamed_rao(y), examiner.mk_seasonal(monthly(y))
    assert mk.h is False and mk.trend == "no trend"
    assert smk.h is False and smk.trend == "no trend"
    c = examiner.consistency(mk, smk)
    assert c["same_significance"] is True and c["mk_significant"] is False


def test_correlated_seasonal_accounts_for_month_to_month_correlation():
    """Plain seasonal MK understates Var(S) under monthly autocorrelation;
    the correlated test must give a larger Var(S) for the same S."""
    rng = np.random.default_rng(5)
    e = np.zeros(360)
    for i in range(1, 360):
        e[i] = 0.8 * e[i - 1] + rng.normal()
    raw = monthly(10 + 5 * np.sin(2 * np.pi * np.arange(360) / 12) + e)
    res = examiner.mk_seasonal(raw)
    plain = pmk.seasonal_test(raw.to_numpy(), period=12)
    assert res.s == plain.s
    assert res.var_s > 2 * plain.var_s


def test_consistency_reports_disagreement():
    base = examiner.mk_hamed_rao(random_series()[1])
    mk = examiner.MKResult(**{**base.to_dict(), "s": 120.0, "h": True})
    smk = examiner.MKResult(**{**base.to_dict(), "test": "correlated_seasonal", "lag": None,
                               "period": 12, "s": -40.0, "h": False})
    c = examiner.consistency(mk, smk)
    assert (c["mk_sign"], c["smk_sign"]) == (1, -1)
    assert c["same_sign"] is False and c["same_significance"] is False and c["agree"] is False
    assert json.loads(json.dumps(c, allow_nan=False)) == c


def test_consistency_rejects_swapped_or_mismatched_inputs():
    mk = examiner.mk_hamed_rao(random_series()[1])
    smk = examiner.mk_seasonal(seasonal_series())
    with pytest.raises(ValueError, match="hamed_rao, correlated_seasonal"):
        examiner.consistency(smk, mk)
    with pytest.raises(ValueError, match="alpha"):
        examiner.consistency(mk, examiner.mk_seasonal(seasonal_series(), alpha=0.1))


def test_mk_results_round_trip_through_json():
    mk = examiner.mk_hamed_rao(random_series()[1])
    mk_all = examiner.mk_hamed_rao(random_series()[1], lag=None)
    smk = examiner.mk_seasonal(seasonal_series())
    for res in (mk, mk_all, smk):
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
    with pytest.raises(ValueError, match="alpha"):
        examiner.mk_seasonal(seasonal_series(), alpha=0)
    for lag in (0, -1, 2.5, True, len(y)):
        with pytest.raises(ValueError, match="lag"):
            examiner.mk_hamed_rao(y, lag=lag)


def test_hamed_rao_rejects_broken_variance_correction():
    """The Hamed-Rao n/n* factor can make Var(S) negative; the library then
    returns NaN z/p with h=False, 'no trend'. The wrapper must raise."""
    y = np.random.default_rng(400).standard_normal(480)
    with np.errstate(invalid="ignore"):
        ref = pmk.hamed_rao_modification_test(y)
    assert ref.var_s < 0 and np.isnan(ref.p) and not ref.h  # library breakdown reproduced
    with pytest.raises(ValueError, match="non-positive or undefined"):
        examiner.mk_hamed_rao(y, lag=None)
    res = examiner.mk_hamed_rao(y)  # the default lag=3 avoids it
    assert res.var_s > 0 and np.isfinite(res.p)
    with pytest.raises(ValueError, match="non-positive or undefined"):
        examiner.mk_hamed_rao(np.ones(30))  # constant series: Var(S) is NaN


def test_seasonal_requires_three_complete_years():
    with pytest.raises(ValueError, match="3 complete years"):
        examiner.mk_seasonal(seasonal_series(n=35))  # 2 complete years + 11 months
    gappy = seasonal_series(n=60)
    gappy.iloc[[5, 17, 29]] = np.nan  # 3 of 5 years incomplete
    with pytest.raises(ValueError, match="3 complete years"):
        examiner.mk_seasonal(gappy)
    res = examiner.mk_seasonal(seasonal_series(n=36))
    assert res.n == 36 and np.isfinite(res.p)


def test_seasonal_rejects_constant_series():
    with pytest.raises(ValueError, match="non-positive or undefined"):
        examiner.mk_seasonal(monthly(np.ones(60)))
