import numpy as np
import pandas as pd
import pytest

from trendcourt import clerk, crosscheck


def seasonal(n_years=30, trend_per_decade=0.3, noise=0.5, seed=0, start="1981-01"):
    index = pd.date_range(start, periods=12 * n_years, freq="MS")
    t = clerk.time_in_years(index)
    rng = np.random.default_rng(seed)
    cycle = 10 * np.sin(2 * np.pi * (index.month - 1) / 12)
    return pd.Series(20 + cycle + trend_per_decade / 10 * (t - t[0]) + rng.normal(0, noise, len(index)),
                     index=index)


def test_identical_series_pass_with_zero_bias():
    s = seasonal()
    r = crosscheck.compare(s, s)
    assert r.bias == 0 and r.rmse == 0
    assert r.r_raw == pytest.approx(1) and r.r_anomaly == pytest.approx(1)
    assert r.slope_diff == 0 and r.intervals_overlap and r.passed


def test_constant_offset_is_bias_not_correlation_or_slope():
    s = seasonal()
    r = crosscheck.compare(s + 0.7, s)
    assert r.bias == pytest.approx(0.7) and r.rmse == pytest.approx(0.7)
    assert r.r_anomaly == pytest.approx(1)
    assert r.slope_diff == pytest.approx(0, abs=1e-9)
    assert r.passed


def test_bias_over_tolerance_fails_only_bias():
    s = seasonal()
    r = crosscheck.compare(s - 1.5, s)
    assert r.checks() == {"anomaly_r": True, "abs_bias": False, "slope_diff": True}
    assert not r.passed


def test_slope_difference_over_tolerance_fails():
    a, b = seasonal(trend_per_decade=0.8, noise=0.05), seasonal(trend_per_decade=0.3, noise=0.05)
    r = crosscheck.compare(a, b)
    assert r.slope_diff == pytest.approx(0.5, abs=0.05)
    assert r.slope_allowance == pytest.approx(0.25 * r.slope_a.slope_per_decade)
    assert not r.checks()["slope_diff"]


def test_uncorrelated_anomalies_fail_r():
    r = crosscheck.compare(seasonal(seed=1, trend_per_decade=0), seasonal(seed=2, trend_per_decade=0))
    assert r.r_raw > 0.95  # the shared seasonal cycle inflates raw r
    assert not r.checks()["anomaly_r"]


def test_slope_allowance_is_larger_of_absolute_and_relative():
    tol = crosscheck.GATE1
    assert tol.slope_allowance(0.2) == pytest.approx(0.1)
    assert tol.slope_allowance(0.8) == pytest.approx(0.2)
    assert tol.slope_allowance(-0.8) == pytest.approx(0.2)


def test_compare_uses_only_shared_non_nan_months():
    a = seasonal(n_years=30)
    b = seasonal(n_years=30, start="1986-01").copy()
    b.iloc[:12] = np.nan
    r = crosscheck.compare(a, b)
    assert r.start == "1987-01" and r.end == "2010-12"
    assert r.n == 24 * 12


def test_compare_rejects_short_overlap():
    s = seasonal(n_years=1)
    with pytest.raises(ValueError, match="24 shared"):
        crosscheck.compare(s, s)


def test_points_in_box_includes_edges():
    cols = pd.MultiIndex.from_tuples([(32.1, -115.8), (32.1, -115.4), (32.9, -115.0), (33.5, -115.0)],
                                     names=["lat", "lon"])
    df = pd.DataFrame(np.zeros((2, 4)), columns=cols)
    kept = crosscheck.points_in_box(df, (-115.625, 32.0, -115.0, 33.0))
    assert list(kept.columns) == [(32.1, -115.4), (32.9, -115.0)]
    with pytest.raises(ValueError, match="no points"):
        crosscheck.points_in_box(df, (0, 0, 1, 1))


def test_to_dict_is_strict_json():
    import json
    s = seasonal()
    json.dumps(crosscheck.compare(s + 0.1, s).to_dict(), allow_nan=False)


# --- acceptance: Gate 1 (b) on the real files -------------------------------


@pytest.mark.parametrize("region", ["mexicali", "iberia", "amazon"])
def test_gate1b_power_tracks_giovanni_merra2(region):
    merra2 = clerk.load_giovanni_csv(clerk.GIOVANNI_DIR / f"{region}_M2TMNXSLV_T2M.csv")
    r = crosscheck.compare(clerk.prepare(region, "T2M").raw, merra2)
    assert r.n == 540
    assert r.checks() == {"anomaly_r": True, "abs_bias": True, "slope_diff": True}
    assert r.intervals_overlap


def test_mexicali_data_box_removes_bias():
    path = clerk.GIOVANNI_DIR / "mexicali_M2TMNXSLV_T2M.csv"
    boxed, n = crosscheck.power_in_box("mexicali", clerk.read_giovanni_header(path).data_bbox)
    r = crosscheck.compare(boxed, clerk.load_giovanni_csv(path))
    assert n == 6
    assert abs(r.bias) < 0.02
