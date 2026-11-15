from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trendcourt import clerk

FIXTURE = Path(__file__).parent / "fixtures" / "power_point_tiny.json"


def monthly(n_months, start="1981-01", values=None):
    index = pd.date_range(start, periods=n_months, freq="MS")
    if values is None:
        values = np.random.default_rng(0).normal(size=n_months)
    return pd.Series(values, index=index, dtype=float)


def knock_out(series, n_missing):
    """Set ``n_missing`` interior months to NaN, keeping the first and last valid."""
    out = series.copy()
    idx = np.linspace(1, len(series) - 2, n_missing).round().astype(int)
    assert len(set(idx)) == n_missing
    out.iloc[idx] = np.nan
    return out


# --- load_power_point -----------------------------------------------------


def test_load_point_drops_annual_key_and_masks_fill_value():
    s = clerk.load_power_point(FIXTURE, "T2M", fill_value=-999.0)

    assert isinstance(s.index, pd.DatetimeIndex)
    assert len(s) == 12  # YYYY13 dropped
    assert s.index[0] == pd.Timestamp("2020-01-01")
    assert s.index[-1] == pd.Timestamp("2020-12-01")
    assert np.isnan(s.loc["2020-03-01"])
    assert s.isna().sum() == 1
    assert (s.dropna() != -999.0).all()
    assert 15.5 not in s.to_numpy()  # the annual value


def test_load_point_reads_fill_value_from_manifest_by_default():
    s = clerk.load_power_point(FIXTURE, "temperature")
    assert clerk.load_manifest()["fill_value"] == -999.0
    assert np.isnan(s.loc["2020-03-01"])


# --- aggregate ------------------------------------------------------------


def test_aggregate_equal_latitudes_is_plain_mean():
    df = pd.DataFrame(np.random.default_rng(1).normal(size=(24, 3)),
                      index=pd.date_range("2000-01", periods=24, freq="MS"))
    out = clerk.aggregate(df, [32.5, 32.5, 32.5])
    np.testing.assert_allclose(out.to_numpy(), df.mean(axis=1).to_numpy())


def test_aggregate_unequal_latitudes_uses_cos_lat_weights():
    lats = np.array([0.0, 60.0])  # weights 1 and 0.5
    df = pd.DataFrame({"a": [3.0, 1.0], "b": [0.0, 4.0]},
                      index=pd.date_range("2000-01", periods=2, freq="MS"))
    out = clerk.aggregate(df, lats)
    w = np.cos(np.deg2rad(lats))
    expected = (df.to_numpy() * w).sum(axis=1) / w.sum()
    np.testing.assert_allclose(out.to_numpy(), expected)
    np.testing.assert_allclose(out.to_numpy(), [2.0, 2.0])


def test_aggregate_renormalizes_over_remaining_points_and_counts_them():
    lats = [0.0, 60.0, 30.0]
    df = pd.DataFrame({"a": [3.0, np.nan, np.nan],
                       "b": [0.0, 4.0, np.nan],
                       "c": [1.0, 2.0, np.nan]},
                      index=pd.date_range("2000-01", periods=3, freq="MS"))
    out, counts = clerk.aggregate(df, lats, return_counts=True)

    w = np.cos(np.deg2rad(lats))
    assert out.iloc[1] == pytest.approx((4.0 * w[1] + 2.0 * w[2]) / (w[1] + w[2]))
    assert np.isnan(out.iloc[2])
    assert counts.tolist() == [3, 2, 0]


def test_aggregate_rejects_mismatched_latitudes():
    df = pd.DataFrame(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        clerk.aggregate(df, [1.0, 2.0])


# --- deseasonalize --------------------------------------------------------


def test_pure_seasonal_cycle_deseasonalizes_to_zero():
    s = monthly(30 * 12)
    s[:] = 15 + 10 * np.sin(2 * np.pi * (s.index.month - 1) / 12)
    anomalies, clim = clerk.deseasonalize(s, return_climatology=True)

    np.testing.assert_allclose(anomalies.to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(clim.loc[s.index.month].to_numpy(), s.to_numpy())
    assert list(clim.index) == list(range(1, 13))


def test_each_calendar_month_anomalies_average_to_zero():
    s = monthly(25 * 12)
    s = s + 10 * np.cos(2 * np.pi * s.index.month / 12) + 0.03 * np.arange(len(s))
    s = knock_out(s, 20)  # gaps must not break the property
    anomalies = clerk.deseasonalize(s)

    by_month = anomalies.groupby(anomalies.index.month).mean()
    np.testing.assert_allclose(by_month.to_numpy(), 0.0, atol=1e-12)
    assert anomalies.isna().sum() == 20


# --- quality gate ---------------------------------------------------------


def test_quality_gate_rejects_19_years():
    q = clerk.quality_gate(monthly(19 * 12))
    assert not q.passed
    assert q.verdict == clerk.INSUFFICIENT
    assert q.years_covered == 19
    assert "19.0 years" in q.reason


def test_quality_gate_rejects_11_percent_missing():
    s = knock_out(monthly(25 * 12), 33)  # 33 / 300 = 11%
    q = clerk.quality_gate(s)
    assert not q.passed
    assert q.verdict == clerk.INSUFFICIENT
    assert q.share_missing == pytest.approx(0.11)
    assert "missing" in q.reason


def test_quality_gate_accepts_20_years_with_10_percent_missing():
    s = knock_out(monthly(20 * 12), 24)  # 24 / 240 = 10%
    q = clerk.quality_gate(s)
    assert q.passed
    assert q.reason is None
    assert q.years_covered == 20
    assert q.share_missing == pytest.approx(0.10)
    assert (q.start, q.end) == (s.index[0], s.index[-1])


def test_quality_gate_counts_months_absent_from_index_as_missing():
    s = monthly(20 * 12).drop(pd.date_range("1990-01", periods=30, freq="MS"))
    q = clerk.quality_gate(s)
    assert q.n_missing == 30
    assert not q.passed


def test_quality_gate_all_nan():
    q = clerk.quality_gate(monthly(30 * 12, values=np.full(360, np.nan)))
    assert not q.passed and q.years_covered == 0


# --- time axis ------------------------------------------------------------


def test_time_in_years_is_mid_month_decimal_year():
    idx = pd.DatetimeIndex(["2000-01-01", "2000-12-01", "2003-07-01"])
    np.testing.assert_allclose(clerk.time_in_years(idx),
                               [2000 + 0.5 / 12, 2000 + 11.5 / 12, 2003 + 6.5 / 12])


def test_time_in_years_keeps_gaps():
    s = monthly(36).drop(pd.date_range("1982-01", periods=12, freq="MS"))
    t = clerk.time_in_years(s.index)
    assert t[12] - t[11] == pytest.approx(13 / 12)


# --- prepare (acceptance, real data) --------------------------------------


def test_prepare_mexicali_temperature():
    p = clerk.prepare("mexicali", "temperature")

    assert p.raw.index[0] == pd.Timestamp("1981-01-01")
    assert p.raw.index[-1] == pd.Timestamp("2025-12-01")
    assert len(p.raw) == len(p.anomalies) == len(p.time) == 45 * 12
    assert p.raw.notna().all()
    assert 10 < p.raw.mean() < 35  # degC, hot desert

    assert p.quality.passed
    assert p.quality.years_covered == 45

    assert list(p.climatology.index) == list(range(1, 13))
    assert p.climatology.idxmax() in (7, 8)  # peak in summer
    np.testing.assert_allclose(
        p.anomalies.groupby(p.anomalies.index.month).mean(), 0.0, atol=1e-10)

    assert p.time[0] == pytest.approx(1981 + 0.5 / 12)
    assert (p.n_points == 9).all()

    prov = p.provenance
    manifest = clerk.load_manifest()
    assert prov.source == manifest["source"]
    assert prov.accessed_utc == manifest["accessed_utc"]
    assert prov.power_code == "T2M"
    assert prov.units == "degC"
    assert prov.n_points == 9
    assert prov.period == "1981-01 to 2025-12"
    assert len(prov.urls) == 9
    assert "45 years" in prov.clerk_text() and "9 points" in prov.clerk_text()


def test_load_power_region_keeps_point_coordinates():
    df = clerk.load_power_region("mexicali", "T2M")
    assert df.shape == (540, 9)
    assert df.columns.names == ["lat", "lon"]
    assert sorted(set(df.columns.get_level_values("lat"))) == [32.1, 32.5, 32.9]


def test_unknown_region_and_variable():
    with pytest.raises(KeyError):
        clerk.load_power_region("atlantis", "temperature")
    with pytest.raises(KeyError):
        clerk.power_code("wind")


# --- load_giovanni_csv ------------------------------------------------------

GIOVANNI_HEADER = """Title:,"Time Series, Area-Averaged of 2-meter air temperature"
User Start Date:,1981-01-01T00:00:00Z
User End Date:,1981-06-30T23:59:59Z
User Bounding Box:,"-116,31.9,-114.8,33.1"
Data Bounding Box:,"-115.625,32,-115,33"
URL to Reproduce Results:,"https://giovanni.gsfc.nasa.gov/giovanni/#bbox=-116,31.9,-114.8,33.1"
Fill Value (mean_M2TMNXSLV_5_12_4_T2M):, 1000000000000000

time, mean_M2TMNXSLV_5_12_4_T2M
"""


def write_giovanni(tmp_path, rows):
    path = tmp_path / "g.csv"
    path.write_text(GIOVANNI_HEADER + "".join(f"{t},{v}\n" for t, v in rows))
    return path


def test_load_giovanni_skips_header_converts_kelvin_and_masks_fill(tmp_path):
    path = write_giovanni(tmp_path, [
        ("1981-01-01 00:00:00", "273.15"),
        ("1981-02-01 00:00:00", "1000000000000000"),
        ("1981-03-01 00:00:00", "300.0"),
        # April absent from the file
        ("1981-05-01 00:00:00", "1e+15"),
        ("1981-06-01 00:00:00", "250.65"),
    ])
    s = clerk.load_giovanni_csv(path)

    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.name == "time"
    assert list(s.index) == list(pd.date_range("1981-01-01", periods=6, freq="MS"))
    np.testing.assert_allclose(s.to_numpy(), [0.0, np.nan, 26.85, np.nan, np.nan, -22.5], equal_nan=True)


def test_read_giovanni_header_parses_boxes_and_fill(tmp_path):
    h = clerk.read_giovanni_header(write_giovanni(tmp_path, [("1981-01-01 00:00:00", "280")]))
    assert h.user_bbox == (-116.0, 31.9, -114.8, 33.1)
    assert h.data_bbox == (-115.625, 32.0, -115.0, 33.0)
    assert h.fill_value == 1e15
    assert h.column == "mean_M2TMNXSLV_5_12_4_T2M"
    assert h.n_header_lines == 8


def test_load_giovanni_rejects_mid_month_timestamps(tmp_path):
    path = write_giovanni(tmp_path, [("1981-01-15 00:00:00", "280"), ("1981-02-15 00:00:00", "281")])
    with pytest.raises(ValueError, match="month starts"):
        clerk.load_giovanni_csv(path)


def test_load_giovanni_without_time_row_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Title:,x\n1981-01-01,280\n")
    with pytest.raises(ValueError, match="'time'"):
        clerk.load_giovanni_csv(path)


@pytest.mark.parametrize("region", ["mexicali", "iberia", "amazon"])
def test_load_giovanni_real_files(region):
    s = clerk.load_giovanni_csv(clerk.GIOVANNI_DIR / f"{region}_M2TMNXSLV_T2M.csv")
    assert s.index[0] == pd.Timestamp("1981-01-01")
    assert s.index[-1] == pd.Timestamp("2025-12-01")
    assert len(s) == 540 and s.notna().all()
    assert -10 < s.min() and s.max() < 45  # °C, not Kelvin
