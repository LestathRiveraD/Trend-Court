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


# --- witness B (real files, no network) --------------------------------------

REGIONS = ["mexicali", "mexico_city", "yucatan", "sahel", "amazon", "iberia"]
WITNESS_VARS = ["temperature", "precipitation"]


@pytest.mark.parametrize("region", REGIONS)
@pytest.mark.parametrize("var", WITNESS_VARS)
def test_witness_loads_every_region_whole_years_no_nan_in_shared_period(region, var):
    b = clerk.load_witness(region, var)
    assert isinstance(b.index, pd.DatetimeIndex)
    assert (b.index == b.index.to_period("M").start_time).all()
    assert b.index[0] == pd.Timestamp("1981-01-01")
    assert b.index[-1] == pd.Timestamp("2025-12-01")
    assert len(b) % 12 == 0

    p = clerk.prepare_witness(region, var)
    assert p.raw.notna().all() and p.anomalies.notna().all()
    assert len(p.shared) % 12 == 0
    assert p.shared[0].month == 1 and p.shared[-1].month == 12
    assert len(p.raw) == len(p.anomalies) == len(p.time) == len(p.n_cells) == len(p.shared)
    assert (p.n_cells == p.provenance.grid.n_cells).all()
    assert p.quality.passed
    np.testing.assert_allclose(
        p.anomalies.groupby(p.anomalies.index.month).mean(), 0.0, atol=1e-10)


def test_witness_spot_check_matches_hand_computation_from_netcdf_cells():
    import netCDF4

    # GPCP Mexicali, July 2000: 2.5 deg cells centered at 31.25/33.75 N and
    # 243.75/246.25 E (-116.25/-113.75). Region box 31.9..33.1 N, -116..-114.8.
    with netCDF4.Dataset(clerk.REPO_ROOT / "data/raw/witness/gpcp_v23/mexicali.nc") as nc:
        nc.set_auto_maskandscale(False)
        times = netCDF4.num2date(nc["time"][:], nc["time"].units)
        k = next(i for i, t in enumerate(times) if (t.year, t.month) == (2000, 7))
        cell = {(float(la), float(lo)): float(nc["precip"][k, i, j])
                for i, la in enumerate(nc["lat"][:]) for j, lo in enumerate(nc["lon"][:])}

    # share of each cell inside the box: lat cell 30..32.5 -> 0.6/2.5, 32.5..35 -> 0.6/2.5;
    # lon cell -117.5..-115 -> 1.0/2.5, -115..-112.5 -> 0.2/2.5
    f_lat = {31.25: 0.6 / 2.5, 33.75: 0.6 / 2.5}
    f_lon = {243.75: 1.0 / 2.5, 246.25: 0.2 / 2.5}
    w = {(la, lo): np.cos(np.deg2rad(la)) * f_lat[la] * f_lon[lo] for la, lo in cell}
    expected = sum(w[c] * cell[c] for c in cell) / sum(w.values())

    got = clerk.load_witness("mexicali", "precipitation").loc["2000-07-01"]
    assert got == pytest.approx(expected, rel=1e-6)


def test_witness_spot_check_temperature_cells_inside_box():
    import netCDF4

    # GHCN_CAMS Amazon, Jan 1990: the 0.5 deg cells tile the box exactly
    # (-4..-2 N, -61..-59 E), so it is a plain cos(lat) mean of 16 cells, in degC.
    with netCDF4.Dataset(clerk.REPO_ROOT / "data/raw/witness/ghcncams/amazon.nc") as nc:
        nc.set_auto_maskandscale(False)
        times = netCDF4.num2date(nc["time"][:], nc["time"].units)
        k = next(i for i, t in enumerate(times) if (t.year, t.month) == (1990, 1))
        lats, lons = np.asarray(nc["lat"][:], float), np.asarray(nc["lon"][:], float) - 360
        air = np.asarray(nc["air"][k], float)
    li = (lats > -4) & (lats < -2)
    lj = (lons > -61) & (lons < -59)
    w = np.cos(np.deg2rad(lats[li]))[:, None] * np.ones(lj.sum())
    expected = (air[np.ix_(li, lj)] * w).sum() / w.sum() - 273.15

    grid = clerk.load_witness_grid("amazon", "temperature")[2]
    assert grid.n_cells == 16
    assert clerk.load_witness("amazon", "temperature").loc["1990-01-01"] == pytest.approx(expected, rel=1e-6)


def test_witness_temperature_is_celsius():
    b = clerk.load_witness("mexicali", "temperature")
    july = b[b.index.month == 7].mean()
    assert 25 < july < 40


def test_witness_soil_moisture_is_none():
    assert clerk.load_witness("mexicali", "soil_moisture") is None
    assert clerk.load_witness("mexicali", "GWETROOT") is None
    assert clerk.prepare_witness("mexicali", "soil_moisture") is None


def test_witness_accepts_power_code():
    pd.testing.assert_series_equal(
        clerk.load_witness("iberia", "T2M"), clerk.load_witness("iberia", "temperature"))


def test_witness_grid_records_cells_and_covered_box():
    _, _, grid = clerk.load_witness_grid("mexicali", "precipitation")
    assert grid.region_box == (31.9, -116.0, 33.1, -114.8)
    assert grid.requested_box == (30.65, -117.25, 34.35, -113.55)
    assert grid.covered_box == (30.0, -117.5, 35.0, -112.5)
    assert grid.grid_step == (2.5, 2.5)
    assert grid.n_cells == 4
    assert sum(c[2] for c in grid.cells) == pytest.approx(1.0)
    assert all(-180 <= c[1] <= 180 for c in grid.cells)
    assert grid.source_units == "mm/day"

    # GHCN_CAMS: a cell centered outside the box with no overlap is dropped
    _, _, t = clerk.load_witness_grid("mexicali", "temperature")
    assert all(lon != -116.25 for _, lon, _ in t.cells)
    assert t.covered_box == (31.5, -116.0, 33.5, -114.5)


def test_witness_provenance():
    manifest = clerk.load_witness_manifest()
    t = clerk.prepare_witness("sahel", "temperature").provenance
    p = clerk.prepare_witness("sahel", "precipitation").provenance

    assert t.independence == "independent"
    assert p.independence == "partly_independent"
    assert t.dataset_name == manifest["datasets"]["ghcncams"]
    assert p.dataset_name == manifest["datasets"]["gpcp_v23"]
    assert t.accessed_utc == p.accessed_utc == manifest["accessed_utc"]
    assert t.url.startswith("https://psl.noaa.gov/") and "ghcncams" in t.url
    assert t.units == "degC" and t.source_units == "degK"
    assert t.period == "1981-01 to 2025-12"
    assert "Independent of POWER" in t.clerk_text()
    assert "Partly independent of POWER" in p.clerk_text()


def test_mask_invalid_handles_fill_missing_and_valid_range():
    attrs = {"missing_value": np.float32(-9.96921e36), "_FillValue": np.float32(1e20),
             "valid_range": np.array([150.0, 400.0], dtype=np.float32)}
    raw = np.array([np.float32(-9.96921e36), np.float32(1e20), 100.0, 300.0, 500.0], dtype=np.float32)
    out = clerk._mask_invalid(raw, attrs)
    assert np.isnan(out[[0, 1, 2, 4]]).all()
    assert out[3] == 300.0


def test_prepare_witness_restricts_to_shared_months():
    a = clerk.prepare("yucatan", "precipitation").raw.copy()
    a.loc["1995-01-01":"1995-12-01"] = np.nan  # a hole in witness A
    a = a.loc[:"2020-12-01"]  # witness A ends early

    p = clerk.prepare_witness("yucatan", "precipitation", witness_a=a)
    assert p.shared[-1] == pd.Timestamp("2020-12-01")
    assert not any(t.year == 1995 for t in p.shared)
    assert len(p.raw) == (40 - 1) * 12
    assert p.quality.n_missing == 12  # the 1995 hole counts against the gate
    full = clerk.load_witness("yucatan", "precipitation")
    pd.testing.assert_series_equal(p.raw, full.loc[p.shared], check_names=False)
    # climatology over the shared months only
    np.testing.assert_allclose(
        p.climatology.to_numpy(), p.raw.groupby(p.raw.index.month).mean().to_numpy())
