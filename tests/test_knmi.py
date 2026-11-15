import math

import numpy as np
import pandas as pd
import pytest

from trendcourt import examiner, knmi


def monthly(n, start="2000-01", values=None):
    index = pd.date_range(start, periods=n, freq="MS")
    if values is None:
        values = np.arange(n, dtype=float)
    return pd.Series(values, index=index, dtype=float)


def parse_knmi(text):
    """Read KNMI ASCII back: ``{year: [values]}``, -999.9 as NaN."""
    rows = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        year, *vals = line.split()
        rows[int(year)] = [math.nan if float(v) == knmi.MISSING else float(v) for v in vals]
    return rows


# --- format ---------------------------------------------------------------


def test_monthly_format_pads_partial_years_and_marks_missing():
    s = monthly(14, start="2000-03")
    s.iloc[2] = np.nan  # 2000-05
    text = knmi.to_knmi_monthly(s, ["hello"])

    assert text.splitlines()[0] == "# hello"
    rows = parse_knmi(text)
    assert sorted(rows) == [2000, 2001]
    assert all(len(v) == 12 for v in rows.values())
    assert np.isnan(rows[2000][:2]).all()  # Jan, Feb before the start
    assert np.isnan(rows[2000][4])  # the NaN month
    assert rows[2000][2] == 0.0 and rows[2001][3] == 13.0
    assert "-999.9" in text


def test_monthly_format_fills_whole_missing_years():
    s = pd.concat([monthly(12, "2000-01"), monthly(12, "2002-01")])
    rows = parse_knmi(knmi.to_knmi_monthly(s, []))
    assert sorted(rows) == [2000, 2001, 2002]
    assert np.isnan(rows[2001]).all()


def test_monthly_format_rejects_mid_month_and_duplicates():
    s = monthly(3)
    with pytest.raises(ValueError, match="month-start"):
        knmi.to_knmi_monthly(pd.Series([1.0], index=pd.DatetimeIndex(["2000-01-15"])), [])
    with pytest.raises(ValueError, match="duplicate"):
        knmi.to_knmi_monthly(pd.concat([s, s]), [])


def test_annual_means_require_twelve_months():
    s = monthly(30, start="2000-01", values=np.ones(30))
    s.iloc[13] = np.nan  # 2001 incomplete; 2002 has only 6 months
    means = knmi.annual_means(s)
    assert list(means.index) == [2000, 2001, 2002]
    assert means[2000] == 1.0
    assert np.isnan(means[2001]) and np.isnan(means[2002])


def test_annual_format_fills_gaps():
    s = pd.Series([1.0, np.nan, 3.0], index=[2000, 2001, 2003])
    rows = parse_knmi(knmi.to_knmi_annual(s, ["x"]))
    assert sorted(rows) == [2000, 2001, 2002, 2003]
    assert rows[2000] == [1.0] and np.isnan(rows[2001][0]) and np.isnan(rows[2002][0])


# --- compare --------------------------------------------------------------


def fit(slope_per_year):
    t = np.arange(2000, 2020, dtype=float)
    return examiner.ols(t, slope_per_year * t + np.sin(t))


def test_compare_converts_units_and_applies_tolerance():
    ours = fit(0.03)
    per_year = knmi.compare("x", "annual", {"slope": ours.slope * 1.005, "units": "per_year"}, ours)
    assert per_year.passed
    assert per_year.knmi_per_decade == pytest.approx(ours.slope_per_decade * 1.005)
    assert per_year.rel_diff == pytest.approx(0.005)

    per_decade = knmi.compare("x", "annual", {"slope": ours.slope_per_decade * 1.02, "units": "per_decade", "stderr": 0.1}, ours)
    assert not per_decade.passed
    assert per_decade.knmi_stderr_per_decade == 0.1


def test_compare_rejects_bad_units_and_series():
    ours = fit(0.03)
    with pytest.raises(ValueError, match="units"):
        knmi.compare("x", "annual", {"slope": 0.03, "units": "per_month"}, ours)
    with pytest.raises(ValueError, match="series"):
        knmi.compare("x", "daily", {"slope": 0.03, "units": "per_year"}, ours)


def test_compare_all_skips_unfilled_entries():
    slopes = {"cases": {"iberia": {"monthly": {"slope": None, "units": "per_year"}}}}
    assert knmi.compare_all(slopes) == []


def test_slopes_file_lists_every_case_and_series():
    slopes = knmi.load_knmi_slopes()
    assert set(slopes["cases"]) == set(knmi.CASES)
    for by_series in slopes["cases"].values():
        assert set(by_series) == set(knmi.SERIES)
        for entry in by_series.values():
            assert entry["units"] in knmi.UNIT_FACTORS
    assert slopes["tolerance"] == knmi.REL_TOL
    knmi.compare_all(slopes)  # filled entries must at least be well-formed


# --- the committed upload files match what ols() sees ---------------------


@pytest.mark.parametrize("region", knmi.CASES)
def test_committed_upload_files_round_trip_to_our_slope(region):
    s = knmi.series_for(region)

    rows = parse_knmi(knmi.upload_path(region, "monthly").read_text(encoding="utf-8"))
    years = sorted(rows)
    t = np.array([y + (m - 0.5) / 12 for y in years for m in range(1, 13)])
    y = np.array([v for yr in years for v in rows[yr]])
    assert examiner.ols(t, y).slope == pytest.approx(s["ols"]["monthly"].slope, rel=1e-4)

    rows = parse_knmi(knmi.upload_path(region, "annual").read_text(encoding="utf-8"))
    years = np.array(sorted(rows), dtype=float)
    y = np.array([rows[int(yr)][0] for yr in years])
    assert examiner.ols(years, y).slope == pytest.approx(s["ols"]["annual"].slope, rel=1e-4)
