import json
import math
import subprocess
import sys
from pathlib import Path

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
    knmi_slope = round(ours.slope * 1.005, 8)
    per_year = knmi.compare("x", "annual", {"slope": knmi_slope, "units": "per_year", "resolution": 1e-8}, ours)
    assert per_year.passed
    assert per_year.bound == "relative"
    assert per_year.knmi_per_decade == pytest.approx(ours.slope_per_decade * 1.005)
    assert per_year.rel_diff == pytest.approx(0.005, rel=1e-4)

    knmi_slope = round(ours.slope_per_decade * 1.02, 6)
    entry = {"slope": knmi_slope, "units": "per_decade", "resolution": 1e-6, "stderr": 0.1}
    per_decade = knmi.compare("x", "annual", entry, ours)
    assert not per_decade.passed
    assert per_decade.knmi_stderr_per_decade == 0.1


def test_compare_allows_half_a_display_step():
    # KNMI showing 3 decimals in degC/yr: our iberia slope 0.034367 displays as
    # 0.034, 1.07% off. Correct, but it would fail a strict 1% test.
    t = np.arange(1981, 2026, dtype=float)
    ours = examiner.ols(t, 0.034367 * t + 0.01 * np.sin(t))
    entry = {"slope": round(ours.slope, 3), "units": "per_year", "resolution": 0.001}
    c = knmi.compare("iberia", "annual", entry, ours)
    assert c.rel_diff > knmi.REL_TOL
    assert c.passed and c.bound == "display"
    assert c.allowed == pytest.approx(0.0005 * examiner.DECADE)

    # One display step off is a real disagreement, not rounding.
    entry["slope"] = round(ours.slope, 3) - 0.001
    assert not knmi.compare("iberia", "annual", entry, ours).passed


def test_compare_requires_a_consistent_resolution():
    ours = fit(0.03)
    with pytest.raises(ValueError, match="resolution"):
        knmi.compare("x", "annual", {"slope": 0.03, "units": "per_year"}, ours)
    with pytest.raises(ValueError, match="positive"):
        knmi.compare("x", "annual", {"slope": 0.03, "units": "per_year", "resolution": 0}, ours)
    with pytest.raises(ValueError, match="more digits"):
        knmi.compare("x", "annual", {"slope": 0.0304, "units": "per_year", "resolution": 0.001}, ours)


def test_compare_rejects_bad_units_and_series():
    ours = fit(0.03)
    with pytest.raises(ValueError, match="units"):
        knmi.compare("x", "annual", {"slope": 0.03, "units": "per_month", "resolution": 0.001}, ours)
    with pytest.raises(ValueError, match="series"):
        knmi.compare("x", "daily", {"slope": 0.03, "units": "per_year", "resolution": 0.001}, ours)


def test_compare_all_skips_unfilled_entries():
    slopes = {"cases": {"iberia": {"monthly": {"slope": None, "units": "per_year"}}}}
    assert knmi.compare_all(slopes) == []


def test_compare_all_requires_the_record_behind_a_slope():
    entry = {"slope": 0.034, "units": "per_year", "resolution": 0.001, "knmi_settings": None, "evidence": "x.png"}
    with pytest.raises(ValueError, match="knmi_settings"):
        knmi.compare_all({"cases": {"iberia": {"annual": entry}}})


@pytest.mark.parametrize("slope, resolution", [(0.0, 0.1), (0.03, 0.01), (0.0, 1.0)])
def test_compare_coarse_display_is_inconclusive_not_a_pass(slope, resolution):
    t = np.arange(1981, 2026, dtype=float)
    ours = examiner.ols(t, 0.034384 * t + 0.01 * np.sin(t))  # iberia annual
    c = knmi.compare("iberia", "annual", {"slope": slope, "units": "per_year", "resolution": resolution}, ours)
    assert not c.passed
    assert c.inconclusive and c.bound == "inconclusive"
    assert knmi.gate_status([c] + [_row(True)] * 5, []) == knmi.GATE_PENDING


@pytest.mark.parametrize("region", knmi.CASES)
@pytest.mark.parametrize("series", knmi.SERIES)
def test_three_decimal_per_year_display_is_decisive_for_our_slopes(region, series):
    ours = knmi.series_for(region)["ols"][series]
    c = knmi.compare(region, series, {"slope": round(ours.slope, 3), "units": "per_year", "resolution": 0.001}, ours)
    assert c.passed and not c.inconclusive


def _filled(evidence):
    return {"slope": 0.034, "units": "per_year", "resolution": 0.001, "knmi_settings": "together", "evidence": evidence}


def test_compare_all_requires_evidence_file_under_docs_gate1(tmp_path):
    (tmp_path / "docs" / "gate1").mkdir(parents=True)
    (tmp_path / "docs" / "gate1" / "knmi_iberia_annual.png").write_bytes(b"png")
    (tmp_path / "elsewhere.png").write_bytes(b"png")
    for bad in ("docs/gate1/never_committed.png", "elsewhere.png", "docs/gate1/../../elsewhere.png", "docs/gate1"):
        with pytest.raises(ValueError, match="evidence"):
            knmi.compare_all({"cases": {"iberia": {"annual": _filled(bad)}}}, root=tmp_path)
    rows = knmi.compare_all({"cases": {"iberia": {"annual": _filled("docs/gate1/knmi_iberia_annual.png")}}}, root=tmp_path)
    assert len(rows) == 1 and rows[0].passed


def test_compare_script_exits_3_on_a_malformed_entry(tmp_path):
    slopes = knmi.load_knmi_slopes()
    slopes["cases"]["iberia"]["annual"]["slope"] = 0.034  # no resolution, settings or evidence
    path = tmp_path / "slopes.json"
    path.write_text(json.dumps(slopes), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "knmi_compare.py"
    proc = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True)
    assert proc.returncode == knmi.GATE_INVALID == 3
    assert "INVALID" in proc.stderr and "resolution" in proc.stderr


def _run_compare_script(tmp_path, text):
    path = tmp_path / "slopes.json"
    path.write_text(text, encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "knmi_compare.py"
    return subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True)


def _committed_with(edit):
    slopes = knmi.load_knmi_slopes()
    edit(slopes)
    return json.dumps(slopes)


def _rename_series(s):
    s["cases"]["iberia"]["daily"] = s["cases"]["iberia"].pop("annual")
    s["cases"]["iberia"]["daily"]["slope"] = 0.034


def _rename_case(s):
    s["cases"]["Iberia"] = s["cases"].pop("iberia")


def _drop_units(s):
    del s["cases"]["iberia"]["annual"]["units"]


def _numeric_evidence(s):
    s["cases"]["iberia"]["annual"].update(slope=0.034, resolution=0.001, knmi_settings="together", evidence=5)


@pytest.mark.parametrize(
    "make_text, message",
    [
        (lambda: knmi.SLOPES_FILE.read_text(encoding="utf-8").replace('"tolerance": 0.01,', '"tolerance": 0.01,,'), "Expecting"),
        (lambda: _committed_with(_rename_series), "unknown series 'daily'"),
        (lambda: _committed_with(_rename_case), "unknown case 'Iberia'"),
        (lambda: _committed_with(_drop_units), "missing 'units'"),
        (lambda: _committed_with(_numeric_evidence), "evidence must be a string"),
    ],
    ids=["json-syntax", "series-key", "case-key", "no-units", "evidence-type"],
)
def test_compare_script_exits_3_on_a_malformed_file(tmp_path, make_text, message):
    proc = _run_compare_script(tmp_path, make_text())
    assert proc.returncode == knmi.GATE_INVALID
    assert "INVALID" in proc.stderr and message in proc.stderr


def test_compare_script_exits_3_on_a_missing_file(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "knmi_compare.py"
    proc = subprocess.run([sys.executable, str(script), str(tmp_path / "nope.json")], capture_output=True, text=True)
    assert proc.returncode == knmi.GATE_INVALID


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"slope": "0.034", "units": "per_year"}, "slope must be a number"),
        ({"slope": True, "units": "per_year"}, "slope must be a number"),
        ({"slope": None, "units": "per_year", "resolution": [0.001]}, "resolution must be a number"),
        ({"slope": None, "units": "per_month"}, "units must be one of"),
        ({"units": "per_year"}, "missing 'slope'"),
        ({"slope": None, "units": "per_year", "knmi_settings": {"start": "together"}}, "knmi_settings must be a string"),
    ],
)
def test_validate_slopes_rejects_bad_entries(entry, message):
    with pytest.raises(ValueError, match=message):
        knmi.validate_slopes({"cases": {"iberia": {"annual": entry}}})


def test_validate_slopes_accepts_committed_file():
    knmi.validate_slopes(knmi.load_knmi_slopes())


# --- gate status / exit code ----------------------------------------------


def _row(passed):
    return knmi.Comparison("x", "annual", 0.3, 0.3, None, 0.0, 0.0, 0.003, "relative", knmi.REL_TOL, passed)


def all_filled():
    return {"cases": {r: {s: {"slope": 0.03} for s in knmi.SERIES} for r in knmi.CASES}}


def test_pending_entries_lists_null_and_absent_entries():
    slopes = all_filled()
    assert knmi.pending_entries(slopes) == []
    slopes["cases"]["iberia"]["monthly"]["slope"] = None
    del slopes["cases"]["amazon"]
    assert knmi.pending_entries(slopes) == [("iberia", "monthly"), ("amazon", "monthly"), ("amazon", "annual")]


def test_gate_status_outcomes():
    six_pass = [_row(True)] * 6
    assert knmi.gate_status(six_pass, []) == knmi.GATE_PASS == 0
    # one filled-in entry, five pending: not a pass (round-1 finding 2)
    assert knmi.gate_status([_row(True)], [("x", "s")] * 5) == knmi.GATE_PENDING == 2
    assert knmi.gate_status([], [("x", "s")] * 6) == knmi.GATE_PENDING
    assert knmi.gate_status([], []) == knmi.GATE_PENDING
    # a failure is reported even while other entries are pending
    assert knmi.gate_status([_row(True), _row(False)], [("x", "s")] * 4) == knmi.GATE_FAIL == 1
    assert knmi.gate_status([_row(False)] + [_row(True)] * 5, []) == knmi.GATE_FAIL
    # inconclusive rows are pending, but do not mask a real failure
    inconclusive = knmi.Comparison("x", "annual", 0.3, 0.0, None, 0.3, 1.0, 0.5, "inconclusive", knmi.REL_TOL, False, True)
    assert knmi.gate_status([inconclusive] + [_row(True)] * 5, []) == knmi.GATE_PENDING
    assert knmi.gate_status([inconclusive, _row(False)] + [_row(True)] * 4, []) == knmi.GATE_FAIL


def test_committed_slopes_file_accounts_for_every_entry():
    slopes = knmi.load_knmi_slopes()
    assert len(knmi.pending_entries(slopes)) + len(knmi.compare_all(slopes)) == len(knmi.CASES) * len(knmi.SERIES)


def test_slopes_file_lists_every_case_and_series():
    slopes = knmi.load_knmi_slopes()
    assert set(slopes["cases"]) == set(knmi.CASES)
    for by_series in slopes["cases"].values():
        assert set(by_series) == set(knmi.SERIES)
        for entry in by_series.values():
            assert entry["units"] in knmi.UNIT_FACTORS
            assert {"slope", "resolution", "stderr", "knmi_settings", "url", "accessed_utc", "evidence"} <= set(entry)
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
