"""Gate 1 (a), ``project.md`` §5: on a real series, the engine's output
equals calling ``scipy`` and ``pymannkendall`` by hand, exactly (``==``).

Case: Mexicali Valley, temperature (POWER T2M, 1981-01 to 2025-12).
The direct calls below use the same arguments the Examiner documents
(95% interval, alpha 0.05, Hamed–Rao ``lag=3``) on the series loaded by
``clerk.prepare``.

The seasonal check is the *correlated* seasonal Mann–Kendall since the
statistics review on PR #12, so it is compared with
``pymannkendall.correlated_seasonal_test`` (the library's own entry
point, which reshapes the monthly vector to years × 12 itself), not with
``seasonal_test``.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pymannkendall as pmk
import pytest
from scipy import stats

import trendcourt
from trendcourt import case as case_module
from trendcourt import clerk

REGION, VARIABLE = "mexicali", "temperature"
REFERENCE = clerk.REPO_ROOT / "docs" / "gate1" / f"{REGION}_{VARIABLE}.json"
MK_FIELDS = ("trend", "h", "p", "z", "Tau", "s", "var_s", "slope", "intercept")


@pytest.fixture(scope="module")
def prepared():
    return clerk.prepare(REGION, VARIABLE)


@pytest.fixture(scope="module")
def case():
    return trendcourt.examine(REGION, VARIABLE)


def test_case_series_is_complete(prepared):
    """The direct calls below get the full arrays with no NaN-dropping, so
    any difference from the engine can only come from the engine."""
    assert prepared.quality.passed
    assert prepared.raw.index[0].month == 1
    assert len(prepared.raw) == 540
    assert not np.isnan(prepared.raw.to_numpy()).any()
    assert not np.isnan(prepared.anomalies.to_numpy()).any()


def test_theil_sen_equals_scipy(prepared, case):
    t, y = prepared.time, prepared.anomalies.to_numpy()
    ref = stats.theilslopes(y, t, alpha=0.95, method="separate")
    ts = case["theil_sen"]

    assert ts["slope"] == ref.slope
    assert ts["intercept"] == ref.intercept
    assert ts["low_slope"] == ref.low_slope
    assert ts["high_slope"] == ref.high_slope
    assert ts["slope_per_decade"] == ref.slope * 10
    assert ts["low_slope_per_decade"] == ref.low_slope * 10
    assert ts["high_slope_per_decade"] == ref.high_slope * 10
    assert ts["confidence"] == 0.95
    assert ts["method"] == "separate"
    assert ts["n"] == len(t)
    assert ts["t_start"] == t[0]
    assert ts["t_end"] == t[-1]


def test_ols_equals_scipy(prepared, case):
    t, y = prepared.time, prepared.anomalies.to_numpy()
    ref = stats.linregress(t, y)
    lr = case["ols"]

    assert lr["slope"] == ref.slope
    assert lr["intercept"] == ref.intercept
    assert lr["r"] == ref.rvalue
    assert lr["p"] == ref.pvalue
    assert lr["stderr"] == ref.stderr
    assert lr["intercept_stderr"] == ref.intercept_stderr
    assert lr["slope_per_decade"] == ref.slope * 10
    assert lr["stderr_per_decade"] == ref.stderr * 10
    assert lr["n"] == len(t)
    assert lr["t_start"] == t[0]
    assert lr["t_end"] == t[-1]


def test_mk_hamed_rao_equals_pymannkendall(prepared, case):
    y = prepared.anomalies.to_numpy()
    ref = pmk.hamed_rao_modification_test(y, alpha=0.05, lag=3)
    mk = case["mk_hamed_rao"]

    for field in MK_FIELDS:
        assert mk[field] == getattr(ref, field), field
    assert mk["test"] == "hamed_rao"
    assert mk["alpha"] == 0.05
    assert mk["lag"] == 3
    assert mk["n"] == len(y)
    assert mk["period"] is None


def test_mk_seasonal_equals_pymannkendall(prepared, case):
    y = prepared.raw.to_numpy()
    ref = pmk.correlated_seasonal_test(y, period=12, alpha=0.05)
    smk = case["mk_seasonal"]

    for field in MK_FIELDS:
        assert smk[field] == getattr(ref, field), field
    assert smk["test"] == "correlated_seasonal"
    assert smk["alpha"] == 0.05
    assert smk["period"] == 12
    assert smk["n"] == len(y)
    assert smk["lag"] is None


def test_consistency_follows_the_tests(case):
    mk, smk, c = case["mk_hamed_rao"], case["mk_seasonal"], case["consistency"]
    assert c["mk_sign"] == int(np.sign(mk["s"]))
    assert c["smk_sign"] == int(np.sign(smk["s"]))
    assert c["mk_significant"] == mk["h"]
    assert c["smk_significant"] == smk["h"]


def test_series_and_metadata_come_from_the_clerk(prepared, case):
    s = case["series"]
    assert s["raw"] == prepared.raw.tolist()
    assert s["anomaly"] == prepared.anomalies.tolist()
    assert s["time"] == prepared.time.tolist()
    assert s["climatology"] == prepared.climatology.tolist()
    assert s["n_points"] == [9] * 540
    assert s["months"][0] == "1981-01" and s["months"][-1] == "2025-12"
    assert case["region"] == REGION
    assert case["variable"] == VARIABLE
    assert case["quality"]["verdict"] == clerk.SUFFICIENT
    assert case["provenance"]["power_code"] == "T2M"


def test_case_is_strict_json_and_round_trips(case):
    """Strict JSON (no NaN/Infinity), and every float reads back bit-for-bit."""
    text = case_module.to_json(case)
    assert json.loads(text) == case


def test_examine_is_deterministic(case):
    clerk._read_json.cache_clear()  # second run re-reads every file from disk
    again = trendcourt.examine(REGION, VARIABLE)
    assert case_module.to_json(again) == case_module.to_json(case)


def test_reference_file_is_current(case):
    """``docs/gate1/mexicali_temperature.json`` is this run's output. If the
    data or pinned libraries change, regenerate it with ``python -m trendcourt mexicali temperature``."""
    assert REFERENCE.read_text(encoding="utf-8") == case_module.to_json(case)


def test_failed_quality_gate_skips_statistics(prepared, monkeypatch):
    quality = dataclasses.replace(prepared.quality, passed=False, verdict=clerk.INSUFFICIENT,
                                  reason="test")
    monkeypatch.setattr(clerk, "prepare",
                        lambda region, var: dataclasses.replace(prepared, quality=quality))
    out = trendcourt.examine(REGION, VARIABLE)
    assert out["quality"]["verdict"] == clerk.INSUFFICIENT
    for key in ("theil_sen", "ols", "mk_hamed_rao", "mk_seasonal", "consistency"):
        assert out[key] is None
    json.dumps(out, allow_nan=False)


def test_cli_writes_canonical_json(case, capsys):
    assert case_module.main([REGION, VARIABLE]) == 0
    assert capsys.readouterr().out == case_module.to_json(case)
    assert case_module.main([]) == 2
