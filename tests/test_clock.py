"""The Clock (``trendcourt/clock.py``, issue #45)."""

from __future__ import annotations

import json
import statistics

import numpy as np
import pytest

from trendcourt import benchmark, cache, clerk, clock, examiner

POWER = json.loads((clerk.REPO_ROOT / "docs" / "benchmark" / "power.json").read_text())


def observed(n_years: int, phi: float, slope_per_decade: float, seed, sigma: float = 1.0):
    """``(t, anomalies, ts)`` of a ``benchmark.synthetic_series`` record whose
    Theil–Sen slope is exactly ``slope_per_decade``.

    Theil–Sen is regression-equivariant (adding ``b * t`` adds ``b`` to every
    pairwise slope), so shifting the anomalies by the gap between the true
    and the fitted slope makes the fit exact. The Clock's trend assumption
    then holds by construction, and comparisons isolate what is varied.
    """
    s = benchmark.synthetic_series(n_years, phi, sigma, 10.0, slope_per_decade, seed=seed)
    a = clerk.deseasonalize(s).to_numpy()
    t = clerk.time_in_years(s.index)
    a = a + (slope_per_decade / examiner.DECADE - examiner.theil_sen(t, a).slope) * t
    ts = examiner.theil_sen(t, a)
    assert ts.slope_per_decade == pytest.approx(slope_per_decade, abs=1e-9)
    return t, a, ts


def run(t, a, ts, **kw):
    kw = {"alpha": 0.05, "horizon_years": 60, "n_sims": 100, "seed": 1, "workers": 1} | kw
    return clock.years_to_verdict(t, a, ts, **kw)


def test_fixed_seed_gives_identical_output():
    t, a, ts = observed(15, 0.3, 0.2, seed=3)
    first = run(t, a, ts, seed=11)
    assert run(t, a, ts, seed=11) == first
    assert run(t, a, ts, seed=11, workers=2) == first  # the worker count does not matter
    assert first["years_median"] is not None
    assert set(first) == {"years_median", "years_p80", "beyond_horizon", "alpha",
                          "horizon_years", "n_sims", "seed", "assumption"}
    assert first["assumption"] == clock.ASSUMPTION
    assert first["years_median"] <= first["years_p80"]


def test_simulated_futures_are_reproducible_and_continue_the_line():
    t, a, ts = observed(20, 0.0, 0.3, seed=4)
    f1 = clock.simulate_futures(t, a, ts, horizon_years=50, n_sims=400, seed=9)
    f2 = clock.simulate_futures(t, a, ts, horizon_years=50, n_sims=400, seed=9)
    assert np.array_equal(f1, f2)
    assert f1.shape == (400, 600)
    # The mean future follows the Theil-Sen line continued (noise averages out).
    t_last = t[-1] + 50.0
    assert f1[:, -1].mean() == pytest.approx(ts.intercept + ts.slope * t_last, abs=0.15)


def test_larger_slope_gives_fewer_years():
    # Same noise draws and record length; only the slope differs.
    small = run(*observed(15, 0.3, 0.1, seed=5))
    large = run(*observed(15, 0.3, 0.3, seed=5))
    assert large["years_median"] < small["years_median"]


def test_higher_r1_gives_more_years():
    # Same slope and the same stationary noise SD; only the persistence differs.
    low = run(*observed(15, 0.0, 0.2, seed=6))
    high = run(*observed(15, 0.7, 0.2, seed=6))
    assert high["years_median"] > low["years_median"]


def test_zero_slope_is_beyond_horizon():
    out = run(*observed(20, 0.3, 0.0, seed=7), horizon_years=40)
    assert out == out | {"years_median": None, "years_p80": None, "beyond_horizon": True}


def _power_crossing(phi: float, slope: float) -> float:
    """Record length at which the #18 power (alpha 0.05) crosses 50%, interpolated."""
    pts = sorted((c["n_years"], c["rates"]["0.05"]["rejection"]["rate"])
                 for c in POWER["cells"] if c["phi"] == phi and c["slope_per_decade"] == slope)
    for (n0, r0), (n1, r1) in zip(pts, pts[1:]):
        if r0 < 0.5 <= r1:
            return n0 + (0.5 - r0) * (n1 - n0) / (r1 - r0)
    raise AssertionError("no 50% crossing on the grid")


CROSSING_TOLERANCE_YEARS = 3.0
"""How far the Clock's 50% point may sit from the power curve's 50% crossing.

The Clock is conditional on one observed record and uses 200 simulations,
the power curve is unconditional (1000 replicates per cell); the median of
5 records agreed within 1 year when this test was written."""


@pytest.mark.parametrize(("phi", "slope", "n_observed"), [(0.0, 0.2, 10), (0.3, 0.2, 15)])
def test_agrees_with_power_by_record_length(phi, slope, n_observed):
    totals = []
    for rep in range(5):
        out = run(*observed(n_observed, phi, slope, seed=[7, rep]), n_sims=200, seed=rep)
        assert out["years_median"] is not None
        totals.append(n_observed + out["years_median"])
    assert statistics.median(totals) == pytest.approx(_power_crossing(phi, slope),
                                                      abs=CROSSING_TOLERANCE_YEARS)


def test_fast_p_is_bit_identical_to_the_examiner():
    rng = np.random.default_rng(3)
    obs = clerk.prepare("mexicali", "soil_moisture").anomalies.to_numpy()  # many ties
    for i in range(30):
        n = int(rng.integers(5, 700))
        if i % 3 == 0:
            x = np.concatenate([obs, np.round(rng.normal(size=n) * 0.01, 3)])
        elif i % 3 == 1:
            x = rng.normal(size=n) + 0.003 * np.arange(n)
        else:
            x = np.round(rng.normal(size=n), 1)  # heavy ties
        assert clock.hamed_rao_p(x) == examiner.mk_hamed_rao(x).p


def test_fast_p_defers_to_the_examiner_on_missing_values_and_bad_input():
    x = np.r_[np.arange(10.0), np.nan, np.arange(10.0)]
    assert clock.hamed_rao_p(x) == examiner.mk_hamed_rao(x).p
    with pytest.raises(ValueError):
        clock.hamed_rao_p(np.ones(20))  # constant: the examiner refuses it


@pytest.mark.parametrize("kw", [{"alpha": 0.0}, {"alpha": 1.0}, {"horizon_years": 0},
                                {"n_sims": 0}, {"workers": 0}, {"alpha": True}])
def test_rejects_bad_arguments(kw):
    with pytest.raises(ValueError):
        run(*observed(10, 0.0, 0.2, seed=8), **kw)


def test_committed_cache_has_a_clock_for_inconclusive_cases_only():
    for path in sorted(cache.OUT_DIR.glob("*.json")):
        if path.name == "index.json":
            continue
        case = json.loads(path.read_text())
        block = case["clock"]
        if case["verdict"]["label"] != "inconclusive":
            assert block is None, path.name
            continue
        assert block["assumption"] in (clock.ASSUMPTION, clock.NO_CLAIM), path.name
        assert block["seed"] == benchmark.case_seed_key(case["region"], case["variable"])
        if block["alpha"] is None:
            assert block == block | {"years_median": None, "years_p80": None,
                                     "beyond_horizon": True, "n_sims": 0}
        else:
            assert block["n_sims"] == clock.N_SIMS
            assert block["beyond_horizon"] == (block["years_median"] is None)
