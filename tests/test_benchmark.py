import math
import json

import numpy as np
import pandas as pd
import pytest

from trendcourt import benchmark, case, clerk, examiner, thresholds


# --- synthetic_series -------------------------------------------------------


def test_synthetic_series_is_deterministic_january_whole_years():
    a = benchmark.synthetic_series(30, 0.6, 1.0, 10.0, 0.2, seed=123)
    b = benchmark.synthetic_series(30, 0.6, 1.0, 10.0, 0.2, seed=123)
    c = benchmark.synthetic_series(30, 0.6, 1.0, 10.0, 0.2, seed=124)

    pd.testing.assert_series_equal(a, b, check_exact=True)
    assert not np.array_equal(a.to_numpy(), c.to_numpy())

    assert isinstance(a.index, pd.DatetimeIndex)
    assert a.index[0] == pd.Timestamp("1981-01-01")
    assert a.index[0].month == 1 and a.index[-1].month == 12
    assert len(a) == 30 * 12
    assert (a.index == pd.date_range("1981-01-01", periods=360, freq="MS")).all()
    assert np.isfinite(a.to_numpy()).all()


def test_synthetic_series_tuple_seed_matches_replicate_seed():
    seed = benchmark.replicate_seed(17, 0.9, 20, 5)
    assert seed == (17, 900, 20, 5)
    a = benchmark.synthetic_series(20, 0.9, 1.0, 10.0, seed=seed)
    b = benchmark.synthetic_series(20, 0.9, 1.0, 10.0, seed=(17, 900, 20, 5))
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_synthetic_series_noise_free_parts():
    """phi = 0 and sigma = 0: exactly the seasonal cycle plus the line."""
    y = benchmark.synthetic_series(20, 0.0, 0.0, 5.0, 0.3, seed=0)
    t = clerk.time_in_years(y.index)
    m = y.index.month.to_numpy()
    expected = 5.0 * np.sin(2 * np.pi * (m - 1) / 12) + 0.03 * (t - t[0])
    np.testing.assert_allclose(y.to_numpy(), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n_years", [20, 30, 45])
@pytest.mark.parametrize("slope", [0.25, -0.4])
def test_theil_sen_recovers_injected_slope_without_noise(n_years, slope):
    """Through the Clerk's deseasonalizing, as in a real case. The anomalies
    of a pure line are a yearly staircase, yet the median pairwise slope is
    still the injected one (equal up to float rounding of the inputs). The
    staircase spreads the pairwise slopes, so the interval is not a point,
    but it contains the slope."""
    y = benchmark.synthetic_series(n_years, 0.0, 0.0, 8.0, slope, seed=0)
    r = benchmark.run_case(y)
    assert r["ts_slope_per_decade"] == pytest.approx(slope, rel=1e-9)
    assert r["ts_low_per_decade"] <= r["ts_slope_per_decade"] <= r["ts_high_per_decade"]


def test_lag1_autocorr_of_long_ar1_series():
    y = benchmark.synthetic_series(1000, 0.6, 1.0, 10.0, seed=2026)
    r1 = benchmark.lag1_autocorr(clerk.deseasonalize(y))
    assert abs(r1 - 0.6) <= 0.05


def test_sigma_is_stationary_standard_deviation():
    for phi in (0.0, 0.9):
        y = benchmark.synthetic_series(2000, phi, 2.0, 0.0, seed=7)
        assert np.std(y.to_numpy()) == pytest.approx(2.0, rel=0.05)


@pytest.mark.parametrize("kwargs", [
    {"n_years": 0}, {"n_years": 2.5}, {"n_years": True},
    {"phi": 1.0}, {"phi": -1.0}, {"phi": float("nan")},
    {"sigma": -0.1}, {"sigma": float("inf")},
    {"seasonal_amp": float("nan")}, {"slope_per_decade": float("inf")},
])
def test_synthetic_series_rejects_bad_parameters(kwargs):
    args = {"n_years": 20, "phi": 0.3, "sigma": 1.0, "seasonal_amp": 1.0, "slope_per_decade": 0.0}
    args.update(kwargs)
    with pytest.raises(ValueError):
        benchmark.synthetic_series(**args, seed=0)


# --- run_case is the case.examine code path ---------------------------------


def _prepared_from(series: pd.Series) -> clerk.PreparedSeries:
    anomalies, clim = clerk.deseasonalize(series, return_climatology=True)
    quality = clerk.quality_gate(series)
    provenance = clerk.Provenance(
        source="synthetic", api="", region="synthetic", region_name="Synthetic", variable="temperature",
        power_code="T2M", longname="synthetic", units="degC", source_units="C", n_points=1,
        start=quality.start, end=quality.end, years=quality.years_covered,
        accessed_utc="2026-01-01T00:00:00+00:00", fill_value=-999.0, urls=())
    return clerk.PreparedSeries(
        raw=series, anomalies=anomalies, time=clerk.time_in_years(series.index), climatology=clim,
        n_points=pd.Series(1, index=series.index), quality=quality, provenance=provenance)


@pytest.mark.parametrize("phi,slope", [(0.0, 0.0), (0.6, 0.2), (0.9, -0.1)])
def test_run_case_equals_case_examine(monkeypatch, phi, slope):
    """Feed case.examine the same series (and so the same anomalies) through
    a stubbed clerk.prepare; every number must be identical."""
    y = benchmark.synthetic_series(30, phi, 1.0, 10.0, slope, seed=99)
    monkeypatch.setattr(clerk, "prepare", lambda region, variable: _prepared_from(y))

    c = case.examine("synthetic", "temperature")
    r = benchmark.run_case(y)

    assert c["quality"]["passed"]
    assert c["series"]["anomaly"] == [float(v) for v in clerk.deseasonalize(y)]
    ts, mk, smk = c["theil_sen"], c["mk_hamed_rao"], c["mk_seasonal"]
    assert r["ts_slope"] == ts["slope"]
    assert r["ts_low"] == ts["low_slope"]
    assert r["ts_high"] == ts["high_slope"]
    assert r["ts_slope_per_decade"] == ts["slope_per_decade"]
    assert r["ts_low_per_decade"] == ts["low_slope_per_decade"]
    assert r["ts_high_per_decade"] == ts["high_slope_per_decade"]
    assert r["hr_p"] == mk["p"] and r["hr_h"] == mk["h"]
    assert r["smk_p"] == smk["p"] and r["smk_h"] == smk["h"]
    assert mk["lag"] == 3 and mk["alpha"] == 0.05 and smk["alpha"] == 0.05


def test_run_case_uses_examiner_defaults_directly():
    y = benchmark.synthetic_series(20, 0.3, 1.0, 10.0, seed=5)
    a = clerk.deseasonalize(y)
    r = benchmark.run_case(y)
    assert r["hr_p"] == examiner.mk_hamed_rao(a, alpha=0.05, lag=3).p
    assert r["smk_p"] == examiner.mk_seasonal(y, alpha=0.05).p
    assert r["ts_slope"] == examiner.theil_sen(clerk.time_in_years(y.index), a, confidence=0.95).slope


# --- rejection summary and the experiment -----------------------------------


def test_rejection_summary_counts_and_flags():
    p = [0.001] * 20 + [0.5] * 80
    s = benchmark.rejection_summary(p, 0.05)
    assert (s["n"], s["rejections"], s["rate"]) == (100, 20, 0.2)
    assert s["ci95"][0] < 0.2 < s["ci95"][1]
    assert s["inflated"] is True

    s = benchmark.rejection_summary([0.03] * 5 + [0.5] * 95, 0.05)
    assert s["rate"] == 0.05 and s["inflated"] is False

    s = benchmark.rejection_summary([0.05, 0.01, 0.0099], 0.01)
    assert s["rejections"] == 1  # strict p < alpha


def test_rejection_summary_exact_interval():
    from scipy.stats import beta

    s = benchmark.rejection_summary([0.0] * 7 + [1.0] * 93, 0.05)
    assert s["ci95"][0] == pytest.approx(beta.ppf(0.025, 7, 94))
    assert s["ci95"][1] == pytest.approx(beta.ppf(0.975, 8, 93))


def test_grid_bracket():
    g = benchmark.FPR_PHIS
    assert benchmark._grid_bracket(-0.1, g) == [0.0]
    assert benchmark._grid_bracket(0.0, g) == [0.0]
    assert benchmark._grid_bracket(0.45, g) == [0.3, 0.6]
    assert benchmark._grid_bracket(0.6, g) == [0.6]
    assert benchmark._grid_bracket(0.95, g) == [0.9]


def test_fpr_experiment_small_is_reproducible_and_worker_independent():
    kw = dict(phis=(0.0, 0.9), years=(20,), seed=3, chunk=4)
    a = benchmark.fpr_experiment(10, workers=1, **kw)
    b = benchmark.fpr_experiment(10, workers=2, **kw)
    assert a == b
    json.dumps(a, allow_nan=False)

    assert a["grid"] == {"phi": [0.0, 0.9], "n_years": [20]}
    assert a["replicates"] == 10 and a["seeds"]["master"] == 3
    assert len(a["cells"]) == 2
    for cell in a["cells"]:
        assert cell["replicates"] == 10
        for test in benchmark.TESTS:
            for alpha in ("0.05", "0.01"):
                s = cell["rates"][test][alpha]
                assert s["n"] + cell["n_failed"] == 10
                assert 0.0 <= s["ci95"][0] <= s["rate"] <= s["ci95"][1] <= 1.0
                assert (f"{test}@{alpha}" in cell["inflated"]) == s["inflated"]


def test_fpr_cell_rates_match_run_case_by_hand():
    """The experiment's p-values are exactly run_case on the seeded series."""
    res = benchmark.fpr_experiment(6, phis=(0.6,), years=(20,), seed=11, chunk=4)
    cell = res["cells"][0]
    p_hr, p_smk = [], []
    for rep in range(6):
        y = benchmark.synthetic_series(20, 0.6, benchmark.FPR_SIGMA, benchmark.FPR_SEASONAL_AMP,
                                       seed=benchmark.replicate_seed(11, 0.6, 20, rep))
        r = benchmark.run_case(y)
        p_hr.append(r["hr_p"])
        p_smk.append(r["smk_p"])
    assert cell["rates"]["hamed_rao"]["0.05"] == benchmark.rejection_summary(p_hr, 0.05)
    assert cell["rates"]["seasonal"]["0.01"] == benchmark.rejection_summary(p_smk, 0.01)


def test_fpr_experiment_rejects_zero_replicates():
    with pytest.raises(ValueError):
        benchmark.fpr_experiment(0)


# --- committed results ------------------------------------------------------


def test_committed_fpr_json_is_complete():
    path = clerk.REPO_ROOT / "docs" / "benchmark" / "fpr.json"
    res = json.loads(path.read_text(encoding="utf-8"))
    assert res["grid"] == {"phi": list(benchmark.FPR_PHIS), "n_years": list(benchmark.FPR_YEARS)}
    assert res["replicates"] >= 1000
    assert res["alphas"] == [0.05, 0.01]
    assert len(res["cells"]) == len(benchmark.FPR_PHIS) * len(benchmark.FPR_YEARS)
    for cell in res["cells"]:
        for test in benchmark.TESTS:
            for alpha in ("0.05", "0.01"):
                s = cell["rates"][test][alpha]
                assert s["inflated"] == (s["ci95"][0] > s["alpha"])
    assert len(res["real_cases"]) == 18
    assert all(-1.0 <= row["r1"] <= 1.0 for row in res["real_cases"])


# --- experiments (b) coverage and (c) power ---------------------------------


def test_trend_replicate_seed_scheme():
    assert benchmark.trend_replicate_seed(18, 0.6, 0.3, 45, 7) == (18, 600, 300, 45, 7)
    assert benchmark.trend_replicate_seed(18, 0.6, 0.1, 45, 7) != benchmark.trend_replicate_seed(18, 0.6, 0.3, 45, 7)


def test_run_case_reports_hamed_rao_z():
    y = benchmark.synthetic_series(30, 0.3, 1.0, 10.0, 0.5, seed=8)
    r = benchmark.run_case(y)
    assert r["hr_z"] == examiner.mk_hamed_rao(clerk.deseasonalize(y)).z


def test_proportion_summary_exact_interval():
    from scipy.stats import beta

    s = benchmark.proportion_summary(940, 1000)
    assert (s["n"], s["k"], s["rate"]) == (1000, 940, 0.94)
    assert s["ci95"][0] == pytest.approx(beta.ppf(0.025, 940, 61))
    assert s["ci95"][1] == pytest.approx(beta.ppf(0.975, 941, 60))
    assert benchmark.proportion_summary(0, 0)["rate"] is None


def test_coverage_cell_matches_run_case_by_hand():
    """Coverage counts exactly the replicates whose run_case interval holds slope / 10."""
    phi, slope, n, seed, reps = 0.6, 0.3, 20, 5, 8
    res = benchmark.coverage_experiment(reps, phis=(phi,), slopes=(slope,), years=(n,), seed=seed, chunk=3)
    cell = res["cells"][0]
    covered, widened, est = 0, 0, []
    for rep in range(reps):
        y = benchmark.synthetic_series(n, phi, benchmark.TREND_SIGMA, benchmark.TREND_SEASONAL_AMP, slope,
                                       seed=benchmark.trend_replicate_seed(seed, phi, slope, n, rep))
        r = benchmark.run_case(y)
        covered += r["ts_low"] <= slope / 10 <= r["ts_high"]
        lo, hi = thresholds.widened_interval(r["ts_slope"], r["ts_low"], r["ts_high"], r["r1_detrended"])
        widened += lo <= slope / 10 <= hi
        est.append(r["ts_slope_per_decade"])
    assert cell["coverage"]["k"] == covered and cell["coverage"]["n"] == reps
    assert cell["widened_coverage"]["k"] == widened and cell["widened_coverage"]["n"] == reps
    assert widened >= covered
    assert cell["miss_low"] + cell["miss_high"] == reps - covered
    assert cell["slope_median_per_decade"] == pytest.approx(np.median(est))


def test_coverage_experiment_reproducible_worker_independent_and_flags():
    kw = dict(phis=(0.0, 0.9), slopes=(0.5,), years=(20,), seed=4, chunk=4)
    a = benchmark.coverage_experiment(12, workers=1, **kw)
    b = benchmark.coverage_experiment(12, workers=2, **kw)
    assert a == b
    json.dumps(a, allow_nan=False)
    assert a["grid"] == {"phi": [0.0, 0.9], "slope_per_decade": [0.5], "n_years": [20]}
    assert a["seeds"]["master"] == 4
    for c in a["cells"]:
        assert c["below_flag"] == (c["coverage"]["rate"] < benchmark.COVERAGE_FLAG)
        assert c["below_nominal"] == (c["coverage"]["ci95"][1] < 0.95)
    flagged = [c["phi"] for c in a["cells"] if c["below_flag"]]
    assert a["onset"] == [{"slope_per_decade": 0.5, "n_years": 20, "phi": min(flagged) if flagged else None}]
    assert {(f["phi"], f["n_years"]) for f in a["flagged_cells"]} == \
        {(c["phi"], c["n_years"]) for c in a["cells"] if c["below_flag"]}


def test_coverage_onset_is_first_flagged_phi():
    """With flag_below = 1.01 every cell is flagged, so the onset is the lowest phi;
    with flag_below = 0 none is, so there is no onset."""
    kw = dict(phis=(0.9, 0.3), slopes=(0.3,), years=(20,), seed=1, chunk=5)
    assert benchmark.coverage_experiment(5, flag_below=1.01, **kw)["onset"][0]["phi"] == 0.3
    res = benchmark.coverage_experiment(5, flag_below=0.0, **kw)
    assert res["onset"][0]["phi"] is None and res["onset_phi"] is None and res["flagged_cells"] == []


def test_power_cell_matches_run_case_by_hand():
    phi, slope, n, seed, reps = 0.3, 0.5, 25, 6, 8
    res = benchmark.power_experiment(reps, phis=(phi,), slopes=(slope,), years=(n,), seed=seed, chunk=3)
    cell = res["cells"][0]
    p, z = [], []
    for rep in range(reps):
        y = benchmark.synthetic_series(n, phi, benchmark.TREND_SIGMA, benchmark.TREND_SEASONAL_AMP, slope,
                                       seed=benchmark.trend_replicate_seed(seed, phi, slope, n, rep))
        r = benchmark.run_case(y)
        p.append(r["hr_p"])
        z.append(r["hr_z"])
    p, z = np.array(p), np.array(z)
    for a in benchmark.ALPHAS:
        rates = cell["rates"][f"{a:g}"]
        assert rates["rejection"]["k"] == int((p < a).sum())
        assert rates["detection"]["k"] == int(((p < a) & (z > 0)).sum())
        assert rates["rejection"] == benchmark.proportion_summary(int((p < a).sum()), reps)


def test_power_experiment_reproducible_and_worker_independent():
    kw = dict(phis=(0.0,), slopes=(0.1, 0.5), years=(20, 45), seed=2, chunk=4)
    a = benchmark.power_experiment(10, workers=1, **kw)
    b = benchmark.power_experiment(10, workers=2, **kw)
    assert a == b
    json.dumps(a, allow_nan=False)
    assert len(a["cells"]) == 4
    for c in a["cells"]:
        for alpha in ("0.05", "0.01"):
            r = c["rates"][alpha]
            assert r["detection"]["k"] <= r["rejection"]["k"]
            assert r["rejection"]["n"] + c["n_failed"] == 10


@pytest.mark.parametrize("fn", [benchmark.coverage_experiment, benchmark.power_experiment])
def test_trend_experiments_reject_bad_arguments(fn):
    with pytest.raises(ValueError):
        fn(0)
    with pytest.raises(ValueError):
        fn(5, slopes=(0.0,))
    with pytest.raises(ValueError):
        fn(5, slopes=(float("nan"),))


def test_committed_coverage_json_is_complete():
    res = json.loads((clerk.REPO_ROOT / "docs" / "benchmark" / "coverage.json").read_text(encoding="utf-8"))
    assert res["grid"] == {"phi": list(benchmark.TREND_PHIS),
                           "slope_per_decade": list(benchmark.COVERAGE_SLOPES),
                           "n_years": list(benchmark.COVERAGE_YEARS)}
    assert res["replicates"] >= 1000 and res["seeds"]["master"] == benchmark.COVERAGE_SEED
    assert len(res["cells"]) == len(benchmark.TREND_PHIS) * len(benchmark.COVERAGE_SLOPES) * len(benchmark.COVERAGE_YEARS)
    for c in res["cells"]:
        cov = c["coverage"]
        assert cov["n"] + c["n_failed"] == res["replicates"]
        assert c["below_flag"] == (cov["rate"] < res["flag_below"])
        assert c["miss_low"] + c["miss_high"] == cov["n"] - cov["k"]
    assert len(res["onset"]) == len(benchmark.COVERAGE_SLOPES) * len(benchmark.COVERAGE_YEARS)

    # The summary fields must agree with the cells (catches a stale or hand-edited file).
    assert res["flagged_cells"] == [
        {"phi": c["phi"], "slope_per_decade": c["slope_per_decade"], "n_years": c["n_years"],
         "coverage": c["coverage"]["rate"]}
        for c in res["cells"] if c["below_flag"]
    ]
    expected_onset = []
    for s in benchmark.COVERAGE_SLOPES:
        for n in benchmark.COVERAGE_YEARS:
            flagged = [c["phi"] for c in res["cells"]
                       if c["slope_per_decade"] == s and c["n_years"] == n and c["below_flag"]]
            expected_onset.append({"slope_per_decade": s, "n_years": n,
                                   "phi": min(flagged) if flagged else None})
    assert res["onset"] == expected_onset
    onset_phis = [o["phi"] for o in expected_onset if o["phi"] is not None]
    assert res["onset_phi"] == (min(onset_phis) if onset_phis else None)


def test_committed_power_json_is_complete():
    res = json.loads((clerk.REPO_ROOT / "docs" / "benchmark" / "power.json").read_text(encoding="utf-8"))
    assert res["grid"] == {"phi": list(benchmark.TREND_PHIS),
                           "slope_per_decade": list(benchmark.POWER_SLOPES),
                           "n_years": list(range(20, 46, 5))}
    assert res["replicates"] >= 1000 and res["seeds"]["master"] == benchmark.POWER_SEED
    assert res["alphas"] == [0.05, 0.01]
    assert len(res["cells"]) == len(benchmark.TREND_PHIS) * len(benchmark.POWER_SLOPES) * len(benchmark.POWER_YEARS)
    for c in res["cells"]:
        for alpha in ("0.05", "0.01"):
            r = c["rates"][alpha]
            assert r["rejection"]["n"] + c["n_failed"] == res["replicates"]
            assert r["detection"]["k"] <= r["rejection"]["k"]


# --- scripts/benchmark.py CLI defaults ----------------------------------------


def _load_benchmark_cli():
    import importlib.util

    path = clerk.REPO_ROOT / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("_benchmark_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("experiment, seed, replicates", [
    ("fpr", 17, 1000),
    ("coverage", 18, 1000),
    ("power", 19, 1000),
    ("applied", 20, 1000),
    ("noise", 21, 1000),
])
def test_cli_per_experiment_defaults(monkeypatch, experiment, seed, replicates):
    """Without --seed/--replicates/--out the CLI uses the experiment's own master
    seed, replicate count and docs/benchmark/<experiment>.json. The run_* functions
    are replaced, so nothing is simulated or written."""
    cli = _load_benchmark_cli()
    captured = {}

    def fake(name):
        def run(args):
            captured[name] = args
            return 0
        return run

    for name in ("fpr", "coverage", "power", "applied", "noise"):
        monkeypatch.setattr(cli, f"run_{name}", fake(name))

    assert cli.main(["--experiment", experiment, "--workers", "1"]) == 0
    assert list(captured) == [experiment]
    args = captured[experiment]
    assert args.seed == seed
    assert args.replicates == replicates
    assert args.out == clerk.REPO_ROOT / "docs" / "benchmark" / f"{experiment}.json"
    assert args.workers == 1


def test_cli_explicit_arguments_override_defaults(monkeypatch, tmp_path):
    cli = _load_benchmark_cli()
    captured = []
    monkeypatch.setattr(cli, "run_coverage", lambda args: captured.append(args) or 0)
    out = tmp_path / "c.json"
    assert cli.main(["--experiment", "coverage", "--replicates", "3", "--seed", "5",
                     "--workers", "2", "--out", str(out)]) == 0
    (args,) = captured
    assert (args.replicates, args.seed, args.workers, args.out) == (3, 5, 2, out)
    with pytest.raises(SystemExit):
        cli.main(["--experiment", "coverage", "--replicates", "0"])


# --- r1 in run_case -----------------------------------------------------------


def test_run_case_reports_detrended_r1_and_can_skip_the_seasonal_test():
    y = benchmark.synthetic_series(30, 0.6, 1.0, 10.0, 0.3, seed=8)
    full = benchmark.run_case(y)
    quick = benchmark.run_case(y, seasonal=False)
    a = clerk.deseasonalize(y)
    t = clerk.time_in_years(y.index)
    ts = examiner.theil_sen(t, a)
    assert full["r1_detrended"] == benchmark.lag1_autocorr(a.to_numpy() - (ts.intercept + ts.slope * t))
    assert quick["smk_p"] is None and quick["smk_h"] is None
    assert {k: v for k, v in quick.items() if not k.startswith("smk")} == \
        {k: v for k, v in full.items() if not k.startswith("smk")}


def test_committed_coverage_json_has_widened_coverage():
    res = json.loads((clerk.REPO_ROOT / "docs" / "benchmark" / "coverage.json").read_text(encoding="utf-8"))
    for c in res["cells"]:
        w = c["widened_coverage"]
        assert w["n"] == c["coverage"]["n"] and w["k"] >= c["coverage"]["k"]
        assert c["widening_factor_mean"] >= 1.0


# --- experiment (d): the policy as applied --------------------------------------


def test_applied_experiment_reproducible_and_matches_policy_by_hand():
    th = thresholds.load()
    kw = dict(phis=(0.3, 0.85), years=(20,), seed=6, chunk=3)
    a = benchmark.applied_experiment(7, workers=1, **kw)
    b = benchmark.applied_experiment(7, workers=2, **kw)
    assert a == b
    json.dumps(a, allow_nan=False)
    assert a["thresholds_used"] == benchmark._thresholds_used(th)
    for cell in a["cells"]:
        phi = cell["phi"]
        claims = strong = widened = 0
        r1s = []
        for rep in range(7):
            y = benchmark.synthetic_series(20, phi, 1.0, 10.0, seed=benchmark.replicate_seed(6, phi, 20, rep))
            r = benchmark.run_case(y, seasonal=False)
            c = thresholds.significance_claim(r["r1_detrended"], r["hr_p"], th)
            claims += c is not None
            strong += c == "strong"
            lo, hi = thresholds.widened_interval(r["ts_slope"], r["ts_low"], r["ts_high"], r["r1_detrended"])
            widened += lo <= 0 <= hi
            r1s.append(r["r1_detrended"])
        assert cell["false_claim"]["k"] == claims and cell["false_strong"]["k"] == strong
        assert cell["widened_coverage"]["k"] == widened
        assert cell["r1_detrended_mean"] == pytest.approx(np.mean(r1s))
        assert cell["r1_detrended_bias"] == pytest.approx(np.mean(r1s) - phi)
        assert sum(cell["band_shares"].values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        benchmark.applied_experiment(0)


def test_committed_applied_json_supports_the_policy():
    """The evidence for closing open point 3: with r1 estimated per record,
    the policy keeps false significance claims within max_fpr_moderate and
    false Strong claims within max_fpr_strong at every true phi simulated,
    on and off the band grid, including above it."""
    res = json.loads((clerk.REPO_ROOT / "docs" / "benchmark" / "applied.json").read_text(encoding="utf-8"))
    th = thresholds.load()
    assert res["grid"] == {"phi": list(benchmark.APPLIED_PHIS), "n_years": list(benchmark.APPLIED_YEARS)}
    assert res["replicates"] >= 1000 and res["seeds"]["master"] == benchmark.APPLIED_SEED
    assert len(res["cells"]) == len(benchmark.APPLIED_PHIS) * len(benchmark.APPLIED_YEARS)
    p = th["fpr_policy"]
    for c in res["cells"]:
        assert c["n_failed"] == 0
        assert c["false_claim"]["rate"] <= p["max_fpr_moderate"], c["phi"]
        assert c["false_strong"]["rate"] <= p["max_fpr_strong"], c["phi"]
        assert c["false_strong"]["k"] <= c["false_claim"]["k"]


# --- experiment (e): the real cases' noise --------------------------------------


def test_case_seed_key_is_stable():
    assert benchmark.case_seed_key("amazon", "precipitation") == benchmark.case_seed_key("amazon", "precipitation")
    import zlib

    assert benchmark.case_seed_key("sahel", "soil_moisture") == zlib.crc32(b"sahel/soil_moisture")


def test_yule_walker_and_long_run_ratio_on_ar1():
    rng = np.random.default_rng(0)
    from scipy import signal

    x = signal.lfilter([1.0], [1.0, -0.6], rng.normal(size=20000))
    coef, s2 = benchmark.yule_walker(x, 1)
    assert coef[0] == pytest.approx(0.6, abs=0.02)
    assert benchmark.ar_long_run_ratio([0.6], 1 - 0.36, 1.0) == pytest.approx(1.6 / 0.4)
    assert math.sqrt(benchmark.ar_long_run_ratio([0.8], 1 - 0.64, 1.0)) == pytest.approx(
        thresholds.widening_factor(0.8))


def test_bartlett_vif():
    assert benchmark.bartlett_vif([0.0] * 5, 5) == 1.0
    assert benchmark.bartlett_vif([0.5, 0.25], 2) == pytest.approx(1 + 2 * (0.5 * 2 / 3 + 0.25 * 1 / 3))
    r = 0.7
    rho = r ** np.arange(1, 2000)
    # untruncated limit is the AR(1) long-run ratio; truncation biases it low
    assert benchmark.bartlett_vif(rho, 36) < (1 + r) / (1 - r)


def test_fit_case_noise_on_a_real_case():
    f = benchmark.fit_case_noise("sahel", "precipitation")
    assert f["n_months"] == 540 and 1 <= f["ar_order"] <= benchmark.NOISE_MAX_ORDER
    assert len(f["ar_coef"]) == f["ar_order"] and len(f["month_sd"]) == 12
    assert f["acf"]["1"] == pytest.approx(f["r1_detrended"])
    assert f["ar1_widening"] == pytest.approx(thresholds.widening_factor(f["r1_detrended"]))
    y = benchmark._noise_series(f, (1, 2, 3))
    assert len(y) == 540 and y.index[0] == pd.Timestamp(f["index_start"])
    assert np.array_equal(y.to_numpy(), benchmark._noise_series(f, (1, 2, 3)).to_numpy())


def test_noise_experiment_small_is_reproducible():
    kw = dict(cases=[("sahel", "precipitation")], seed=4, chunk=2)
    a = benchmark.noise_experiment(5, workers=1, **kw)
    b = benchmark.noise_experiment(5, workers=2, **kw)
    assert a == b
    json.dumps(a, allow_nan=False)
    (c,) = a["cases"]
    assert c["replicates"] == 5 and c["false_claim"]["n"] + c["n_failed"] == 5
    assert c["stated_fpr"] == {k: v["rate"] for k, v in
                               thresholds.band_for(c["r1_detrended"])["hamed_rao_fpr"].items()}


def test_committed_noise_json_is_complete():
    res = json.loads((clerk.REPO_ROOT / "docs" / "benchmark" / "noise.json").read_text(encoding="utf-8"))
    assert res["replicates"] >= 1000 and res["seeds"]["master"] == benchmark.NOISE_SEED
    assert len(res["cases"]) == 18
    fpr = json.loads(thresholds.FPR_PATH.read_text(encoding="utf-8"))
    r1 = {(r["region"], r["variable"]): r["r1_detrended"] for r in fpr["real_cases"]}
    for c in res["cases"]:
        assert c["n_failed"] == 0
        assert c["r1_detrended"] == pytest.approx(r1[(c["region"], c["variable"])], abs=1e-12)
