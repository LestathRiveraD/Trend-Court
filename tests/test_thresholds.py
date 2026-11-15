import copy
import json
import math

import pytest

from trendcourt import benchmark, benchmark_report, clerk, thresholds


@pytest.fixture(scope="module")
def th():
    return thresholds.load()


@pytest.fixture(scope="module")
def fpr():
    return json.loads(thresholds.FPR_PATH.read_text(encoding="utf-8"))


# --- thresholds.json ---------------------------------------------------------


def test_committed_thresholds_load_and_point_to_the_doc(th):
    assert th["source"] == "docs/benchmark.md"
    assert (clerk.REPO_ROOT / th["source"]).is_file()
    assert th["alpha_strong"] == 0.01 and th["alpha_moderate"] == 0.05
    assert th["window_agreement_strong"] == 0.80
    assert set(th["no_trend_margin"]) == set(clerk.load_regions()["variables"])
    assert th["no_trend_margin_units"] == {"temperature": "degC/decade",
                                           "precipitation": "mm/day/decade",
                                           "soil_moisture": "fraction/decade"}


def test_committed_bands_are_derived_from_fpr_json(th, fpr):
    """The bands are not hand-typed: they are exactly what the policy gives on
    the committed false-positive benchmark."""
    p = th["fpr_policy"]
    derived = thresholds.derive_bands(fpr, th["alpha_strong"], th["alpha_moderate"],
                                      p["max_fpr_strong"], p["max_fpr_moderate"])
    assert th["autocorrelation_bands"] == json.loads(json.dumps(derived))


def test_band_caps_follow_the_policy(th):
    caps = [(b["r1_max"], b["max_verdict"], b["moderate_alpha"]) for b in th["autocorrelation_bands"]]
    assert caps == [(0.3, "strong", 0.05), (0.6, "strong", 0.05),
                    (0.8, "moderate", 0.01), (0.9, "inconclusive", None)]
    p = th["fpr_policy"]
    for b in th["autocorrelation_bands"]:
        f = b["hamed_rao_fpr"]
        strong_ok = f["0.01"]["rate"] <= p["max_fpr_strong"]
        assert (b["max_verdict"] == "strong") == strong_ok
        if b["moderate_alpha"] is not None:
            assert f[f"{b['moderate_alpha']:g}"]["rate"] <= p["max_fpr_moderate"]


def test_worst_fpr_takes_the_highest_rate_over_lengths(fpr):
    w = thresholds.worst_fpr(fpr, 0.8, 0.01)
    rates = [c["rates"]["hamed_rao"]["0.01"]["rate"] for c in fpr["cells"] if c["phi"] == 0.8]
    assert w["rate"] == max(rates)
    with pytest.raises(ValueError, match="not on the FPR grid"):
        thresholds.worst_fpr(fpr, 0.5, 0.05)


def test_derive_bands_policy_edges(fpr):
    """A looser policy lets every band claim Strong; a stricter one caps more."""
    loose = thresholds.derive_bands(fpr, 0.01, 0.05, 1.0, 1.0)
    assert all(b["max_verdict"] == "strong" and b["moderate_alpha"] == 0.05 for b in loose)
    strict = thresholds.derive_bands(fpr, 0.01, 0.05, 0.0, 0.0)
    assert all(b["max_verdict"] == "inconclusive" and b["moderate_alpha"] is None for b in strict)


# --- band_for ------------------------------------------------------------------


@pytest.mark.parametrize("r1, upper", [
    (-1.0, 0.3), (-0.2, 0.3), (0.0, 0.3), (0.3, 0.3), (0.3000001, 0.6),
    (0.6, 0.6), (0.79, 0.8), (0.8, 0.8), (0.807, 0.9), (0.9, 0.9),
])
def test_band_for_edges(th, r1, upper):
    assert thresholds.band_for(r1, th)["r1_max"] == upper


def test_band_for_outside_the_benchmark(th):
    assert thresholds.band_for(0.95, th) is None
    with pytest.raises(ValueError):
        thresholds.band_for(math.nan, th)


def test_band_for_default_loads_committed_file():
    assert thresholds.band_for(0.5)["r1_max"] == 0.6


def test_load_returns_independent_copies():
    a = thresholds.load()
    a["alpha_strong"] = 0.5
    assert thresholds.load()["alpha_strong"] == 0.01


# --- widening and the no-trend rule -------------------------------------------


def test_widening_factor():
    assert thresholds.widening_factor(0.0) == 1.0
    assert thresholds.widening_factor(-0.3) == 1.0
    assert thresholds.widening_factor(0.6) == pytest.approx(2.0)
    assert thresholds.widening_factor(0.8) == pytest.approx(3.0)
    for bad in (1.0, 1.2, math.nan, math.inf):
        with pytest.raises(ValueError):
            thresholds.widening_factor(bad)


def test_widened_interval_keeps_asymmetry():
    lo, hi = thresholds.widened_interval(0.1, 0.0, 0.3, 0.6)
    assert lo == pytest.approx(0.1 - 2 * 0.1)
    assert hi == pytest.approx(0.1 + 2 * 0.2)
    assert thresholds.widened_interval(0.1, 0.0, 0.3, 0.0) == (0.0, pytest.approx(0.3))


def test_no_trend_needs_nonsignificance_and_widened_interval_inside_margin(th):
    # temperature margin 0.1 degC/decade
    assert thresholds.no_trend_allowed("temperature", 0.0, -0.04, 0.04, 0.3, 0.5, th)
    # raw interval fits, widened one (factor 3 at r1 = 0.8) does not
    assert not thresholds.no_trend_allowed("temperature", 0.0, -0.04, 0.04, 0.8, 0.5, th)
    # significant at alpha_moderate: never "no trend"
    assert not thresholds.no_trend_allowed("temperature", 0.0, -0.01, 0.01, 0.0, 0.049, th)
    # p between 0.01 and 0.05 is still significant for this rule, whatever the band
    assert not thresholds.no_trend_allowed("temperature", 0.0, -0.01, 0.01, 0.7, 0.03, th)
    assert thresholds.no_trend_allowed("temperature", 0.0, -0.01, 0.01, 0.0, 0.05, th)
    with pytest.raises(ValueError, match="no no_trend_margin"):
        thresholds.no_trend_allowed("wind", 0.0, -0.01, 0.01, 0.0, 0.5, th)


def test_no_trend_never_outside_the_benchmark(th):
    """r1 above the grid: band_for is None, so never "no trend", however narrow
    the interval (the widening was only checked up to phi = 0.9)."""
    top = th["autocorrelation_bands"][-1]["r1_max"]
    assert top == 0.9
    for r1 in (0.9000001, 0.95, 0.999, 1.0, 1.5):
        assert thresholds.band_for(r1, th) is None
        assert not thresholds.no_trend_allowed("temperature", 0.0, -0.01, 0.01, r1, 0.5, th)
    with pytest.raises(ValueError):
        thresholds.no_trend_allowed("temperature", 0.0, -0.01, 0.01, math.nan, 0.5, th)


def test_no_trend_still_allowed_in_the_inconclusive_capped_band(th):
    """(0.8, 0.9] is capped at Inconclusive for significance, but the no-trend
    rule rests on the widened interval, which is checked up to phi = 0.9."""
    assert thresholds.band_for(0.85, th)["max_verdict"] == "inconclusive"
    f = thresholds.widening_factor(0.85)  # about 3.51
    assert thresholds.no_trend_allowed("temperature", 0.0, -0.02, 0.02, 0.85, 0.5, th)
    assert 0.02 * f < 0.1
    assert thresholds.no_trend_allowed("temperature", 0.0, -0.02, 0.02, 0.9, 0.5, th)
    assert not thresholds.no_trend_allowed("temperature", 0.0, -0.03, 0.03, 0.85, 0.5, th)


# --- schema checks ---------------------------------------------------------------


def _broken(th, change):
    bad = copy.deepcopy(th)
    change(bad)
    return bad


@pytest.mark.parametrize("change, match", [
    (lambda t: t.pop("alpha_strong"), "missing"),
    (lambda t: t.update(alpha_strong=0.05, alpha_moderate=0.01), "alpha_strong < alpha_moderate"),
    (lambda t: t.update(window_agreement_strong=1.5), "window_agreement_strong"),
    (lambda t: t["no_trend_margin"].update(temperature=-0.1), "positive"),
    (lambda t: t["no_trend_margin_units"].pop("temperature"), "same variables"),
    (lambda t: t.update(autocorrelation_bands=[]), "empty"),
    (lambda t: t["autocorrelation_bands"][1].update(r1_min_exclusive=0.2), "contiguous"),
    (lambda t: t["autocorrelation_bands"][0].update(max_verdict="certain"), "unknown max_verdict"),
    (lambda t: t["autocorrelation_bands"][0].update(moderate_alpha=0.02), "moderate_alpha must be"),
    (lambda t: t["autocorrelation_bands"][3].update(moderate_alpha=0.01), "inconclusive band"),
    (lambda t: t.pop("fpr_policy"), "missing"),
    (lambda t: t.pop("interval_widening"), "missing"),
    (lambda t: t["fpr_policy"].pop("max_fpr_strong"), "max_fpr_strong must be"),
    (lambda t: t["fpr_policy"].pop("max_fpr_moderate"), "max_fpr_moderate must be"),
    (lambda t: t["fpr_policy"].update(max_fpr_moderate="0.1"), "max_fpr_moderate must be"),
    (lambda t: t["fpr_policy"].update(max_fpr_strong=1.5), "max_fpr_strong must be"),
    (lambda t: t["fpr_policy"].update(max_fpr_strong=0.1, max_fpr_moderate=0.05),
     "max_fpr_strong <= fpr_policy.max_fpr_moderate"),
    (lambda t: t["fpr_policy"].update(test="seasonal"), "fpr_policy.test must be 'hamed_rao'"),
    (lambda t: t["fpr_policy"].pop("test"), "fpr_policy.test must be 'hamed_rao'"),
    (lambda t: t["interval_widening"].update(method="block_bootstrap"),
     "interval_widening.method must be 'ar1'"),
    (lambda t: t["interval_widening"].pop("method"), "interval_widening.method must be 'ar1'"),
])
def test_check_rejects_broken_files(th, tmp_path, change, match):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(_broken(th, change)), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        thresholds.load(path)


def test_no_verdict_threshold_is_hard_coded_in_the_package():
    """Only thresholds.json holds the verdict thresholds; the package refers to
    them by name. (Experiment grids in benchmark.py and Examiner defaults are not
    verdict thresholds.)"""
    pkg = clerk.REPO_ROOT / "trendcourt"
    for path in pkg.glob("*.py"):
        if path.name == "thresholds.py":
            continue
        text = path.read_text(encoding="utf-8")
        for name in ("window_agreement_strong", "no_trend_margin", "max_fpr_strong", "max_fpr_moderate"):
            assert f"{name} =" not in text and f"{name}=" not in text, (path.name, name)


# --- report: figures and tables ---------------------------------------------------


def test_committed_doc_tables_match_the_json(th):
    """docs/benchmark.md's generated block is what --figures writes today."""
    doc = benchmark_report.DOC_PATH.read_text(encoding="utf-8")
    expected = benchmark_report.replace_tables(doc, benchmark_report.tables_markdown(
        benchmark_report.load_results(), th))
    assert doc == expected


def test_real_case_no_trend_columns(th, fpr):
    """Table 6's no-trend columns, from the committed records: "no trend" is
    unreachable for 10 of 18 cases and allowed for none (docs/benchmark.md,
    the no_trend_margin decision)."""
    nt = benchmark_report.real_case_no_trend(fpr, th)
    assert len(nt) == 18
    unreachable = sorted(k for k, v in nt.items() if not v["reachable"])
    assert unreachable == sorted([
        ("mexicali", "temperature"), ("amazon", "temperature"), ("iberia", "temperature"),
        ("yucatan", "precipitation"), ("amazon", "precipitation"), ("iberia", "precipitation"),
        ("mexico_city", "soil_moisture"), ("yucatan", "soil_moisture"),
        ("amazon", "soil_moisture"), ("iberia", "soil_moisture")])
    assert not any(v["allowed"] for v in nt.values())
    for (region, var), v in nt.items():
        assert v["margin"] == th["no_trend_margin"][var]
        assert v["reachable"] == (v["half_width"] <= v["margin"])  # all 18 are inside the grid
    mc = nt[("mexico_city", "precipitation")]
    assert mc["reachable"] and mc["hr_p"] >= th["alpha_moderate"] and not mc["allowed"]


def test_update_doc_fails_on_stale_bands(th, tmp_path):
    results = benchmark_report.load_results()
    benchmark_report.check_bands_current(results, th)  # committed bands are current
    stale = copy.deepcopy(th)
    stale["autocorrelation_bands"][2]["max_verdict"] = "strong"
    doc = tmp_path / "doc.md"
    original = f"{benchmark_report.TABLES_START}\n{benchmark_report.TABLES_END}\n"
    doc.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="derive_bands"):
        benchmark_report.update_doc(results, stale, doc)
    assert doc.read_text(encoding="utf-8") == original


def test_committed_figures_exist():
    for name in benchmark_report.FIGURES.values():
        path = benchmark_report.BENCH_DIR / name
        assert path.is_file() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name


def test_figures_regenerate_from_json_without_simulation(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("the figures must not rerun the simulation")

    for name in ("fpr_experiment", "coverage_experiment", "power_experiment", "run_case",
                 "synthetic_series", "real_case_autocorr"):
        monkeypatch.setattr(benchmark, name, boom)
    paths = benchmark_report.write_figures(benchmark_report.load_results(), tmp_path)
    assert sorted(p.name for p in paths) == sorted(benchmark_report.FIGURES.values())
    for p in paths:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # same input, same bytes
    first = {p.name: p.read_bytes() for p in paths}
    benchmark_report.write_figures(benchmark_report.load_results(), tmp_path)
    assert {p.name: p.read_bytes() for p in paths} == first


def test_ar1_prediction_and_implied_widening():
    assert benchmark_report.ar1_predicted_coverage(0.0) == pytest.approx(0.95)
    assert benchmark_report.implied_widening(0.95) == pytest.approx(1.0)
    # the prediction inverts the implied widening
    for phi in (0.3, 0.6, 0.8):
        c = benchmark_report.ar1_predicted_coverage(phi)
        assert benchmark_report.implied_widening(c) == pytest.approx(thresholds.widening_factor(phi))


def test_committed_coverage_matches_the_ar1_prediction():
    """The finding behind the widened interval: pooled coverage at each phi is
    within 0.03 of what an interval ignoring AR(1) would give."""
    cov = benchmark_report.load_results()["coverage"]
    for phi in cov["grid"]["phi"]:
        observed = benchmark_report._pooled_coverage(cov, phi)
        assert observed == pytest.approx(benchmark_report.ar1_predicted_coverage(phi), abs=0.03), phi


def test_real_case_ranges(fpr):
    ranges = benchmark_report.real_case_ranges(fpr)
    assert set(ranges) == {"temperature", "precipitation", "soil_moisture"}
    for lo, hi in ranges.values():
        assert -1 < lo <= hi < 1


def test_replace_tables_requires_markers_once():
    s, e = benchmark_report.TABLES_START, benchmark_report.TABLES_END
    assert benchmark_report.replace_tables(f"a\n{s}\nold\n{e}\nb", "new") == f"a\n{s}\n\nnew\n\n{e}\nb"
    for bad in ("no markers", f"{s}{s}{e}", f"{s}"):
        with pytest.raises(ValueError):
            benchmark_report.replace_tables(bad, "x")


# --- CLI ---------------------------------------------------------------------------


def _cli():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_benchmark_cli_fig", clerk.REPO_ROOT / "scripts" / "benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_figures_mode(monkeypatch):
    cli = _cli()
    called = []
    monkeypatch.setattr(cli, "run_figures", lambda args: called.append(args) or 0)
    assert cli.main(["--figures"]) == 0 and len(called) == 1
    for argv in (["--figures", "--seed", "3"], ["--figures", "--experiment", "fpr"], []):
        with pytest.raises(SystemExit):
            cli.main(argv)
