"""Verdict module v0 (issue #20): one hand-built case per label, the
boundaries of each rule, and the verdict block in the case JSON."""

import copy
import json

import pytest

import trendcourt
from trendcourt import case as case_module
from trendcourt import thresholds, verdict

TH = thresholds.load()
A_STRONG, A_MOD = TH["alpha_strong"], TH["alpha_moderate"]
TEMP_MARGIN = TH["no_trend_margin"]["temperature"]


def make_case(*, p=0.001, slope=0.3, low=None, high=None, s=None, r1=0.1, agree=True,
              variable="temperature", region="synthetic", climatology=None, windows=None,
              witness=None, passed=True):
    """A case dict with just the blocks ``decide`` reads (slopes per decade)."""
    low = slope - 0.1 if low is None else low
    high = slope + 0.1 if high is None else high
    s = (1 if slope > 0 else -1 if slope < 0 else 0) * 100 if s is None else s
    c = {
        "region": region,
        "variable": variable,
        "quality": {"passed": passed, "reason": "too short"},
        "series": {"climatology": [2.0] * 12 if climatology is None else climatology},
        "theil_sen": {"slope_per_decade": slope, "low_slope_per_decade": low,
                      "high_slope_per_decade": high},
        "mk_hamed_rao": {"p": p, "s": s},
        "consistency": {"agree": agree},
        "autocorrelation": {"r1_detrended": r1},
    }
    if windows is not None:
        c["windows"] = {"same_sign_share": windows}
    if witness is not None:
        c["witness"] = witness
    if not passed:
        for k in ("theil_sen", "mk_hamed_rao", "consistency", "autocorrelation"):
            c[k] = None
    return c


AGREEING_WITNESS = {"sign": 1, "significant": True}


def checks(v):
    return {c["name"]: c for c in v["checks"]}


# --- one case per label --------------------------------------------------------


def test_strong_evidence_needs_every_check():
    v = verdict.decide(make_case(windows=0.9, witness=AGREEING_WITNESS), TH)
    assert v["label"] == "strong_evidence"
    assert not v["capped"]
    assert all(c["status"] == "pass" for c in v["checks"])


def test_capped_strong_while_window_and_witness_are_pending():
    v = verdict.decide(make_case(), TH)
    assert v["label"] == "moderate"
    assert v["capped"] is True
    ch = checks(v)
    assert ch["hamed_rao_p"] == {"name": "hamed_rao_p", "status": "pass", "value": 0.001,
                                 "threshold": A_STRONG}
    assert ch["window_agreement"]["status"] == "pending"
    assert ch["window_agreement"]["threshold"] == TH["window_agreement_strong"]
    assert ch["witness_agreement"]["status"] == "pending"
    assert "window agreement and witness agreement" in v["reason"]


def test_capped_with_only_one_check_pending():
    v = verdict.decide(make_case(windows=0.95), TH)
    assert v["label"] == "moderate" and v["capped"]
    assert "witness agreement check is done" in v["reason"]


def test_moderate():
    v = verdict.decide(make_case(p=0.03), TH)
    assert v["label"] == "moderate"
    assert v["capped"] is False  # p is not below alpha_strong: Strong was never possible
    assert checks(v)["hamed_rao_p"]["threshold"] == A_MOD


def test_inconclusive():
    v = verdict.decide(make_case(p=0.3, slope=0.05, low=-0.2, high=0.3), TH)
    assert v["label"] == "inconclusive"
    assert checks(v)["hamed_rao_p"]["status"] == "fail"


def test_no_detectable_trend():
    v = verdict.decide(make_case(p=0.6, slope=0.0, low=-0.05, high=0.05, s=3), TH)
    assert v["label"] == "no_detectable_trend"
    assert v["no_trend_margin"] == TEMP_MARGIN


def test_witnesses_disagree():
    v = verdict.decide(make_case(witness={"sign": -1, "significant": True}), TH)
    assert v["label"] == "witnesses_disagree"
    assert checks(v)["witness_agreement"]["status"] == "fail"


def test_insufficient_record():
    v = verdict.decide(make_case(passed=False), TH)
    assert v["label"] == "insufficient_record"
    assert [c["status"] for c in v["checks"]] == ["pending"] * 4
    assert v["capped"] is False
    assert v["reason"] == "The record is too incomplete to test: too short."


def test_every_label_is_reachable_and_known():
    cases = [make_case(windows=0.9, witness=AGREEING_WITNESS), make_case(p=0.03),
             make_case(p=0.3, slope=0.05, low=-0.2, high=0.3),
             make_case(witness={"sign": -1, "significant": True}),
             make_case(p=0.6, slope=0.0, low=-0.05, high=0.05), make_case(passed=False)]
    assert sorted(verdict.decide(c, TH)["label"] for c in cases) == sorted(verdict.LABELS)


# --- boundaries -----------------------------------------------------------------


def test_p_exactly_at_alpha_strong_is_not_strong():
    v = verdict.decide(make_case(p=A_STRONG, windows=1.0, witness=AGREEING_WITNESS), TH)
    assert v["label"] == "moderate"
    assert v["capped"] is False


def test_p_exactly_at_alpha_moderate_is_not_significant():
    v = verdict.decide(make_case(p=A_MOD, slope=0.2, low=0.05, high=0.35), TH)
    assert v["label"] == "inconclusive"
    assert checks(v)["hamed_rao_p"]["status"] == "fail"


def test_p_exactly_at_alpha_moderate_allows_no_trend():
    v = verdict.decide(make_case(p=A_MOD, slope=0.0, low=-0.05, high=0.05), TH)
    assert v["label"] == "no_detectable_trend"


def test_interval_exactly_at_the_margin():
    # r1 <= 0: the widening factor is 1, so the interval is used as given.
    at = make_case(p=0.5, slope=0.0, low=-TEMP_MARGIN, high=TEMP_MARGIN, r1=-0.1)
    assert verdict.decide(at, TH)["label"] == "no_detectable_trend"
    beyond = make_case(p=0.5, slope=0.0, low=-TEMP_MARGIN, high=TEMP_MARGIN * 1.0001, r1=-0.1)
    assert verdict.decide(beyond, TH)["label"] == "inconclusive"


def test_window_share_exactly_at_threshold_passes():
    share = TH["window_agreement_strong"]
    v = verdict.decide(make_case(windows=share, witness=AGREEING_WITNESS), TH)
    assert v["label"] == "strong_evidence"


# --- the autocorrelation bands and caveats --------------------------------------


def test_band_capped_at_moderate_needs_its_own_alpha():
    # r1 = 0.7 is in the (0.6, 0.8] band: Moderate needs p < alpha_strong.
    assert verdict.decide(make_case(p=0.03, r1=0.7), TH)["label"] == "inconclusive"
    v = verdict.decide(make_case(p=0.005, r1=0.7, windows=1.0, witness=AGREEING_WITNESS), TH)
    assert v["label"] == "moderate"
    assert v["capped"] is False  # the band, not a pending check, stops Strong
    assert v["band_max_verdict"] == "moderate"


@pytest.mark.parametrize("r1", [0.85, 0.95])
def test_no_claim_in_the_inconclusive_band_or_above_the_grid(r1):
    v = verdict.decide(make_case(p=1e-6, r1=r1), TH)
    assert v["label"] == "inconclusive"
    assert checks(v)["hamed_rao_p"]["threshold"] is None
    assert "autocorrelated" in v["reason"]


def test_interval_optimistic_case_cannot_be_no_trend():
    c = make_case(p=0.6, slope=0.0, low=-0.01, high=0.01, region="yucatan")
    v = verdict.decide(c, TH)
    assert v["label"] == "inconclusive"
    assert v["caveats"] == ["interval_optimistic"]


def test_precipitation_margin_is_relative_to_the_regional_mean():
    rel = TH["no_trend_margin"]["precipitation"]["relative_to_mean"]
    c = make_case(p=0.5, slope=0.0, low=-0.05, high=0.05, variable="precipitation",
                  climatology=[2.0] * 12)  # margin = rel * 2 mm/day
    v = verdict.decide(c, TH)
    assert v["no_trend_margin"] == pytest.approx(rel * 2.0)
    assert v["label"] == ("no_detectable_trend" if 0.05 <= rel * 2.0 else "inconclusive")


# --- the other checks -----------------------------------------------------------


def test_failed_window_check_blocks_strong_without_capping():
    v = verdict.decide(make_case(windows=0.5, witness=AGREEING_WITNESS), TH)
    assert v["label"] == "moderate" and v["capped"] is False
    assert checks(v)["window_agreement"] == {"name": "window_agreement", "status": "fail",
                                             "value": 0.5, "threshold": TH["window_agreement_strong"]}


def test_failed_consistency_blocks_strong_without_capping():
    v = verdict.decide(make_case(agree=False), TH)
    assert v["label"] == "moderate" and v["capped"] is False
    assert checks(v)["mk_seasonal_consistency"]["status"] == "fail"


def test_witness_with_opposite_but_insignificant_sign_is_not_disagreement():
    v = verdict.decide(make_case(witness={"sign": -1, "significant": False}), TH)
    assert v["label"] == "inconclusive"  # Moderate needs the same sign across witnesses
    assert checks(v)["witness_agreement"]["status"] == "fail"


def test_insignificant_witness_a_cannot_disagree():
    v = verdict.decide(make_case(p=0.4, slope=0.1, low=-0.1, high=0.3,
                                 witness={"sign": -1, "significant": True}), TH)
    assert v["label"] == "inconclusive"


def test_witnesses_disagree_rule():
    assert verdict.witnesses_disagree(1, True, -1, True)
    assert not verdict.witnesses_disagree(1, True, 1, True)
    assert not verdict.witnesses_disagree(1, False, -1, True)
    assert not verdict.witnesses_disagree(1, True, -1, False)
    assert not verdict.witnesses_disagree(1, True, 0, True)


def test_rules_read_the_thresholds_passed_in():
    th = copy.deepcopy(TH)
    th["alpha_strong"], th["alpha_moderate"] = 0.001, 0.002
    for b in th["autocorrelation_bands"]:
        if b["moderate_alpha"] is not None:
            b["moderate_alpha"] = 0.002 if b["moderate_alpha"] == A_MOD else 0.001
    assert verdict.decide(make_case(p=0.0015), th)["label"] == "moderate"
    assert verdict.decide(make_case(p=0.0015), th)["capped"] is False
    assert verdict.decide(make_case(p=0.003), th)["label"] == "inconclusive"


def test_checks_are_listed_in_order_and_strict_json():
    for c in (make_case(), make_case(passed=False)):
        v = verdict.decide(c, TH)
        assert [x["name"] for x in v["checks"]] == list(verdict.CHECK_NAMES)
        assert set(v) == {"label", "checks", "capped", "reason", "band_max_verdict",
                          "no_trend_margin", "caveats"}
        json.dumps(v, allow_nan=False)


@pytest.mark.parametrize("bad", [{"windows": 1.5}, {"witness": {"sign": 2, "significant": True}}])
def test_malformed_phase3_inputs_raise(bad):
    c = make_case()
    if "windows" in bad:
        c["windows"] = {"same_sign_share": bad["windows"]}
    else:
        c["witness"] = bad["witness"]
    with pytest.raises(ValueError):
        verdict.decide(c, TH)


# --- the reason sentence (review round 1) ----------------------------------------


def test_moderate_reason_names_p_when_p_is_not_below_alpha_strong():
    # Sahel temperature: p = 0.016 in a Strong band, no failed check.
    v = verdict.decide(make_case(p=0.016, r1=0.3), TH)
    assert v["label"] == "moderate" and not v["capped"]
    assert v["reason"] == (f"The trend is significant (p = 0.016 < {A_MOD:g}), "
                           f"but p is not below {A_STRONG:g} as Strong requires.")


def test_moderate_reason_names_the_band_cap():
    # Amazon temperature: band capped at Moderate, every check pass or pending.
    v = verdict.decide(make_case(p=0.005, r1=0.7), TH)
    assert v["label"] == "moderate" and not v["capped"]
    assert "fail" not in [c["status"] for c in v["checks"]]
    assert v["reason"] == (f"The trend is significant (p = 0.005 < {A_STRONG:g}), "
                           "but this record is too autocorrelated for more than Moderate.")


def test_moderate_reason_names_the_failed_check():
    v = verdict.decide(make_case(agree=False), TH)
    assert v["reason"] == (f"The trend is significant (p = 0.001 < {A_STRONG:g}), "
                           "but the seasonal-test consistency check failed.")
    v = verdict.decide(make_case(windows=0.5, witness=AGREEING_WITNESS), TH)
    assert v["reason"].endswith("but the window agreement check failed.")
    v = verdict.decide(make_case(agree=False, windows=0.5, witness=AGREEING_WITNESS), TH)
    assert v["reason"].endswith("but the seasonal-test consistency and window agreement "
                                "checks failed.")


def test_moderate_reason_lists_every_blocker():
    # Sahel precipitation: p not below alpha_strong and consistency failed.
    v = verdict.decide(make_case(p=0.0155, agree=False), TH)
    assert v["reason"].endswith(f"but p is not below {A_STRONG:g} as Strong requires "
                                "and the seasonal-test consistency check failed.")
    v = verdict.decide(make_case(p=0.005, r1=0.7, windows=0.5), TH)
    assert v["reason"].endswith("but this record is too autocorrelated for more than Moderate "
                                "and the window agreement check failed.")


def test_inconclusive_reason_names_the_wide_interval():
    v = verdict.decide(make_case(p=0.3, slope=0.05, low=-0.2, high=0.3), TH)
    assert v["label"] == "inconclusive"
    assert v["reason"] == ("The test cannot tell this trend from noise (p = 0.3), and the "
                           "slope's interval, widened for autocorrelation, reaches past "
                           f"±{TEMP_MARGIN:.3g} per decade, so a trend that size is not ruled out.")
    assert "zero change" not in v["reason"]


def test_inconclusive_reason_names_interval_optimistic():
    v = verdict.decide(make_case(p=0.6, slope=0.0, low=-0.01, high=0.01, region="yucatan"), TH)
    assert v["label"] == "inconclusive"
    assert v["reason"].endswith("and this record's noise makes its interval too optimistic "
                                "to rule out a trend.")


def test_inconclusive_reason_above_the_grid():
    v = verdict.decide(make_case(p=0.6, slope=0.0, low=-0.01, high=0.01, r1=0.95), TH)
    assert v["label"] == "inconclusive"
    assert v["reason"].endswith("and the record is more autocorrelated than the benchmark covers.")


@pytest.mark.parametrize("p", [0.0, 1e-9, 9.99e-7])
def test_tiny_p_is_printed_as_below_the_floor(p):
    v = verdict.decide(make_case(p=p), TH)
    assert v["reason"].startswith("The trend is significant (p < 1e-6), ")
    assert "p = 0" not in v["reason"]
    assert checks(v)["hamed_rao_p"]["value"] == p  # the raw value is kept


def test_p_at_the_floor_is_printed():
    assert verdict._p_text(verdict.P_FLOOR, A_STRONG) == f"p = 1e-06 < {A_STRONG:g}"
    assert verdict._p_text(0.0, A_STRONG) == "p < 1e-6"
    assert verdict._p_text(0.0123) == "p = 0.0123"


# --- the case JSON ----------------------------------------------------------------


@pytest.fixture(scope="module")
def mexicali():
    return trendcourt.examine("mexicali", "temperature")


def test_examine_adds_the_verdict_block(mexicali):
    assert mexicali["schema_version"] == case_module.SCHEMA_VERSION == 2
    assert mexicali["verdict"] == verdict.decide(mexicali, thresholds.load())
    assert mexicali["verdict"]["label"] == "moderate" and mexicali["verdict"]["capped"]
    assert mexicali["verdict"]["band_max_verdict"] == "strong"


def test_failed_quality_gate_gives_insufficient_record(monkeypatch):
    import dataclasses

    from trendcourt import clerk
    prepared = clerk.prepare("mexicali", "temperature")
    quality = dataclasses.replace(prepared.quality, passed=False, verdict=clerk.INSUFFICIENT,
                                  reason="test reason")
    monkeypatch.setattr(clerk, "prepare",
                        lambda region, var: dataclasses.replace(prepared, quality=quality))
    out = trendcourt.examine("mexicali", "temperature")
    assert out["autocorrelation"] is None
    assert out["verdict"]["label"] == "insufficient_record"
    assert out["verdict"]["reason"] == "The record is too incomplete to test: test reason."
