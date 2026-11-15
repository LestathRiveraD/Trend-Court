"""The verdict: one label per case, from the case JSON and ``thresholds.json``.

v0 (issue #20). The labels come from ``project.md`` §3:

* ``strong_evidence``: the significance test allows Strong
  (:func:`thresholds.significance_claim` is ``"strong"``: the record's
  autocorrelation band allows Strong and the Hamed–Rao p is below
  ``alpha_strong``), the Hamed–Rao and seasonal tests are consistent, at
  least ``window_agreement_strong`` of the windows have the same sign,
  and witness B agrees.
* ``moderate``: the significance test allows a claim (p below the band's
  ``moderate_alpha``) but not every Strong check passed. While window
  agreement or witness agreement is still missing, a case that would
  otherwise be Strong stays Moderate and ``capped`` is true
  (``project.md`` §7: "single-witness mode with the verdict capped at
  Moderate").
* ``witnesses_disagree``: witness A (POWER) and witness B both give a
  significant trend, of opposite signs.
* ``no_detectable_trend``: :func:`thresholds.no_trend_allowed`: p at or
  above ``alpha_moderate`` and the AR(1)-widened Theil–Sen interval inside
  ``±no_trend_margin`` for the variable, within the benchmarked
  autocorrelation range and for a case not flagged ``interval_optimistic``.
* ``inconclusive``: everything else, including a small p in an
  autocorrelation band that allows no claim (or caps it below that p).
* ``insufficient_record``: the record failed the Clerk's quality gate; no
  statistics were run.

The autocorrelation bands refine the issue's plain "p < alpha_moderate ->
Moderate": a band whose ``moderate_alpha`` is ``alpha_strong`` needs
p < 0.01 for Moderate, and a band capped at Inconclusive (or ``r1`` above
the benchmark grid) allows no significance claim at all. Every number comes
from ``thresholds.json`` through :mod:`trendcourt.thresholds`.

Window agreement and witness B arrive in Phase 3. Until then a case has no
``windows`` / ``witness`` block, their checks are ``"pending"``, and the
verdict never claims they passed. Their provisional inputs, read when
present:

* ``case["windows"] = {"same_sign_share": float}``: share of windows whose
  Theil–Sen slope has the same sign as the full record's;
* ``case["witness"] = {"sign": -1 | 0 | 1, "significant": bool}``: the sign
  of witness B's trend and whether its own test is significant.
"""

from __future__ import annotations

import math

from trendcourt import thresholds as thresholds_module

LABELS = ("strong_evidence", "moderate", "inconclusive", "witnesses_disagree",
          "no_detectable_trend", "insufficient_record")

CHECK_NAMES = ("hamed_rao_p", "mk_seasonal_consistency", "window_agreement", "witness_agreement")
"""The checks every verdict lists, in this order."""

PASS, FAIL, PENDING = "pass", "fail", "pending"

P_FLOOR = 1e-6
"""Below this a reason prints ``p < 1e-6``, not the value: ``pymannkendall``
underflows to exactly 0 on strong trends, and "p = 0" would be false."""

_CHECK_TEXT = {"mk_seasonal_consistency": "seasonal-test consistency",
               "window_agreement": "window agreement"}


def _p_text(p: float, threshold: float | None = None) -> str:
    """``p`` for a reason: ``"p = 0.016 < 0.05"``, or ``"p < 1e-6"`` below
    :data:`P_FLOOR` (0 included)."""
    if p < P_FLOOR:
        return f"p < {P_FLOOR:.0e}".replace("e-0", "e-")
    return f"p = {p:.3g}" + ("" if threshold is None else f" < {threshold:g}")


def _check(name: str, status: str, value=None, threshold=None) -> dict:
    return {"name": name, "status": status, "value": value, "threshold": threshold}


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def witnesses_disagree(sign_a: int, significant_a: bool, sign_b: int, significant_b: bool) -> bool:
    """Significant results of opposite sign (``project.md`` §3).

    A non-significant witness cannot disagree: the sign of a trend that is
    not significant is noise.
    """
    return bool(significant_a and significant_b and sign_a * sign_b < 0)


def _window_check(case: dict, th: dict) -> dict:
    w = case.get("windows")
    threshold = th["window_agreement_strong"]
    if w is None:
        return _check("window_agreement", PENDING, None, threshold)
    share = float(w["same_sign_share"])
    if not (math.isfinite(share) and 0 <= share <= 1):
        raise ValueError(f"windows.same_sign_share must be in [0, 1], got {share!r}")
    return _check("window_agreement", PASS if share >= threshold else FAIL, share, threshold)


def _witness_check(case: dict, sign_a: int) -> dict:
    w = case.get("witness")
    if w is None:
        return _check("witness_agreement", PENDING)
    sign_b = int(w["sign"])
    if sign_b not in (-1, 0, 1):
        raise ValueError(f"witness.sign must be -1, 0 or 1, got {w['sign']!r}")
    agree = sign_a != 0 and sign_b == sign_a
    return _check("witness_agreement", PASS if agree else FAIL, agree)


def _insufficient(case: dict) -> dict:
    reason = ((case.get("quality") or {}).get("reason") or "it failed the quality gate").rstrip(".")
    return {
        "label": "insufficient_record",
        "checks": [_check(n, PENDING) for n in CHECK_NAMES],
        "capped": False,
        "reason": f"The record is too incomplete to test: {reason}.",
        "band_max_verdict": None,
        "no_trend_margin": None,
        "caveats": [],
    }


def decide(case: dict, thresholds: dict) -> dict:
    """The verdict of one case.

    ``case`` is the case JSON of :func:`trendcourt.case.examine` (its
    ``autocorrelation.r1_detrended`` is the ``r1`` of the band rule), or a
    hand-built dict with the same blocks. ``thresholds`` is
    :func:`trendcourt.thresholds.load`. Returns a plain dict:

    * ``label``: one of :data:`LABELS`;
    * ``checks``: ``[{name, status, value, threshold}]`` for
      :data:`CHECK_NAMES`, ``status`` ``"pass"``, ``"fail"`` or
      ``"pending"`` (input not available yet);
    * ``capped``: true when only pending checks stopped a Strong label;
    * ``reason``: one plain sentence for the card;
    * ``band_max_verdict``, ``no_trend_margin``: the numbers behind the label
      (the band of ``autocorrelation.r1_detrended``; ``None`` above the
      benchmark grid; the margin per decade, in the variable's units);
    * ``caveats``: the case's ``noise_caveats`` flags; non-empty means the
      card must print ``noise_model_check.card_text``.
    """
    th = thresholds
    if not (case.get("quality") or {}).get("passed"):
        return _insufficient(case)

    ts, mk, cons = case["theil_sen"], case["mk_hamed_rao"], case["consistency"]
    region, variable = case["region"], case["variable"]
    p = float(mk["p"])
    r1 = float(case["autocorrelation"]["r1_detrended"])
    slope = float(ts["slope_per_decade"])
    sign_a = _sign(float(mk["s"]))

    band = thresholds_module.band_for(r1, th)
    claim = thresholds_module.significance_claim(r1, p, th)
    if claim == "strong":
        p_threshold = th["alpha_strong"]
    elif band is not None and band["moderate_alpha"] is not None:
        p_threshold = band["moderate_alpha"]
    else:
        p_threshold = None  # the band allows no significance claim
    checks = [
        _check("hamed_rao_p", PASS if claim else FAIL, p, p_threshold),
        _check("mk_seasonal_consistency", PASS if cons["agree"] else FAIL, bool(cons["agree"])),
        _window_check(case, th),
        _witness_check(case, sign_a),
    ]
    by_name = {c["name"]: c for c in checks}

    regional_mean = None
    if isinstance(th["no_trend_margin"].get(variable), dict):
        regional_mean = thresholds_module.regional_mean(case["series"]["climatology"])
    margin = thresholds_module.no_trend_margin(variable, th, regional_mean=regional_mean)
    caveats = thresholds_module.noise_caveat(region, variable, th)

    capped = False
    witness = case.get("witness")
    if witness is not None and witnesses_disagree(sign_a, claim is not None,
                                                  int(witness["sign"]), bool(witness["significant"])):
        label = "witnesses_disagree"
        reason = ("Both datasets show a significant trend, but in opposite directions, "
                  "so the evidence contradicts itself.")
    elif claim is not None:
        robustness = [by_name[n]["status"] for n in
                      ("mk_seasonal_consistency", "window_agreement", "witness_agreement")]
        if by_name["witness_agreement"]["status"] == FAIL:
            label = "inconclusive"
            reason = (f"The trend is significant ({_p_text(p)}), but the second dataset "
                      "does not show the same direction.")
        elif claim == "strong" and all(s == PASS for s in robustness):
            label = "strong_evidence"
            reason = (f"The trend is significant ({_p_text(p, th['alpha_strong'])}) "
                      "and holds across time windows and a second dataset.")
        else:
            label = "moderate"
            capped = claim == "strong" and FAIL not in robustness
            if capped:
                missing = [n.replace("_", " ") for n in ("window_agreement", "witness_agreement")
                           if by_name[n]["status"] == PENDING]
                reason = (f"The trend is significant ({_p_text(p, th['alpha_strong'])}), "
                          f"but it stays Moderate until the {' and '.join(missing)} "
                          f"check{'s are' if len(missing) > 1 else ' is'} done.")
            else:
                # Name what actually stopped Strong: the band's cap or p, and any failed check.
                blockers = []
                if band["max_verdict"] != "strong":
                    blockers.append("this record is too autocorrelated for more than Moderate")
                elif claim != "strong":
                    blockers.append(f"p is not below {th['alpha_strong']:g} as Strong requires")
                failed = [_CHECK_TEXT[n] for n in ("mk_seasonal_consistency", "window_agreement")
                          if by_name[n]["status"] == FAIL]
                if failed:
                    blockers.append(f"the {' and '.join(failed)} "
                                    f"check{'s' if len(failed) > 1 else ''} failed")
                reason = (f"The trend is significant ({_p_text(p, p_threshold)}), "
                          f"but {' and '.join(blockers)}.")
    else:
        ts_low, ts_high = float(ts["low_slope_per_decade"]), float(ts["high_slope_per_decade"])
        if thresholds_module.no_trend_allowed(variable, slope, ts_low, ts_high, r1, p, th,
                                              region=region, regional_mean=regional_mean):
            label = "no_detectable_trend"
            reason = (f"The test finds no trend ({_p_text(p)}) and the slope's interval stays "
                      f"within ±{margin:.3g} per decade of zero.")
        elif p < th["alpha_moderate"]:
            label = "inconclusive"
            reason = (f"The p-value is small ({_p_text(p)}), but this record is too autocorrelated "
                      "for that p-value to count as evidence.")
        else:
            label = "inconclusive"
            # Name what stopped "No detectable trend", in no_trend_allowed's terms.
            if band is None:
                blocker = "the record is more autocorrelated than the benchmark covers"
            elif "interval_optimistic" in caveats:
                blocker = ("this record's noise makes its interval too optimistic "
                           "to rule out a trend")
            else:
                blocker = (f"the slope's interval, widened for autocorrelation, reaches past "
                           f"±{margin:.3g} per decade, so a trend that size is not ruled out")
            reason = f"The test cannot tell this trend from noise ({_p_text(p)}), and {blocker}."

    return {
        "label": label,
        "checks": checks,
        "capped": capped,
        "reason": reason,
        "band_max_verdict": None if band is None else band["max_verdict"],
        "no_trend_margin": margin,
        "caveats": caveats,
    }
