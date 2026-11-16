"""The case: one region and variable taken through the Clerk and the Examiner.

:func:`examine` is the first draft of the case JSON that the cache build
and the site read. It adds no statistics of its own: every number comes
straight from :func:`trendcourt.clerk.prepare` and the
:mod:`trendcourt.examiner` wrappers, so Gate 1 (a) (``project.md`` §5)
can check it against direct ``scipy`` and ``pymannkendall`` calls.

Regenerate a reference file with::

    python -m trendcourt mexicali temperature > docs/gate1/mexicali_temperature.json
"""

from __future__ import annotations

import json
import math
import sys

import pandas as pd

from trendcourt import benchmark, clerk, clock, examiner, thresholds, verdict

SCHEMA_VERSION = 2
"""Version 2: ``autocorrelation`` and ``verdict`` blocks (verdict module v0, issue #20).

The ``clock`` block (issue #45) landed before the Phase 3 bump (#41), so
it joins schema 3 there rather than bumping the version itself."""


def _month(ts: pd.Timestamp | None) -> str | None:
    return None if ts is None else f"{ts:%Y-%m}"


def _floats(values) -> list[float | None]:
    """Plain floats for strict JSON; NaN (missing) becomes ``None``."""
    return [float(v) if math.isfinite(v) else None for v in values]


def _provenance(p: clerk.Provenance) -> dict:
    return {
        "source": p.source,
        "api": p.api,
        "region_name": p.region_name,
        "power_code": p.power_code,
        "longname": p.longname,
        "units": p.units,
        "source_units": p.source_units,
        "n_points": p.n_points,
        "period": p.period,
        "start": _month(p.start),
        "end": _month(p.end),
        "years": p.years,
        "accessed_utc": p.accessed_utc,
        "fill_value": p.fill_value,
        "urls": list(p.urls),
        "clerk_text": p.clerk_text(),
    }


def _quality(q: clerk.QualityResult) -> dict:
    return {
        "passed": q.passed,
        "verdict": q.verdict,
        "reason": q.reason,
        "years_covered": q.years_covered,
        "share_missing": q.share_missing,
        "start": _month(q.start),
        "end": _month(q.end),
        "n_months": q.n_months,
        "n_missing": q.n_missing,
    }


def examine(region: str, variable: str) -> dict:
    """Run the Clerk and the Examiner on one region and variable.

    Calls ``clerk.prepare(region, variable)``, then, with ``t`` the
    prepared decimal-year time axis:

    * ``theil_sen(t, anomalies)`` and ``ols(t, anomalies)``: slopes;
    * ``mk_hamed_rao(anomalies)``: the main significance test;
    * ``mk_seasonal(raw)``: the correlated seasonal check, on the raw
      monthly series;
    * ``consistency`` of the two tests.

    All with the Examiner's defaults (95% interval, alpha 0.05, Hamed–Rao
    ``lag=3``). Returns a plain dict that ``json.dumps(...,
    allow_nan=False)`` accepts. If the record fails the quality gate the
    statistics are not run and their keys are ``None``; the quality
    block says why.

    Then ``autocorrelation.r1_detrended``, the lag-1 autocorrelation of the
    anomalies after removing the Theil–Sen line
    (``benchmark.detrended_r1``, the ``r1`` the thresholds' band rule
    reads), and ``verdict``, :func:`trendcourt.verdict.decide` on the case
    with ``thresholds.load()``. Neither changes any other number.

    Last, ``clock``: for an Inconclusive verdict, :func:`clock_block` (years
    of further data until the trend would likely be significant); ``None``
    for every other label.
    """
    prepared = clerk.prepare(region, variable)
    quality = prepared.quality

    case = {
        "schema_version": SCHEMA_VERSION,
        "region": region,
        "variable": prepared.provenance.variable or prepared.provenance.power_code,
        "provenance": _provenance(prepared.provenance),
        "quality": _quality(quality),
        "series": {
            "months": [f"{ts:%Y-%m}" for ts in prepared.raw.index],
            "time": _floats(prepared.time),
            "raw": _floats(prepared.raw),
            "anomaly": _floats(prepared.anomalies),
            "n_points": [int(n) for n in prepared.n_points],
            "climatology": _floats(prepared.climatology),
        },
        "theil_sen": None,
        "ols": None,
        "mk_hamed_rao": None,
        "mk_seasonal": None,
        "consistency": None,
        "autocorrelation": None,
    }
    if not quality.passed:
        case["verdict"] = verdict.decide(case, thresholds.load())
        case["clock"] = None
        return case

    t, anomalies = prepared.time, prepared.anomalies
    mk = examiner.mk_hamed_rao(anomalies)
    smk = examiner.mk_seasonal(prepared.raw)
    ts = examiner.theil_sen(t, anomalies)
    case.update(
        theil_sen=ts.to_dict(),
        ols=examiner.ols(t, anomalies).to_dict(),
        mk_hamed_rao=mk.to_dict(),
        mk_seasonal=smk.to_dict(),
        consistency=examiner.consistency(mk, smk),
        autocorrelation={"r1_detrended": benchmark.detrended_r1(t, anomalies, ts)},
    )
    th = thresholds.load()
    case["verdict"] = verdict.decide(case, th)
    case["clock"] = None
    if case["verdict"]["label"] == "inconclusive":
        case["clock"] = clock_block(region, variable, t, anomalies, ts,
                                    case["autocorrelation"]["r1_detrended"], th)
    return case


def clock_block(region: str, variable: str, t, anomalies, ts, r1: float, th: dict) -> dict:
    """The ``clock`` block of an Inconclusive case (issue #45).

    :func:`trendcourt.clock.years_to_verdict` with ``alpha`` the
    ``moderate_alpha`` of the record's autocorrelation band and the seed
    ``benchmark.case_seed_key(region, variable)``. When the band allows no
    significance claim (``moderate_alpha`` is ``None``, or ``r1`` is above
    the benchmark grid), no record length can give a verdict under the
    Clock's assumption, so nothing is simulated: ``beyond_horizon`` is true,
    the year counts and ``alpha`` are ``None``, ``n_sims`` is 0, and
    ``assumption`` says why.
    """
    band = thresholds.band_for(r1, th)
    alpha = None if band is None else band["moderate_alpha"]
    seed = benchmark.case_seed_key(region, variable)
    if alpha is None:
        return {
            "years_median": None,
            "years_p80": None,
            "beyond_horizon": True,
            "alpha": None,
            "horizon_years": clock.HORIZON_YEARS,
            "n_sims": 0,
            "seed": seed,
            "assumption": clock.NO_CLAIM,
        }
    return clock.years_to_verdict(t, anomalies, ts, alpha=alpha, seed=seed)


def to_json(case: dict) -> str:
    """Canonical strict-JSON text of a case: sorted keys, 2-space indent,
    trailing newline. Floats are written with ``repr``, so they read back
    bit-for-bit."""
    return json.dumps(case, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m trendcourt REGION VARIABLE", file=sys.stderr)
        return 2
    sys.stdout.write(to_json(examine(*argv)))
    return 0

