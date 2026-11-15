"""Calibrated verdict thresholds (issue #19, ``docs/benchmark.md``).

The numbers live in ``trendcourt/thresholds.json``; this module only reads
and checks them. Nothing else in the package may hard-code a verdict
threshold: the verdict module (#20) calls :func:`load` and
:func:`band_for`.

The file holds

* ``alpha_strong`` / ``alpha_moderate``: Hamed–Rao p-value cut-offs.
* ``window_agreement_strong``: share of windows with the same sign that
  Strong needs (Phase 3; recorded now so the schema does not change).
* ``no_trend_margin``: per variable, in the variable's units per decade
  (``no_trend_margin_units``). A number is an absolute margin; an object
  ``{"relative_to_mean": r}`` (precipitation) means ``r`` times the
  region's mean, as ``no_trend_margin_formula`` states
  (:func:`no_trend_margin`, :func:`regional_mean`). "No detectable trend"
  is allowed only when the Hamed–Rao test is not significant at
  ``alpha_moderate`` *and* the AR(1)-widened Theil–Sen interval
  (:func:`widened_interval`) lies inside ``[-margin, +margin]``, for
  records inside the benchmarked autocorrelation range
  (:func:`no_trend_allowed`).
* ``interval_widening``: how that interval is widened; ``method`` must
  be ``"ar1"``, the only one implemented.
* ``fpr_policy`` and ``autocorrelation_bands``: the largest false-positive
  rate each verdict may carry, and, for each range of lag-1
  autocorrelation, the verdict cap and the Moderate cut-off that follow
  from ``docs/benchmark/fpr.json`` (:func:`derive_bands` rebuilds them).
  ``fpr_policy.test`` must be ``"hamed_rao"``, the test the bands use.
  :func:`significance_claim` applies them to one record.
* ``noise_model_check`` and ``noise_caveats``: the rule that flags a real
  case whose own noise is more persistent than the AR(1) model behind the
  bands and the widening, and the flagged cases it gives on
  ``docs/benchmark/noise.json`` (:func:`derive_noise_caveats` rebuilds
  them; :func:`noise_caveat` looks one case up). The card of a flagged
  case must say that its stated false-positive rate and interval are
  optimistic (``noise_model_check.card_text``).
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_PATH = Path(__file__).resolve().with_name("thresholds.json")
FPR_PATH = REPO_ROOT / "docs" / "benchmark" / "fpr.json"
NOISE_PATH = REPO_ROOT / "docs" / "benchmark" / "noise.json"

SCHEMA_VERSION = 2
"""Version 2: relative precipitation margin, noise-model caveats (statistics review, issue #19)."""

VERDICT_ORDER = ("inconclusive", "moderate", "strong")
"""Verdict caps from weakest to strongest."""

TEST = "hamed_rao"
"""The significance test the verdict uses (``project.md`` §3, step 6)."""

WIDENING_METHOD = "ar1"
"""The interval widening :func:`widening_factor` implements."""

REQUIRED_KEYS = ("schema_version", "source", "alpha_strong", "alpha_moderate",
                 "window_agreement_strong", "no_trend_margin", "no_trend_margin_units",
                 "no_trend_margin_formula", "interval_widening", "fpr_policy",
                 "autocorrelation_bands", "noise_model_check", "noise_caveats")

RELATIVE_MARGIN_KEY = "relative_to_mean"
"""Key of a relative ``no_trend_margin`` entry: the margin per decade as a
fraction of the regional mean."""

NOISE_FLAGS = ("interval_optimistic", "fpr_optimistic")
"""Caveat flags :func:`derive_noise_caveats` can give a case."""


def worst_fpr(fpr: dict, phi: float, alpha: float, test: str = TEST) -> dict:
    """Largest rejection rate over record lengths at one ``phi`` and ``alpha``.

    Returns ``{"rate", "ci95", "n_years"}`` from the committed FPR cell
    with the highest rate (the shortest such record on a tie).
    """
    cells = [c for c in fpr["cells"] if c["phi"] == phi]
    if not cells:
        raise ValueError(f"phi {phi} is not on the FPR grid {fpr['grid']['phi']}")
    key = f"{alpha:g}"
    best = max(cells, key=lambda c: (c["rates"][test][key]["rate"], -c["n_years"]))
    s = best["rates"][test][key]
    return {"rate": s["rate"], "ci95": list(s["ci95"]), "n_years": best["n_years"]}


def derive_bands(fpr: dict, alpha_strong: float, alpha_moderate: float,
                 max_fpr_strong: float, max_fpr_moderate: float) -> list[dict]:
    """Autocorrelation bands and verdict caps from the FPR benchmark.

    Band ``i`` covers ``r1`` in ``(phi[i-1], phi[i]]`` of the benchmark
    grid (the first band starts at -1), and takes the false-positive
    rates of its upper edge ``phi[i]``, the worst case over record
    lengths (:func:`worst_fpr`). The rate rises with ``phi``, so each
    band reports the rate of its most autocorrelated end, never less.
    The ``phi = 0`` point is folded into the first nonzero band.

    * ``max_verdict`` is ``"strong"`` when the rate at ``alpha_strong``
      is at most ``max_fpr_strong``; else ``"moderate"`` when
      ``alpha_moderate`` or ``alpha_strong`` keeps the rate at most
      ``max_fpr_moderate``; else ``"inconclusive"``.
    * ``moderate_alpha`` is the larger of the two alphas whose rate is at
      most ``max_fpr_moderate`` (``None`` when neither is).

    Records with ``r1`` above the largest grid ``phi`` are outside the
    benchmark: :func:`band_for` returns ``None`` for them.
    """
    grid = sorted(float(p) for p in fpr["grid"]["phi"])
    edges = [p for p in grid if p > 0] or grid
    bands = []
    lower = -1.0
    for phi in edges:
        at = {f"{a:g}": worst_fpr(fpr, phi, a) for a in (alpha_moderate, alpha_strong)}
        moderate_alpha = next((a for a in (alpha_moderate, alpha_strong)
                               if at[f"{a:g}"]["rate"] <= max_fpr_moderate), None)
        if at[f"{alpha_strong:g}"]["rate"] <= max_fpr_strong:
            cap = "strong"
        elif moderate_alpha is not None:
            cap = "moderate"
        else:
            cap = "inconclusive"
        bands.append({"r1_min_exclusive": lower, "r1_max": phi, "benchmark_phi": phi,
                      "max_verdict": cap, "moderate_alpha": moderate_alpha,
                      "hamed_rao_fpr": at})
        lower = phi
    return bands


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_margins(th: dict) -> None:
    margins = th["no_trend_margin"]
    if set(margins) != set(th["no_trend_margin_units"]):
        raise ValueError("no_trend_margin and no_trend_margin_units must name the same variables")
    relative = []
    for var, m in margins.items():
        if isinstance(m, dict):
            r = m.get(RELATIVE_MARGIN_KEY)
            if set(m) != {RELATIVE_MARGIN_KEY} or not (_is_number(r) and 0 < r < 1):
                raise ValueError(f"no_trend_margin[{var!r}] must be a positive number or "
                                 f"{{{RELATIVE_MARGIN_KEY!r}: r}} with 0 < r < 1, got {m!r}")
            relative.append(var)
        elif not (_is_number(m) and m > 0):
            raise ValueError(f"no_trend_margin[{var!r}] must be a positive number, got {m!r}")
    formula = th["no_trend_margin_formula"]
    if not isinstance(formula, dict) or set(formula) != set(relative) or not all(
            isinstance(formula[v], str) and formula[v] for v in relative):
        raise ValueError("no_trend_margin_formula must state the formula of every relative margin, "
                         f"and only those ({sorted(relative)})")


def _check_noise(th: dict) -> None:
    rule = th["noise_model_check"]
    if not isinstance(rule, dict):
        raise ValueError("noise_model_check must be an object")
    v = rule.get("min_widened_coverage")
    if not (_is_number(v) and 0 < v < 1):
        raise ValueError(f"noise_model_check.min_widened_coverage must be in (0, 1), got {v!r}")
    for key in ("source", "card_text"):
        if not (isinstance(rule.get(key), str) and rule[key]):
            raise ValueError(f"noise_model_check.{key} must be a non-empty string")
    seen = set()
    caveats = th["noise_caveats"]
    if not isinstance(caveats, list):
        raise ValueError("noise_caveats must be a list")
    for c in caveats:
        if not (isinstance(c, dict) and set(c) == {"region", "variable", "flags"}):
            raise ValueError(f"each noise_caveats entry needs exactly region, variable, flags, got {c!r}")
        key = (c["region"], c["variable"])
        if key in seen:
            raise ValueError(f"noise_caveats lists {key} twice")
        seen.add(key)
        if not c["flags"] or any(f not in NOISE_FLAGS for f in c["flags"]) or c["flags"] != sorted(set(c["flags"])):
            raise ValueError(f"noise_caveats flags must be a sorted, non-empty subset of {NOISE_FLAGS}, "
                             f"got {c['flags']!r}")


def _check(th: dict) -> dict:
    missing = [k for k in REQUIRED_KEYS if k not in th]
    if missing:
        raise ValueError(f"thresholds.json is missing {missing}")
    if th["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}, got {th['schema_version']!r}")
    a_s, a_m = th["alpha_strong"], th["alpha_moderate"]
    if not (0 < a_s < a_m < 1):
        raise ValueError(f"need 0 < alpha_strong < alpha_moderate < 1, got {a_s}, {a_m}")
    if not (0 < th["window_agreement_strong"] <= 1):
        raise ValueError("window_agreement_strong must be in (0, 1]")
    policy = th["fpr_policy"]
    for key in ("max_fpr_strong", "max_fpr_moderate"):
        v = policy.get(key)
        if not (isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1):
            raise ValueError(f"fpr_policy.{key} must be a number in [0, 1], got {v!r}")
    if policy["max_fpr_strong"] > policy["max_fpr_moderate"]:
        raise ValueError("need fpr_policy.max_fpr_strong <= fpr_policy.max_fpr_moderate, got "
                         f"{policy['max_fpr_strong']}, {policy['max_fpr_moderate']}")
    if policy.get("test") != TEST:
        raise ValueError(f"fpr_policy.test must be {TEST!r} (the only test derive_bands uses), "
                         f"got {policy.get('test')!r}")
    if th["interval_widening"].get("method") != WIDENING_METHOD:
        raise ValueError(f"interval_widening.method must be {WIDENING_METHOD!r} (the only one "
                         f"widening_factor implements), got {th['interval_widening'].get('method')!r}")
    _check_margins(th)
    _check_noise(th)
    bands = th["autocorrelation_bands"]
    if not bands:
        raise ValueError("autocorrelation_bands is empty")
    prev = -1.0
    for b in bands:
        if b["r1_min_exclusive"] != prev or not b["r1_max"] > prev:
            raise ValueError("autocorrelation_bands must be contiguous and increasing from -1")
        if b["max_verdict"] not in VERDICT_ORDER:
            raise ValueError(f"unknown max_verdict {b['max_verdict']!r}")
        if b["moderate_alpha"] not in (None, a_s, a_m):
            raise ValueError(f"moderate_alpha must be alpha_strong, alpha_moderate or null, "
                             f"got {b['moderate_alpha']!r}")
        if (b["max_verdict"] == "inconclusive") != (b["moderate_alpha"] is None):
            raise ValueError("an inconclusive band has moderate_alpha null, and only it")
        prev = b["r1_max"]
    return th


@lru_cache(maxsize=1)
def _load_cached() -> str:
    return THRESHOLDS_PATH.read_text(encoding="utf-8")


def load(path: str | Path | None = None) -> dict:
    """The thresholds, checked. Default path: ``trendcourt/thresholds.json``.

    Returns a fresh dict each call, so callers may not mutate shared state.
    """
    text = _load_cached() if path is None else Path(path).read_text(encoding="utf-8")
    return _check(json.loads(text))


def band_for(r1: float, thresholds: dict | None = None) -> dict | None:
    """The autocorrelation band of a record with lag-1 autocorrelation ``r1``.

    ``r1`` is the lag-1 autocorrelation of the anomalies after removing
    the Theil–Sen line (``benchmark.real_case_autocorr``'s
    ``r1_detrended``). Returns ``None`` when ``r1`` is above every band:
    the record is more persistent than anything benchmarked, and the
    verdict must not claim significance.
    """
    if not math.isfinite(r1):
        raise ValueError(f"r1 must be finite, got {r1}")
    th = load() if thresholds is None else thresholds
    for b in th["autocorrelation_bands"]:
        if b["r1_min_exclusive"] < r1 <= b["r1_max"]:
            return b
    if r1 <= th["autocorrelation_bands"][0]["r1_min_exclusive"]:
        return th["autocorrelation_bands"][0]
    return None


def significance_claim(r1: float, hr_p: float, thresholds: dict | None = None) -> str | None:
    """The strongest significance label the record's band allows for this p.

    ``"strong"`` when the band (:func:`band_for` of ``r1``) allows Strong
    and ``hr_p < alpha_strong``; else ``"moderate"`` when the band allows a
    claim and ``hr_p`` is below its ``moderate_alpha``; else ``None`` (no
    significance claim: outside the bands, a band capped at Inconclusive,
    or p too large). This is the significance part of the verdict only:
    Strong also needs window and witness agreement (Phase 3), which only
    remove claims. ``docs/benchmark/applied.json`` simulates this function
    on trend-free series.
    """
    if not math.isfinite(hr_p):
        raise ValueError(f"hr_p must be finite, got {hr_p}")
    th = load() if thresholds is None else thresholds
    band = band_for(r1, th)
    if band is None or band["max_verdict"] == "inconclusive":
        return None
    if band["max_verdict"] == "strong" and hr_p < th["alpha_strong"]:
        return "strong"
    if hr_p < band["moderate_alpha"]:
        return "moderate"
    return None


def widening_factor(r1: float) -> float:
    """AR(1) variance-inflation factor ``sqrt((1 + r1) / (1 - r1))``.

    ``r1`` at or below 0 gives 1: the interval is never narrowed.
    """
    if not (math.isfinite(r1) and r1 < 1):
        raise ValueError(f"r1 must be finite and below 1, got {r1}")
    r = max(r1, 0.0)
    return math.sqrt((1.0 + r) / (1.0 - r))


def widened_interval(slope: float, low: float, high: float, r1: float) -> tuple[float, float]:
    """Theil–Sen interval widened about the slope by :func:`widening_factor`.

    Each side is scaled separately, so an asymmetric interval stays
    asymmetric.
    """
    f = widening_factor(r1)
    return slope - f * (slope - low), slope + f * (high - slope)


def regional_mean(climatology) -> float:
    """The regional mean a relative margin refers to: the mean of the 12
    calendar-month climatology values (``clerk.prepare(...).climatology``,
    whose base period is the record itself), in the variable's units."""
    values = [float(v) for v in climatology]
    if len(values) != 12 or not all(math.isfinite(v) for v in values):
        raise ValueError("the climatology must have 12 finite monthly values")
    return sum(values) / 12.0


def no_trend_margin(variable: str, thresholds: dict | None = None, *,
                    regional_mean: float | None = None) -> float:
    """``no_trend_margin`` for ``variable``, per decade, in its units.

    An absolute margin is returned as is. A relative one
    (``{"relative_to_mean": r}``, precipitation) is ``r * regional_mean``;
    ``regional_mean`` (:func:`regional_mean` of the case's climatology)
    is then required and must be positive.
    """
    th = load() if thresholds is None else thresholds
    if variable not in th["no_trend_margin"]:
        raise ValueError(f"no no_trend_margin for variable {variable!r}")
    m = th["no_trend_margin"][variable]
    if not isinstance(m, dict):
        return float(m)
    if regional_mean is None or not (math.isfinite(regional_mean) and regional_mean > 0):
        raise ValueError(f"the {variable} margin is relative to the regional mean: pass a positive "
                         f"regional_mean, got {regional_mean!r}")
    return m[RELATIVE_MARGIN_KEY] * regional_mean


def no_trend_allowed(variable: str, slope: float, low: float, high: float, r1: float,
                     hr_p: float, thresholds: dict | None = None, *, region: str,
                     regional_mean: float | None = None) -> bool:
    """Whether the evidence supports "No detectable trend".

    True only when the case (``region``, ``variable``) is not flagged
    ``interval_optimistic`` in ``noise_caveats`` (:func:`noise_caveat`),
    ``r1`` is inside the benchmarked bands
    (:func:`band_for` is not ``None``), the Hamed–Rao test is not
    significant at ``alpha_moderate``, and the widened interval lies
    inside ``[-margin, +margin]`` with ``margin`` from
    :func:`no_trend_margin` (same units, per decade, as the slope and
    interval passed in; ``regional_mean`` is required for precipitation,
    whose margin is relative).

    * ``r1`` above the benchmark grid (``band_for`` is ``None``, so above
      0.9 with the committed bands): always False. The AR(1) widening was
      checked only up to ``phi = 0.9``, so the widened interval cannot be
      trusted there; the verdict is Inconclusive.
    * Inside the benchmarked bands, including the band capped at
      Inconclusive for significance claims ((0.8, 0.9] with the committed
      bands), "No detectable trend" stays allowed. That cap concerns the
      false-positive rate of the significance test, not the interval.
      Simulated with ``r1`` estimated from each record
      (``docs/benchmark/applied.json``, ``docs/benchmark.md`` Tables 3 and 7),
      the widened interval covers close to 95% on 45-year records up to
      ``phi = 0.9`` (0.93 at 0.9), but only about 0.91–0.93 on 20-year
      records at ``phi`` 0.85–0.9: Phase 3 windows of 20 years or less in
      that band must not reuse this rule unchanged.
    * A case flagged ``interval_optimistic``: always False. Under its own
      fitted noise the widened interval covers less than
      ``noise_model_check.min_widened_coverage``, so the interval is too
      narrow to support a claim of absence (``docs/benchmark.md``,
      Table 8). ``region`` is a required keyword so the check cannot be
      skipped by accident; a region not in ``noise_caveats`` (e.g. a
      synthetic case) is not flagged.
    """
    th = load() if thresholds is None else thresholds
    m = no_trend_margin(variable, th, regional_mean=regional_mean)
    if "interval_optimistic" in noise_caveat(region, variable, th):
        return False
    if band_for(r1, th) is None:
        return False
    lo, hi = widened_interval(slope, low, high, r1)
    return hr_p >= th["alpha_moderate"] and -m <= lo and hi <= m


def derive_noise_caveats(noise: dict, thresholds: dict | None = None) -> list[dict]:
    """The ``noise_caveats`` that ``noise_model_check`` gives on ``noise.json``.

    For each case of ``docs/benchmark/noise.json`` (trend-free records
    simulated from the case's own fitted AR(p) noise, the policy applied
    with each replicate's own ``r1``):

    * ``interval_optimistic``: the widened interval's simulated coverage is
      below ``noise_model_check.min_widened_coverage``;
    * ``fpr_optimistic``: in a band that allows a claim, the lower end of
      the exact 95% interval of the simulated false-claim rate is above the
      rate the band states at its Moderate cut-off, or (in a band that
      allows Strong) that of the false-Strong rate is above the stated rate
      at ``alpha_strong``. The lower end, so Monte Carlo noise alone does
      not flag a case.

    Returns ``[{"region", "variable", "flags"}]`` for the flagged cases
    only, in the file's order, flags sorted.
    """
    th = load() if thresholds is None else thresholds
    rule = th["noise_model_check"]
    out = []
    for c in noise["cases"]:
        flags = set()
        if c["widened_coverage"]["rate"] < rule["min_widened_coverage"]:
            flags.add("interval_optimistic")
        if c["max_verdict"] != "inconclusive":
            stated = c["stated_fpr"]
            if c["false_claim"]["ci95"][0] > stated[f"{c['moderate_alpha']:g}"]:
                flags.add("fpr_optimistic")
            if (c["max_verdict"] == "strong"
                    and c["false_strong"]["ci95"][0] > stated[f"{th['alpha_strong']:g}"]):
                flags.add("fpr_optimistic")
        if flags:
            out.append({"region": c["region"], "variable": c["variable"], "flags": sorted(flags)})
    return out


def noise_caveat(region: str, variable: str, thresholds: dict | None = None) -> list[str]:
    """The caveat flags of one real case (empty when it is not flagged).

    A non-empty list means the card must print ``noise_model_check.card_text``:
    the case's noise varies on longer time scales than AR(1) allows, so its
    stated false-positive rate and interval are optimistic.
    """
    th = load() if thresholds is None else thresholds
    for c in th["noise_caveats"]:
        if c["region"] == region and c["variable"] == variable:
            return list(c["flags"])
    return []
