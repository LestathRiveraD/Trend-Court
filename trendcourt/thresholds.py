"""Calibrated verdict thresholds (issue #19, ``docs/benchmark.md``).

The numbers live in ``trendcourt/thresholds.json``; this module only reads
and checks them. Nothing else in the package may hard-code a verdict
threshold: the verdict module (#20) calls :func:`load` and
:func:`band_for`.

The file holds

* ``alpha_strong`` / ``alpha_moderate``: Hamed–Rao p-value cut-offs.
* ``window_agreement_strong``: share of windows with the same sign that
  Strong needs (Phase 3; recorded now so the schema does not change).
* ``no_trend_margin``: per variable, in the variable's units per decade.
  "No detectable trend" is allowed only when the Hamed–Rao test is not
  significant at ``alpha_moderate`` *and* the AR(1)-widened Theil–Sen
  interval (:func:`widened_interval`) lies inside ``[-margin, +margin]``,
  for records inside the benchmarked autocorrelation range
  (:func:`no_trend_allowed`).
* ``interval_widening``: how that interval is widened; ``method`` must
  be ``"ar1"``, the only one implemented.
* ``fpr_policy`` and ``autocorrelation_bands``: the largest false-positive
  rate each verdict may carry, and, for each range of lag-1
  autocorrelation, the verdict cap and the Moderate cut-off that follow
  from ``docs/benchmark/fpr.json`` (:func:`derive_bands` rebuilds them).
  ``fpr_policy.test`` must be ``"hamed_rao"``, the test the bands use.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_PATH = Path(__file__).resolve().with_name("thresholds.json")
FPR_PATH = REPO_ROOT / "docs" / "benchmark" / "fpr.json"

VERDICT_ORDER = ("inconclusive", "moderate", "strong")
"""Verdict caps from weakest to strongest."""

TEST = "hamed_rao"
"""The significance test the verdict uses (``project.md`` §3, step 6)."""

WIDENING_METHOD = "ar1"
"""The interval widening :func:`widening_factor` implements."""

REQUIRED_KEYS = ("schema_version", "source", "alpha_strong", "alpha_moderate",
                 "window_agreement_strong", "no_trend_margin", "no_trend_margin_units",
                 "interval_widening", "fpr_policy", "autocorrelation_bands")


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


def _check(th: dict) -> dict:
    missing = [k for k in REQUIRED_KEYS if k not in th]
    if missing:
        raise ValueError(f"thresholds.json is missing {missing}")
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
    margins = th["no_trend_margin"]
    if set(margins) != set(th["no_trend_margin_units"]):
        raise ValueError("no_trend_margin and no_trend_margin_units must name the same variables")
    for var, m in margins.items():
        if not (isinstance(m, (int, float)) and math.isfinite(m) and m > 0):
            raise ValueError(f"no_trend_margin[{var!r}] must be a positive number, got {m!r}")
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


def no_trend_allowed(variable: str, slope: float, low: float, high: float, r1: float,
                     hr_p: float, thresholds: dict | None = None) -> bool:
    """Whether the evidence supports "No detectable trend".

    True only when ``r1`` is inside the benchmarked bands
    (:func:`band_for` is not ``None``), the Hamed–Rao test is not
    significant at ``alpha_moderate``, and the widened interval lies
    inside ``[-no_trend_margin, +no_trend_margin]`` for ``variable``
    (same units, per decade, as the slope and interval passed in).

    * ``r1`` above the benchmark grid (``band_for`` is ``None``, so above
      0.9 with the committed bands): always False. The AR(1) widening was
      checked only up to ``phi = 0.9`` (``docs/benchmark.md``, Table 3),
      so the widened interval cannot be trusted there; the verdict is
      Inconclusive.
    * Inside the benchmarked bands, including the band capped at
      Inconclusive for significance claims ((0.8, 0.9] with the committed
      bands), "No detectable trend" stays allowed. That cap concerns the
      false-positive rate of the significance test, not the interval; the
      widening factor matches the simulated coverage up to ``phi = 0.9``
      and is slightly conservative there (4.36 against the 4.17 needed).
    """
    th = load() if thresholds is None else thresholds
    if variable not in th["no_trend_margin"]:
        raise ValueError(f"no no_trend_margin for variable {variable!r}")
    if band_for(r1, th) is None:
        return False
    m = th["no_trend_margin"][variable]
    lo, hi = widened_interval(slope, low, high, r1)
    return hr_p >= th["alpha_moderate"] and -m <= lo and hi <= m
