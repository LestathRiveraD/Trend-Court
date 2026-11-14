"""The Examiner: statistics.

Thin wrappers over ``scipy.stats`` (``theilslopes``, ``linregress``) and
``pymannkendall`` (``hamed_rao_modification_test``, ``seasonal_test``).
No data loading lives here, and no statistics are reimplemented: the
wrappers only drop missing values, convert units (per year to per
decade) and return stable, JSON-serializable result objects.

Usage (``project.md`` §3, step 5, plus the OLS slope for the KNMI check)::

    t = clerk.time_in_years(anomalies.index)
    ts = theil_sen(t, anomalies)        # slope, 95% interval, per decade
    lr = ols(t, anomalies)              # KNMI cross-check
    json.dumps(ts.to_dict())
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

DECADE = 10.0


def _json_safe(d: dict) -> dict:
    """Map non-finite floats to ``None`` so ``json.dumps`` emits ``null``
    (strict JSON) instead of the invalid tokens ``NaN`` / ``Infinity``."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


@dataclass(frozen=True)
class SlopeResult:
    """Theil–Sen slope from ``scipy.stats.theilslopes``.

    ``slope``, ``low_slope`` and ``high_slope`` are in units of ``y`` per
    year (``t`` in decimal years); the ``*_per_decade`` fields are the
    same values times 10. ``n`` is the number of points used after
    dropping NaN pairs; ``t_start`` and ``t_end`` are the first and last
    of those times.
    """

    slope: float
    intercept: float
    low_slope: float
    high_slope: float
    slope_per_decade: float
    low_slope_per_decade: float
    high_slope_per_decade: float
    confidence: float
    method: str
    n: int
    t_start: float
    t_end: float

    def to_dict(self) -> dict:
        """Plain dict for strict JSON; non-finite floats become ``None``."""
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class OLSResult:
    """Least-squares slope from ``scipy.stats.linregress``.

    ``p`` is linregress's two-sided p-value for a zero slope. ``stderr``
    is the standard error of the slope (per year), ``intercept_stderr``
    that of the intercept.
    """

    slope: float
    intercept: float
    r: float
    p: float
    stderr: float
    intercept_stderr: float
    slope_per_decade: float
    stderr_per_decade: float
    n: int
    t_start: float
    t_end: float

    def to_dict(self) -> dict:
        """Plain dict for strict JSON; non-finite floats become ``None``."""
        return _json_safe(asdict(self))


def _clean(t_years, y) -> tuple[np.ndarray, np.ndarray]:
    """Float arrays with every pair containing a NaN dropped.

    NaN means "missing" and is dropped; ±inf is bad data and raises.
    Raises if fewer than 3 pairs or fewer than 2 distinct times remain,
    since scipy would otherwise return an all-NaN fit.
    """
    t = np.asarray(t_years, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or t.shape != y.shape:
        raise ValueError(f"t_years and y must be 1-D and the same length; got {t.shape} and {y.shape}")
    if np.isinf(t).any() or np.isinf(y).any():
        raise ValueError("t_years and y must not contain infinite values")
    keep = ~(np.isnan(t) | np.isnan(y))
    t, y = t[keep], y[keep]
    if len(t) < 3:
        raise ValueError(f"need at least 3 non-NaN points, got {len(t)}")
    if np.unique(t).size < 2:
        raise ValueError("need at least 2 distinct times after dropping NaN")
    return t, y


def _span(t: np.ndarray) -> dict:
    return {"n": int(len(t)), "t_start": float(t.min()), "t_end": float(t.max())}


def theil_sen(t_years, y, confidence: float = 0.95, method: str = "separate") -> SlopeResult:
    """Theil–Sen slope with its confidence interval.

    Calls ``scipy.stats.theilslopes(y, t_years, alpha=confidence,
    method=method)`` on the NaN-free pairs. In scipy, ``alpha`` is the
    confidence level (0.95 gives the 95% interval), not 1 − confidence.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    t, y = _clean(t_years, y)
    res = stats.theilslopes(y, t, alpha=confidence, method=method)
    slope, intercept = float(res.slope), float(res.intercept)
    low, high = float(res.low_slope), float(res.high_slope)
    return SlopeResult(
        slope=slope,
        intercept=intercept,
        low_slope=low,
        high_slope=high,
        slope_per_decade=slope * DECADE,
        low_slope_per_decade=low * DECADE,
        high_slope_per_decade=high * DECADE,
        confidence=float(confidence),
        method=method,
        **_span(t),
    )


def ols(t_years, y) -> OLSResult:
    """Ordinary least-squares slope, via ``scipy.stats.linregress(t_years, y)``
    on the NaN-free pairs. Used for the KNMI cross-check."""
    t, y = _clean(t_years, y)
    res = stats.linregress(t, y)
    slope, stderr = float(res.slope), float(res.stderr)
    return OLSResult(
        slope=slope,
        intercept=float(res.intercept),
        r=float(res.rvalue),
        p=float(res.pvalue),
        stderr=stderr,
        intercept_stderr=float(res.intercept_stderr),
        slope_per_decade=slope * DECADE,
        stderr_per_decade=stderr * DECADE,
        **_span(t),
    )
