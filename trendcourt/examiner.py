"""The Examiner: statistics.

Thin wrappers over ``scipy.stats`` (``theilslopes``, ``linregress``) and
``pymannkendall`` (``hamed_rao_modification_test``, ``seasonal_test``).
No data loading lives here, and no statistics are reimplemented: the
wrappers only validate input, drop missing values, convert units (per
year to per decade) and return stable, JSON-serializable result objects.

Usage (``project.md`` §3, steps 5 and 6, plus the OLS slope for the
KNMI check)::

    t = clerk.time_in_years(anomalies.index)
    ts = theil_sen(t, anomalies)        # slope, 95% interval, per decade
    lr = ols(t, anomalies)              # KNMI cross-check
    mk = mk_hamed_rao(anomalies)        # main test: deseasonalized anomalies
    smk = mk_seasonal(raw)              # check: raw monthly series
    consistency(mk, smk)                # sign / significance agreement
    json.dumps(ts.to_dict())
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import pymannkendall as pmk
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


# --- Mann–Kendall (project.md §3, step 6) ----------------------------------


@dataclass(frozen=True)
class MKResult:
    """A ``pymannkendall`` result as a plain, JSON-serializable record.

    ``trend``, ``h``, ``p``, ``z``, ``Tau``, ``s`` and ``var_s`` are the
    library's fields unchanged (``h`` is ``abs(z)`` above the two-sided
    critical value at ``alpha``). ``slope`` and ``intercept`` are the
    library's Sen estimate, kept for reference only: the reported slope
    comes from :func:`theil_sen`. Note that the library's slope is per
    time step (per month for Hamed–Rao, per year for seasonal), not per
    decimal year.

    ``test`` is ``"hamed_rao"`` or ``"seasonal"``. ``n`` is the number of
    non-NaN values the library used. ``lag`` is the Hamed–Rao ``lag``
    argument as passed (``None`` means the library default, all lags);
    it is ``None`` for the seasonal test. ``period`` is the seasonal
    period, ``None`` for Hamed–Rao.
    """

    test: str
    trend: str
    h: bool
    p: float
    z: float
    Tau: float
    s: float
    var_s: float
    slope: float
    intercept: float
    alpha: float
    n: int
    lag: int | None
    period: int | None

    def to_dict(self) -> dict:
        """Plain dict for strict JSON; non-finite floats become ``None``."""
        return _json_safe(asdict(self))


def _mk_values(y) -> np.ndarray:
    """1-D float values for pymannkendall. NaN (missing) passes through,
    since the library skips it; ±inf raises."""
    v = np.asarray(y, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {v.shape}")
    if np.isinf(v).any():
        raise ValueError("y must not contain infinite values")
    return v


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _mk_result(res, *, test: str, alpha: float, n: int, lag, period) -> MKResult:
    return MKResult(
        test=test,
        trend=str(res.trend),
        h=bool(res.h),
        p=float(res.p),
        z=float(res.z),
        Tau=float(res.Tau),
        s=float(res.s),
        var_s=float(res.var_s),
        slope=float(res.slope),
        intercept=float(res.intercept),
        alpha=float(alpha),
        n=n,
        lag=lag,
        period=period,
    )


def mk_hamed_rao(y, alpha: float = 0.05, lag: int | None = None) -> MKResult:
    """Mann–Kendall with the Hamed–Rao (1998) autocorrelation correction.

    Expects the **deseasonalized anomalies** (``clerk.deseasonalize``),
    in time order at a regular step. This is the main significance test.

    Calls ``pymannkendall.hamed_rao_modification_test(y, alpha=alpha,
    lag=lag)`` on the values unchanged. ``lag=None`` is the library
    default (all lags up to n − 1); Hamed and Rao suggest 3.

    NaN handling in pymannkendall 1.4.3: missing values are dropped and
    S, Var(S) and the rank autocorrelation are computed on the remaining
    values as if they were consecutive. Gaps do not raise. ±inf raises
    here, as does fewer than 3 non-NaN values.
    """
    _check_alpha(alpha)
    if lag is not None and (isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 1):
        raise ValueError(f"lag must be a positive integer or None, got {lag!r}")
    v = _mk_values(y)
    n = int(np.count_nonzero(~np.isnan(v)))
    if n < 3:
        raise ValueError(f"need at least 3 non-NaN values, got {n}")
    if lag is not None and lag >= n:
        raise ValueError(f"lag must be smaller than the number of non-NaN values ({n}), got {lag}")
    res = pmk.hamed_rao_modification_test(v, alpha=alpha, lag=lag)
    return _mk_result(res, test="hamed_rao", alpha=alpha, n=n,
                      lag=None if lag is None else int(lag), period=None)


def _monthly_periods(index) -> pd.PeriodIndex:
    if isinstance(index, pd.PeriodIndex):
        return index.asfreq("M")
    if isinstance(index, pd.DatetimeIndex):
        return index.to_period("M")
    raise TypeError("y_raw must be a pandas Series with a DatetimeIndex or PeriodIndex, "
                    f"got index of type {type(index).__name__}")


def mk_seasonal(y_raw: pd.Series, period: int = 12, alpha: float = 0.05) -> MKResult:
    """Seasonal Mann–Kendall (Hirsch and Slack 1984), the consistency check.

    Expects the **raw monthly series** (not anomalies), as a pandas
    Series indexed by month (``DatetimeIndex`` or ``PeriodIndex``).

    Calls ``pymannkendall.seasonal_test(values, period=12, alpha=alpha)``.
    The library groups values by position (it reshapes to years × 12),
    so the index is checked first: it must be contiguous monthly with no
    duplicates, and it must start in January, otherwise this raises
    rather than silently mixing months. A partial final year is padded
    with NaN to December here.

    NaN handling in pymannkendall 1.4.3: the reshape happens before
    missing values are dropped, and each calendar month is then tested
    with its own NaN skipped, so interior gaps keep the month alignment.
    """
    _check_alpha(alpha)
    if period != 12:
        raise ValueError(f"only monthly data is supported (period=12), got period={period}")
    if not isinstance(y_raw, pd.Series):
        raise TypeError(f"y_raw must be a pandas Series indexed by month, got {type(y_raw).__name__}")
    months = _monthly_periods(y_raw.index)
    if len(months) == 0:
        raise ValueError("y_raw is empty")
    if len(months) > 1 and not (np.diff(months.asi8) == 1).all():
        raise ValueError("y_raw index must be contiguous monthly in increasing order, with no gaps "
                         "or duplicates (reindex missing months to NaN)")
    if months[0].month != 1:
        raise ValueError(f"y_raw must start in January for the seasonal test; starts {months[0]}")
    v = _mk_values(y_raw.to_numpy())
    v = np.pad(v, (0, -len(v) % period), constant_values=np.nan)
    n = int(np.count_nonzero(~np.isnan(v)))
    if n < 3:
        raise ValueError(f"need at least 3 non-NaN values, got {n}")
    res = pmk.seasonal_test(v, period=period, alpha=alpha)
    return _mk_result(res, test="seasonal", alpha=alpha, n=n, lag=None, period=period)


def _sign(x: float) -> int:
    return int(np.sign(x))


def consistency(mk: MKResult, smk: MKResult) -> dict:
    """Do the Hamed–Rao and seasonal tests agree?

    Sign is the sign of each test's S (−1, 0 or +1); significance is the
    library's ``h`` at the shared ``alpha``. Returns a plain dict:
    ``alpha``, ``mk_sign``, ``smk_sign``, ``same_sign``,
    ``mk_significant``, ``smk_significant``, ``same_significance`` and
    ``agree`` (both of the above). Consumed by the verdict module.
    """
    if mk.test != "hamed_rao" or smk.test != "seasonal":
        raise ValueError(f"expected (hamed_rao, seasonal) results, got ({mk.test}, {smk.test})")
    if mk.alpha != smk.alpha:
        raise ValueError(f"tests were run at different alpha: {mk.alpha} and {smk.alpha}")
    mk_sign, smk_sign = _sign(mk.s), _sign(smk.s)
    same_sign = mk_sign == smk_sign
    same_significance = mk.h == smk.h
    return {
        "alpha": mk.alpha,
        "mk_sign": mk_sign,
        "smk_sign": smk_sign,
        "same_sign": same_sign,
        "mk_significant": mk.h,
        "smk_significant": smk.h,
        "same_significance": same_significance,
        "agree": same_sign and same_significance,
    }
