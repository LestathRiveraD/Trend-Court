"""The Examiner: statistics.

Thin wrappers over ``scipy.stats`` (``theilslopes``, ``linregress``) and
``pymannkendall`` (``hamed_rao_modification_test``,
``correlated_multivariate_test`` for the correlated seasonal test).
No data loading lives here, and no statistics are reimplemented: the
wrappers only validate input, drop missing values, convert units (per
year to per decade) and return stable, JSON-serializable result objects.

Usage (``project.md`` §3, steps 5 and 6, plus the OLS slope for the
KNMI check)::

    t = clerk.time_in_years(anomalies.index)
    ts = theil_sen(t, anomalies)        # slope, 95% interval, per decade
    lr = ols(t, anomalies)              # KNMI cross-check
    mk = mk_hamed_rao(anomalies)        # main test: deseasonalized anomalies
    smk = mk_seasonal(raw)              # check: raw monthly series (correlated seasonal MK)
    consistency(mk, smk)                # sign / significance agreement
    json.dumps(ts.to_dict())
"""

from __future__ import annotations

import math
import os
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

    ``test`` is ``"hamed_rao"`` or ``"correlated_seasonal"``. ``n`` is
    the number of values the library used (for the seasonal test, 12 ×
    the number of complete years). ``lag`` is the Hamed–Rao ``lag``
    argument as passed (``None`` means all lags, the library default);
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


def mk_hamed_rao(y, alpha: float = 0.05, lag: int | None = 3) -> MKResult:
    """Mann–Kendall with the Hamed–Rao (1998) autocorrelation correction.

    Expects the **deseasonalized anomalies** (``clerk.deseasonalize``),
    in time order at a regular step. This is the main significance test.

    Calls ``pymannkendall.hamed_rao_modification_test(y, alpha=alpha,
    lag=lag)`` on the values unchanged. The default ``lag=3`` (the first
    3 autocorrelation lags, as Hamed and Rao suggest) replaces the
    library's default of all lags (``lag=None``, still accepted): with
    every lag eligible, about 5% of them pass the significance filter
    by chance, and the test rejected 10–25% of trend-free AR(1) series
    at alpha 0.05, against 5–17% with ``lag=3`` (statistics review on
    PR #12). Even at ``lag=3`` the test is liberal for strongly
    persistent series; Phase 2 validation (a) calibrates the choice.

    NaN handling in pymannkendall 1.4.3: missing values are dropped and
    S, Var(S) and the rank autocorrelation are computed on the remaining
    values as if they were consecutive. Gaps do not raise. ±inf raises
    here, as does fewer than 3 non-NaN values.

    The Hamed–Rao correction can break down: the n/n* factor can make
    the corrected Var(S) negative (seen with ``lag=None`` on white noise
    of length 480), or undefined (NaN) for a constant series.
    The library then returns NaN z and p but still reports ``h=False``,
    ``'no trend'``. This raises ``ValueError`` instead, so a broken test
    is never read as "no trend"; a short ``lag`` (the default 3) usually
    avoids it.
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
    with np.errstate(invalid="ignore", divide="ignore"):
        res = pmk.hamed_rao_modification_test(v, alpha=alpha, lag=lag)
    var_s, z, p = float(res.var_s), float(res.z), float(res.p)
    if not (np.isfinite(var_s) and var_s > 0 and np.isfinite(z) and np.isfinite(p)):
        raise ValueError(
            f"Hamed-Rao corrected variance of S is non-positive or undefined (var_s={var_s}, "
            f"z={z}, p={p}, lag={lag}); the test result is not usable. Try a short lag "
            "(e.g. lag=3), and check the series is not constant")
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
    """Correlated seasonal Mann–Kendall (Hirsch and Slack 1984), the
    consistency check.

    Expects the **raw monthly series** (not anomalies), as a pandas
    Series indexed by month (``DatetimeIndex`` or ``PeriodIndex``). The
    test compares each calendar month only with itself, so subtracting a
    monthly climatology would not change the result; month alignment is
    what matters.

    The plain seasonal test (``pymannkendall.seasonal_test``) treats the
    12 monthly S scores as independent. With month-to-month
    autocorrelation that understates Var(S): it rejected 30–50% of
    trend-free AR(1) series with phi ≥ 0.6 (statistics review on PR #12).
    This uses the covariance-corrected version instead, which rejected
    4–10%.

    The values are reshaped to years × 12 (as
    ``pymannkendall.correlated_seasonal_test`` does) and passed to
    ``pymannkendall.correlated_multivariate_test(x, alpha=alpha)``.
    ``correlated_seasonal_test`` itself is not called because in 1.4.3
    it does not forward ``alpha`` (``h`` and ``trend`` are always at
    0.05); at alpha 0.05 the two are identical.

    Because the library groups values by position, the index is checked
    first: it must be contiguous monthly with no duplicates, and it must
    start in January, otherwise this raises rather than silently mixing
    months. A partial final year is padded with NaN.

    NaN handling in pymannkendall 1.4.3: the correlated test drops every
    year (row) that has any NaN month, so a partial final year and any
    year with a gap are left out entirely; ``n`` records the values
    actually used. At least 3 complete years are required, and a
    degenerate covariance (e.g. a constant series) raises ``ValueError``.
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
    x = v.reshape(-1, period)
    complete_years = int(np.count_nonzero(~np.isnan(x).any(axis=1)))
    if complete_years < 3:
        raise ValueError("the correlated seasonal test needs at least 3 complete years (all 12 "
                         f"months non-NaN); got {complete_years}")
    with np.errstate(invalid="ignore", divide="ignore"):
        res = pmk.correlated_multivariate_test(x, alpha=alpha)
    var_s, z, p = float(res.var_s), float(res.z), float(res.p)
    if not (np.isfinite(var_s) and var_s > 0 and np.isfinite(z) and np.isfinite(p)):
        raise ValueError(
            f"correlated seasonal variance of S is non-positive or undefined (var_s={var_s}, "
            f"z={z}, p={p}); the test result is not usable. Check the series is not constant")
    return _mk_result(res, test="correlated_seasonal", alpha=alpha, n=complete_years * period,
                      lag=None, period=period)


def _sign(x: float) -> int:
    return int(np.sign(x))


def consistency(mk: MKResult, smk: MKResult) -> dict:
    """Do the Hamed–Rao and correlated seasonal tests agree?

    Sign is the sign of each test's S (−1, 0 or +1); significance is the
    library's ``h`` at the shared ``alpha``. Returns a plain dict:
    ``alpha``, ``mk_sign``, ``smk_sign``, ``same_sign``,
    ``mk_significant``, ``smk_significant``, ``same_significance`` and
    ``agree`` (both of the above). Consumed by the verdict module.

    Note that ``agree`` also requires the same sign when neither test is
    significant, where the sign of S is noise. The verdict's "witnesses
    disagree" rule should key on significant results of opposite sign,
    not on ``agree`` alone.
    """
    if mk.test != "hamed_rao" or smk.test != "correlated_seasonal":
        raise ValueError("expected (hamed_rao, correlated_seasonal) results, "
                         f"got ({mk.test}, {smk.test})")
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


# --- Window sensitivity (project.md §3, step 7) -----------------------------


def _window_stats(chunk: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[float, float] | None]:
    """``(slope per decade, Hamed–Rao p)`` for each ``(t, y)`` window, with the
    full-record wrappers and defaults; ``None`` where a test cannot run or
    gives a non-finite slope or p, so a null cell is always an untested one."""
    out: list[tuple[float, float] | None] = []
    for t, y in chunk:
        try:
            b, pv = float(theil_sen(t, y).slope_per_decade), float(mk_hamed_rao(y).p)
        except ValueError:
            out.append(None)
            continue
        out.append((b, pv) if math.isfinite(b) and math.isfinite(pv) else None)
    return out


def _cpu_count() -> int:
    """CPUs this process may run on (its affinity mask where the OS has one)."""
    try:
        return len(os.sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _whole_years(index: pd.DatetimeIndex) -> list[int]:
    """Calendar years whose 12 months are all in ``index`` (values may be NaN)."""
    months = pd.Series(1, index=index)
    per_year = months.groupby([index.year, index.month]).size().groupby(level=0).size()
    return [int(y) for y, n in per_year.items() if n == 12]


def windows(t_years, anomalies: pd.Series, min_years: int = 15, alpha: float = 0.05,
            workers: int | None = None) -> dict:
    """Window sensitivity: the trend over every start / end year pair.

    For every pair of whole calendar years ``(start, end)`` with
    ``end - start + 1 >= min_years``, takes the months of those years and
    runs the same wrappers, with the same defaults, as the full record:
    :func:`theil_sen` (slope per decade) and :func:`mk_hamed_rao`
    (``lag=3``, its p). Nothing is reimplemented; ``workers`` processes
    only share out the windows (``1`` runs them in this process, ``None``
    uses every CPU the process may run on), and the result does not
    depend on it.

    ``anomalies`` is the deseasonalized monthly series (a pandas Series
    with a ``DatetimeIndex``, as from ``clerk.prepare``) and ``t_years`` its
    decimal-year times. A whole year is one whose 12 months are all in the
    index. Each window is gated over its full calendar span, ``12 * (end -
    start + 1)`` months, with the Clerk's threshold: at most
    ``clerk.MAX_MISSING`` of those months may be NaN or absent, and its
    first and last years must each have a valid month. (The span is at
    least ``min_years`` by construction.) ``clerk.quality_gate`` itself is
    not used here because it measures a record from its first to its last
    valid month: edge gaps would be trimmed rather than counted, so a
    window labelled ``start``-``end`` could be tested on a shorter span
    and repeat that span's cell. The edge-year rule stops a short edge
    gap, within the missing threshold, doing the same. Both tests must
    also run and give finite results; otherwise the window is ``None`` in
    the grid and left out of both shares.

    The reference is the full input record: its Theil–Sen slope sign and
    whether its Hamed–Rao p is below ``alpha`` (the verdict's
    ``alpha_moderate``). Returns a plain dict for strict JSON:

    * ``same_sign_share``: share of tested windows whose slope has the
      full record's sign (``None`` when no window could be tested);
    * ``same_significance_share``: share whose p is on the same side of
      ``alpha`` as the full record's;
    * ``n_windows``: every pair long enough; ``n_tested``: those in the
      shares;
    * ``start_years``, ``end_years`` and the ``slope`` (per decade) and
      ``p`` matrices, ``slope[i][j]`` for ``start_years[i]`` to
      ``end_years[j]``, ``None`` where the window is too short or untested;
    * ``min_years``, ``alpha``, ``full_slope_per_decade``, ``full_p``.
    """
    from trendcourt import clerk  # no cycle: the Clerk does not import the Examiner

    if isinstance(min_years, bool) or not isinstance(min_years, (int, np.integer)) or min_years < 1:
        raise ValueError(f"min_years must be a positive integer, got {min_years!r}")
    _check_alpha(alpha)
    if workers is not None and (isinstance(workers, bool)
                                or not isinstance(workers, (int, np.integer)) or workers < 1):
        raise ValueError(f"workers must be a positive integer or None, got {workers!r}")
    if not isinstance(anomalies, pd.Series) or not isinstance(anomalies.index, pd.DatetimeIndex):
        raise TypeError("anomalies must be a pandas Series with a DatetimeIndex")
    t = np.asarray(t_years, dtype=float)
    if t.shape != (len(anomalies),):
        raise ValueError(f"t_years must have one value per month; got {t.shape} for {len(anomalies)}")
    y = anomalies.to_numpy(dtype=float)
    full_slope = theil_sen(t, y).slope_per_decade
    full_p = mk_hamed_rao(y).p
    full_sign, full_significant = _sign(full_slope), full_p < alpha

    years = _whole_years(anomalies.index)
    start_years = [s for s in years if s + min_years - 1 <= years[-1]] if years else []
    end_years = [e for e in years if e - min_years + 1 >= years[0]] if years else []
    year_of = anomalies.index.year.to_numpy()
    valid = ~np.isnan(y)
    valid_years = set(year_of[valid].tolist())

    pairs, jobs = [], []
    for s in start_years:
        for e in end_years:
            if e - s + 1 < min_years:
                continue
            pairs.append((s, e))
            if s not in valid_years or e not in valid_years:
                continue
            keep = (year_of >= s) & (year_of <= e)
            n_span = 12 * (e - s + 1)
            if 1 - int(valid[keep].sum()) / n_span <= clerk.MAX_MISSING:
                jobs.append(((s, e), (t[keep], y[keep])))

    if workers is None:
        workers = _cpu_count()
    data = [job for _, job in jobs]
    if workers <= 1 or len(data) < 2:
        results = _window_stats(data)
    else:
        from concurrent.futures import ProcessPoolExecutor

        size = -(-len(data) // (workers * 4))
        chunks = [data[i:i + size] for i in range(0, len(data), size)]
        with ProcessPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
            results = [r for part in pool.map(_window_stats, chunks) for r in part]
    tested = {key: r for (key, _), r in zip(jobs, results) if r is not None}

    col = {e: j for j, e in enumerate(end_years)}
    row = {s: i for i, s in enumerate(start_years)}
    slope = [[None] * len(end_years) for _ in start_years]
    p = [[None] * len(end_years) for _ in start_years]
    same_sign = same_sig = 0
    for (s, e), (b, pv) in tested.items():
        slope[row[s]][col[e]] = b
        p[row[s]][col[e]] = pv
        same_sign += _sign(b) == full_sign
        same_sig += (pv < alpha) == full_significant
    n = len(tested)
    return {
        "min_years": int(min_years),
        "alpha": float(alpha),
        "full_slope_per_decade": float(full_slope),
        "full_p": float(full_p),
        "n_windows": len(pairs),
        "n_tested": n,
        "same_sign_share": same_sign / n if n else None,
        "same_significance_share": same_sig / n if n else None,
        "start_years": start_years,
        "end_years": end_years,
        "slope": slope,
        "p": p,
    }
