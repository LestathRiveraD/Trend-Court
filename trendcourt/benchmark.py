"""Synthetic benchmark: the Clerk and the Examiner on series with known truth.

``project.md`` §3, validation: (a) on trend-free AR(1) noise with a
seasonal cycle, the trend tests should reject about as often as their
nominal level; (b) with an injected linear trend, the Theil–Sen 95%
interval should contain the true slope about 95% of the time; (c) the
power of the Hamed–Rao test by record length. This module builds those series and runs them through
**the same code path as a real case** (:func:`trendcourt.case.examine`):
``clerk.deseasonalize``, then the Examiner wrappers with their defaults.
The trend statistics come only from those wrappers: nothing here calls
``scipy.stats.theilslopes`` or ``pymannkendall`` directly. (``scipy`` is
used only to generate the AR(1) noise and for the binomial interval
on each rejection rate.)

The experiment driver is ``scripts/benchmark.py``::

    python scripts/benchmark.py --experiment fpr       # docs/benchmark/fpr.json
    python scripts/benchmark.py --experiment coverage  # docs/benchmark/coverage.json
    python scripts/benchmark.py --experiment power     # docs/benchmark/power.json
    python scripts/benchmark.py --experiment applied   # docs/benchmark/applied.json
    python scripts/benchmark.py --experiment noise     # docs/benchmark/noise.json

``applied`` runs the verdict policy of ``trendcourt/thresholds.json`` as
it will be applied (band chosen by each record's own estimated ``r1``) on
trend-free AR(1) series; ``noise`` fits an AR(p) noise model to each real
case and runs the same policy on series simulated from it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import signal

from trendcourt import clerk, examiner

START = "1981-01"
"""First month of every synthetic series (a January, like the real record)."""

BURN_IN = 600
"""AR(1) steps drawn and discarded before the series starts. At
``phi = 0.9`` the start-up transient decays as 0.9**600, about 1e-28."""


def synthetic_series(
    n_years: int,
    phi: float,
    sigma: float,
    seasonal_amp: float,
    slope_per_decade: float = 0.0,
    *,
    seed,
) -> pd.Series:
    """A monthly series with a seasonal cycle, a linear trend and AR(1) noise.

    * ``n_years`` whole years starting in January 1981, on a month-start
      ``DatetimeIndex``, so :func:`clerk.deseasonalize` and
      :func:`examiner.mk_seasonal` accept it unchanged.
    * Seasonal cycle: ``seasonal_amp * sin(2*pi*(month - 1) / 12)``.
    * Trend: ``slope_per_decade / 10`` per year, on the Clerk's decimal-year
      axis (:func:`clerk.time_in_years`), zero at the first month.
    * Noise: ``x[t] = phi * x[t-1] + e[t]`` with Gaussian innovations of
      standard deviation ``sigma * sqrt(1 - phi**2)``, so ``sigma`` is the
      *stationary* standard deviation of the noise whatever ``phi`` is.
      :data:`BURN_IN` steps are drawn first and discarded, so the noise is
      stationary from the first month.

    Every random draw comes from ``numpy.random.default_rng(seed)``: the
    same ``seed`` gives the same series, bit for bit. ``seed`` is
    anything ``default_rng`` accepts (an int, a sequence of ints, a
    ``SeedSequence``).
    """
    if isinstance(n_years, bool) or not isinstance(n_years, (int, np.integer)) or n_years < 1:
        raise ValueError(f"n_years must be a positive integer, got {n_years!r}")
    if not (math.isfinite(phi) and -1.0 < phi < 1.0):
        raise ValueError(f"phi must be in (-1, 1) for stationary AR(1) noise, got {phi}")
    if not (math.isfinite(sigma) and sigma >= 0.0):
        raise ValueError(f"sigma must be finite and non-negative, got {sigma}")
    if not (math.isfinite(seasonal_amp) and math.isfinite(slope_per_decade)):
        raise ValueError("seasonal_amp and slope_per_decade must be finite")

    n = 12 * int(n_years)
    index = pd.date_range(START, periods=n, freq="MS", name="time")
    t = clerk.time_in_years(index)

    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, sigma * math.sqrt(1.0 - phi * phi), size=BURN_IN + n)
    noise = signal.lfilter([1.0], [1.0, -phi], innovations)[BURN_IN:]

    seasonal = seasonal_amp * np.sin(2.0 * np.pi * (index.month.to_numpy() - 1) / 12.0)
    trend = slope_per_decade / examiner.DECADE * (t - t[0])
    return pd.Series(seasonal + trend + noise, index=index, name="value")


def detrended_r1(t, anomalies, ts) -> float:
    """:func:`lag1_autocorr` of ``anomalies`` minus the Theil–Sen line ``ts``.

    The ``r1`` the verdict rule reads (``thresholds.band_for``,
    ``thresholds.widening_factor``): a real trend raises the raw ``r1``,
    so the line is removed first.
    """
    return lag1_autocorr(np.asarray(anomalies, dtype=float) - (ts.intercept + ts.slope * np.asarray(t)))


def run_case(series: pd.Series, *, seasonal: bool = True) -> dict:
    """The statistics :func:`trendcourt.case.examine` reports, on one series.

    Same calls and defaults as ``examine``: the anomalies from
    ``clerk.deseasonalize(series)`` and the time axis
    ``clerk.time_in_years(series.index)``, then

    * ``examiner.theil_sen(t, anomalies)``: slope and 95% interval;
    * ``examiner.mk_hamed_rao(anomalies)``: alpha 0.05, ``lag=3``;
    * ``examiner.mk_seasonal(series)``: correlated seasonal MK on the
      raw series, alpha 0.05.

    Returns a flat dict: Theil–Sen ``ts_slope``, ``ts_low``, ``ts_high``
    (per year) and their ``*_per_decade`` versions; ``hr_p``, ``hr_h``,
    ``hr_z`` (the sign gives the trend direction);
    ``smk_p``, ``smk_h``; and ``r1_detrended`` (:func:`detrended_r1`),
    the autocorrelation the verdict's band and interval widening use.
    With ``seasonal=False`` the seasonal test is skipped (``smk_p`` and
    ``smk_h`` are ``None``); the verdict does not read it, and it is the
    slowest of the three. A broken Hamed–Rao variance raises
    ``ValueError``, as it does in ``examine``.
    """
    anomalies = clerk.deseasonalize(series)
    t = clerk.time_in_years(series.index)
    ts = examiner.theil_sen(t, anomalies)
    mk = examiner.mk_hamed_rao(anomalies)
    smk = examiner.mk_seasonal(series) if seasonal else None
    return {
        "ts_slope": ts.slope,
        "ts_low": ts.low_slope,
        "ts_high": ts.high_slope,
        "ts_slope_per_decade": ts.slope_per_decade,
        "ts_low_per_decade": ts.low_slope_per_decade,
        "ts_high_per_decade": ts.high_slope_per_decade,
        "hr_p": mk.p,
        "hr_h": mk.h,
        "hr_z": mk.z,
        "smk_p": None if smk is None else smk.p,
        "smk_h": None if smk is None else smk.h,
        "r1_detrended": detrended_r1(t, anomalies, ts),
    }


def lag1_autocorr(x) -> float:
    """Sample lag-1 autocorrelation, ``pandas.Series.autocorr(lag=1)``.

    That is the Pearson correlation of ``x[t]`` with ``x[t+1]`` over the
    pairs where both are present, so gaps drop pairs rather than joining
    values across them.
    """
    return float(pd.Series(np.asarray(x, dtype=float)).autocorr(lag=1))


# --------------------------------------------------------------------------
# Experiment (a): false-positive rate on trend-free AR(1) noise
# --------------------------------------------------------------------------

FPR_PHIS = (0.0, 0.3, 0.6, 0.8, 0.9)
FPR_YEARS = (20, 30, 45)
FPR_REPLICATES = 1000
FPR_SIGMA = 1.0
FPR_SEASONAL_AMP = 10.0
FPR_SEED = 17
ALPHAS = (0.05, 0.01)
TESTS = ("hamed_rao", "seasonal")
CI_LEVEL = 0.95


def replicate_seed(master: int, phi: float, n_years: int, rep: int) -> tuple[int, int, int, int]:
    """Seed of one replicate: ``(master, round(1000 * phi), n_years, rep)``.

    Depends only on the cell's parameters and the replicate number, not
    on the grid's order, the worker count or the chunking, so any one
    replicate can be regenerated on its own::

        synthetic_series(n_years, phi, FPR_SIGMA, FPR_SEASONAL_AMP,
                         seed=replicate_seed(FPR_SEED, phi, n_years, rep))
    """
    return (int(master), int(round(1000 * phi)), int(n_years), int(rep))


def _fpr_chunk(task: tuple) -> list[tuple[float, float] | str]:
    """Run replicates ``start .. stop-1`` of one cell. Top level so it pickles."""
    master, phi, n_years, start, stop = task
    out: list[tuple[float, float] | str] = []
    for rep in range(start, stop):
        y = synthetic_series(n_years, phi, FPR_SIGMA, FPR_SEASONAL_AMP,
                             seed=replicate_seed(master, phi, n_years, rep))
        try:
            r = run_case(y)
        except ValueError as exc:  # broken Hamed–Rao / seasonal variance: count, never hide
            out.append(f"replicate {rep}: {exc}")
            continue
        out.append((r["hr_p"], r["smk_p"]))
    return out


def rejection_summary(p_values, alpha: float, level: float = CI_LEVEL) -> dict:
    """Rejection rate ``p < alpha`` with a Clopper–Pearson (exact binomial) interval.

    ``inflated`` is true when the whole interval lies above ``alpha``:
    the test rejects trend-free series more often than its nominal level,
    beyond Monte Carlo noise.
    """
    from scipy.stats import binomtest

    p = np.asarray(p_values, dtype=float)
    n = int(p.size)
    k = int(np.count_nonzero(p < alpha))
    if n == 0:
        return {"alpha": alpha, "n": 0, "rejections": 0, "rate": None, "ci95": None, "inflated": None}
    ci = binomtest(k, n).proportion_ci(confidence_level=level, method="exact")
    low, high = float(ci.low), float(ci.high)
    return {"alpha": alpha, "n": n, "rejections": k, "rate": k / n,
            "ci95": [low, high], "inflated": low > alpha}


def fpr_experiment(
    replicates: int = FPR_REPLICATES,
    *,
    phis=FPR_PHIS,
    years=FPR_YEARS,
    seed: int = FPR_SEED,
    workers: int = 1,
    chunk: int = 50,
) -> dict:
    """Rejection rates of both trend tests on trend-free AR(1) series.

    For each ``(phi, n_years)`` cell, ``replicates`` series from
    :func:`synthetic_series` with ``slope_per_decade = 0`` go through
    :func:`run_case`; the result reports, for Hamed–Rao and for the
    correlated seasonal MK, the rejection rate at each alpha in
    :data:`ALPHAS` with its exact 95% binomial interval
    (:func:`rejection_summary`). A replicate whose test raises is
    counted in ``n_failed`` (with the messages in ``failures``) and left
    out of the rates.

    With ``workers > 1`` the replicates run in a process pool. The
    numbers do not depend on ``workers`` or ``chunk``: each replicate
    has its own seed (:func:`replicate_seed`) and results are collected
    in order.
    """
    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, got {replicates}")
    cells = [(float(phi), int(n)) for phi in phis for n in years]
    tasks = [(seed, phi, n, a, min(a + chunk, replicates))
             for phi, n in cells for a in range(0, replicates, chunk)]

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(_fpr_chunk, tasks))
    else:
        chunks = [_fpr_chunk(t) for t in tasks]

    by_cell: dict[tuple[float, int], list] = {c: [] for c in cells}
    for task, res in zip(tasks, chunks):
        by_cell[(task[1], task[2])].extend(res)

    out_cells = []
    for (phi, n), res in by_cell.items():
        ok = [r for r in res if not isinstance(r, str)]
        failures = [r for r in res if isinstance(r, str)]
        p = {"hamed_rao": [r[0] for r in ok], "seasonal": [r[1] for r in ok]}
        rates = {test: {f"{a:g}": rejection_summary(p[test], a) for a in ALPHAS} for test in TESTS}
        out_cells.append({
            "phi": phi,
            "n_years": n,
            "replicates": replicates,
            "n_failed": len(failures),
            "failures": failures,
            "rates": rates,
            "inflated": sorted(f"{test}@{a:g}" for test in TESTS for a in ALPHAS
                               if rates[test][f"{a:g}"]["inflated"]),
        })

    return {
        "experiment": "fpr",
        "description": ("Rejection rate of the trend tests on trend-free monthly series "
                        "(seasonal cycle + stationary AR(1) noise), through trendcourt.benchmark.run_case: "
                        "clerk.deseasonalize, examiner.mk_hamed_rao(anomalies, lag=3), "
                        "examiner.mk_seasonal(raw). A test rejects when p < alpha. "
                        "'inflated' means the whole exact 95% binomial interval lies above alpha."),
        "grid": {"phi": [float(p) for p in phis], "n_years": [int(n) for n in years]},
        "replicates": replicates,
        "series": {"start": START, "sigma": FPR_SIGMA, "seasonal_amp": FPR_SEASONAL_AMP,
                   "slope_per_decade": 0.0, "burn_in": BURN_IN,
                   "sigma_meaning": "stationary standard deviation of the AR(1) noise"},
        "seeds": {"master": seed,
                  "scheme": "numpy.random.default_rng((master, round(1000 * phi), n_years, replicate))",
                  "replicate_range": [0, replicates - 1]},
        "alphas": list(ALPHAS),
        "ci": {"level": CI_LEVEL, "method": "Clopper-Pearson (scipy.stats.binomtest, method='exact')"},
        "cells": out_cells,
        "inflated_cells": [{"phi": c["phi"], "n_years": c["n_years"], "tests": c["inflated"]}
                           for c in out_cells if c["inflated"]],
    }


def _grid_bracket(r: float, phis) -> list[float]:
    """The grid ``phi`` values either side of ``r`` (one value if outside the grid or on a point)."""
    grid = sorted(float(p) for p in phis)
    if r <= grid[0]:
        return [grid[0]]
    if r >= grid[-1]:
        return [grid[-1]]
    for lo, hi in zip(grid, grid[1:]):
        if r == lo:
            return [lo]
        if lo < r < hi:
            return [lo, hi]
    return [grid[-1]]


def real_case_autocorr(phis=FPR_PHIS) -> list[dict]:
    """Lag-1 autocorrelation of the deseasonalized anomalies for every real case.

    One entry per region in ``data/regions.json`` and per variable, from
    :func:`clerk.prepare`. ``r1`` is :func:`lag1_autocorr` of the
    anomalies, the series the Hamed–Rao test sees. ``r1_detrended`` is
    the same after subtracting the case's Theil–Sen line, since a real
    trend itself raises ``r1``; it is ``None`` when the record fails the
    quality gate, as no slope is reported then. ``phi_bracket`` gives
    the benchmark grid values either side of ``r1``.
    """
    regions = clerk.load_regions()
    rows = []
    for region in regions["regions"]:
        for variable in regions["variables"]:
            prepared = clerk.prepare(region, variable)
            anomalies = prepared.anomalies
            r1 = lag1_autocorr(anomalies)
            r1_detrended = None
            if prepared.quality.passed:
                ts = examiner.theil_sen(prepared.time, anomalies)
                r1_detrended = detrended_r1(prepared.time, anomalies, ts)
            rows.append({
                "region": region,
                "variable": variable,
                "quality_passed": prepared.quality.passed,
                "n_months": int(anomalies.notna().sum()),
                "r1": r1,
                "r1_detrended": r1_detrended,
                "phi_bracket": _grid_bracket(r1, phis),
            })
    return rows


# --------------------------------------------------------------------------
# Experiments (b) and (c): interval coverage and power with injected trends
# --------------------------------------------------------------------------

TREND_PHIS = FPR_PHIS
"""AR(1) coefficients of the coverage and power grids (same as experiment (a))."""

TREND_SIGMA = FPR_SIGMA
"""Stationary noise SD. It is 1, so a slope in units per decade is also the
slope in noise SDs per decade, and the results carry over to every variable."""

TREND_SEASONAL_AMP = FPR_SEASONAL_AMP

COVERAGE_SLOPES = (0.1, 0.3, 0.5)
"""Injected slopes, in noise SDs per decade (issue #18)."""
COVERAGE_YEARS = (20, 30, 45)
COVERAGE_REPLICATES = 1000
COVERAGE_SEED = 18
COVERAGE_LEVEL = 0.95
"""Nominal level of the Theil–Sen interval (``examiner.theil_sen`` default)."""
COVERAGE_FLAG = 0.93
"""A cell is flagged when its estimated coverage falls below this."""

POWER_SLOPES = (0.1, 0.2, 0.3, 0.5)
"""Injected slopes, in noise SDs per decade."""
POWER_YEARS = (20, 25, 30, 35, 40, 45)
POWER_REPLICATES = 1000
POWER_SEED = 19


def trend_replicate_seed(master: int, phi: float, slope: float, n_years: int, rep: int) -> tuple:
    """Seed of one replicate with an injected trend:
    ``(master, round(1000 * phi), round(1000 * slope), n_years, rep)``.

    Like :func:`replicate_seed`, it depends only on the cell and the
    replicate number, so any replicate can be regenerated on its own::

        synthetic_series(n_years, phi, TREND_SIGMA, TREND_SEASONAL_AMP, slope,
                         seed=trend_replicate_seed(master, phi, slope, n_years, rep))
    """
    return (int(master), int(round(1000 * phi)), int(round(1000 * slope)), int(n_years), int(rep))


def _trend_chunk(task: tuple) -> list[tuple[float, float, float, float, float] | str]:
    """Replicates ``start .. stop-1`` of one ``(phi, slope, n_years)`` cell.

    Returns per replicate ``(ts_slope, ts_low, ts_high, hr_p, hr_z,
    r1_detrended)`` from :func:`run_case` (slopes per year), or the error
    message if a test raised. Top level so it pickles.
    """
    master, phi, slope, n_years, start, stop = task
    out: list = []
    for rep in range(start, stop):
        y = synthetic_series(n_years, phi, TREND_SIGMA, TREND_SEASONAL_AMP, slope,
                             seed=trend_replicate_seed(master, phi, slope, n_years, rep))
        try:
            r = run_case(y)
        except ValueError as exc:  # count, never hide
            out.append(f"replicate {rep}: {exc}")
            continue
        out.append((r["ts_slope"], r["ts_low"], r["ts_high"], r["hr_p"], r["hr_z"], r["r1_detrended"]))
    return out


def _run_trend_grid(cells, replicates: int, seed: int, workers: int, chunk: int) -> dict:
    """Run every cell through :func:`_trend_chunk`; results in replicate order per cell."""
    tasks = [(seed, phi, slope, n, a, min(a + chunk, replicates))
             for phi, slope, n in cells for a in range(0, replicates, chunk)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(_trend_chunk, tasks))
    else:
        chunks = [_trend_chunk(t) for t in tasks]
    by_cell: dict[tuple, list] = {c: [] for c in cells}
    for task, res in zip(tasks, chunks):
        by_cell[task[1:4]].extend(res)
    return by_cell


def proportion_summary(k: int, n: int, level: float = CI_LEVEL) -> dict:
    """``k`` successes out of ``n`` with a Clopper–Pearson (exact binomial) interval."""
    from scipy.stats import binomtest

    if n == 0:
        return {"n": 0, "k": 0, "rate": None, "ci95": None}
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=level, method="exact")
    return {"n": int(n), "k": int(k), "rate": k / n, "ci95": [float(ci.low), float(ci.high)]}


def _check_grid(replicates: int, slopes) -> None:
    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, got {replicates}")
    if any(not math.isfinite(s) or s <= 0 for s in slopes):
        raise ValueError(f"injected slopes must be finite and positive, got {list(slopes)}")


def _seeds(seed: int, replicates: int) -> dict:
    return {"master": seed,
            "scheme": ("numpy.random.default_rng((master, round(1000 * phi), "
                       "round(1000 * slope), n_years, replicate))"),
            "replicate_range": [0, replicates - 1]}


def _series_meta() -> dict:
    return {"start": START, "sigma": TREND_SIGMA, "seasonal_amp": TREND_SEASONAL_AMP,
            "burn_in": BURN_IN, "sigma_meaning": "stationary standard deviation of the AR(1) noise",
            "slope_units": "noise SDs per decade (sigma = 1, so also units of the series per decade)"}


def coverage_experiment(
    replicates: int = COVERAGE_REPLICATES,
    *,
    phis=TREND_PHIS,
    slopes=COVERAGE_SLOPES,
    years=COVERAGE_YEARS,
    seed: int = COVERAGE_SEED,
    flag_below: float = COVERAGE_FLAG,
    workers: int = 1,
    chunk: int = 50,
) -> dict:
    """Coverage of the Theil–Sen 95% interval on series with a known linear trend.

    For each ``(phi, slope, n_years)`` cell, ``replicates`` series from
    :func:`synthetic_series` with ``slope_per_decade = slope`` go through
    :func:`run_case`, which calls ``examiner.theil_sen`` with its defaults
    (95%, ``method="separate"``). A replicate covers when
    ``ts_low <= slope / 10 <= ts_high`` (per year, the injected value).
    Coverage is the share of covering replicates, with an exact 95%
    binomial interval (:func:`proportion_summary`). A cell is flagged
    (``below_flag``) when the coverage estimate is below ``flag_below``;
    ``onset`` gives, for each ``(slope, n_years)``, the smallest ``phi``
    of the grid at which coverage is below ``flag_below``.

    Also reported per cell: the share of misses below and above the
    interval, the mean and median Theil–Sen slope (bias) and the median
    interval width, all per decade. Failed replicates are counted in
    ``n_failed`` and left out. The numbers do not depend on ``workers``
    or ``chunk``.
    """
    _check_grid(replicates, slopes)
    cells = [(float(phi), float(s), int(n)) for phi in phis for s in slopes for n in years]
    by_cell = _run_trend_grid(cells, replicates, seed, workers, chunk)

    out_cells = []
    for (phi, slope, n), res in by_cell.items():
        ok = np.array([r[:3] + r[5:6] for r in res if not isinstance(r, str)], dtype=float).reshape(-1, 4)
        failures = [r for r in res if isinstance(r, str)]
        true = slope / examiner.DECADE
        est, low, high, r1d = ok[:, 0], ok[:, 1], ok[:, 2], ok[:, 3]
        covered = (low <= true) & (true <= high)
        cov = proportion_summary(int(covered.sum()), len(ok))
        widened = _widened_cover(est, low, high, r1d, true)
        d = examiner.DECADE
        out_cells.append({
            "phi": phi,
            "slope_per_decade": slope,
            "n_years": n,
            "replicates": replicates,
            "n_failed": len(failures),
            "failures": failures,
            "coverage": cov,
            "widened_coverage": proportion_summary(int(widened["covered"].sum()), len(ok)),
            "widening_factor_mean": widened["factor_mean"],
            "r1_detrended_mean": float(r1d.mean()) if len(ok) else None,
            "miss_low": int((high < true).sum()),
            "miss_high": int((low > true).sum()),
            "slope_mean_per_decade": float(est.mean() * d) if len(ok) else None,
            "slope_median_per_decade": float(np.median(est) * d) if len(ok) else None,
            "width_median_per_decade": float(np.median(high - low) * d) if len(ok) else None,
            "below_flag": cov["rate"] is not None and cov["rate"] < flag_below,
            "below_nominal": cov["ci95"] is not None and cov["ci95"][1] < COVERAGE_LEVEL,
        })

    onset = []
    for s in slopes:
        for n in years:
            row = sorted((c for c in out_cells if c["slope_per_decade"] == float(s) and c["n_years"] == int(n)),
                         key=lambda c: c["phi"])
            first = next((c["phi"] for c in row if c["below_flag"]), None)
            onset.append({"slope_per_decade": float(s), "n_years": int(n), "phi": first})
    flagged_phis = [o["phi"] for o in onset if o["phi"] is not None]

    return {
        "experiment": "coverage",
        "description": ("Coverage of the Theil-Sen 95% interval (examiner.theil_sen defaults: "
                        "confidence 0.95, method='separate') on monthly series with a known linear "
                        "trend, seasonal cycle and stationary AR(1) noise, through "
                        "trendcourt.benchmark.run_case (Theil-Sen on clerk.deseasonalize anomalies). "
                        "A replicate covers when low <= true slope <= high. 'below_flag' means the "
                        f"coverage estimate is below {flag_below}; 'below_nominal' means the whole "
                        f"exact 95% binomial interval lies below {COVERAGE_LEVEL}. "
                        "miss_low counts intervals entirely below the true slope, miss_high entirely above. "
                        "'widened_coverage' is the coverage of the same interval widened as the verdict's "
                        "no-trend rule widens it (trendcourt.thresholds.widened_interval), with r1 the "
                        "detrended lag-1 autocorrelation estimated from each replicate, not the true phi; "
                        "'widening_factor_mean' is the mean factor applied."),
        "grid": {"phi": [float(p) for p in phis], "slope_per_decade": [float(s) for s in slopes],
                 "n_years": [int(n) for n in years]},
        "replicates": replicates,
        "nominal": COVERAGE_LEVEL,
        "flag_below": flag_below,
        "series": _series_meta(),
        "seeds": _seeds(seed, replicates),
        "ci": {"level": CI_LEVEL, "method": "Clopper-Pearson (scipy.stats.binomtest, method='exact')"},
        "cells": out_cells,
        "flagged_cells": [{"phi": c["phi"], "slope_per_decade": c["slope_per_decade"],
                           "n_years": c["n_years"], "coverage": c["coverage"]["rate"]}
                          for c in out_cells if c["below_flag"]],
        "onset": onset,
        "onset_phi": min(flagged_phis) if flagged_phis else None,
    }


def power_experiment(
    replicates: int = POWER_REPLICATES,
    *,
    phis=TREND_PHIS,
    slopes=POWER_SLOPES,
    years=POWER_YEARS,
    seed: int = POWER_SEED,
    workers: int = 1,
    chunk: int = 50,
) -> dict:
    """Power of the Hamed–Rao test on series with a known positive linear trend.

    For each ``(phi, slope, n_years)`` cell, ``replicates`` series go
    through :func:`run_case` (``examiner.mk_hamed_rao`` on the anomalies,
    ``lag=3``). For each alpha in :data:`ALPHAS`, ``rejection`` is the
    share with ``p < alpha`` (what the verdict calls significant) and
    ``detection`` the share with ``p < alpha`` and ``z > 0`` (significant
    in the injected direction), each with an exact 95% binomial interval.
    Failed replicates are counted in ``n_failed`` and left out. The
    numbers do not depend on ``workers`` or ``chunk``.

    Rates at short records and high ``phi`` include the test's own size
    inflation (see ``docs/benchmark/fpr.json``), so they overstate power
    there.
    """
    _check_grid(replicates, slopes)
    cells = [(float(phi), float(s), int(n)) for phi in phis for s in slopes for n in years]
    by_cell = _run_trend_grid(cells, replicates, seed, workers, chunk)

    out_cells = []
    for (phi, slope, n), res in by_cell.items():
        ok = np.array([r[3:5] for r in res if not isinstance(r, str)], dtype=float).reshape(-1, 2)
        failures = [r for r in res if isinstance(r, str)]
        p, z = ok[:, 0], ok[:, 1]
        rates = {}
        for a in ALPHAS:
            sig = p < a
            rates[f"{a:g}"] = {"alpha": a,
                               "rejection": proportion_summary(int(sig.sum()), len(ok)),
                               "detection": proportion_summary(int((sig & (z > 0)).sum()), len(ok))}
        out_cells.append({
            "phi": phi,
            "slope_per_decade": slope,
            "n_years": n,
            "replicates": replicates,
            "n_failed": len(failures),
            "failures": failures,
            "rates": rates,
        })

    return {
        "experiment": "power",
        "description": ("Power of examiner.mk_hamed_rao (lag=3) on monthly series with a known positive "
                        "linear trend, seasonal cycle and stationary AR(1) noise, through "
                        "trendcourt.benchmark.run_case. 'rejection' is the share with p < alpha; "
                        "'detection' the share with p < alpha and z > 0 (the injected direction). "
                        "Where the test is liberal on trend-free noise (docs/benchmark/fpr.json), "
                        "these rates overstate power."),
        "grid": {"phi": [float(p) for p in phis], "slope_per_decade": [float(s) for s in slopes],
                 "n_years": [int(n) for n in years]},
        "replicates": replicates,
        "alphas": list(ALPHAS),
        "series": _series_meta(),
        "seeds": _seeds(seed, replicates),
        "ci": {"level": CI_LEVEL, "method": "Clopper-Pearson (scipy.stats.binomtest, method='exact')"},
        "cells": out_cells,
    }


def _widened_cover(est, low, high, r1d, true) -> dict:
    """Which intervals, widened by the verdict's rule with each replicate's own
    ``r1``, contain ``true``; and the mean widening factor."""
    from trendcourt import thresholds

    if len(est) == 0:
        return {"covered": np.zeros(0, dtype=bool), "factor_mean": None}
    factors = np.array([thresholds.widening_factor(r) for r in r1d])
    wl, wh = est - factors * (est - low), est + factors * (high - est)
    return {"covered": (wl <= true) & (true <= wh), "factor_mean": float(factors.mean())}


def _policy_summary(rows: np.ndarray, th: dict, true_phi: float | None = None) -> dict:
    """The verdict policy on trend-free replicates.

    ``rows`` has one row per replicate: ``(ts_slope, ts_low, ts_high, hr_p,
    r1_detrended)`` (slopes per year; the true slope is 0). Each replicate
    is put in a band by its **own** ``r1_detrended``
    (:func:`trendcourt.thresholds.significance_claim`), as the verdict will
    be. Reports, each with an exact 95% binomial interval:

    * ``false_claim``: share of replicates that get any significance claim
      (Moderate or Strong) from their band;
    * ``false_strong``: share that pass the Strong test of a band that allows
      Strong (``p < alpha_strong``);
    * ``fpr``: plain ``p < alpha`` for each alpha in :data:`ALPHAS`;
    * ``raw_coverage`` and ``widened_coverage``: share of raw Theil–Sen
      intervals, and of intervals widened with the replicate's own ``r1``
      (:func:`trendcourt.thresholds.widened_interval`), that contain 0;
      with ``true_phi``, also ``widened_coverage_true_phi`` (widened with
      the factor at the true ``phi``);
    * ``band_shares``: share of replicates in each band (keyed by the
      band's ``r1_max``, ``"outside"`` above the grid);
    * the mean and SD of ``r1_detrended`` and the mean widening factor.
    """
    from trendcourt import thresholds

    n = len(rows)
    est, low, high, p, r1d = (rows[:, i] for i in range(5))
    claims = [thresholds.significance_claim(r, pv, th) for r, pv in zip(r1d, p)]
    bands = [thresholds.band_for(r, th) for r in r1d]
    keys = [f"{b['r1_max']:g}" for b in th["autocorrelation_bands"]] + ["outside"]
    shares = {k: 0 for k in keys}
    for b in bands:
        shares["outside" if b is None else f"{b['r1_max']:g}"] += 1
    widened = _widened_cover(est, low, high, r1d, 0.0)
    out = {
        "false_claim": proportion_summary(sum(c is not None for c in claims), n),
        "false_strong": proportion_summary(sum(c == "strong" for c in claims), n),
        "fpr": {f"{a:g}": proportion_summary(int((p < a).sum()), n) for a in ALPHAS},
        "raw_coverage": proportion_summary(int(((low <= 0) & (0 <= high)).sum()), n),
        "widened_coverage": proportion_summary(int(widened["covered"].sum()), n),
        "widening_factor_mean": widened["factor_mean"],
        "band_shares": {k: v / n for k, v in shares.items()} if n else None,
        "r1_detrended_mean": float(r1d.mean()) if n else None,
        "r1_detrended_sd": float(r1d.std(ddof=1)) if n > 1 else None,
    }
    if true_phi is not None:
        f = thresholds.widening_factor(true_phi)
        cov_true = (est - f * (est - low) <= 0) & (0 <= est + f * (high - est))
        out["widened_coverage_true_phi"] = proportion_summary(int(cov_true.sum()), n)
        out["widening_factor_true_phi"] = f
        out["r1_detrended_bias"] = out["r1_detrended_mean"] - true_phi if n else None
    return out


def _thresholds_used(th: dict) -> dict:
    """The part of ``thresholds.json`` a policy experiment depends on, stored
    with the result so a stale result can be detected."""
    return {"alpha_strong": th["alpha_strong"], "alpha_moderate": th["alpha_moderate"],
            "interval_widening": th["interval_widening"]["method"],
            "bands": [{"r1_min_exclusive": b["r1_min_exclusive"], "r1_max": b["r1_max"],
                       "max_verdict": b["max_verdict"], "moderate_alpha": b["moderate_alpha"]}
                      for b in th["autocorrelation_bands"]]}


def _run_pool(fn, tasks, workers: int) -> list:
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, tasks))
    return [fn(t) for t in tasks]


# --------------------------------------------------------------------------
# Experiment (d): the verdict policy as applied, on trend-free AR(1) noise
# --------------------------------------------------------------------------

APPLIED_PHIS = (0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
"""True AR(1) coefficients, on and off the band grid and across its edges."""
APPLIED_YEARS = FPR_YEARS
APPLIED_REPLICATES = 1000
APPLIED_SEED = 20


def _applied_chunk(task: tuple) -> list:
    """Replicates ``start .. stop-1`` of one ``(phi, n_years)`` cell, trend-free.

    Seeds as in experiment (a) (:func:`replicate_seed`), with master
    :data:`APPLIED_SEED`. Returns ``(ts_slope, ts_low, ts_high, hr_p,
    r1_detrended)`` per replicate, or the error message.
    """
    master, phi, n_years, start, stop = task
    out: list = []
    for rep in range(start, stop):
        y = synthetic_series(n_years, phi, FPR_SIGMA, FPR_SEASONAL_AMP,
                             seed=replicate_seed(master, phi, n_years, rep))
        try:
            r = run_case(y, seasonal=False)
        except ValueError as exc:  # count, never hide
            out.append(f"replicate {rep}: {exc}")
            continue
        out.append((r["ts_slope"], r["ts_low"], r["ts_high"], r["hr_p"], r["r1_detrended"]))
    return out


def applied_experiment(
    replicates: int = APPLIED_REPLICATES,
    *,
    phis=APPLIED_PHIS,
    years=APPLIED_YEARS,
    seed: int = APPLIED_SEED,
    thresholds_dict: dict | None = None,
    workers: int = 1,
    chunk: int = 50,
) -> dict:
    """The verdict policy as it will be applied, on trend-free AR(1) series.

    The bands of ``thresholds.json`` were set from the false-positive rate
    at the grid ``phi`` (experiment (a)). A real record is put in a band by
    its **estimated** ``r1_detrended``, which is noisy and biased low, and
    its true ``phi`` can be anywhere in or between bands. Here, for each
    true ``phi`` (including values off the grid and above it) and record
    length, ``replicates`` trend-free series go through :func:`run_case`
    (without the seasonal test) and :func:`_policy_summary`: the rate of
    false significance claims and false Strong claims under the policy,
    the coverage of the widened interval with the estimated ``r1``, and
    the distribution of ``r1_detrended``.

    The result stores the thresholds it used (``thresholds_used``);
    :func:`trendcourt.benchmark_report.check_policy_results_current`
    refuses to build tables from a result whose thresholds differ from the
    committed ones. The numbers do not depend on ``workers`` or ``chunk``.
    """
    from trendcourt import thresholds

    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, got {replicates}")
    th = thresholds.load() if thresholds_dict is None else thresholds_dict
    cells = [(float(phi), int(n)) for phi in phis for n in years]
    tasks = [(seed, phi, n, a, min(a + chunk, replicates))
             for phi, n in cells for a in range(0, replicates, chunk)]
    chunks = _run_pool(_applied_chunk, tasks, workers)
    by_cell: dict[tuple, list] = {c: [] for c in cells}
    for task, res in zip(tasks, chunks):
        by_cell[(task[1], task[2])].extend(res)

    out_cells = []
    for (phi, n), res in by_cell.items():
        ok = np.array([r for r in res if not isinstance(r, str)], dtype=float).reshape(-1, 5)
        failures = [r for r in res if isinstance(r, str)]
        out_cells.append({"phi": phi, "n_years": n, "replicates": replicates,
                          "n_failed": len(failures), "failures": failures,
                          **_policy_summary(ok, th, true_phi=phi)})

    return {
        "experiment": "applied",
        "description": ("The verdict policy of trendcourt/thresholds.json as it will be applied, on "
                        "trend-free monthly series (seasonal cycle + stationary AR(1) noise): each "
                        "replicate goes through trendcourt.benchmark.run_case (clerk.deseasonalize, "
                        "examiner.theil_sen, examiner.mk_hamed_rao lag=3; no seasonal test) and is put in "
                        "a band by its own r1_detrended (thresholds.significance_claim). 'false_claim': "
                        "any significance claim (Moderate or Strong) the band allows; 'false_strong': "
                        "p < alpha_strong in a band that allows Strong. 'widened_coverage': the Theil-Sen "
                        "interval widened with the replicate's own r1 (thresholds.widened_interval) "
                        "contains 0; 'widened_coverage_true_phi': widened with the factor at the true phi."),
        "grid": {"phi": [float(p) for p in phis], "n_years": [int(n) for n in years]},
        "replicates": replicates,
        "series": {"start": START, "sigma": FPR_SIGMA, "seasonal_amp": FPR_SEASONAL_AMP,
                   "slope_per_decade": 0.0, "burn_in": BURN_IN,
                   "sigma_meaning": "stationary standard deviation of the AR(1) noise"},
        "seeds": {"master": seed,
                  "scheme": "numpy.random.default_rng((master, round(1000 * phi), n_years, replicate))",
                  "replicate_range": [0, replicates - 1]},
        "thresholds_used": _thresholds_used(th),
        "ci": {"level": CI_LEVEL, "method": "Clopper-Pearson (scipy.stats.binomtest, method='exact')"},
        "cells": out_cells,
    }


# --------------------------------------------------------------------------
# Experiment (e): does AR(1) describe the real noise?
# --------------------------------------------------------------------------

NOISE_ACF_LAGS = (1, 2, 3, 6, 12)
"""Lags at which the real cases' autocorrelation is reported next to ``r1**k``."""
NOISE_VIF_LAGS = 36
"""Lags in the Bartlett-weighted variance-inflation estimate."""
NOISE_MAX_ORDER = 12
"""Largest AR order tried when fitting each case's noise model (AIC)."""
NOISE_REPLICATES = 1000
NOISE_SEED = 21


def case_seed_key(region: str, variable: str) -> int:
    """A stable integer for one case, ``zlib.crc32("region/variable")``.

    Unlike Python's ``hash``, it is the same in every process and run, so
    the noise experiment is bit-reproducible.
    """
    import zlib

    return zlib.crc32(f"{region}/{variable}".encode("utf-8"))


def acf(x, max_lag: int) -> np.ndarray:
    """Sample autocorrelations at lags ``1 .. max_lag`` (:func:`lag1_autocorr`'s
    definition at every lag: ``pandas.Series.autocorr``)."""
    s = pd.Series(np.asarray(x, dtype=float))
    return np.array([s.autocorr(lag=k) for k in range(1, max_lag + 1)])


def bartlett_vif(rho, max_lag: int) -> float:
    """``1 + 2 * sum_k (1 - k / (max_lag + 1)) * rho[k]``, ``k = 1 .. max_lag``.

    The factor by which autocorrelation inflates the variance of a mean
    or a slope, estimated with Bartlett weights. Applied to ``r1**k`` it
    gives the AR(1) value with the same truncation, so the two can be
    compared without the truncation's own downward bias.
    """
    rho = np.asarray(rho, dtype=float)[:max_lag]
    k = np.arange(1, len(rho) + 1)
    return float(1.0 + 2.0 * np.sum((1.0 - k / (max_lag + 1)) * rho))


def yule_walker(z, order: int) -> tuple[np.ndarray, float]:
    """Yule–Walker AR(``order``) coefficients and innovation variance of ``z``
    (demeaned; biased autocovariances, so the fit is always stationary)."""
    z = np.asarray(z, dtype=float)
    z = z - z.mean()
    n = len(z)
    r = np.array([np.dot(z[: n - k], z[k:]) / n for k in range(order + 1)])
    R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
    coef = np.linalg.solve(R, r[1: order + 1])
    return coef, float(r[0] - coef @ r[1: order + 1])


def ar_long_run_ratio(coef, innovation_var: float, variance: float) -> float:
    """Long-run variance over variance of an AR(p) process,
    ``innovation_var / (1 - sum(coef))**2 / variance``.

    For AR(1) it is ``(1 + phi) / (1 - phi)``, the square of
    ``thresholds.widening_factor``: the square root is the widening the
    fitted model calls for.
    """
    return float(innovation_var / (1.0 - float(np.sum(coef))) ** 2 / variance)


def fit_case_noise(region: str, variable: str, max_order: int = NOISE_MAX_ORDER) -> dict | None:
    """The noise model of one real case, or ``None`` if it fails the quality gate.

    The residuals ``e`` are the deseasonalized anomalies (``clerk.prepare``)
    minus their Theil–Sen line. They are divided by their calendar-month
    standard deviation, ``z = e / sd[month]`` (precipitation in particular
    is far more variable in the wet season), and an AR(p) model is fitted
    to ``z`` by Yule–Walker, the order chosen by AIC from 1 to
    ``max_order``. Returns the fit and the diagnostics the doc reports:
    the ACF of ``e`` at :data:`NOISE_ACF_LAGS` against ``r1**k``, the
    Bartlett variance inflation over :data:`NOISE_VIF_LAGS` lags against
    the AR(1) value with the same weights, and the widening the AR(p) model
    calls for against the AR(1) factor. Records with missing months are
    not supported (all 18 have 540).
    """
    from trendcourt import thresholds

    prepared = clerk.prepare(region, variable)
    if not prepared.quality.passed:
        return None
    a = prepared.anomalies
    if a.isna().any():
        raise ValueError(f"{region}/{variable}: the noise fit needs a record without gaps")
    t = prepared.time
    ts = examiner.theil_sen(t, a)
    e = a.to_numpy() - (ts.intercept + ts.slope * t)
    months = a.index.month.to_numpy()
    sd = pd.Series(e).groupby(months).std(ddof=1)
    month_sd = sd.loc[months].to_numpy()
    z = e / month_sd

    best = None
    for q in range(1, max_order + 1):
        coef, s2 = yule_walker(z, q)
        aic = len(z) * math.log(s2) + 2 * q
        if best is None or aic < best[0]:
            best = (aic, q, coef, s2)
    _, order, coef, s2 = best

    r1 = lag1_autocorr(e)
    rho = acf(e, max(NOISE_VIF_LAGS, max(NOISE_ACF_LAGS)))
    rho_ar1 = r1 ** np.arange(1, len(rho) + 1)
    vif = bartlett_vif(rho, NOISE_VIF_LAGS)
    vif_ar1 = bartlett_vif(rho_ar1, NOISE_VIF_LAGS)
    zc = z - z.mean()
    ar_factor = math.sqrt(ar_long_run_ratio(coef, s2, float(np.dot(zc, zc) / len(zc))))
    return {
        "region": region,
        "variable": variable,
        "n_months": int(len(e)),
        "r1_detrended": r1,
        "acf": {str(k): float(rho[k - 1]) for k in NOISE_ACF_LAGS},
        "acf_ar1": {str(k): float(r1 ** k) for k in NOISE_ACF_LAGS},
        "vif_bartlett": vif,
        "vif_bartlett_ar1": vif_ar1,
        "vif_ratio": vif / vif_ar1,
        "ar_order": int(order),
        "ar_coef": [float(c) for c in coef],
        "ar_innovation_var": s2,
        "ar_widening": ar_factor,
        "ar1_widening": thresholds.widening_factor(r1),
        "month_sd": [float(sd.loc[m]) for m in range(1, 13)],
        "index_start": str(a.index[0].date()),
    }


def _noise_series(fit: dict, seed) -> pd.Series:
    """One trend-free record from a case's fitted noise model: AR(p) in the
    standardized units (600 burn-in steps), times the calendar-month SD."""
    n = fit["n_months"]
    index = pd.date_range(fit["index_start"], periods=n, freq="MS", name="time")
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, math.sqrt(fit["ar_innovation_var"]), size=BURN_IN + n)
    z = signal.lfilter([1.0], np.r_[1.0, -np.asarray(fit["ar_coef"])], innovations)[BURN_IN:]
    sd = np.asarray(fit["month_sd"])[index.month.to_numpy() - 1]
    return pd.Series(z * sd, index=index, name="value")


def _noise_chunk(task: tuple) -> list:
    """Replicates ``start .. stop-1`` of one case. Seed:
    ``(master, case_seed_key(region, variable), replicate)``."""
    master, fit, start, stop = task
    key = case_seed_key(fit["region"], fit["variable"])
    out: list = []
    for rep in range(start, stop):
        y = _noise_series(fit, (int(master), key, int(rep)))
        try:
            r = run_case(y, seasonal=False)
        except ValueError as exc:  # count, never hide
            out.append(f"replicate {rep}: {exc}")
            continue
        out.append((r["ts_slope"], r["ts_low"], r["ts_high"], r["hr_p"], r["r1_detrended"]))
    return out


def noise_experiment(
    replicates: int = NOISE_REPLICATES,
    *,
    seed: int = NOISE_SEED,
    cases=None,
    thresholds_dict: dict | None = None,
    workers: int = 1,
    chunk: int = 50,
) -> dict:
    """Per real case: how far its noise is from AR(1), and what that does to the policy.

    For every case of ``data/regions.json`` that passes the quality gate
    (or the ``(region, variable)`` pairs in ``cases``),
    :func:`fit_case_noise` gives the ACF diagnostics and an AR(p) noise
    model with calendar-month variance; ``replicates`` trend-free records
    are simulated from that model on the case's own calendar
    (:func:`_noise_series`) and go through :func:`run_case` (without the
    seasonal test) and :func:`_policy_summary`. The case's band is the one
    of its observed ``r1_detrended``; ``stated_fpr`` is the rate that band
    states (what the card will print).

    The caveat flags derived from this result live in
    ``thresholds.json`` (``noise_caveats``, rebuilt by
    :func:`trendcourt.thresholds.derive_noise_caveats`). The numbers do
    not depend on ``workers`` or ``chunk``.
    """
    from trendcourt import thresholds

    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, got {replicates}")
    th = thresholds.load() if thresholds_dict is None else thresholds_dict
    if cases is None:
        regions = clerk.load_regions()
        cases = [(r, v) for r in regions["regions"] for v in regions["variables"]]
    fits = [f for f in (fit_case_noise(r, v) for r, v in cases) if f is not None]
    tasks = [(seed, f, a, min(a + chunk, replicates)) for f in fits for a in range(0, replicates, chunk)]
    chunks = _run_pool(_noise_chunk, tasks, workers)
    by_case: dict[tuple, list] = {(f["region"], f["variable"]): [] for f in fits}
    for task, res in zip(tasks, chunks):
        by_case[(task[1]["region"], task[1]["variable"])].extend(res)

    out_cases = []
    for f in fits:
        res = by_case[(f["region"], f["variable"])]
        ok = np.array([r for r in res if not isinstance(r, str)], dtype=float).reshape(-1, 5)
        failures = [r for r in res if isinstance(r, str)]
        band = thresholds.band_for(f["r1_detrended"], th)
        out_cases.append({
            **f,
            "band_r1_max": None if band is None else band["r1_max"],
            "max_verdict": "inconclusive" if band is None else band["max_verdict"],
            "moderate_alpha": None if band is None else band["moderate_alpha"],
            "stated_fpr": None if band is None else {k: v["rate"] for k, v in band["hamed_rao_fpr"].items()},
            "replicates": replicates,
            "n_failed": len(failures),
            "failures": failures,
            **_policy_summary(ok, th),
        })

    return {
        "experiment": "noise",
        "description": ("Does AR(1) describe the real noise? Per real case: the ACF of the Theil-Sen-"
                        "detrended anomalies against r1**k, the Bartlett variance inflation "
                        f"({NOISE_VIF_LAGS} lags) against the AR(1) value with the same weights, and an "
                        f"AR(p) model (Yule-Walker, order by AIC up to {NOISE_MAX_ORDER}) fitted to the "
                        "residuals divided by their calendar-month SD. Trend-free records simulated from "
                        "that model (times the month SD, on the case's calendar) go through "
                        "trendcourt.benchmark.run_case (no seasonal test) and the verdict policy as "
                        "applied (band from each replicate's own r1_detrended), as in "
                        "docs/benchmark/applied.json. 'stated_fpr' is the rate of the band of the case's "
                        "observed r1_detrended. 'ar_widening' is sqrt of the AR(p) long-run variance "
                        "ratio, the widening the fitted model calls for; 'ar1_widening' the factor the "
                        "rule applies."),
        "replicates": replicates,
        "acf_lags": list(NOISE_ACF_LAGS),
        "vif_lags": NOISE_VIF_LAGS,
        "max_order": NOISE_MAX_ORDER,
        "burn_in": BURN_IN,
        "seeds": {"master": seed,
                  "scheme": ("numpy.random.default_rng((master, zlib.crc32('region/variable'), "
                             "replicate))"),
                  "replicate_range": [0, replicates - 1]},
        "thresholds_used": _thresholds_used(th),
        "ci": {"level": CI_LEVEL, "method": "Clopper-Pearson (scipy.stats.binomtest, method='exact')"},
        "cases": out_cases,
    }
