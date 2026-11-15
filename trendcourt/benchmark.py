"""Synthetic benchmark: the Clerk and the Examiner on series with known truth.

``project.md`` §3, validation (a): on trend-free AR(1) noise with a
seasonal cycle, the trend tests should reject about as often as their
nominal level. This module builds those series and runs them through
**the same code path as a real case** (:func:`trendcourt.case.examine`):
``clerk.deseasonalize``, then the Examiner wrappers with their defaults.
The trend statistics come only from those wrappers: nothing here calls
``scipy.stats.theilslopes`` or ``pymannkendall`` directly. (``scipy`` is
used only to generate the AR(1) noise and for the binomial interval
on each rejection rate.)

The experiment driver is ``scripts/benchmark.py``::

    python scripts/benchmark.py --experiment fpr    # writes docs/benchmark/fpr.json
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


def run_case(series: pd.Series) -> dict:
    """The statistics :func:`trendcourt.case.examine` reports, on one series.

    Same calls and defaults as ``examine``: the anomalies from
    ``clerk.deseasonalize(series)`` and the time axis
    ``clerk.time_in_years(series.index)``, then

    * ``examiner.theil_sen(t, anomalies)``: slope and 95% interval;
    * ``examiner.mk_hamed_rao(anomalies)``: alpha 0.05, ``lag=3``;
    * ``examiner.mk_seasonal(series)``: correlated seasonal MK on the
      raw series, alpha 0.05.

    Returns a flat dict: Theil–Sen ``ts_slope``, ``ts_low``, ``ts_high``
    (per year) and their ``*_per_decade`` versions; ``hr_p``, ``hr_h``;
    ``smk_p``, ``smk_h``. A broken Hamed–Rao variance raises
    ``ValueError``, as it does in ``examine``.
    """
    anomalies = clerk.deseasonalize(series)
    t = clerk.time_in_years(series.index)
    ts = examiner.theil_sen(t, anomalies)
    mk = examiner.mk_hamed_rao(anomalies)
    smk = examiner.mk_seasonal(series)
    return {
        "ts_slope": ts.slope,
        "ts_low": ts.low_slope,
        "ts_high": ts.high_slope,
        "ts_slope_per_decade": ts.slope_per_decade,
        "ts_low_per_decade": ts.low_slope_per_decade,
        "ts_high_per_decade": ts.high_slope_per_decade,
        "hr_p": mk.p,
        "hr_h": mk.h,
        "smk_p": smk.p,
        "smk_h": smk.h,
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
                r1_detrended = lag1_autocorr(anomalies.to_numpy() - (ts.intercept + ts.slope * prepared.time))
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
