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
    (per year) and their ``*_per_decade`` versions; ``hr_p``, ``hr_h``,
    ``hr_z`` (the sign gives the trend direction);
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
        "hr_z": mk.z,
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

    Returns per replicate ``(ts_slope, ts_low, ts_high, hr_p, hr_z)`` from
    :func:`run_case` (slopes per year), or the error message if a test
    raised. Top level so it pickles.
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
        out.append((r["ts_slope"], r["ts_low"], r["ts_high"], r["hr_p"], r["hr_z"]))
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
        ok = np.array([r[:3] for r in res if not isinstance(r, str)], dtype=float).reshape(-1, 3)
        failures = [r for r in res if isinstance(r, str)]
        true = slope / examiner.DECADE
        est, low, high = ok[:, 0], ok[:, 1], ok[:, 2]
        covered = (low <= true) & (true <= high)
        cov = proportion_summary(int(covered.sum()), len(ok))
        d = examiner.DECADE
        out_cells.append({
            "phi": phi,
            "slope_per_decade": slope,
            "n_years": n,
            "replicates": replicates,
            "n_failed": len(failures),
            "failures": failures,
            "coverage": cov,
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
                        "miss_low counts intervals entirely below the true slope, miss_high entirely above."),
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
        ok = np.array([r[3:] for r in res if not isinstance(r, str)], dtype=float).reshape(-1, 2)
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
