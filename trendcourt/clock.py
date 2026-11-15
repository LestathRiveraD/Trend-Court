"""The Clock: how many more years of data an Inconclusive case needs.

``project.md`` §3, step 9: "Clock module: AR(1) Monte Carlo for years to
significance." An Inconclusive verdict is shown "with time-to-verdict":
not significant yet, about N more years would settle it.

:func:`years_to_verdict` answers that under one stated assumption
(:data:`ASSUMPTION`): the trend continues at the record's current
Theil–Sen rate, and the noise around it is AR(1), with the lag-1
autocorrelation ``r1`` (:func:`trendcourt.benchmark.detrended_r1`) and the
standard deviation of the anomalies after removing the Theil–Sen line.
Each simulation appends ``k`` future years (``12 * k`` months) to the
**observed** anomalies: the Theil–Sen line continued, plus AR(1) noise that
starts from the last observed residual. It then runs the same test as a
real case, :func:`trendcourt.examiner.mk_hamed_rao` with its defaults
(alpha 0.05 for its autocorrelation filter, ``lag=3``), and calls the
simulation significant when ``p < alpha``. ``alpha`` is the band's Moderate
cut-off from ``thresholds.json``.

Every simulation draws its whole ``horizon_years`` future once, and a
shorter ``k`` uses the first ``12 * k`` months of it, so the share of
significant simulations is evaluated on the same draws at every ``k``.
That share rises with ``k`` (a longer record of a continuing trend is
easier to detect), which is what makes the bisection search valid.

All draws come from ``numpy.random.default_rng(seed)``; the case passes
``benchmark.case_seed_key(region, variable)``, so a cache rebuild is
byte-identical.
"""

from __future__ import annotations

import contextlib
import math
import os

import numpy as np
from scipy import signal

from trendcourt import benchmark, examiner

HORIZON_YEARS = 100
"""Default search limit. Past it the Clock says ``beyond_horizon``."""

N_SIMS = 500
"""Default number of simulated futures."""

PILOT_SIMS = 100
"""Simulations the first stage of the search uses to locate each point."""

SHARE_MEDIAN = 0.5
SHARE_P80 = 0.8
"""The two shares of simulations the Clock reports a year count for."""

ASSUMPTION = (
    "Assumes the trend continues at the current Theil-Sen rate, and that the "
    "noise around it stays like the record's: AR(1), with the lag-1 "
    "autocorrelation and the standard deviation of the anomalies after "
    "removing the Theil-Sen line. Each simulation adds future years to the "
    "observed record and reruns the same Hamed-Rao test. The years are counted "
    "from the end of the record."
)
"""The wording the case JSON carries in ``clock.assumption``."""


def _noise_fit(t: np.ndarray, y: np.ndarray, ts) -> tuple[float, float, float]:
    """``(r1, residual_sd, last_residual)`` of ``y`` about the Theil–Sen line."""
    resid = y - (ts.intercept + ts.slope * t)
    finite = resid[np.isfinite(resid)]
    if finite.size < 3:
        raise ValueError(f"need at least 3 finite anomalies, got {finite.size}")
    r1 = benchmark.detrended_r1(t, y, ts)
    if not (math.isfinite(r1) and -1.0 < r1 < 1.0):
        raise ValueError(f"detrended r1 must be in (-1, 1) for AR(1) noise, got {r1}")
    sd = float(np.std(finite, ddof=1))
    return float(r1), sd, float(finite[-1])


def simulate_futures(t, anomalies, ts, *, horizon_years: int = HORIZON_YEARS,
                     n_sims: int = N_SIMS, seed) -> np.ndarray:
    """``n_sims`` simulated futures of ``12 * horizon_years`` months each.

    Row ``i`` is the Theil–Sen line continued past the last month of ``t``
    plus AR(1) noise ``x[j] = r1 * x[j-1] + e[j]``, with ``x[-1]`` the last
    observed residual and Gaussian innovations of standard deviation
    ``residual_sd * sqrt(1 - r1**2)``, so the noise keeps the record's
    stationary standard deviation. Every draw comes from
    ``numpy.random.default_rng(seed)``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(anomalies, dtype=float)
    if t.shape != y.shape or t.ndim != 1:
        raise ValueError("t and anomalies must be 1-D arrays of the same length")
    r1, sd, last = _noise_fit(t, y, ts)
    m = 12 * int(horizon_years)
    t_future = t[-1] + np.arange(1, m + 1) / 12.0
    line = ts.intercept + ts.slope * t_future

    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, sd * math.sqrt(1.0 - r1 * r1), size=(int(n_sims), m))
    noise = signal.lfilter([1.0], [1.0, -r1], innovations, axis=1,
                           zi=np.full((int(n_sims), 1), r1 * last))[0]
    return line + noise


def _check_args(alpha, horizon_years, n_sims) -> None:
    if not (isinstance(alpha, float | int) and not isinstance(alpha, bool) and 0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    for name, v in (("horizon_years", horizon_years), ("n_sims", n_sims)):
        if isinstance(v, bool) or not isinstance(v, int | np.integer) or v < 1:
            raise ValueError(f"{name} must be a positive integer, got {v!r}")


_WORKER: dict = {}


def _init_worker(y: np.ndarray, futures: np.ndarray, alpha: float) -> None:
    _WORKER.update(y=y, futures=futures, alpha=alpha)


def _significant(task: tuple[int, int]) -> bool:
    """Simulation ``i`` extended by ``k`` years reaches ``p < alpha``."""
    i, k = task
    w = _WORKER
    series = np.concatenate([w["y"], w["futures"][i, : 12 * k]])
    return bool(examiner.mk_hamed_rao(series).p < w["alpha"])


def default_workers() -> int:
    """Processes the Clock uses by default: every CPU this process may run on."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not Linux
        return max(1, os.cpu_count() or 1)


def years_to_verdict(t, anomalies, ts, *, alpha: float, horizon_years: int = HORIZON_YEARS,
                     n_sims: int = N_SIMS, seed, workers: int | None = None) -> dict:
    """Years of further data until the Hamed–Rao test is likely significant.

    ``t`` and ``anomalies`` are the case's time axis and deseasonalized
    anomalies (``clerk.prepare``); ``ts`` is their
    :func:`examiner.theil_sen` result; ``alpha`` the band's Moderate
    cut-off. See the module docstring for the simulation.

    Returns a plain dict:

    * ``years_median``: the smallest whole number of years ``k`` at which at
      least 50% of the simulations reach ``p < alpha``; ``None`` if that is
      beyond ``horizon_years``;
    * ``years_p80``: the same for 80% of the simulations; ``None`` beyond
      ``horizon_years``;
    * ``beyond_horizon``: true when the 50% point is beyond the horizon;
    * ``alpha``, ``horizon_years``, ``n_sims``, ``seed``, and
      ``assumption`` (:data:`ASSUMPTION`).

    Each point is found by a doubling search on ``k`` (1, 2, 4, ...)
    followed by bisection, valid because the share of significant
    simulations rises with ``k``. Each share is counted in simulation order
    and stops as soon as the comparison with the target share is decided,
    which gives the same answer as counting every simulation. ``workers``
    (default :func:`default_workers`) processes run the tests; the result
    does not depend on it.
    """
    _check_args(alpha, horizon_years, n_sims)
    workers = default_workers() if workers is None else workers
    if isinstance(workers, bool) or not isinstance(workers, int | np.integer) or workers < 1:
        raise ValueError(f"workers must be a positive integer, got {workers!r}")
    horizon = int(horizon_years)
    y = np.asarray(anomalies, dtype=float)
    futures = simulate_futures(t, y, ts, horizon_years=horizon, n_sims=n_sims, seed=seed)
    seen: dict[int, list[bool]] = {}

    with contextlib.ExitStack() as stack:
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            pool = stack.enter_context(ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker, initargs=(y, futures, float(alpha))))
            # A pilot share runs in one round; a full share in a few.
            batch = max(PILOT_SIMS, 8 * workers)

            def run(tasks):
                return list(pool.map(_significant, tasks,
                                     chunksize=max(1, len(tasks) // (2 * workers))))
        else:
            _init_worker(y, futures, float(alpha))
            batch = 1

            def run(tasks):
                return [_significant(task) for task in tasks]

        def reaches(k: int, share: float, m: int) -> bool:
            """At least ``share`` of the first ``m`` simulations have ``p < alpha`` at ``k`` years.

            Exact: it stops once the count decides the comparison, and
            extra results from a full batch only make the count longer.
            """
            need = math.ceil(share * m - 1e-9)
            hits = seen.setdefault(k, [])
            while True:
                count = sum(hits[:m])
                done = min(len(hits), m)
                if count >= need:
                    return True
                if count + (m - done) < need:
                    return False
                start = len(hits)
                hits.extend(run([(i, k) for i in range(start, min(start + batch, m))]))

        def search(share: float, m: int, lo: int, start: int | None = None) -> int | None:
            """Smallest ``k`` in ``(lo, horizon]`` where :func:`reaches` holds for the first
            ``m`` simulations, ``None`` if there is none. ``lo`` is known not to (0: no
            added years). ``start`` is a guess: the search doubles its step away from
            it (down if it holds, up if not), then bisects the bracket."""
            if start is not None and lo < start <= horizon:
                if reaches(start, share, m):
                    hi, step = start, 1
                    while hi - step > lo:
                        if not reaches(hi - step, share, m):
                            lo = hi - step
                            break
                        hi, step = hi - step, 2 * step
                    return bisect(share, m, lo, hi)
                lo = start
            step = 1
            while True:
                k = min(lo + step, horizon)
                if k <= lo:
                    return None
                if reaches(k, share, m):
                    return bisect(share, m, lo, k)
                lo, step = k, 2 * step

        def bisect(share: float, m: int, lo: int, hi: int) -> int:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if reaches(mid, share, m):
                    hi = mid
                else:
                    lo = mid
            return hi

        # Stage 1 locates each point with the first PILOT_SIMS simulations (cheap);
        # stage 2 finds the exact point for all n_sims, starting from that guess.
        # Only stage 2 decides the answer; stage 1 only saves tests, and its
        # results are reused.
        pilot = min(PILOT_SIMS, n_sims)
        guess = search(SHARE_MEDIAN, pilot, 0)
        median = search(SHARE_MEDIAN, n_sims, 0, horizon if guess is None else guess)
        p80 = None
        if median is not None:
            guess = search(SHARE_P80, pilot, median - 1)
            p80 = search(SHARE_P80, n_sims, median - 1, horizon if guess is None else guess)

    return {
        "years_median": median,
        "years_p80": p80,
        "beyond_horizon": median is None,
        "alpha": float(alpha),
        "horizon_years": horizon,
        "n_sims": int(n_sims),
        "seed": int(seed),
        "assumption": ASSUMPTION,
    }
