"""Cross-checks of our series against independent references (Gate 1).

Gate 1 (b), ``project.md`` §5: the area-averaged POWER T2M should track
Giovanni's MERRA-2 T2M area average for the same box, since POWER's
meteorology is derived from MERRA-2. The tolerances are fixed in
``docs/gate1.md`` before comparing; :class:`Tolerances` holds the same
values.

Usage::

    power = clerk.prepare("mexicali", "T2M").raw
    merra2 = clerk.load_giovanni_csv(clerk.GIOVANNI_DIR / "mexicali_M2TMNXSLV_T2M.csv")
    result = compare(power, merra2)
    result.checks(), result.passed
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from trendcourt import clerk, examiner


@dataclass(frozen=True)
class Tolerances:
    """Pass criteria for Gate 1 (b); keep in sync with ``docs/gate1.md``.

    The slope criterion is ``|Δslope| <= max(slope_abs, slope_rel × |POWER slope|)``,
    in °C per decade.
    """

    min_anomaly_r: float = 0.95
    max_abs_bias: float = 1.0
    slope_abs: float = 0.1
    slope_rel: float = 0.25

    def slope_allowance(self, power_slope: float) -> float:
        return max(self.slope_abs, self.slope_rel * abs(power_slope))


GATE1 = Tolerances()


@dataclass(frozen=True)
class CrossCheckResult:
    """Comparison of two monthly series over the months both have.

    ``bias`` and ``rmse`` are ``a − b`` on the raw values. Anomalies are
    each series minus its own calendar-month climatology over the shared
    months, so the climatologies cover the same period. Slopes are
    Theil–Sen on the anomalies (:func:`examiner.theil_sen`), per decade,
    with their 95% intervals.
    """

    n: int
    start: str
    end: str
    r_raw: float
    r_anomaly: float
    bias: float
    rmse: float
    slope_a: examiner.SlopeResult
    slope_b: examiner.SlopeResult
    tolerances: Tolerances

    @property
    def slope_diff(self) -> float:
        """``a`` minus ``b`` slope, °C per decade."""
        return self.slope_a.slope_per_decade - self.slope_b.slope_per_decade

    @property
    def slope_allowance(self) -> float:
        return self.tolerances.slope_allowance(self.slope_a.slope_per_decade)

    @property
    def intervals_overlap(self) -> bool:
        a, b = self.slope_a, self.slope_b
        return a.low_slope_per_decade <= b.high_slope_per_decade and b.low_slope_per_decade <= a.high_slope_per_decade

    def checks(self) -> dict[str, bool]:
        """Each Gate 1 (b) tolerance, pass or fail."""
        tol = self.tolerances
        return {
            "anomaly_r": bool(self.r_anomaly >= tol.min_anomaly_r),
            "abs_bias": bool(abs(self.bias) <= tol.max_abs_bias),
            "slope_diff": bool(abs(self.slope_diff) <= self.slope_allowance),
        }

    @property
    def passed(self) -> bool:
        return all(self.checks().values())

    def to_dict(self) -> dict:
        """Plain dict for strict JSON."""
        return {
            "n": self.n,
            "start": self.start,
            "end": self.end,
            "r_raw": self.r_raw,
            "r_anomaly": self.r_anomaly,
            "bias": self.bias,
            "rmse": self.rmse,
            "slope_a": self.slope_a.to_dict(),
            "slope_b": self.slope_b.to_dict(),
            "slope_diff": self.slope_diff,
            "slope_allowance": self.slope_allowance,
            "intervals_overlap": self.intervals_overlap,
            "checks": self.checks(),
            "passed": self.passed,
            "tolerances": asdict(self.tolerances),
        }


def compare(a: pd.Series, b: pd.Series, tolerances: Tolerances = GATE1) -> CrossCheckResult:
    """Compare monthly series ``a`` (ours) against ``b`` (the reference).

    Both need a month-start ``DatetimeIndex``. Only months where both are
    non-NaN are used; at least 24 shared months are required so every
    calendar month has two values for the climatology.
    """
    for name, s in (("a", a), ("b", b)):
        if not isinstance(s.index, pd.DatetimeIndex):
            raise TypeError(f"{name} must have a DatetimeIndex")
        if s.index.duplicated().any():
            raise ValueError(f"{name} has duplicate timestamps")
    both = pd.concat({"a": a, "b": b}, axis=1, join="inner").dropna()
    if len(both) < 24:
        raise ValueError(f"need at least 24 shared non-NaN months, got {len(both)}")

    anom = both.apply(clerk.deseasonalize)
    t = clerk.time_in_years(both.index)
    diff = both["a"] - both["b"]
    return CrossCheckResult(
        n=len(both),
        start=f"{both.index[0]:%Y-%m}",
        end=f"{both.index[-1]:%Y-%m}",
        r_raw=float(both["a"].corr(both["b"])),
        r_anomaly=float(anom["a"].corr(anom["b"])),
        bias=float(diff.mean()),
        rmse=float(np.sqrt((diff**2).mean())),
        slope_a=examiner.theil_sen(t, anom["a"]),
        slope_b=examiner.theil_sen(t, anom["b"]),
        tolerances=tolerances,
    )


def points_in_box(df: pd.DataFrame, bbox, eps: float = 1e-9) -> pd.DataFrame:
    """Columns of a :func:`clerk.load_power_region` frame whose point lies
    inside ``bbox = (west, south, east, north)`` (Giovanni's order), edges
    included."""
    west, south, east, north = bbox
    lat = df.columns.get_level_values("lat").to_numpy(dtype=float)
    lon = df.columns.get_level_values("lon").to_numpy(dtype=float)
    keep = (lat >= south - eps) & (lat <= north + eps) & (lon >= west - eps) & (lon <= east + eps)
    if not keep.any():
        raise ValueError(f"no points inside box {bbox}")
    return df.loc[:, keep]


def power_in_box(region: str, bbox, var: str = "T2M") -> tuple[pd.Series, int]:
    """POWER series re-averaged over only the region's points inside ``bbox``;
    returns the series and the number of points used."""
    df = points_in_box(clerk.load_power_region(region, var), bbox)
    series = clerk.aggregate(df, df.columns.get_level_values("lat"))
    series.name = clerk.power_code(var)
    return series, df.shape[1]


def plot_comparison(a: pd.Series, b: pd.Series, result: CrossCheckResult, path, *,
                    title: str, label_a: str = "POWER T2M", label_b: str = "Giovanni MERRA-2 T2M"):
    """Raw series overlaid (top) and anomalies with Theil–Sen lines (bottom), saved to ``path``.

    Imports matplotlib lazily, so the library modules do not need it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    both = pd.concat({"a": a, "b": b}, axis=1, join="inner").dropna()
    anom = both.apply(clerk.deseasonalize)
    t = clerk.time_in_years(both.index)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    top.plot(both.index, both["b"], color="tab:orange", lw=1.6, label=label_b)
    top.plot(both.index, both["a"], color="tab:blue", lw=0.9, label=label_a)
    top.set_ylabel("Monthly mean (°C)")
    top.set_title(f"{title}: r = {result.r_raw:.4f}, bias (POWER − MERRA-2) = {result.bias:+.2f} °C, "
                  f"RMSE = {result.rmse:.2f} °C", fontsize=10)
    top.legend(loc="upper left", fontsize=9)

    for key, slope, color, label in (("b", result.slope_b, "tab:orange", label_b),
                                     ("a", result.slope_a, "tab:blue", label_a)):
        bottom.plot(both.index, anom[key], color=color, lw=0.8, alpha=0.7)
        bottom.plot(both.index, slope.intercept + slope.slope * t, color=color, lw=2,
                    label=f"{label}: {slope.slope_per_decade:+.3f} °C/decade "
                          f"[{slope.low_slope_per_decade:+.3f}, {slope.high_slope_per_decade:+.3f}]")
    bottom.axhline(0, color="0.6", lw=0.6)
    bottom.set_ylabel("Anomaly (°C)")
    bottom.set_title(f"Anomalies: r = {result.r_anomaly:.4f}; Theil–Sen slopes with 95% intervals", fontsize=10)
    bottom.legend(loc="upper left", fontsize=9)

    fig.savefig(path, dpi=120)
    plt.close(fig)
