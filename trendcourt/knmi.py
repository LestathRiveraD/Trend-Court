"""Gate 1 (c): our OLS slope against KNMI Climate Explorer's.

KNMI fits a least-squares trend, so the like-for-like check is
``examiner.ols`` against KNMI's regression slope on identical input. This
module writes our prepared series in KNMI's user-time-series format and
compares the slope KNMI reports (copied by hand into
``docs/gate1/knmi_slopes.json``) with ours.

No statistics are reimplemented here: slopes come from ``examiner.ols``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trendcourt import clerk, examiner

CASES = ("mexicali", "iberia", "amazon")
VAR = "temperature"
MISSING = -999.9
REL_TOL = 0.01  # identical input: slopes agree to within 1%, or half a KNMI display step (docs/gate1.md)
UPLOAD_DIR = clerk.DATA_DIR / "knmi" / "upload"
SLOPES_FILE = clerk.REPO_ROOT / "docs" / "gate1" / "knmi_slopes.json"
SERIES = ("monthly", "annual")
UNIT_FACTORS = {"per_year": examiner.DECADE, "per_decade": 1.0}


# --------------------------------------------------------------------------
# The series we upload
# --------------------------------------------------------------------------


def annual_means(monthly: pd.Series) -> pd.Series:
    """Calendar-year means of a monthly series; NaN for any incomplete year."""
    index = pd.DatetimeIndex(monthly.index)
    grouped = pd.Series(monthly.to_numpy(dtype=float), index=index).groupby(index.year)
    means = grouped.mean()
    means[grouped.count() < 12] = np.nan
    means.index.name = "year"
    return means


def series_for(region: str) -> dict:
    """The prepared anomalies and their annual means, with our OLS fit of each.

    Monthly time is mid-month decimal years (``clerk.time_in_years``);
    annual time is ``year + 0.5``. A constant offset in the time axis does
    not change the slope, so KNMI's convention only matters through gaps.
    """
    p = clerk.prepare(region, VAR)
    annual = annual_means(p.anomalies)
    return {
        "prepared": p,
        "monthly": p.anomalies,
        "annual": annual,
        "ols": {
            "monthly": examiner.ols(p.time, p.anomalies),
            "annual": examiner.ols(annual.index.to_numpy(dtype=float) + 0.5, annual),
        },
    }


def _fmt(v: float) -> str:
    return f"{MISSING:10.1f}" if math.isnan(v) else f"{v:10.4f}"


def to_knmi_monthly(series: pd.Series, header: list[str]) -> str:
    """``yyyy v1 .. v12`` lines, ``#`` header, -999.9 for missing months."""
    index = pd.DatetimeIndex(series.index)
    if len(index) and not (index.day == 1).all():
        raise ValueError("expected month-start timestamps")
    frame = pd.DataFrame({"year": index.year, "month": index.month, "v": series.to_numpy(dtype=float)})
    if frame.duplicated(["year", "month"]).any():
        raise ValueError("duplicate months in series")
    table = frame.pivot(index="year", columns="month", values="v").reindex(columns=range(1, 13))
    full_years = range(int(table.index.min()), int(table.index.max()) + 1) if len(table) else []
    table = table.reindex(full_years)
    lines = [f"# {h}" for h in header]
    lines += [f"{year:4d} " + " ".join(_fmt(v) for v in row) for year, row in zip(table.index, table.to_numpy(dtype=float))]
    return "\n".join(lines) + "\n"


def to_knmi_annual(series: pd.Series, header: list[str]) -> str:
    """``yyyy v`` lines, ``#`` header, -999.9 for missing years."""
    years = series.index.to_numpy(dtype=int)
    if len(set(years)) != len(years):
        raise ValueError("duplicate years in series")
    full = series.reindex(range(int(years.min()), int(years.max()) + 1)) if len(years) else series
    lines = [f"# {h}" for h in header]
    lines += [f"{int(y):4d} {_fmt(float(v))}" for y, v in full.items()]
    return "\n".join(lines) + "\n"


def upload_path(region: str, series: str, out_dir: Path = UPLOAD_DIR) -> Path:
    return out_dir / f"{region}_{clerk.power_code(VAR)}_anom_{series}.dat"


def export(region: str, out_dir: Path = UPLOAD_DIR) -> list[Path]:
    """Write the monthly and annual upload files for one region."""
    s = series_for(region)
    prov = s["prepared"].provenance
    base = [
        f"Trend Court: NASA POWER {prov.power_code} anomalies, {prov.region_name}",
        f"cos(lat)-weighted mean of {prov.n_points} points; monthly climatology over {prov.period} removed",
        f"source {prov.source}, accessed {prov.accessed_utc}",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fmt in (("monthly", to_knmi_monthly), ("annual", to_knmi_annual)):
        header = base + [f"{name} values; missing = {MISSING}", f"{prov.power_code}_anom [{prov.units}] {name} anomaly"]
        path = upload_path(region, name, out_dir)
        path.write_text(fmt(s[name], header), encoding="utf-8")
        paths.append(path)
    return paths


# --------------------------------------------------------------------------
# Comparison against the slopes KNMI reports
# --------------------------------------------------------------------------


GATE_PASS = 0  # every case and series filled in, all within tolerance
GATE_FAIL = 1  # at least one filled-in entry is outside tolerance
GATE_PENDING = 2  # nothing out of tolerance, but some entries still await KNMI
REQUIRED_RECORD = ("resolution", "knmi_settings", "evidence")


@dataclass(frozen=True)
class Comparison:
    region: str
    series: str
    ours_per_decade: float
    knmi_per_decade: float
    knmi_stderr_per_decade: float | None
    abs_diff: float  # |KNMI - ours|, per decade
    rel_diff: float
    allowed: float  # max(rel_tol * |ours|, resolution / 2), per decade
    bound: str  # "relative" or "display" (which term of ``allowed`` applied)
    tolerance: float
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _per_decade(value: float, units: str) -> float:
    try:
        return float(value) * UNIT_FACTORS[units]
    except KeyError:
        raise ValueError(f"units must be one of {sorted(UNIT_FACTORS)}, got {units!r}") from None


def _check_resolution(slope: float, resolution) -> float:
    if resolution is None:
        raise ValueError("resolution (the last displayed digit of KNMI's slope, e.g. 0.001) is required")
    resolution = float(resolution)
    if not (math.isfinite(resolution) and resolution > 0):
        raise ValueError(f"resolution must be a positive number, got {resolution!r}")
    steps = float(slope) / resolution
    if abs(steps - round(steps)) > 1e-6:
        raise ValueError(f"slope {slope!r} has more digits than resolution {resolution!r}")
    return resolution


def compare(region: str, series: str, knmi: dict, ours: examiner.OLSResult, tolerance: float = REL_TOL) -> Comparison:
    """KNMI's slope against ours, allowing for KNMI's display rounding.

    ``knmi`` holds ``slope`` exactly as KNMI displays it, ``units``
    (``per_year`` or ``per_decade``), ``resolution`` (the place value of the
    last displayed digit, in the same units, e.g. ``0.001``) and optionally
    ``stderr`` in the same units. Passes when, per decade,
    ``|knmi - ours| <= max(tolerance * |ours|, resolution / 2)``: a correct
    slope can be off by up to half a display step after rounding.
    """
    if series not in SERIES:
        raise ValueError(f"series must be one of {SERIES}, got {series!r}")
    resolution = _per_decade(_check_resolution(knmi["slope"], knmi.get("resolution")), knmi["units"])
    knmi_slope = _per_decade(knmi["slope"], knmi["units"])
    stderr = knmi.get("stderr")
    knmi_stderr = None if stderr is None else _per_decade(stderr, knmi["units"])
    ours_slope = ours.slope_per_decade
    if not (math.isfinite(knmi_slope) and math.isfinite(ours_slope)) or ours_slope == 0:
        raise ValueError(f"cannot form a relative difference from {knmi_slope} and {ours_slope}")
    diff = abs(knmi_slope - ours_slope)
    rel_bound = tolerance * abs(ours_slope)
    display_bound = resolution / 2
    bound, allowed = ("display", display_bound) if display_bound > rel_bound else ("relative", rel_bound)
    return Comparison(
        region, series, ours_slope, knmi_slope, knmi_stderr,
        diff, diff / abs(ours_slope), allowed, bound, tolerance, diff <= allowed,
    )


def load_knmi_slopes(path: Path = SLOPES_FILE) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def compare_all(slopes: dict) -> list[Comparison]:
    """Compare every filled-in KNMI entry; entries with ``slope: null`` are skipped.

    A filled-in entry must also record ``resolution``, ``knmi_settings`` and
    ``evidence``, so a pass is always backed by what KNMI showed.
    """
    out = []
    for region, by_series in slopes["cases"].items():
        filled = {k: v for k, v in by_series.items() if v.get("slope") is not None}
        if not filled:
            continue
        for series, knmi in filled.items():
            missing = [k for k in REQUIRED_RECORD if knmi.get(k) in (None, "")]
            if missing:
                raise ValueError(f"{region} {series}: slope filled in but {', '.join(missing)} missing")
        ours = series_for(region)["ols"]
        for series, knmi in filled.items():
            out.append(compare(region, series, knmi, ours[series], slopes.get("tolerance", REL_TOL)))
    return out


def pending_entries(slopes: dict) -> list[tuple[str, str]]:
    """Every expected (case, series) whose KNMI slope is absent or null."""
    cases = slopes.get("cases", {})
    return [
        (region, series)
        for region in CASES
        for series in SERIES
        if cases.get(region, {}).get(series, {}).get("slope") is None
    ]


def gate_status(rows: list[Comparison], pending: list[tuple[str, str]]) -> int:
    """Exit code for Gate 1 (c): ``GATE_FAIL`` if any compared entry fails,
    else ``GATE_PENDING`` while any entry awaits KNMI, else ``GATE_PASS``."""
    if any(not r.passed for r in rows):
        return GATE_FAIL
    if pending or not rows:
        return GATE_PENDING
    return GATE_PASS


def markdown_table(rows: list[Comparison]) -> str:
    head = (
        "| Case | Series | Ours (°C/decade) | KNMI (°C/decade) | KNMI ± s.e. | Abs. diff | Allowed (bound) | Rel. diff | Pass |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    body = [
        f"| {c.region} | {c.series} | {c.ours_per_decade:.4f} | {c.knmi_per_decade:.4f} | "
        f"{'—' if c.knmi_stderr_per_decade is None else f'{c.knmi_stderr_per_decade:.4f}'} | "
        f"{c.abs_diff:.4f} | {c.allowed:.4f} ({c.bound}) | "
        f"{c.rel_diff:.2%} | {'yes' if c.passed else '**no**'} |"
        for c in rows
    ]
    return "\n".join([head, *body])
