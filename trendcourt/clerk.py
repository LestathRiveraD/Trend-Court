"""The Clerk: data handling.

Loads raw series from ``data/``, cleans them, aggregates to the
analysis resolution, and deseasonalizes. No statistics live here.

Pipeline for one region and variable (``project.md`` §3, steps 1 to 4)::

    df = load_power_region("mexicali", "temperature")    # one column per point
    series, n = aggregate(df, lats, return_counts=True)  # cos(lat)-weighted mean
    quality = quality_gate(series)                       # 20 years / 10% missing
    anomalies, clim = deseasonalize(series, return_climatology=True)
    t = time_in_years(series.index)                      # decimal years

or, in one call, ``prepare("mexicali", "temperature")``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
POWER_DIR = DATA_DIR / "raw" / "power"
POWER_MANIFEST = POWER_DIR / "manifest.json"
REGIONS_FILE = DATA_DIR / "regions.json"

INSUFFICIENT = "insufficient record"
SUFFICIENT = "sufficient record"
MIN_YEARS = 20
MAX_MISSING = 0.10


# --------------------------------------------------------------------------
# Metadata files
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(path: Path | str = POWER_MANIFEST) -> dict:
    """The POWER download manifest (source, access time, fill value, files)."""
    return _read_json(str(path))


def load_regions(path: Path | str = REGIONS_FILE) -> dict:
    """Region boxes, analysis period and variable codes."""
    return _read_json(str(path))


def power_code(var: str) -> str:
    """Map a variable name (``"temperature"``) or POWER code (``"T2M"``) to the POWER code."""
    variables = load_regions()["variables"]
    if var in variables:
        return variables[var]["power"]
    if var in {v["power"] for v in variables.values()}:
        return var
    known = sorted(variables) + sorted(v["power"] for v in variables.values())
    raise KeyError(f"unknown variable {var!r}; expected one of {known}")


def _variable_name(code: str) -> str | None:
    for name, v in load_regions()["variables"].items():
        if v["power"] == code:
            return name
    return None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_power_point(
    path: Path | str, var: str, fill_value: float | None = None
) -> pd.Series:
    """One POWER point file as a monthly series.

    ``var`` is the POWER code (``"T2M"``) or the variable name
    (``"temperature"``). The ``YYYY13`` keys (POWER's annual value) are
    dropped. Values equal to ``fill_value`` become NaN; by default the
    fill value comes from the download manifest. The index is a complete
    month-start ``DatetimeIndex``, so months absent from the file show up
    as NaN rather than disappearing.
    """
    if fill_value is None:
        fill_value = load_manifest()["fill_value"]
    parameters = _read_json(str(Path(path)))["properties"]["parameter"]
    code = var if var in parameters else power_code(var)
    raw = parameters[code]

    monthly = {k: v for k, v in raw.items() if int(k[4:]) <= 12}
    index = pd.to_datetime(list(monthly), format="%Y%m")
    values = np.array(list(monthly.values()), dtype=float)
    values[values == fill_value] = np.nan

    series = pd.Series(values, index=index, name=code).sort_index()
    full = pd.date_range(series.index[0], series.index[-1], freq="MS")
    series = series.reindex(full)
    series.index.name = "time"
    return series


def _region_files(region: str, manifest: dict) -> list[dict]:
    files = [f for f in manifest["files"] if f["region"] == region]
    if not files:
        known = sorted({f["region"] for f in manifest["files"]})
        raise KeyError(f"no POWER points for region {region!r}; known regions: {known}")
    return files


def load_power_region(region: str, var: str) -> pd.DataFrame:
    """All POWER points of a region, one column per point.

    Columns are a ``MultiIndex`` with levels ``lat`` and ``lon``, so each
    point's coordinates travel with its data::

        df.columns.get_level_values("lat")
    """
    manifest = load_manifest()
    files = _region_files(region, manifest)
    columns = {
        (f["lat"], f["lon"]): load_power_point(REPO_ROOT / f["path"], var, manifest["fill_value"])
        for f in files
    }
    df = pd.concat(columns, axis=1, names=["lat", "lon"])
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="MS", name="time"))
    return df


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def aggregate(df: pd.DataFrame, lats, *, return_counts: bool = False):
    """Area-weighted mean over points, with weights cos(lat), normalized.

    In a month where some points are NaN, the weights of the remaining
    points are renormalized; a month with no points is NaN. With
    ``return_counts=True``, also returns the number of points used per
    month as an integer series.
    """
    lats = np.asarray(lats, dtype=float)
    if lats.shape != (df.shape[1],):
        raise ValueError(f"got {lats.size} latitudes for {df.shape[1]} columns")

    weights = np.cos(np.deg2rad(lats))
    values = df.to_numpy(dtype=float)
    present = ~np.isnan(values)
    weight_sum = present @ weights
    weighted = np.where(present, values, 0.0) @ weights
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(weight_sum > 0, weighted / weight_sum, np.nan)

    series = pd.Series(mean, index=df.index, name="value")
    if not return_counts:
        return series
    counts = pd.Series(present.sum(axis=1), index=df.index, name="n_points")
    return series, counts


# --------------------------------------------------------------------------
# Quality gate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityResult:
    """Outcome of the quality gate.

    ``years_covered`` is the length of the record (first to last valid
    month, inclusive) in years; ``share_missing`` is the share of months
    inside that span with no value.
    """

    passed: bool
    verdict: str
    reason: str | None
    years_covered: float
    share_missing: float
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    n_months: int
    n_missing: int

    @property
    def period(self) -> str:
        if self.start is None:
            return "no data"
        return f"{self.start:%Y-%m} to {self.end:%Y-%m}"


def quality_gate(
    series: pd.Series, min_years: float = MIN_YEARS, max_missing: float = MAX_MISSING
) -> QualityResult:
    """``"insufficient record"`` if under ``min_years`` or more than ``max_missing`` missing.

    The record runs from the first to the last valid month. Months inside
    it that are NaN, or absent from the index, count as missing.
    """
    valid = series.dropna()
    if valid.empty:
        return QualityResult(False, INSUFFICIENT, "no valid months", 0.0, 1.0, None, None, 0, 0)

    start, end = valid.index.min(), valid.index.max()
    n_months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    n_missing = n_months - len(valid)
    years = n_months / 12
    share = n_missing / n_months

    reasons = []
    if n_months < min_years * 12:
        reasons.append(f"record covers {years:.1f} years, under {min_years:g}")
    if share > max_missing:
        reasons.append(f"{share:.1%} of months missing, over {max_missing:.0%}")

    return QualityResult(
        passed=not reasons,
        verdict=INSUFFICIENT if reasons else SUFFICIENT,
        reason="; ".join(reasons) or None,
        years_covered=years,
        share_missing=share,
        start=start,
        end=end,
        n_months=n_months,
        n_missing=n_missing,
    )


# --------------------------------------------------------------------------
# Deseasonalizing and time axis
# --------------------------------------------------------------------------


def climatology(series: pd.Series) -> pd.Series:
    """Long-term mean of each calendar month over the series' own period (index 1 to 12)."""
    clim = series.groupby(series.index.month).mean()
    clim = clim.reindex(range(1, 13))
    clim.index.name = "month"
    clim.name = "climatology"
    return clim


def deseasonalize(series: pd.Series, *, return_climatology: bool = False):
    """Subtract each calendar month's long-term mean.

    With ``return_climatology=True``, also returns the climatology (see
    :func:`climatology`) so it can be shown alongside the anomalies.
    """
    clim = climatology(series)
    anomalies = series - clim.loc[series.index.month].to_numpy()
    anomalies.name = "anomaly"
    if return_climatology:
        return anomalies, clim
    return anomalies


def time_in_years(index) -> np.ndarray:
    """Decimal years at mid-month: ``year + (month - 0.5) / 12``.

    Regress on this rather than a 0..n row index, so gaps keep their
    true spacing and slopes come out per year.
    """
    index = pd.DatetimeIndex(index) if not isinstance(index, pd.PeriodIndex) else index
    return index.year.to_numpy(dtype=float) + (index.month.to_numpy(dtype=float) - 0.5) / 12


# --------------------------------------------------------------------------
# Provenance and the one-call entry point
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """Where a prepared series came from; feeds the Clerk panel on the site."""

    source: str
    api: str
    region: str
    region_name: str
    variable: str | None
    power_code: str
    longname: str
    units: str
    source_units: str
    n_points: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    years: float
    accessed_utc: str
    fill_value: float
    urls: tuple[str, ...]

    @property
    def period(self) -> str:
        if self.start is None:
            return "no data"
        return f"{self.start:%Y-%m} to {self.end:%Y-%m}"

    def clerk_text(self) -> str:
        """One-line summary for the Clerk panel."""
        what = self.variable or self.power_code
        return (
            f"Filed {self.years:g} years of monthly {what} ({self.longname}, {self.units}) "
            f"for {self.region_name}, {self.period}, averaged over {self.n_points} points. "
            f"Source: {self.source}, accessed {self.accessed_utc[:10]}."
        )


@dataclass(frozen=True)
class PreparedSeries:
    """Everything the Examiner and the site need for one region and variable."""

    raw: pd.Series
    anomalies: pd.Series
    time: np.ndarray
    climatology: pd.Series
    n_points: pd.Series
    quality: QualityResult
    provenance: Provenance


def prepare(region: str, var: str) -> PreparedSeries:
    """Load, aggregate, gate and deseasonalize one POWER region series."""
    code = power_code(var)
    manifest = load_manifest()
    files = _region_files(region, manifest)

    df = load_power_region(region, code)
    raw, counts = aggregate(df, df.columns.get_level_values("lat"), return_counts=True)
    raw.name = code
    anomalies, clim = deseasonalize(raw, return_climatology=True)
    quality = quality_gate(raw)

    meta = _read_json(str(REPO_ROOT / files[0]["path"]))["parameters"][code]
    name = _variable_name(code)
    units = load_regions()["variables"][name]["units"] if name else meta["units"]
    provenance = Provenance(
        source=manifest["source"],
        api=manifest["api"],
        region=region,
        region_name=load_regions()["regions"].get(region, {}).get("name", region),
        variable=name,
        power_code=code,
        longname=meta["longname"],
        units=units,
        source_units=meta["units"],
        n_points=df.shape[1],
        start=quality.start,
        end=quality.end,
        years=quality.years_covered,
        accessed_utc=manifest["accessed_utc"],
        fill_value=manifest["fill_value"],
        urls=tuple(f["url"] for f in files),
    )
    return PreparedSeries(
        raw=raw,
        anomalies=anomalies,
        time=time_in_years(raw.index),
        climatology=clim,
        n_points=counts,
        quality=quality,
        provenance=provenance,
    )
