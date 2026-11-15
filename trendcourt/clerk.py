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

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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


GIOVANNI_DIR = DATA_DIR / "raw" / "giovanni"
GIOVANNI_FILL_VALUE = 1e15
KELVIN_OFFSET = 273.15


@dataclass(frozen=True)
class GiovanniHeader:
    """Metadata lines at the top of a Giovanni time-series CSV.

    Boxes are ``(west, south, east, north)`` in degrees, as Giovanni
    writes them. ``data_bbox`` gives the outermost source-grid cell
    *centers* inside the user box, not the edges of the averaged area:
    the cells extend half a grid step beyond it on each side (for
    MERRA-2, see :func:`trendcourt.crosscheck.data_area`).
    """

    title: str
    user_bbox: tuple[float, float, float, float]
    data_bbox: tuple[float, float, float, float]
    fill_value: float
    column: str
    n_header_lines: int


def _parse_bbox(text: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"expected 4 bounding-box values, got {text!r}")
    return tuple(parts)  # type: ignore[return-value]


def read_giovanni_header(path: Path | str) -> GiovanniHeader:
    """Parse the metadata lines of a Giovanni time-series CSV.

    The header is a variable number of ``key:,value`` lines, then a blank
    line, then the column row starting with ``time``. Its length is found
    by scanning, not assumed.
    """
    meta: dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if row and row[0].strip() == "time":
                if len(row) != 2:
                    raise ValueError(f"{path}: expected 2 columns in {row!r}")
                n_header = i
                column = row[1].strip()
                break
            if row and row[0].strip():
                meta[row[0].strip().rstrip(":")] = ",".join(row[1:]).strip()
        else:
            raise ValueError(f"{path}: no 'time' column row found")

    fill_key = next((k for k in meta if k.startswith("Fill Value")), None)
    fill = float(meta[fill_key]) if fill_key else GIOVANNI_FILL_VALUE
    try:
        user_bbox = _parse_bbox(meta["User Bounding Box"])
        data_bbox = _parse_bbox(meta["Data Bounding Box"])
    except KeyError as exc:
        raise ValueError(f"{path}: header has no {exc.args[0]!r} line") from None
    return GiovanniHeader(
        title=meta.get("Title", ""),
        user_bbox=user_bbox,
        data_bbox=data_bbox,
        fill_value=fill,
        column=column,
        n_header_lines=n_header,
    )


def load_giovanni_csv(path: Path | str) -> pd.Series:
    """A Giovanni area-averaged temperature CSV as a monthly series in °C.

    Skips the metadata header (see :func:`read_giovanni_header`), sets the
    header's fill value (1e15) to NaN, converts Kelvin to °C, and returns
    a complete month-start ``DatetimeIndex`` named ``time``, so months
    absent from the file show up as NaN. Values are compared to the fill
    value with a relative tolerance, since the file writes it as an
    integer and a float round-trip may not be exact.

    Only for Kelvin temperature variables such as MERRA-2 ``T2M``; the
    conversion is not checked against the file's units, which Giovanni
    does not write.
    """
    header = read_giovanni_header(path)
    df = pd.read_csv(path, skiprows=header.n_header_lines, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    values = df[header.column].to_numpy(dtype=float, copy=True)
    values[np.isclose(values, header.fill_value, rtol=1e-6, atol=0.0)] = np.nan

    index = pd.to_datetime(df["time"])
    if not (index == index.dt.to_period("M").dt.start_time).all():
        raise ValueError(f"{path}: timestamps are not month starts")
    if index.duplicated().any():
        raise ValueError(f"{path}: duplicate months")

    series = pd.Series(values - KELVIN_OFFSET, index=pd.DatetimeIndex(index), name="T2M").sort_index()
    series = series.reindex(pd.date_range(series.index[0], series.index[-1], freq="MS"))
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


# --------------------------------------------------------------------------
# Witness B (NOAA PSL gridded files, pulled the night before)
# --------------------------------------------------------------------------

WITNESS_DIR = DATA_DIR / "raw" / "witness"
WITNESS_MANIFEST = WITNESS_DIR / "manifest_psl.json"

INDEPENDENT = "independent"
PARTLY_INDEPENDENT = "partly_independent"
SAME_SOURCE = "same_source"


@dataclass(frozen=True)
class WitnessSpec:
    """How to read witness B for one variable."""

    dataset: str  # key in the PSL manifest and folder under WITNESS_DIR
    netcdf_var: str
    grid_step: float  # degrees, lat and lon
    grid_origin: float  # native cell centers are grid_origin + k * grid_step (lat and lon)
    kelvin: bool  # convert K to degC
    units: str  # units of the returned series
    independence: str
    independence_note: str


WITNESS_B: dict[str, WitnessSpec] = {
    "temperature": WitnessSpec(
        dataset="ghcncams",
        netcdf_var="air",
        grid_step=0.5,
        grid_origin=0.25,  # centers at 0.25 + 0.5k (lat/lon actual_range 89.75, 0.25..359.75)
        kelvin=True,
        units="degC",
        independence=INDEPENDENT,
        independence_note="station based (GHCN + CAMS), no reanalysis input",
    ),
    "precipitation": WitnessSpec(
        dataset="gpcp_v23",
        netcdf_var="precip",
        grid_step=2.5,
        grid_origin=1.25,  # centers at 1.25 + 2.5k (lat/lon actual_range 88.75, 1.25..358.75)
        kelvin=False,
        units="mm/day",
        independence=PARTLY_INDEPENDENT,
        independence_note="POWER's corrected precipitation uses gauge-based corrections",
    ),
}


def load_witness_manifest(path: Path | str = WITNESS_MANIFEST) -> dict:
    """The NOAA PSL witness download manifest (source, access time, files)."""
    return _read_json(str(path))


def _variable_key(var: str) -> str:
    """Variable name for a name or POWER code (``"T2M"`` -> ``"temperature"``)."""
    name = _variable_name(power_code(var))
    assert name is not None
    return name


def _witness_file(region: str, dataset: str, manifest: dict) -> dict:
    files = [f for f in manifest["files"] if f["dataset"] == dataset and f["region"] == region]
    if not files:
        known = sorted({f["region"] for f in manifest["files"] if f["dataset"] == dataset})
        raise KeyError(f"no {dataset} witness file for region {region!r}; known regions: {known}")
    if len(files) > 1:
        raise ValueError(f"{len(files)} {dataset} witness files for region {region!r}")
    return files[0]


def _cell_edges(centers: np.ndarray, default_step: float) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper edges of cells on a regular grid, from their centers."""
    centers = np.asarray(centers, dtype=float)
    if centers.size > 1:
        steps = np.abs(np.diff(np.sort(centers)))
        if not np.allclose(steps, steps[0], rtol=1e-4):
            raise ValueError(f"irregular grid spacing: {np.unique(steps.round(6))}")
        step = float(steps[0])
    else:
        step = default_step
    return centers - step / 2, centers + step / 2


def _check_native_grid(
    centers: np.ndarray, origin: float, step: float, *, axis: str, path: str, actual_range=None
) -> None:
    """Raise unless every cell center is ``origin + k * step`` (and the axis spacing is ``step``).

    NCSS subsets can come back with a relabelled axis: a GPCP request that
    crossed 0 deg E was labelled from the requested west edge (359.75,
    362.25, ...) instead of the native centers (358.75, 1.25, ...). Such an
    axis has the right spacing but the wrong origin, and would silently
    average the wrong area. ``actual_range`` (the axis attribute, which
    describes the full dataset) is checked too, so a wrong spec is caught.
    """
    centers = np.asarray(centers, dtype=float)
    checks = [("centers", centers)]
    if actual_range is not None:
        checks.append(("actual_range", np.asarray(actual_range, dtype=float)))
    for what, values in checks:
        k = (values - origin) / step
        off = ~np.isclose(k, np.round(k), rtol=0.0, atol=1e-4)
        if off.any():
            raise ValueError(
                f"{path}: {axis} {what} {values[off].tolist()} are off the native grid "
                f"{origin:g} + k * {step:g}; the file's {axis} axis is mislabelled"
            )
    if centers.size > 1:
        steps = np.diff(centers)
        if axis == "lon":  # a correctly labelled subset may wrap at 0/360 (358.75, 1.25, ...)
            steps = (steps + 180.0) % 360.0 - 180.0
        steps = np.abs(steps)
        if not np.allclose(steps, step, rtol=1e-4):
            raise ValueError(f"{path}: {axis} spacing {np.unique(steps.round(6)).tolist()}, expected {step:g}")


def _overlap_fraction(lo: np.ndarray, hi: np.ndarray, box_lo: float, box_hi: float) -> np.ndarray:
    """Share of each cell ``[lo, hi]`` that lies inside ``[box_lo, box_hi]``."""
    inside = np.clip(np.minimum(hi, box_hi) - np.maximum(lo, box_lo), 0.0, None)
    return inside / (hi - lo)


def _mask_invalid(values: np.ndarray, attrs: dict) -> np.ndarray:
    """NaN for fill/missing values and anything outside ``valid_range``.

    Reads ``_FillValue``, ``missing_value`` and ``valid_range`` (or
    ``valid_min``/``valid_max``) from the variable's attributes. Fill
    values are compared with a relative tolerance, since float32
    round-trips of ~1e36 are not exact.
    """
    values = np.array(values, dtype=float, copy=True)
    for key in ("_FillValue", "missing_value"):
        if key in attrs:
            for fill in np.atleast_1d(attrs[key]).astype(float):
                values[np.isclose(values, fill, rtol=1e-6, atol=0.0)] = np.nan
    lo, hi = -np.inf, np.inf
    if "valid_range" in attrs:
        lo, hi = (float(x) for x in np.asarray(attrs["valid_range"]))
    lo = float(attrs.get("valid_min", lo))
    hi = float(attrs.get("valid_max", hi))
    with np.errstate(invalid="ignore"):
        values[(values < lo) | (values > hi)] = np.nan
    return values


@dataclass(frozen=True)
class WitnessGrid:
    """The grid cells averaged for witness B and the area they cover.

    ``cells`` lists ``(lat, lon, weight)`` for every cell that overlaps
    the region box, with ``lon`` in -180..180 and weights normalized to
    sum to 1 (cos(lat) times the share of the cell inside the box).
    Boxes are ``(south, west, north, east)``, like ``regions.json``.
    ``covered_box`` is the outer edge of those cells: the area whose
    data actually enters the mean, larger than ``region_box`` when the
    grid is coarse (GPCP's 2.5 deg cells).
    """

    region_box: tuple[float, float, float, float]
    requested_box: tuple[float, float, float, float] | None
    covered_box: tuple[float, float, float, float]
    cells: tuple[tuple[float, float, float], ...]
    grid_step: tuple[float, float]  # (lat, lon) degrees
    source_units: str  # the file's ``units`` attribute, before conversion

    @property
    def n_cells(self) -> int:
        return len(self.cells)


def _requested_box(url: str) -> tuple[float, float, float, float] | None:
    q = parse_qs(urlparse(url).query)
    try:
        return tuple(float(q[k][0]) for k in ("south", "west", "north", "east"))  # type: ignore[return-value]
    except (KeyError, ValueError):
        return None


def load_witness_grid(region: str, var: str) -> tuple[pd.Series, pd.Series, WitnessGrid] | None:
    """Witness B for one region as ``(series, n_cells, grid)``; ``None`` without a witness.

    See :func:`load_witness` for the series. ``n_cells`` is the number
    of cells with a valid value each month.
    """
    import xarray as xr

    name = _variable_key(var)
    spec = WITNESS_B.get(name)
    if spec is None:
        return None
    entry = _witness_file(region, spec.dataset, load_witness_manifest())
    south, west, north, east = load_regions()["regions"][region]["box"]

    with xr.open_dataset(REPO_ROOT / entry["path"], mask_and_scale=False, decode_times=True) as ds:
        da = ds[spec.netcdf_var].transpose("time", "lat", "lon")
        attrs = dict(da.attrs)
        raw = da.to_numpy()
        lats = ds["lat"].to_numpy().astype(float)
        lons = ds["lon"].to_numpy().astype(float)
        lat_range = ds["lat"].attrs.get("actual_range")
        lon_range = ds["lon"].attrs.get("actual_range")
        times = pd.DatetimeIndex(ds["time"].to_numpy())

    values = _mask_invalid(raw, attrs)
    values = values * float(attrs.get("scale_factor", 1.0)) + float(attrs.get("add_offset", 0.0))
    if spec.kelvin:
        units = str(attrs.get("units", "")).lower()
        if units not in {"k", "degk", "kelvin"}:
            raise ValueError(f"{entry['path']}: expected Kelvin, file says units={attrs.get('units')!r}")
        values = values - KELVIN_OFFSET

    for axis, centers, rng in (("lat", lats, lat_range), ("lon", lons, lon_range)):
        _check_native_grid(centers, spec.grid_origin, spec.grid_step, axis=axis,
                           path=entry["path"], actual_range=rng)

    lons = np.where(lons > 180, lons - 360, lons)
    lat_lo, lat_hi = _cell_edges(lats, spec.grid_step)
    lon_lo, lon_hi = _cell_edges(lons, spec.grid_step)
    f_lat = _overlap_fraction(lat_lo, lat_hi, south, north)
    f_lon = _overlap_fraction(lon_lo, lon_hi, west, east)
    weights = np.outer(np.cos(np.deg2rad(lats)) * f_lat, f_lon)  # (lat, lon)
    use = weights > 0
    if not use.any():
        raise ValueError(f"{entry['path']}: no grid cell overlaps the {region} box")

    flat_w = weights[use]
    flat_v = values[:, use]  # (time, cell)
    present = ~np.isnan(flat_v)
    w_sum = present @ flat_w
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(w_sum > 0, np.where(present, flat_v, 0.0) @ flat_w / w_sum, np.nan)

    if not (times == times.to_period("M").start_time).all():
        raise ValueError(f"{entry['path']}: timestamps are not month starts")
    if times.duplicated().any():
        raise ValueError(f"{entry['path']}: duplicate months")
    series = pd.Series(mean, index=times, name=spec.netcdf_var).sort_index()
    counts = pd.Series(present.sum(axis=1), index=times, name="n_cells").sort_index()

    period = load_regions()["period"]
    full = pd.date_range(f"{period['start']}-01-01", f"{period['end']}-12-01", freq="MS", name="time")
    series = series.reindex(full)
    counts = counts.reindex(full, fill_value=0).astype(int)

    ii, jj = np.nonzero(use)
    cells = tuple(
        (float(lats[i]), float(lons[j]), float(weights[i, j] / flat_w.sum())) for i, j in zip(ii, jj)
    )
    grid = WitnessGrid(
        region_box=(south, west, north, east),
        requested_box=_requested_box(entry["url"]),
        covered_box=(
            float(lat_lo[ii].min()),
            float(lon_lo[jj].min()),
            float(lat_hi[ii].max()),
            float(lon_hi[jj].max()),
        ),
        cells=cells,
        grid_step=(float(lat_hi[0] - lat_lo[0]), float(lon_hi[0] - lon_lo[0])),
        source_units=str(attrs.get("units", "")),
    )
    return series, counts, grid


def load_witness(region: str, var: str) -> pd.Series | None:
    """Witness B for one region and variable as a monthly series, or ``None``.

    Opens the region's NetCDF file (``data/raw/witness/<dataset>/<region>.nc``),
    masks fill/missing values and values outside ``valid_range``, and
    takes the area mean over every grid cell that overlaps the region
    box, weighting each by cos(lat) times the share of the cell inside
    the box (renormalized over the cells present each month). GHCN_CAMS
    temperature is converted from K to °C; GPCP precipitation is already
    mm/day. The index is a complete month-start ``DatetimeIndex`` named
    ``time`` over the analysis period in ``regions.json``.

    Returns ``None`` for variables without a witness B (soil moisture),
    which run in single-witness mode. Use :func:`load_witness_grid` for
    the cells used and the box actually covered.
    """
    out = load_witness_grid(region, var)
    return None if out is None else out[0]


@dataclass(frozen=True)
class WitnessProvenance:
    """Where witness B came from; feeds the cross-examination panel."""

    dataset: str
    dataset_name: str
    source: str
    url: str
    accessed_utc: str
    independence: str
    independence_note: str
    region: str
    region_name: str
    variable: str
    units: str
    source_units: str
    grid: WitnessGrid
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    years: float

    @property
    def period(self) -> str:
        if self.start is None:
            return "no data"
        return f"{self.start:%Y-%m} to {self.end:%Y-%m}"

    def clerk_text(self) -> str:
        """One-line summary for the cross-examination panel."""
        s, w, n, e = self.grid.covered_box
        return (
            f"Witness B: {self.dataset_name}, {self.years:g} years of monthly {self.variable} "
            f"({self.units}) for {self.region_name}, {self.period} (months shared with POWER), "
            f"averaged over {self.grid.n_cells} grid cell{'s' if self.grid.n_cells != 1 else ''} covering {s:g} to {n:g} N, "
            f"{w:g} to {e:g} E. {self.independence.replace('_', ' ').capitalize()} of POWER. "
            f"Source: {self.source}, accessed {self.accessed_utc[:10]}."
        )


@dataclass(frozen=True)
class PreparedWitness:
    """Witness B through the same gate and deseasonalizing as witness A.

    Every series is restricted to the months both witnesses have
    (``shared``); ``climatology`` is computed over those months only.
    """

    raw: pd.Series
    anomalies: pd.Series
    time: np.ndarray
    climatology: pd.Series
    n_cells: pd.Series
    quality: QualityResult
    provenance: WitnessProvenance
    shared: pd.DatetimeIndex


def prepare_witness(region: str, var: str, witness_a: pd.Series | None = None) -> PreparedWitness | None:
    """Gate and deseasonalize witness B over the months it shares with witness A.

    ``witness_a`` is the POWER series (``prepare(region, var).raw``);
    it is loaded when not given. Months where either series is NaN are
    dropped from both before the quality gate, the climatology and the
    time axis, so the two witnesses are compared on the same months.
    Returns ``None`` for variables without a witness B.
    """
    out = load_witness_grid(region, var)
    if out is None:
        return None
    series_b, counts, grid = out
    name = _variable_key(var)
    spec = WITNESS_B[name]
    if witness_a is None:
        witness_a = prepare(region, name).raw

    shared = series_b.dropna().index.intersection(witness_a.dropna().index).sort_values()
    raw = series_b.loc[shared]
    anomalies, clim = deseasonalize(raw, return_climatology=True)
    quality = quality_gate(raw)

    manifest = load_witness_manifest()
    entry = _witness_file(region, spec.dataset, manifest)
    provenance = WitnessProvenance(
        dataset=spec.dataset,
        dataset_name=manifest["datasets"][spec.dataset],
        source=manifest["source"],
        url=entry["url"],
        accessed_utc=entry.get("accessed_utc", manifest["accessed_utc"]),  # per-file re-fetch
        independence=spec.independence,
        independence_note=spec.independence_note,
        region=region,
        region_name=load_regions()["regions"].get(region, {}).get("name", region),
        variable=name,
        units=spec.units,
        source_units=grid.source_units,
        grid=grid,
        start=quality.start,
        end=quality.end,
        years=quality.years_covered,
    )
    return PreparedWitness(
        raw=raw,
        anomalies=anomalies,
        time=time_in_years(raw.index),
        climatology=clim,
        n_cells=counts.loc[shared],
        quality=quality,
        provenance=provenance,
        shared=shared,
    )
