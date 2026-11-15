/**
 * Display helpers for the evidence panel (#24).
 *
 * Nothing here estimates anything. These helpers only round, label and place
 * values that the case JSON already carries (`theil_sen`, `mk_hamed_rao`, `series`).
 */

import type { CaseFile, MannKendall, Series, TheilSen } from "./data";

/** Unit strings as the case JSON writes them, turned into display labels. */
export interface UnitLabel {
  /** Axis title suffix, e.g. "°C". */
  axis: string;
  /** Inline unit after a number. Starts with a space, or is "" if the unit reads better on the axis only. */
  inline: string;
  /** Text after "per decade" for unitless quantities, e.g. " on the 0–1 scale". */
  after: string;
}

export function unitLabel(units: string): UnitLabel {
  switch (units) {
    case "degC":
      return { axis: "°C", inline: " °C", after: "" };
    case "fraction 0-1":
      return { axis: "fraction, 0–1", inline: "", after: " on the 0–1 scale" };
    default:
      return { axis: units, inline: ` ${units}`, after: "" };
  }
}

/** Typographic minus, so negative numbers line up with the plus sign. */
function signed(x: number, decimals: number, plus: boolean): string {
  const s = Math.abs(x).toFixed(decimals);
  // Values that round to zero print without a sign.
  if (Number(s) === 0) return s;
  if (x < 0) return `−${s}`;
  return plus ? `+${s}` : s;
}

/**
 * Decimals for a slope and its interval: enough that the interval width shows
 * two significant figures, so neither the slope nor the bounds collapse together.
 */
export function slopeDecimals(low: number, high: number): number {
  const width = Math.abs(high - low);
  if (!Number.isFinite(width) || width === 0) return 2;
  return Math.min(6, Math.max(0, Math.ceil(-Math.log10(width)) + 1));
}

/** "p < 0.001", "p = 0.012", "p = 0.34". */
export function formatP(p: number): string {
  if (!Number.isFinite(p)) return "p unavailable";
  if (p < 0.001) return "p < 0.001";
  if (p < 0.1) return `p = ${Number(p.toPrecision(2))}`;
  return `p = ${p.toFixed(2)}`;
}

export interface ExaminerLine {
  slope: string;
  interval: string;
  p: string;
  /** Full sentence, e.g. "+0.40 °C per decade (0.30 to 0.50), p < 0.001". */
  text: string;
}

/**
 * The Examiner's one-line finding: Theil–Sen slope per decade, its interval, and
 * the autocorrelation-corrected (Hamed–Rao) Mann–Kendall p-value.
 */
export function examinerLine(ts: TheilSen, mk: MannKendall, units: string): ExaminerLine {
  const u = unitLabel(units);
  const d = slopeDecimals(ts.low_slope_per_decade, ts.high_slope_per_decade);
  const slope = `${signed(ts.slope_per_decade, d, true)}${u.inline} per decade${u.after}`;
  const interval = `${signed(ts.low_slope_per_decade, d, false)} to ${signed(ts.high_slope_per_decade, d, false)}`;
  const p = formatP(mk.p);
  return { slope, interval, p, text: `${slope} (${interval}), ${p}` };
}

/** Confidence as a percentage label, e.g. 0.95 → "95%". */
export function confidenceLabel(ts: TheilSen): string {
  return `${Math.round(ts.confidence * 100)}%`;
}

export interface TrendGeometry {
  /** Theil–Sen line: first and last fitted month. */
  line: { x: [string, string]; y: [number, number] };
  /**
   * Interval band edges through the fitted months [first, pivot, last]: lines with the
   * low and high slopes, pivoting where the Theil–Sen line meets the median fitted time.
   * `null` when the JSON has no interval bounds.
   */
  band: { x: [string, string, string]; upper: [number, number, number]; lower: [number, number, number] } | null;
}

/**
 * Where to draw the Theil–Sen line and its interval band, from `theil_sen.slope`,
 * `theil_sen.intercept` and the bounds, evaluated at the times in `series.time`.
 *
 * The fit uses `y = intercept + slope * t` with `t` in decimal years (scipy's
 * `theilslopes`). With `method="separate"` that line passes through the medians of
 * the fitted times and values, so the band pivots there. The pivot is only where the
 * band is drawn; it is not a new estimate.
 */
export function trendGeometry(series: Series, ts: TheilSen): TrendGeometry | null {
  const fitted: number[] = [];
  for (let i = 0; i < series.months.length; i++) {
    if (series.time[i] != null && series.anomaly[i] != null) fitted.push(i);
  }
  if (fitted.length < 2) return null;
  const first = fitted[0];
  const last = fitted[fitted.length - 1];
  const t0 = series.time[first] as number;
  const t1 = series.time[last] as number;
  const at = (slope: number, intercept: number, t: number) => intercept + slope * t;
  const line = {
    x: [series.months[first], series.months[last]] as [string, string],
    y: [at(ts.slope, ts.intercept, t0), at(ts.slope, ts.intercept, t1)] as [number, number],
  };

  const hasBand = Number.isFinite(ts.low_slope) && Number.isFinite(ts.high_slope);
  if (!hasBand) return { line, band: null };

  // Median fitted time, taken at an actual month so the band has a vertex there.
  const mid = fitted[Math.floor((fitted.length - 1) / 2)];
  const tp = series.time[mid] as number;
  const yp = at(ts.slope, ts.intercept, tp);
  const edge = (s: number, t: number) => yp + s * (t - tp);
  const lo0 = edge(ts.low_slope, t0);
  const hi0 = edge(ts.high_slope, t0);
  const lo1 = edge(ts.low_slope, t1);
  const hi1 = edge(ts.high_slope, t1);
  return {
    line,
    band: {
      x: [series.months[first], series.months[mid], series.months[last]],
      upper: [Math.max(lo0, hi0), yp, Math.max(lo1, hi1)],
      lower: [Math.min(lo0, hi0), yp, Math.min(lo1, hi1)],
    },
  };
}

/** "45 years", "540 months, averaged over 9 points", "accessed 2026-11-14". */
export function clerkFacts(c: CaseFile): { source: string; points: string; accessed: string } {
  const p = c.provenance;
  return {
    source: p.source,
    points: `${p.n_points} grid point${p.n_points === 1 ? "" : "s"}`,
    accessed: p.accessed_utc.slice(0, 10),
  };
}
