/**
 * Typed loaders for the precomputed case JSON (`project.md` §4).
 *
 * The site computes no statistics: every number it shows comes from these files.
 * The types mirror what `trendcourt/case.py` writes (`case.examine()` and
 * `case.to_json()`); `index.json` follows the contract in issue #21.
 *
 * Loaders fail loudly. A file whose `schema_version` is not the one this build
 * was written for throws `SchemaMismatchError` rather than rendering numbers
 * from a layout the UI does not understand.
 */

/** Must equal `SCHEMA_VERSION` in `trendcourt/case.py`. Bump both together. */
export const CASE_SCHEMA_VERSION = 2;
/** Version of the `index.json` layout written by the cache build (#21). */
export const INDEX_SCHEMA_VERSION = 1;

// ---------------------------------------------------------------------------
// Case JSON (`site/public/cases/<region>_<variable>.json`)
// ---------------------------------------------------------------------------

export interface Provenance {
  source: string;
  api: string;
  region_name: string;
  power_code: string;
  longname: string;
  units: string;
  source_units: string;
  n_points: number;
  period: string;
  start: string | null;
  end: string | null;
  years: number;
  accessed_utc: string;
  fill_value: number;
  urls: string[];
  clerk_text: string;
}

export interface Quality {
  passed: boolean;
  verdict: string;
  reason: string | null;
  years_covered: number;
  share_missing: number;
  start: string | null;
  end: string | null;
  n_months: number;
  n_missing: number;
}

/** Monthly series. Missing months are `null` and must be drawn as gaps. */
export interface Series {
  months: string[];
  time: (number | null)[];
  raw: (number | null)[];
  anomaly: (number | null)[];
  n_points: number[];
  climatology: (number | null)[];
}

export interface TheilSen {
  slope: number;
  intercept: number;
  low_slope: number;
  high_slope: number;
  slope_per_decade: number;
  low_slope_per_decade: number;
  high_slope_per_decade: number;
  confidence: number;
  method: string;
  n: number;
  t_start: number;
  t_end: number;
}

export interface Ols {
  slope: number;
  intercept: number;
  stderr: number;
  intercept_stderr: number;
  slope_per_decade: number;
  stderr_per_decade: number;
  r: number;
  p: number;
  n: number;
  t_start: number;
  t_end: number;
}

export interface MannKendall {
  test: string;
  trend: string;
  h: boolean;
  p: number;
  z: number;
  Tau: number;
  s: number;
  var_s: number;
  slope: number;
  intercept: number;
  n: number;
  alpha: number;
  lag: number | null;
  period: number | null;
}

export interface Consistency {
  agree: boolean;
  alpha: number;
  mk_sign: number;
  mk_significant: boolean;
  smk_sign: number;
  smk_significant: boolean;
  same_sign: boolean;
  same_significance: boolean;
}

/** Verdict labels of `trendcourt/verdict.py` (`LABELS`). */
export type VerdictLabel =
  | "strong_evidence"
  | "moderate"
  | "inconclusive"
  | "witnesses_disagree"
  | "no_detectable_trend"
  | "insufficient_record";

export interface VerdictCheck {
  name: string;
  status: "pass" | "fail" | "pending";
  value: number | boolean | null;
  threshold: number | null;
}

export interface Verdict {
  label: VerdictLabel;
  /** hamed_rao_p, mk_seasonal_consistency, window_agreement, witness_agreement, in that order. */
  checks: VerdictCheck[];
  /** True when only pending checks (windows, witness B) stopped a Strong label. */
  capped: boolean;
  /** One plain sentence for the card. */
  reason: string;
  /** `max_verdict` of the record's autocorrelation band; `null` above the benchmark grid or when the record failed the quality gate. */
  band_max_verdict: "strong" | "moderate" | "inconclusive" | null;
  /** No-trend margin per decade, in the variable's units; `null` when the record failed the quality gate. */
  no_trend_margin: number | null;
  /** `noise_caveats` flags; non-empty means the card prints `noise_model_check.card_text`. */
  caveats: ("fpr_optimistic" | "interval_optimistic")[];
}

export interface Autocorrelation {
  /** Lag-1 autocorrelation of the anomalies after removing the Theil–Sen line. */
  r1_detrended: number;
}

export interface CaseFile {
  schema_version: number;
  region: string;
  variable: string;
  provenance: Provenance;
  quality: Quality;
  series: Series;
  /** The statistics blocks are `null` when the record fails the quality gate. */
  theil_sen: TheilSen | null;
  ols: Ols | null;
  mk_hamed_rao: MannKendall | null;
  mk_seasonal: MannKendall | null;
  consistency: Consistency | null;
  /** `null` when the record fails the quality gate. */
  autocorrelation: Autocorrelation | null;
  verdict: Verdict;
}

// ---------------------------------------------------------------------------
// index.json (`site/public/cases/index.json`)
// ---------------------------------------------------------------------------

export interface IndexRegion {
  id: string;
  name: string;
  biome: string;
  /** [south, west, north, east] in degrees, as in `data/regions.json`. */
  box: [number, number, number, number];
}

export interface IndexVariable {
  id: string;
  name: string;
  units: string;
}

export interface IndexCase {
  region: string;
  variable: string;
  /** Bare file name inside `cases/`, e.g. `mexicali_temperature.json`. */
  file: string;
  /** Verdict label (`verdict.label` of the case); `null` only in a cache built before the verdict module (#20). */
  verdict: VerdictLabel | null;
  quality: { passed: boolean; verdict: string };
}

export interface IndexFile {
  schema_version: number;
  /** Git commit of the cache build, and the NASA data access dates (no wall clock). */
  build: { commit: string; accessed: string[] };
  regions: IndexRegion[];
  variables: IndexVariable[];
  cases: IndexCase[];
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export class SchemaMismatchError extends Error {
  constructor(what: string, found: unknown, expected: number) {
    super(
      `${what}: schema_version is ${JSON.stringify(found)}, but this site build expects ${expected}. ` +
        "Rebuild the cache and the site from the same commit.",
    );
    this.name = "SchemaMismatchError";
  }
}

export class CaseDataError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CaseDataError";
  }
}

type Json = Record<string, unknown>;

function isObject(v: unknown): v is Json {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function need(obj: Json, keys: readonly string[], what: string): void {
  const missing = keys.filter((k) => !(k in obj));
  if (missing.length) throw new CaseDataError(`${what}: missing ${missing.join(", ")}`);
}

function checkVersion(obj: Json, expected: number, what: string): void {
  if (obj.schema_version !== expected) throw new SchemaMismatchError(what, obj.schema_version, expected);
}

const CASE_KEYS = [
  "region",
  "variable",
  "provenance",
  "quality",
  "series",
  "theil_sen",
  "ols",
  "mk_hamed_rao",
  "mk_seasonal",
  "consistency",
  "autocorrelation",
  "verdict",
] as const;
const SERIES_KEYS = ["months", "time", "raw", "anomaly", "n_points", "climatology"] as const;
const SAFE_FILE = /^[a-z0-9_]+\.json$/;

/** Check a parsed case file and return it typed. Throws on any mismatch. */
export function parseCase(raw: unknown, what = "case file"): CaseFile {
  if (!isObject(raw)) throw new CaseDataError(`${what}: not a JSON object`);
  checkVersion(raw, CASE_SCHEMA_VERSION, what);
  need(raw, CASE_KEYS, what);
  for (const block of ["provenance", "quality", "series"] as const) {
    if (!isObject(raw[block])) throw new CaseDataError(`${what}: ${block} is not an object`);
  }
  const series = raw.series as Json;
  need(series, SERIES_KEYS, `${what} series`);
  const n = (series.months as unknown[]).length;
  for (const k of ["months", "time", "raw", "anomaly", "n_points"] as const) {
    const arr = series[k];
    if (!Array.isArray(arr) || arr.length !== n) {
      throw new CaseDataError(`${what}: series.${k} must be an array of length ${n} (same as series.months)`);
    }
  }
  if (!isObject(raw.verdict)) throw new CaseDataError(`${what}: verdict is not an object`);
  const quality = raw.quality as Json;
  if (quality.passed === true) {
    for (const block of ["theil_sen", "mk_hamed_rao", "mk_seasonal", "consistency", "autocorrelation"] as const) {
      if (!isObject(raw[block])) throw new CaseDataError(`${what}: quality passed but ${block} is missing`);
    }
  }
  return raw as unknown as CaseFile;
}

/** Check a parsed `index.json` and return it typed. Throws on any mismatch. */
export function parseIndex(raw: unknown, what = "cases/index.json"): IndexFile {
  if (!isObject(raw)) throw new CaseDataError(`${what}: not a JSON object`);
  checkVersion(raw, INDEX_SCHEMA_VERSION, what);
  need(raw, ["build", "regions", "variables", "cases"], what);
  for (const k of ["regions", "variables", "cases"] as const) {
    if (!Array.isArray(raw[k])) throw new CaseDataError(`${what}: ${k} is not an array`);
  }
  const regions = new Set((raw.regions as Json[]).map((r) => r.id));
  const variables = new Set((raw.variables as Json[]).map((v) => v.id));
  for (const c of raw.cases as Json[]) {
    if (typeof c.file !== "string" || !SAFE_FILE.test(c.file)) {
      throw new CaseDataError(`${what}: case file name ${JSON.stringify(c.file)} is not a bare *.json name`);
    }
    if (!regions.has(c.region)) throw new CaseDataError(`${what}: ${c.file} names unknown region ${JSON.stringify(c.region)}`);
    if (!variables.has(c.variable)) {
      throw new CaseDataError(`${what}: ${c.file} names unknown variable ${JSON.stringify(c.variable)}`);
    }
  }
  return raw as unknown as IndexFile;
}

// ---------------------------------------------------------------------------
// Loaders
// ---------------------------------------------------------------------------

type Fetch = (url: string) => Promise<Pick<Response, "ok" | "status" | "statusText" | "json">>;

/**
 * URL of a file in `public/cases/`, relative to the Vite base, so the same
 * build works in dev, in preview and under the GitHub Pages subpath.
 */
export function caseUrl(file: string, base: string = import.meta.env.BASE_URL): string {
  return `${base.endsWith("/") ? base : `${base}/`}cases/${file}`;
}

async function getJson(url: string, fetchFn: Fetch): Promise<unknown> {
  let res;
  try {
    res = await fetchFn(url);
  } catch (err) {
    throw new CaseDataError(`could not fetch ${url}: ${(err as Error).message}`);
  }
  if (!res.ok) throw new CaseDataError(`could not fetch ${url}: HTTP ${res.status} ${res.statusText}`);
  try {
    return await res.json();
  } catch (err) {
    throw new CaseDataError(`${url} is not valid JSON: ${(err as Error).message}`);
  }
}

export async function loadIndex(fetchFn: Fetch = fetch): Promise<IndexFile> {
  const url = caseUrl("index.json");
  return parseIndex(await getJson(url, fetchFn), url);
}

/** Load the case an index entry points at, and check it is the case the entry names. */
export async function loadCase(
  entry: Pick<IndexCase, "file" | "region" | "variable">,
  fetchFn: Fetch = fetch,
): Promise<CaseFile> {
  if (!SAFE_FILE.test(entry.file)) throw new CaseDataError(`refusing to load ${JSON.stringify(entry.file)}`);
  const url = caseUrl(entry.file);
  const c = parseCase(await getJson(url, fetchFn), url);
  if (c.region !== entry.region || c.variable !== entry.variable) {
    throw new CaseDataError(
      `${url} holds ${c.region}/${c.variable}, but index.json lists it as ${entry.region}/${entry.variable}`,
    );
  }
  return c;
}
