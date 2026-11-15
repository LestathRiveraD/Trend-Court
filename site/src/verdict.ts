/**
 * Display helpers for the verdict card (#25).
 *
 * Nothing here decides anything. The label, the checks, the cap and the reason
 * come from the `verdict` block that `trendcourt/verdict.py` writes into the case
 * JSON; these helpers only name, word and place them.
 */

import type { CaseFile, Verdict, VerdictCheck, VerdictLabel } from "./data";

export interface LabelInfo {
  /** The label as the card prints it. */
  text: string;
  /** CSS modifier, `v-<id>`, for the label's colour and icon. */
  cls: string;
}

export const LABELS: Record<VerdictLabel, LabelInfo> = {
  strong_evidence: { text: "Strong evidence", cls: "v-strong" },
  moderate: { text: "Moderate", cls: "v-moderate" },
  inconclusive: { text: "Inconclusive", cls: "v-inconclusive" },
  witnesses_disagree: { text: "Witnesses disagree", cls: "v-disagree" },
  no_detectable_trend: { text: "No detectable trend", cls: "v-none" },
  insufficient_record: { text: "Insufficient record", cls: "v-insufficient" },
};

export function labelInfo(label: VerdictLabel): LabelInfo {
  const info = LABELS[label];
  if (!info) throw new Error(`unknown verdict label ${JSON.stringify(label)}`);
  return info;
}

/**
 * `noise_model_check.card_text` in `trendcourt/thresholds.json`, printed on every card
 * whose `verdict.caveats` is non-empty. `verdict.test.ts` checks the two stay equal.
 */
export const NOISE_CAVEAT_TEXT =
  "This record varies on multi-year time scales more than the noise model behind the benchmark assumes, so the stated false-positive rate and interval are optimistic.";

/** Same floor as `P_FLOOR` in `trendcourt/verdict.py`: below it a p-value prints as a bound. */
export const P_FLOOR = 1e-6;

/**
 * Python's `f"{x:.3g}"`: three significant digits, exponent form when the exponent is
 * below -4 or at least 3 (`1.77e-05`), trailing zeros dropped.
 */
export function g3(x: number): string {
  if (!Number.isFinite(x) || x === 0) return String(x);
  const [mant, expStr] = x.toExponential(2).split("e");
  const exp = Number(expStr);
  if (exp < -4 || exp >= 3) {
    const m = mant.replace(/\.?0+$/, "");
    const sign = exp < 0 ? "-" : "+";
    return `${m}e${sign}${String(Math.abs(exp)).padStart(2, "0")}`;
  }
  return String(Number(x.toPrecision(3)));
}

/**
 * A p-value for the checks list, worded like the verdict's reason (`_p_text` in
 * `trendcourt/verdict.py`): "p = 0.016", "p = 1.77e-05", "p < 1e-6".
 */
export function pText(p: number): string {
  if (!Number.isFinite(p)) return "p unavailable";
  if (p < P_FLOOR) return "p < 1e-6";
  return `p = ${g3(p)}`;
}

export type CheckStatus = VerdictCheck["status"];

export interface CheckRow {
  name: string;
  /** What the check asks, in plain words. */
  title: string;
  status: CheckStatus;
  /** "✓", "✗" or "…": shown with `statusText`, never alone. */
  mark: string;
  statusText: string;
  /** The measured value, or "not measured yet" while pending. */
  value: string;
  /** What the value has to meet. */
  threshold: string;
}

const STATUS: Record<CheckStatus, { mark: string; text: string }> = {
  pass: { mark: "✓", text: "passed" },
  fail: { mark: "✗", text: "failed" },
  pending: { mark: "…", text: "pending (coming soon)" },
};

const CHECK_TITLES: Record<string, string> = {
  hamed_rao_p: "Trend test (Mann–Kendall, Hamed–Rao corrected)",
  mk_seasonal_consistency: "Seasonal test agrees",
  window_agreement: "Time windows agree",
  witness_agreement: "Second dataset agrees",
};

/** Short names for the signature and the cap sentence. */
const CHECK_SHORT: Record<string, string> = {
  hamed_rao_p: "the trend test",
  mk_seasonal_consistency: "the seasonal test",
  window_agreement: "the window check",
  witness_agreement: "the second dataset",
};

const pct = (x: number) => `${Math.round(x * 100)}%`;

function checkValue(c: VerdictCheck): string {
  if (c.status === "pending" || c.value === null) return "not measured yet";
  switch (c.name) {
    case "hamed_rao_p":
      return typeof c.value === "number" ? pText(c.value) : String(c.value);
    case "mk_seasonal_consistency":
      return c.value ? "same result as the main test" : "different result from the main test";
    case "window_agreement":
      return typeof c.value === "number" ? `${pct(c.value)} of windows same sign` : String(c.value);
    case "witness_agreement":
      return c.value ? "same direction" : "not the same direction";
    default:
      return String(c.value);
  }
}

function checkThreshold(c: VerdictCheck): string {
  switch (c.name) {
    case "hamed_rao_p":
      return c.threshold === null ? "no p-value counts at this autocorrelation" : `needs p < ${c.threshold}`;
    case "mk_seasonal_consistency":
      return "needs the same sign and significance";
    case "window_agreement":
      return c.threshold === null ? "—" : `needs ≥ ${pct(c.threshold)}`;
    case "witness_agreement":
      return "needs the same direction";
    default:
      return c.threshold === null ? "—" : String(c.threshold);
  }
}

/** One row per entry of `verdict.checks`, in the order the JSON lists them. */
export function checkRows(v: Verdict): CheckRow[] {
  return v.checks.map((c) => ({
    name: c.name,
    title: CHECK_TITLES[c.name] ?? c.name.replace(/_/g, " "),
    status: c.status,
    mark: STATUS[c.status].mark,
    statusText: STATUS[c.status].text,
    value: checkValue(c),
    threshold: checkThreshold(c),
  }));
}

function joinAnd(items: string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

const short = (name: string) => CHECK_SHORT[name] ?? name.replace(/_/g, " ");

/**
 * Why the label is capped: "Capped at Moderate until the second dataset and the window
 * check are in." `null` when `verdict.capped` is false.
 */
export function cappedText(v: Verdict): string | null {
  if (!v.capped) return null;
  // Second dataset first: it is the bigger missing piece.
  const order = ["witness_agreement", "window_agreement"];
  const pending = v.checks
    .filter((c) => c.status === "pending")
    .map((c) => c.name)
    .sort((a, b) => order.indexOf(a) - order.indexOf(b))
    .map(short);
  if (!pending.length) return "Capped at Moderate.";
  return `Capped at Moderate until ${joinAnd(pending)} ${pending.length > 1 ? "are" : "is"} in.`;
}

/**
 * The Council's signature, naming the checks behind the verdict: the ones that ran
 * (passed or failed), and the ones still pending.
 */
export function signature(c: CaseFile): { by: string; on: string } {
  if (!c.quality.passed || c.verdict.label === "insufficient_record") {
    return { by: "— The Council", on: "Ruled on the Clerk's quality check; no trend test was run." };
  }
  const ran = c.verdict.checks.filter((k) => k.status !== "pending").map((k) => short(k.name));
  const pending = c.verdict.checks.filter((k) => k.status === "pending").map((k) => short(k.name));
  let on = ran.length ? `Ruled on ${joinAnd(ran)}.` : "No check has run yet.";
  if (pending.length) on += ` Still pending: ${joinAnd(pending)}.`;
  return { by: "— The Council", on };
}

/** "Verdict for Temperature, Iberia: Moderate." For the screen-reader announcement. */
export function announcement(c: CaseFile, variableName: string): string {
  return `Verdict for ${variableName}, ${c.provenance.region_name}: ${labelInfo(c.verdict.label).text}.`;
}
