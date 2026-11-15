import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { parseCase, type CaseFile, type Verdict, type VerdictLabel } from "./data";
import { examinerLine } from "./format";
import VerdictCard from "./Verdict";
import { LABELS, NOISE_CAVEAT_TEXT, announcement, cappedText, checkRows, pText, signature } from "./verdict";

const path = (rel: string) => fileURLToPath(new URL(rel, import.meta.url));
const casesDir = path("../public/cases/");
const caseFiles = readdirSync(casesDir).filter((f) => f.endsWith(".json") && f !== "index.json");
const load = (f: string): CaseFile => parseCase(JSON.parse(readFileSync(casesDir + f, "utf8")), f);
const render = (c: CaseFile) => renderToStaticMarkup(createElement(VerdictCard, { c }));

/** React escapes text; undo the few entities it writes so we can search for the raw strings. */
const text = (html: string) =>
  html
    .replace(/<[^>]+>/g, "")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&#x27;/g, "'")
    .replace(/&quot;/g, '"');

function withVerdict(c: CaseFile, patch: Partial<Verdict>): CaseFile {
  return { ...c, verdict: { ...c.verdict, ...patch } };
}

describe("label table", () => {
  it("covers every label of trendcourt/verdict.py", () => {
    const py = readFileSync(path("../../trendcourt/verdict.py"), "utf8");
    const m = py.match(/^LABELS = \(([^)]*)\)/m);
    expect(m).not.toBeNull();
    const labels = [...m![1].matchAll(/"([a-z_]+)"/g)].map((x) => x[1]);
    expect(labels).toHaveLength(6);
    expect(Object.keys(LABELS).sort()).toEqual([...labels].sort());
    // Every label prints a distinct text and gets a distinct style.
    expect(new Set(Object.values(LABELS).map((l) => l.text)).size).toBe(6);
    expect(new Set(Object.values(LABELS).map((l) => l.cls)).size).toBe(6);
  });
});

describe("noise caveat text", () => {
  it("is noise_model_check.card_text in thresholds.json", () => {
    const th = JSON.parse(readFileSync(path("../../trendcourt/thresholds.json"), "utf8"));
    expect(NOISE_CAVEAT_TEXT).toBe(th.noise_model_check.card_text);
  });
});

describe("pText", () => {
  it("bounds p below 1e-6, like trendcourt/verdict.py", () => {
    expect(pText(0)).toBe("p < 1e-6");
    expect(pText(2.4e-7)).toBe("p < 1e-6");
    expect(pText(0.0163)).toBe("p = 0.0163");
    expect(pText(0.41234)).toBe("p = 0.412");
    expect(pText(Number.NaN)).toBe("p unavailable");
  });
});

describe("checks and cap", () => {
  const c = load("iberia_temperature.json");

  it("lists every check with its status, value and threshold", () => {
    const rows = checkRows(c.verdict);
    expect(rows.map((r) => r.name)).toEqual(c.verdict.checks.map((k) => k.name));
    expect(rows[0]).toMatchObject({ status: "pass", mark: "✓", value: "p < 1e-6", threshold: "needs p < 0.01" });
    expect(rows[2]).toMatchObject({
      status: "pending",
      statusText: "pending (coming soon)",
      value: "not measured yet",
      threshold: "needs ≥ 80%",
    });
  });

  it("words failed checks and measured values", () => {
    const v: Verdict = {
      ...c.verdict,
      checks: [
        { name: "hamed_rao_p", status: "fail", value: 0.03, threshold: null },
        { name: "mk_seasonal_consistency", status: "fail", value: false, threshold: null },
        { name: "window_agreement", status: "pass", value: 0.85, threshold: 0.8 },
        { name: "witness_agreement", status: "fail", value: false, threshold: null },
      ],
    };
    const rows = checkRows(v);
    expect(rows.map((r) => r.mark)).toEqual(["✗", "✗", "✓", "✗"]);
    expect(rows[0].threshold).toBe("no p-value counts at this autocorrelation");
    expect(rows[1].value).toBe("different result from the main test");
    expect(rows[2].value).toBe("85% of windows same sign");
    expect(rows[3].value).toBe("not the same direction");
  });

  it("says why a verdict is capped", () => {
    expect(c.verdict.capped).toBe(true);
    expect(cappedText(c.verdict)).toBe("Capped at Moderate until the second dataset and the window check are in.");
    const oneLeft = {
      ...c.verdict,
      checks: c.verdict.checks.map((k) =>
        k.name === "window_agreement" ? { ...k, status: "pass" as const, value: 0.9 } : k,
      ),
    };
    expect(cappedText(oneLeft)).toBe("Capped at Moderate until the second dataset is in.");
    expect(cappedText({ ...c.verdict, capped: false })).toBeNull();
  });

  it("signs as the Council and names the checks behind the verdict", () => {
    expect(signature(c)).toEqual({
      by: "— The Council",
      on: "Ruled on the trend test and the seasonal test. Still pending: the window check and the second dataset.",
    });
  });

  it("announces the label with the case", () => {
    expect(announcement(c, "Temperature")).toBe("Verdict for Temperature, Madrid, Spain (Iberia): Moderate.");
  });
});

describe("verdict card on every cached case", () => {
  it("has all 18 cases", () => {
    expect(caseFiles).toHaveLength(18);
  });

  it.each(caseFiles)("%s: label and numbers match the JSON", (f) => {
    const c = load(f);
    const html = render(c);
    const t = text(html);
    const v = c.verdict;

    expect(html).toContain(`class="card ${LABELS[v.label].cls}"`);
    expect(t).toContain(LABELS[v.label].text);
    expect(t).toContain(v.reason);

    const line = examinerLine(c.theil_sen!, c.mk_hamed_rao!, c.provenance.units);
    expect(t).toContain(line.slope);
    expect(t).toContain(`[${line.interval}]`);
    expect(t).toContain(line.p);

    for (const row of checkRows(v)) {
      expect(t).toContain(`${row.mark} ${row.statusText}${row.title}${row.value} · ${row.threshold}`);
    }
    expect(t.includes("Capped at Moderate")).toBe(v.capped);
    expect(t.includes(NOISE_CAVEAT_TEXT)).toBe(v.caveats.length > 0);
    expect(t).toContain("— The Council");
    expect(t).toContain("Trend, not attribution");
  });
});

describe("insufficient record", () => {
  const base = load("mexicali_temperature.json");
  const c: CaseFile = {
    ...base,
    quality: { ...base.quality, passed: false, reason: "only 12 years of data, 30 are needed", years_covered: 12 },
    theil_sen: null,
    ols: null,
    mk_hamed_rao: null,
    mk_seasonal: null,
    consistency: null,
    autocorrelation: null,
    verdict: {
      label: "insufficient_record",
      checks: base.verdict.checks.map((k) => ({ ...k, status: "pending", value: null, threshold: null })),
      capped: false,
      reason: "The record is too incomplete to test: only 12 years of data, 30 are needed.",
      band_max_verdict: null,
      no_trend_margin: null,
      caveats: [],
    },
  };

  it("shows the quality reason instead of statistics", () => {
    const t = text(render(c));
    expect(t).toContain("Insufficient record");
    expect(t).toContain("only 12 years of data, 30 are needed");
    expect(t).toContain("Years covered12");
    expect(t).not.toContain("Slope");
    expect(t).not.toContain("Corrected p");
    expect(t).not.toContain("The checks");
    expect(t).toContain("Ruled on the Clerk's quality check");
    expect(t).toContain("Trend, not attribution");
  });
});

describe("other labels render", () => {
  const base = load("iberia_temperature.json");
  it.each<VerdictLabel>(["strong_evidence", "witnesses_disagree", "no_detectable_trend"])("%s", (label) => {
    const c = withVerdict(base, { label, capped: false });
    const t = text(render(c));
    expect(t).toContain(LABELS[label].text);
    expect(t.includes("No-trend margin")).toBe(label === "no_detectable_trend");
  });
});

// ---------------------------------------------------------------------------
// Contrast (WCAG 2.1 AA): label text on its badge, and check status text on white.
// ---------------------------------------------------------------------------

const css = readFileSync(path("./styles.css"), "utf8");
const token = (name: string): string => {
  const m = css.match(new RegExp(`--${name}:\\s*(#[0-9A-Fa-f]{6})`));
  if (!m) throw new Error(`token --${name} not found`);
  return m[1];
};
function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

describe("contrast", () => {
  it.each([
    ["v-strong", "v-strong-on"],
    ["v-moderate", "v-moderate-on"],
    ["v-inconclusive", "v-inconclusive-on"],
    ["v-disagree", "v-disagree-on"],
    ["v-none", "v-none-on"],
    ["v-insufficient", "ink"],
  ])("label %s with %s meets AA for text", (bg, fg) => {
    expect(contrast(token(bg), token(fg))).toBeGreaterThanOrEqual(4.5);
  });

  it.each(["pass", "fail", "muted", "indigo"])("--%s on white meets AA for text", (fg) => {
    expect(contrast(token(fg), token("surface"))).toBeGreaterThanOrEqual(4.5);
  });
});
