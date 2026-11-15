import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  CASE_SCHEMA_VERSION,
  CaseDataError,
  SchemaMismatchError,
  caseUrl,
  loadCase,
  loadIndex,
  parseCase,
  parseIndex,
} from "./data";

const cases = (f: string) => fileURLToPath(new URL(`../public/cases/${f}`, import.meta.url));
const read = (f: string) => JSON.parse(readFileSync(cases(f), "utf8"));
const gate1 = () =>
  JSON.parse(readFileSync(fileURLToPath(new URL("../../docs/gate1/mexicali_temperature.json", import.meta.url)), "utf8"));

/** A fake fetch that serves the committed files in public/cases/ under the Pages base. */
function servePublic(overrides: Record<string, unknown> = {}) {
  const urls: string[] = [];
  const fetchFn = async (url: string) => {
    urls.push(url);
    const file = url.replace(/^\/Trend-Court\/cases\//, "");
    if (file in overrides) return { ok: true, status: 200, statusText: "OK", json: async () => overrides[file] };
    try {
      const body = read(file);
      return { ok: true, status: 200, statusText: "OK", json: async () => body };
    } catch {
      return { ok: false, status: 404, statusText: "Not Found", json: async () => ({}) };
    }
  };
  return { fetchFn, urls };
}

describe("caseUrl", () => {
  it("resolves under the Vite base", () => {
    expect(caseUrl("index.json", "/Trend-Court/")).toBe("/Trend-Court/cases/index.json");
    expect(caseUrl("a.json", "/")).toBe("/cases/a.json");
    expect(caseUrl("a.json", "/sub")).toBe("/sub/cases/a.json");
  });
  it("uses the configured GitHub Pages base by default", () => {
    expect(import.meta.env.BASE_URL).toBe("/Trend-Court/");
    expect(caseUrl("index.json")).toBe("/Trend-Court/cases/index.json");
  });
});

describe("committed stand-in data", () => {
  it("the stand-in case is the Gate 1 (a) reference file, unchanged", () => {
    expect(read("mexicali_temperature.json")).toEqual(gate1());
  });

  it("every case in index.json loads and matches its entry", async () => {
    const { fetchFn } = servePublic();
    const index = await loadIndex(fetchFn);
    expect(index.cases.length).toBeGreaterThan(0);
    for (const entry of index.cases) {
      const c = await loadCase(entry, fetchFn);
      expect(c.schema_version).toBe(CASE_SCHEMA_VERSION);
      expect(c.quality.passed).toBe(entry.quality.passed);
    }
  });
});

describe("parseCase", () => {
  it("accepts the reference case and keeps every number", () => {
    const raw = gate1();
    const c = parseCase(raw);
    expect(c.theil_sen?.slope_per_decade).toBe(raw.theil_sen.slope_per_decade);
    expect(c.series.anomaly).toHaveLength(540);
  });

  it.each([0, 2, "1", undefined])("rejects schema_version %j loudly", (v) => {
    expect(() => parseCase({ ...gate1(), schema_version: v })).toThrow(SchemaMismatchError);
  });

  it("rejects a missing block", () => {
    const { mk_seasonal: _, ...rest } = gate1();
    expect(() => parseCase(rest)).toThrow(/missing mk_seasonal/);
  });

  it("rejects ragged series", () => {
    const raw = gate1();
    raw.series.anomaly.pop();
    expect(() => parseCase(raw)).toThrow(/series.anomaly/);
  });

  it("rejects a passed record without statistics", () => {
    expect(() => parseCase({ ...gate1(), theil_sen: null })).toThrow(/theil_sen/);
  });

  it("accepts a failed-quality record whose statistics are null", () => {
    const raw = {
      ...gate1(),
      quality: { ...gate1().quality, passed: false, reason: "too short" },
      theil_sen: null,
      ols: null,
      mk_hamed_rao: null,
      mk_seasonal: null,
      consistency: null,
    };
    expect(parseCase(raw).quality.reason).toBe("too short");
  });

  it("rejects non-objects", () => {
    expect(() => parseCase(null)).toThrow(CaseDataError);
    expect(() => parseCase([])).toThrow(CaseDataError);
  });
});

describe("parseIndex", () => {
  it("rejects a schema mismatch", () => {
    expect(() => parseIndex({ ...read("index.json"), schema_version: 99 })).toThrow(SchemaMismatchError);
  });

  it.each(["../secret.json", "a/b.json", "x.js", ""])("rejects unsafe file name %j", (file) => {
    const idx = read("index.json");
    idx.cases[0].file = file;
    expect(() => parseIndex(idx)).toThrow(/bare \*\.json/);
  });

  it("rejects a case naming an unknown region", () => {
    const idx = read("index.json");
    idx.cases[0].region = "atlantis";
    expect(() => parseIndex(idx)).toThrow(/unknown region/);
  });
});

describe("loaders", () => {
  it("fetch under the Pages base", async () => {
    const { fetchFn, urls } = servePublic();
    const index = await loadIndex(fetchFn);
    await loadCase(index.cases[0], fetchFn);
    expect(urls).toEqual(["/Trend-Court/cases/index.json", "/Trend-Court/cases/mexicali_temperature.json"]);
  });

  it("report HTTP errors with the URL", async () => {
    const { fetchFn } = servePublic();
    await expect(loadCase({ file: "nope.json", region: "x", variable: "y" }, fetchFn)).rejects.toThrow(
      /\/Trend-Court\/cases\/nope.json: HTTP 404/,
    );
  });

  it("report network failures", async () => {
    const fetchFn = async () => {
      throw new TypeError("Failed to fetch");
    };
    await expect(loadIndex(fetchFn)).rejects.toThrow(/could not fetch .*Failed to fetch/);
  });

  it("reject a case file that is not the case the index names", async () => {
    const { fetchFn } = servePublic({ "mexicali_temperature.json": { ...gate1(), region: "sahel" } });
    await expect(
      loadCase({ file: "mexicali_temperature.json", region: "mexicali", variable: "temperature" }, fetchFn),
    ).rejects.toThrow(/holds sahel\/temperature/);
  });

  it("refuse unsafe file names before fetching", async () => {
    const { fetchFn, urls } = servePublic();
    await expect(loadCase({ file: "../index.json", region: "a", variable: "b" }, fetchFn)).rejects.toThrow(/refusing/);
    expect(urls).toEqual([]);
  });
});
