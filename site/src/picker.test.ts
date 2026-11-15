import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { parseIndex, type IndexFile } from "./data";
import { findCase, formatHash, parseHash, regions, worries } from "./picker";

const index = (): IndexFile =>
  parseIndex(JSON.parse(readFileSync(fileURLToPath(new URL("../public/cases/index.json", import.meta.url)), "utf8")));

describe("worries", () => {
  it("offers Heat, Rain and Drought / dry soil in the user's words, each with a definition", () => {
    const w = worries(index());
    expect(w.map((x) => [x.variable, x.label])).toEqual([
      ["temperature", "Heat"],
      ["precipitation", "Rain"],
      ["soil_moisture", "Drought / dry soil"],
    ]);
    for (const x of w) expect(x.definition.length).toBeGreaterThan(20);
  });

  it("only offers variables that have a case, and falls back to the index name for unknown ones", () => {
    const idx = index();
    idx.variables.push({ id: "wind", name: "Wind", units: "m/s" }, { id: "sun", name: "Sunlight", units: "W/m2" });
    idx.cases.push({ region: "mexicali", variable: "wind", file: "mexicali_wind.json", verdict: null, quality: { passed: true, verdict: "" } });
    const w = worries(idx);
    expect(w.map((x) => x.variable)).toEqual(["temperature", "precipitation", "soil_moisture", "wind"]);
    expect(w[3]).toMatchObject({ label: "Wind", definition: "Wind (m/s)." });
  });
});

describe("regions", () => {
  it("offers the six preset regions with name and biome", () => {
    const r = regions(index());
    expect(r).toHaveLength(6);
    for (const x of r) {
      expect(x.name).toBeTruthy();
      expect(x.biome).toBeTruthy();
    }
  });
});

describe("hash", () => {
  it("round-trips every one of the 18 combinations", () => {
    const idx = index();
    let n = 0;
    for (const r of regions(idx)) {
      for (const w of worries(idx)) {
        const sel = { region: r.id, variable: w.variable };
        const res = parseHash(formatHash(sel), idx);
        expect(res).toEqual({ kind: "case", selection: sel, entry: findCase(idx, r.id, w.variable) });
        n++;
      }
    }
    expect(n).toBe(18);
  });

  it("formats the example deep link", () => {
    expect(formatHash({ region: "mexicali", variable: "temperature" })).toBe("#/mexicali/temperature");
    expect(formatHash({ region: "mexicali", variable: null })).toBe("");
  });

  it("treats an empty hash as no selection and tolerates a trailing slash", () => {
    const idx = index();
    expect(parseHash("", idx)).toEqual({ kind: "empty" });
    expect(parseHash("#/", idx)).toEqual({ kind: "empty" });
    expect(parseHash("#/sahel/precipitation/", idx).kind).toBe("case");
  });

  it("reports hashes that name no case instead of guessing", () => {
    const idx = index();
    for (const h of ["#/atlantis/temperature", "#/mexicali/wind", "#/mexicali", "#/a/b/c", "#/%E0%A4%A/x"]) {
      expect(parseHash(h, idx)).toEqual({ kind: "unknown", hash: h });
    }
  });
});
