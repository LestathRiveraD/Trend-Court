import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { parseCase, type Series, type TheilSen } from "./data";
import { examinerLine, formatP, slopeDecimals, trendGeometry, unitLabel } from "./format";

const gate1 = () =>
  parseCase(
    JSON.parse(readFileSync(fileURLToPath(new URL("../../docs/gate1/mexicali_temperature.json", import.meta.url)), "utf8")),
  );

describe("examinerLine", () => {
  it("matches docs/gate1/mexicali_temperature.json once rounded (acceptance, #24)", () => {
    const c = gate1();
    const line = examinerLine(c.theil_sen!, c.mk_hamed_rao!, c.provenance.units);
    // slope_per_decade 0.3999…, interval 0.2974… to 0.5020…, Hamed–Rao p 3.3e-9
    expect(line.text).toBe("+0.40 °C per decade (0.30 to 0.50), p < 0.001");
    expect(Number(line.slope.match(/[\d.]+/)![0])).toBeCloseTo(c.theil_sen!.slope_per_decade, 2);
  });

  it("uses a typographic minus and keeps tiny unitless slopes readable", () => {
    const ts = { slope_per_decade: -0.00142, low_slope_per_decade: -0.00222, high_slope_per_decade: -0.00063 } as TheilSen;
    const mk = { p: 0.0262 } as Parameters<typeof examinerLine>[1];
    expect(examinerLine(ts, mk, "fraction 0-1").text).toBe("−0.0014 per decade on the 0–1 scale (−0.0022 to −0.0006), p = 0.026");
  });

  it("does not print a sign on a slope that rounds to zero", () => {
    const ts = { slope_per_decade: -0.0004, low_slope_per_decade: -0.03, high_slope_per_decade: 0.05 } as TheilSen;
    expect(examinerLine(ts, { p: 0.7 } as never, "mm/day").slope).toBe("0.000 mm/day per decade");
  });
});

describe("formatP", () => {
  it("rounds sensibly", () => {
    expect(formatP(0)).toBe("p < 0.001");
    expect(formatP(3.26e-9)).toBe("p < 0.001");
    expect(formatP(0.00631)).toBe("p = 0.0063");
    expect(formatP(0.0162)).toBe("p = 0.016");
    expect(formatP(0.178)).toBe("p = 0.18");
  });
});

describe("slopeDecimals", () => {
  it("shows two significant figures of the interval width", () => {
    expect(slopeDecimals(0.297, 0.502)).toBe(2);
    expect(slopeDecimals(-0.028, -0.017)).toBe(3);
    expect(slopeDecimals(-0.0022, -0.0006)).toBe(4);
    expect(slopeDecimals(1, 1)).toBe(2);
  });
});

describe("unitLabel", () => {
  it("labels the three variables", () => {
    expect(unitLabel("degC").axis).toBe("°C");
    expect(unitLabel("mm/day").inline).toBe(" mm/day");
    expect(unitLabel("fraction 0-1").axis).toBe("fraction, 0–1");
  });
});

describe("trendGeometry", () => {
  it("draws the Theil–Sen line from slope and intercept at the series times", () => {
    const c = gate1();
    const ts = c.theil_sen!;
    const g = trendGeometry(c.series, ts)!;
    expect(g.line.x).toEqual(["1981-01", "2025-12"]);
    expect(g.line.y[0]).toBeCloseTo(ts.intercept + ts.slope * c.series.time[0]!, 10);
    expect(g.line.y[1]).toBeCloseTo(ts.intercept + ts.slope * c.series.time.at(-1)!, 10);
    // band contains the line at both ends and pinches at the pivot
    const b = g.band!;
    expect(b.lower[0]).toBeLessThan(g.line.y[0]);
    expect(b.upper[0]).toBeGreaterThan(g.line.y[0]);
    expect(b.upper[1]).toBe(b.lower[1]);
    expect(b.upper[2] - b.lower[2]).toBeCloseTo((ts.high_slope - ts.low_slope) * (c.series.time.at(-1)! - c.series.time[269]!), 10);
  });

  it("skips missing months at the ends", () => {
    const series = {
      months: ["2000-01", "2000-02", "2000-03", "2000-04"],
      time: [2000.04, 2000.13, 2000.21, 2000.29],
      raw: [null, 1, 2, null],
      anomaly: [null, 0.1, 0.2, null],
      n_points: [1, 1, 1, 1],
      climatology: [],
    } as Series;
    const g = trendGeometry(series, { slope: 1, intercept: -2000, low_slope: 0.5, high_slope: 1.5 } as TheilSen)!;
    expect(g.line.x).toEqual(["2000-02", "2000-03"]);
  });

  it("omits the band when the bounds are missing", () => {
    const c = gate1();
    const ts = { ...c.theil_sen!, low_slope: Number.NaN };
    expect(trendGeometry(c.series, ts)!.band).toBeNull();
  });
});
