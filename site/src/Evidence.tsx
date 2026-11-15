import { useEffect, useId, useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js-dist-min";
import type { CaseFile } from "./data";
import { clerkFacts, confidenceLabel, examinerLine, trendGeometry, unitLabel } from "./format";
import Plot from "./Plot";

type View = "raw" | "anomaly";

/** Chart colours come from the CSS tokens in styles.css, so a theme change restyles the chart too. */
interface ChartTheme {
  series: string;
  trend: string;
  casing: string;
  band: string;
  grid: string;
  zero: string;
  text: string;
  hoverBg: string;
  hoverText: string;
}

function readTheme(): ChartTheme {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
  return {
    series: v("--c-anomaly", "#4B3BFF"),
    trend: v("--c-trend", "#FFC93C"),
    casing: v("--c-trend-casing", "#1B1446"),
    band: v("--c-band", "rgba(255,201,60,.35)"),
    grid: v("--c-grid", "#E6EAF8"),
    zero: v("--line", "#CBD3F2"),
    text: v("--muted", "#5A5F86"),
    hoverBg: v("--surface", "#FFFFFF"),
    hoverText: v("--ink", "#1B1446"),
  };
}

/** Re-read the tokens when the colour scheme changes. */
function useChartTheme(): ChartTheme {
  const [theme, setTheme] = useState(readTheme);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    if (!mq) return;
    const update = () => setTheme(readTheme());
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return theme;
}

/**
 * Evidence panel (#24): the Clerk's header, the series as Raw and Anomalies views,
 * and the Examiner's line. Every number comes from the case JSON.
 */
export default function Evidence({ c }: { c: CaseFile }) {
  const [view, setView] = useState<View>("anomaly");
  const theme = useChartTheme();
  const tabsId = useId();
  const facts = clerkFacts(c);
  const p = c.provenance;
  const ts = c.theil_sen;
  const mk = c.mk_hamed_rao;
  const examiner = ts && mk ? examinerLine(ts, mk, p.units) : null;

  return (
    <div className="evidence">
      <div className="clerk">
        <p className="role">The Clerk</p>
        <p>{p.clerk_text}</p>
        <ul className="facts" aria-label="Source details">
          <li>{facts.source}</li>
          <li>{facts.points}</li>
          <li>accessed {facts.accessed}</li>
        </ul>
      </div>

      <div className="tabs" role="tablist" aria-label="Series view">
        {(
          [
            ["anomaly", "Anomalies"],
            ["raw", "Raw"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            id={`${tabsId}-${id}`}
            aria-selected={view === id}
            aria-controls={`${tabsId}-panel`}
            className="tab"
            onClick={() => setView(id)}
          >
            {label}
          </button>
        ))}
      </div>

      <div role="tabpanel" id={`${tabsId}-panel`} aria-labelledby={`${tabsId}-${view}`} className="tabpanel">
        <p className="explain">
          {view === "raw"
            ? `Monthly ${c.variable.replace(/_/g, " ")} as filed, seasons and all.`
            : "Each month compared with its usual value: the seasonal cycle is removed, so the slow change shows."}
        </p>
        <EvidenceChart c={c} view={view} theme={theme} />
      </div>

      <div className="examiner">
        <p className="role">The Examiner</p>
        {examiner && ts ? (
          <>
            <p className="finding">
              <strong>{examiner.slope}</strong>{" "}
              <span className="nowrap">({examiner.interval})</span>, <span className="nowrap">{examiner.p}</span>
            </p>
            <p className="muted small">
              Theil–Sen slope with its {confidenceLabel(ts)} interval; Mann–Kendall p-value corrected for autocorrelation
              (Hamed–Rao).
            </p>
          </>
        ) : (
          <p className="muted">No trend was tested: {c.quality.reason ?? "the record did not pass the quality check."}</p>
        )}
      </div>
    </div>
  );
}

function EvidenceChart({ c, view, theme }: { c: CaseFile; view: View; theme: ChartTheme }) {
  const u = unitLabel(c.provenance.units);
  const digits = c.provenance.units === "fraction 0-1" ? 3 : 2;
  const valueFmt = `%{y:.${digits}f}${u.inline}`;

  const data = useMemo<Data[]>(() => {
    const s = c.series;
    const traces: Data[] = [];
    const geom = view === "anomaly" && c.theil_sen ? trendGeometry(s, c.theil_sen) : null;

    traces.push({
      x: s.months,
      y: view === "raw" ? s.raw : s.anomaly,
      type: "scatter",
      mode: "lines",
      // Missing months are null in the JSON and stay as gaps.
      connectgaps: false,
      line: { color: theme.series, width: 1.2 },
      opacity: view === "anomaly" ? 0.75 : 1,
      name: view === "raw" ? "Monthly value" : "Anomaly",
      hovertemplate: `%{x|%b %Y}<br>${valueFmt}<extra></extra>`,
    });

    // The band sits above the busy monthly line so it stays visible; the trend line goes on top.
    if (geom?.band) {
      const ci = confidenceLabel(c.theil_sen!);
      traces.push(
        {
          x: geom.band.x,
          y: geom.band.lower,
          type: "scatter",
          mode: "lines",
          line: { color: theme.casing, width: 1, dash: "dot" },
          hoverinfo: "skip",
          showlegend: false,
        },
        {
          x: geom.band.x,
          y: geom.band.upper,
          type: "scatter",
          mode: "lines",
          fill: "tonexty",
          fillcolor: theme.band,
          line: { color: theme.casing, width: 1, dash: "dot" },
          hoverinfo: "skip",
          name: `${ci} slope range`,
        },
      );
    }

    if (geom) {
      // Dark casing under the gold line keeps it visible on light and dark backgrounds.
      traces.push(
        {
          x: geom.line.x,
          y: geom.line.y,
          type: "scatter",
          mode: "lines",
          line: { color: theme.casing, width: 4 },
          hoverinfo: "skip",
          showlegend: false,
        },
        {
          x: geom.line.x,
          y: geom.line.y,
          type: "scatter",
          mode: "lines",
          line: { color: theme.trend, width: 2 },
          hoverinfo: "skip",
          name: "Theil–Sen trend",
        },
      );
    }
    return traces;
  }, [c, view, theme, valueFmt]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      height: 300,
      margin: { l: 56, r: 8, t: view === "anomaly" ? 32 : 8, b: 36 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "DM Mono, ui-monospace, monospace", size: 11, color: theme.text },
      hovermode: "closest",
      hoverlabel: { bgcolor: theme.hoverBg, bordercolor: theme.casing, font: { color: theme.hoverText } },
      showlegend: view === "anomaly",
      legend: { orientation: "h", traceorder: "normal", x: 0, y: 1, yanchor: "bottom", bgcolor: "rgba(0,0,0,0)" },
      xaxis: { type: "date", gridcolor: theme.grid, title: { text: "" }, fixedrange: true },
      yaxis: {
        gridcolor: theme.grid,
        zerolinecolor: theme.zero,
        zeroline: view === "anomaly",
        fixedrange: true,
        title: { text: view === "raw" ? `Monthly value (${u.axis})` : `Anomaly (${u.axis})` },
      },
    }),
    [view, theme, u.axis],
  );

  const what = view === "raw" ? "Monthly" : "Monthly anomaly of";
  return (
    <Plot
      data={data}
      layout={layout}
      label={`${what} ${c.variable.replace(/_/g, " ")}, ${c.provenance.region_name}, ${c.provenance.period}`}
    />
  );
}
