import { useEffect, useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js-dist-min";
import { loadCase, loadIndex, type CaseFile, type IndexFile } from "./data";
import Picker from "./Picker";
import { EMPTY, findCase, formatHash, parseHash, regions, worries, type Selection } from "./picker";
import Plot from "./Plot";

type Status<T> = { state: "loading" } | { state: "error"; message: string } | { state: "ready"; value: T };

export default function App() {
  const [index, setIndex] = useState<Status<IndexFile>>({ state: "loading" });
  const [selection, setSelection] = useState<Selection>(EMPTY);
  const [badHash, setBadHash] = useState<string | null>(null);
  const [caseFile, setCaseFile] = useState<Status<CaseFile> | null>(null);

  useEffect(() => {
    loadIndex().then(
      (value) => setIndex({ state: "ready", value }),
      (err: Error) => setIndex({ state: "error", message: err.message }),
    );
  }, []);

  // The URL hash (#/<region>/<variable>) is the source of truth for a complete selection,
  // so a reload or a deep link from the video reopens the same case.
  useEffect(() => {
    if (index.state !== "ready") return;
    const sync = () => {
      const res = parseHash(window.location.hash, index.value);
      if (res.kind === "case") {
        setSelection(res.selection);
        setBadHash(null);
      } else if (res.kind === "empty") {
        setSelection(EMPTY);
        setBadHash(null);
      } else {
        setBadHash(res.hash);
      }
    };
    sync();
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, [index]);

  const onPick = (sel: Selection) => {
    setSelection(sel);
    setBadHash(null);
    const hash = formatHash(sel);
    if (hash && hash !== window.location.hash) window.location.hash = hash;
  };

  const selected = index.state === "ready" ? findCase(index.value, selection.region, selection.variable) : null;

  useEffect(() => {
    if (!selected) {
      setCaseFile(null);
      return;
    }
    let cancelled = false;
    setCaseFile({ state: "loading" });
    loadCase(selected).then(
      (value) => !cancelled && setCaseFile({ state: "ready", value }),
      (err: Error) => !cancelled && setCaseFile({ state: "error", message: err.message }),
    );
    return () => {
      cancelled = true;
    };
    // Reload only when the case changes, not when a new index object names the same file.
  }, [selected?.file]);

  const accessed = index.state === "ready" ? index.value.build.accessed.join(", ") : null;

  return (
    <div className="wrap">
      <header className="hero">
        <span className="tag">NASA Space Apps 2026 · Trend Detective</span>
        <h1>Trend Court</h1>
        <p className="lede">Put a climate trend on trial. The Clerk files the evidence, the Examiner tests it, and the Council rules.</p>
        <p className="notice" role="note">
          <strong>Cached NASA data.</strong> Every case was computed ahead of time from NASA data
          {accessed ? ` accessed ${accessed}` : ""}; nothing on this page calls a live service.
        </p>
      </header>

      {index.state === "error" && <ErrorBanner message={index.message} />}

      <section className="slot" aria-labelledby="picker-h">
        <h2 id="picker-h">Pick a case</h2>
        {index.state === "loading" && <p className="muted">Loading the docket…</p>}
        {badHash && (
          <p className="error" role="alert">
            The link <code>{badHash}</code> names no case on the docket. Pick one below.
          </p>
        )}
        {index.state === "ready" && (
          <>
            <Picker index={index.value} selection={selection} onChange={onPick} />
            <OnTheStand index={index.value} selection={selection} />
          </>
        )}
      </section>

      <div className="court">
        <section className="panel" aria-labelledby="evidence-h">
          <h2 id="evidence-h">The evidence</h2>
          {caseFile?.state === "loading" && <p className="muted">The Clerk is fetching the file…</p>}
          {caseFile?.state === "error" && <ErrorBanner message={caseFile.message} />}
          {caseFile?.state === "ready" && <Evidence c={caseFile.value} />}
          {!caseFile && index.state === "ready" && <p className="muted">Pick a worry and a place to see the evidence.</p>}
        </section>

        <section className="panel" aria-labelledby="verdict-h">
          <h2 id="verdict-h">The verdict</h2>
          {caseFile?.state === "ready" ? <VerdictSlot c={caseFile.value} /> : <p className="muted">No case on the stand yet.</p>}
        </section>
      </div>

      <footer>Trend, not attribution. Trend Court tests whether a trend is there, not what caused it.</footer>
    </div>
  );
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="error" role="alert">
      <strong>The court could not open this file.</strong> {message}
    </div>
  );
}

/** Says which case is on the stand, or what is still to pick. Polite live region for screen readers. */
function OnTheStand({ index, selection }: { index: IndexFile; selection: Selection }) {
  const worry = worries(index).find((w) => w.variable === selection.variable);
  const region = regions(index).find((r) => r.id === selection.region);
  let text: string;
  if (worry && region) text = `On the stand: ${worry.label} in ${region.name}.`;
  else if (worry) text = `${worry.label} it is. Now pick where.`;
  else if (region) text = `${region.name} it is. Now pick what worries you.`;
  else text = "Pick what worries you and where, in either order.";
  return (
    <p className="stand" aria-live="polite">
      {text}
    </p>
  );
}

/** Placeholder evidence panel: Clerk text and the anomaly series. Replaced by #24. */
function Evidence({ c }: { c: CaseFile }) {
  const data = useMemo<Data[]>(
    () => [
      {
        x: c.series.months,
        y: c.series.anomaly,
        type: "scatter",
        mode: "lines",
        connectgaps: false,
        line: { color: "#4B3BFF", width: 1.5 },
        hovertemplate: `%{x}: %{y:.2f} ${c.provenance.units}<extra></extra>`,
      },
    ],
    [c],
  );
  const layout = useMemo<Partial<Layout>>(
    () => ({
      height: 280,
      margin: { l: 48, r: 12, t: 8, b: 32 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "DM Mono, ui-monospace, monospace", size: 11, color: "#5A5F86" },
      xaxis: { gridcolor: "#E6EAF8" },
      yaxis: { gridcolor: "#E6EAF8", zerolinecolor: "#CBD3F2", title: { text: `Anomaly (${c.provenance.units})` } },
    }),
    [c],
  );
  return (
    <>
      <p className="clerk">{c.provenance.clerk_text}</p>
      <Plot data={data} layout={layout} label={`Monthly ${c.variable} anomaly, ${c.provenance.region_name}, ${c.provenance.period}`} />
    </>
  );
}

/** Placeholder verdict slot. Replaced by the verdict card (#25). */
function VerdictSlot({ c }: { c: CaseFile }) {
  if (!c.quality.passed) {
    return (
      <p>
        <span className="pill p-insufficient">Insufficient record</span> {c.quality.reason}
      </p>
    );
  }
  if (!c.verdict) {
    return <p className="muted">The verdict card arrives with the verdict module (#20) and card (#25).</p>;
  }
  return (
    <p>
      <span className="pill">{c.verdict.label.replace(/_/g, " ")}</span> {c.verdict.reason}
    </p>
  );
}
