import { useEffect, useState } from "react";
import { loadCase, loadIndex, type CaseFile, type IndexCase, type IndexFile } from "./data";
import Evidence from "./Evidence";
import VerdictCard from "./Verdict";
import { announcement } from "./verdict";

type Status<T> = { state: "loading" } | { state: "error"; message: string } | { state: "ready"; value: T };

export default function App() {
  const [index, setIndex] = useState<Status<IndexFile>>({ state: "loading" });
  const [selected, setSelected] = useState<IndexCase | null>(null);
  const [caseFile, setCaseFile] = useState<Status<CaseFile> | null>(null);

  useEffect(() => {
    loadIndex().then(
      (value) => {
        setIndex({ state: "ready", value });
        setSelected(value.cases[0] ?? null);
      },
      (err: Error) => setIndex({ state: "error", message: err.message }),
    );
  }, []);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    setCaseFile({ state: "loading" });
    loadCase(selected).then(
      (value) => !cancelled && setCaseFile({ state: "ready", value }),
      (err: Error) => !cancelled && setCaseFile({ state: "error", message: err.message }),
    );
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const variableName = (id: string) =>
    (index.state === "ready" ? index.value.variables.find((v) => v.id === id)?.name : undefined) ?? id.replace(/_/g, " ");
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
        {index.state === "ready" && (
          <CaseList index={index.value} selected={selected} onSelect={setSelected} />
        )}
      </section>

      <div className="court">
        <section className="panel" aria-labelledby="evidence-h">
          <h2 id="evidence-h">The evidence</h2>
          {caseFile?.state === "loading" && <p className="muted">The Clerk is fetching the file…</p>}
          {caseFile?.state === "error" && <ErrorBanner message={caseFile.message} />}
          {caseFile?.state === "ready" && <Evidence c={caseFile.value} />}
          {!caseFile && index.state === "ready" && <p className="muted">Pick a case to see its evidence.</p>}
        </section>

        <section className="panel" aria-labelledby="verdict-h">
          <h2 id="verdict-h">The verdict</h2>
          {/* Stays mounted across case changes so screen readers announce each new label. */}
          <p className="sr-only" aria-live="polite" aria-atomic="true">
            {caseFile?.state === "ready" ? announcement(caseFile.value, variableName(caseFile.value.variable)) : ""}
          </p>
          {caseFile?.state === "ready" ? <VerdictCard c={caseFile.value} /> : <p className="muted">No case on the stand yet.</p>}
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

/** Placeholder picker: lists the cases in index.json. Replaced by #23. */
function CaseList({ index, selected, onSelect }: { index: IndexFile; selected: IndexCase | null; onSelect: (c: IndexCase) => void }) {
  const region = (id: string) => index.regions.find((r) => r.id === id)?.name ?? id;
  const variable = (id: string) => index.variables.find((v) => v.id === id)?.name ?? id;
  if (!index.cases.length) return <p className="muted">index.json lists no cases yet.</p>;
  return (
    <div className="cases" role="group" aria-label="Cases">
      {index.cases.map((c) => (
        <button
          key={c.file}
          type="button"
          className="case-btn"
          aria-pressed={selected?.file === c.file}
          onClick={() => onSelect(c)}
        >
          <span className="case-var">{variable(c.variable)}</span>
          <span className="case-region">{region(c.region)}</span>
        </button>
      ))}
    </div>
  );
}
