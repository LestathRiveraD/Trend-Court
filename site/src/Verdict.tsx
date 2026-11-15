import { useId } from "react";
import type { CaseFile, VerdictLabel } from "./data";
import { confidenceLabel, examinerLine, unitLabel } from "./format";
import { NOISE_CAVEAT_TEXT, cappedText, checkRows, labelInfo, signature } from "./verdict";

/**
 * Verdict card v0 (#25): the label, one plain sentence, the numbers behind it, the
 * checks list and the Council's signature. It shows `verdict` from the case JSON as
 * it is, pending checks included; it computes nothing.
 */
export default function VerdictCard({ c }: { c: CaseFile }) {
  const v = c.verdict;
  const info = labelInfo(v.label);
  const headingId = useId();
  const insufficient = v.label === "insufficient_record" || !c.quality.passed;
  const sig = signature(c);
  const capped = cappedText(v);

  return (
    <article className={`card ${info.cls}`} aria-labelledby={headingId}>
      <h3 className="label" id={headingId}>
        <LabelIcon label={v.label} />
        <span>{info.text}</span>
      </h3>

      {insufficient ? (
        <>
          <p className="reason">{c.quality.reason ?? v.reason}</p>
          <QualityNumbers c={c} />
        </>
      ) : (
        <>
          <p className="reason">{v.reason}</p>
          <Numbers c={c} />
          {capped && <p className="capped">{capped}</p>}
          <Checks c={c} />
        </>
      )}

      {v.caveats.length > 0 && (
        <p className="caveat" role="note">
          <strong>Caveat.</strong> {NOISE_CAVEAT_TEXT}
        </p>
      )}

      <p className="signature">
        <span className="sig-by">{sig.by}</span>
        <span className="sig-on">{sig.on}</span>
      </p>
      <p className="footnote">Trend, not attribution: this card says whether a trend is there, not what caused it.</p>
    </article>
  );
}

function Numbers({ c }: { c: CaseFile }) {
  const ts = c.theil_sen;
  const mk = c.mk_hamed_rao;
  if (!ts || !mk) return null;
  const line = examinerLine(ts, mk, c.provenance.units);
  const u = unitLabel(c.provenance.units);
  const margin = c.verdict.no_trend_margin;
  return (
    <dl className="numbers">
      <div>
        <dt>Slope</dt>
        <dd>{line.slope}</dd>
      </div>
      <div>
        <dt>{confidenceLabel(ts)} interval</dt>
        <dd>[{line.interval}]</dd>
      </div>
      <div>
        <dt>Corrected p</dt>
        <dd>{line.p}</dd>
      </div>
      {c.verdict.label === "no_detectable_trend" && margin !== null && (
        <div>
          <dt>No-trend margin</dt>
          <dd>
            ±{Number(margin.toPrecision(3))}
            {u.inline} per decade{u.after}
          </dd>
        </div>
      )}
    </dl>
  );
}

function QualityNumbers({ c }: { c: CaseFile }) {
  const q = c.quality;
  return (
    <dl className="numbers">
      <div>
        <dt>Years covered</dt>
        <dd>{q.years_covered}</dd>
      </div>
      <div>
        <dt>Months missing</dt>
        <dd>
          {q.n_missing} of {q.n_months} ({Math.round(q.share_missing * 100)}%)
        </dd>
      </div>
    </dl>
  );
}

function Checks({ c }: { c: CaseFile }) {
  const rows = checkRows(c.verdict);
  const id = useId();
  return (
    <div className="checks">
      <p className="role" id={id}>
        The checks
      </p>
      <ul aria-labelledby={id}>
        {rows.map((r) => (
          <li key={r.name} className={`check s-${r.status}`}>
            <span className="check-status">
              <span aria-hidden="true">{r.mark}</span> {r.statusText}
            </span>
            <span className="check-title">{r.title}</span>
            <span className="check-value">
              {r.value} <span className="muted">· {r.threshold}</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Decorative: the label text always sits beside it, so meaning never rests on the icon or the colour. */
function LabelIcon({ label }: { label: VerdictLabel }) {
  const common = {
    width: 26,
    height: 26,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2.6,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
    focusable: false,
    className: "label-icon",
  };
  switch (label) {
    case "strong_evidence": // double tick
      return (
        <svg {...common}>
          <path d="M2 13l4 4 8-9" />
          <path d="M10 15l2 2 10-11" />
        </svg>
      );
    case "moderate": // single tick
      return (
        <svg {...common}>
          <path d="M4 12.5l5 5L20 6" />
        </svg>
      );
    case "inconclusive": // question mark
      return (
        <svg {...common}>
          <path d="M8.5 9a3.5 3.5 0 1 1 5 3.2c-1 .5-1.5 1.2-1.5 2.3V15" />
          <circle cx="12" cy="19" r="0.6" fill="currentColor" />
        </svg>
      );
    case "witnesses_disagree": // opposing arrows
      return (
        <svg {...common}>
          <path d="M7 20V4M3 8l4-4 4 4" />
          <path d="M17 4v16M13 16l4 4 4-4" />
        </svg>
      );
    case "no_detectable_trend": // flat arrow
      return (
        <svg {...common}>
          <path d="M3 12h17M16 8l4 4-4 4" />
        </svg>
      );
    case "insufficient_record": // broken record
      return (
        <svg {...common}>
          <path d="M3 12h5M16 12h5" />
          <path d="M10 9l4 6M14 9l-4 6" />
        </svg>
      );
  }
}
