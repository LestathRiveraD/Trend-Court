import type { IndexFile, IndexRegion } from "./data";
import { findCase, regions, worries, type Selection } from "./picker";

interface PickerProps {
  index: IndexFile;
  selection: Selection;
  onChange: (sel: Selection) => void;
}

/**
 * Two native radio groups, "What worries you?" then "Where?". Native radios give
 * screen-reader semantics and arrow-key movement inside each group for free; the
 * checked state shows as the radio dot, never as colour alone.
 */
export default function Picker({ index, selection, onChange }: PickerProps) {
  const ws = worries(index);
  const rs = regions(index);
  if (!ws.length || !rs.length) return <p className="muted">index.json lists no cases yet.</p>;

  return (
    <div className="picker">
      <fieldset className="step">
        <legend>
          <span className="step-n" aria-hidden="true">1</span> What worries you?
        </legend>
        <div className="choices worries">
          {ws.map((w) => (
            <label key={w.variable} className="choice">
              <input
                type="radio"
                name="worry"
                value={w.variable}
                checked={selection.variable === w.variable}
                onChange={() => onChange({ ...selection, variable: w.variable })}
                aria-describedby={`worry-def-${w.variable}`}
              />
              <span className="choice-body">
                <span className="choice-title">{w.label}</span>
                <span className="choice-text" id={`worry-def-${w.variable}`}>
                  {w.definition}
                </span>
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      <fieldset className="step">
        <legend>
          <span className="step-n" aria-hidden="true">2</span> Where?
        </legend>
        <div className="choices regions">
          {rs.map((r) => {
            const entry = selection.variable ? findCase(index, r.id, selection.variable) : null;
            const missing = selection.variable !== null && entry === null;
            const insufficient = entry !== null && !entry.quality.passed;
            const noteId = `region-note-${r.id}`;
            return (
              <label key={r.id} className={`choice${missing ? " is-disabled" : ""}`}>
                <input
                  type="radio"
                  name="region"
                  value={r.id}
                  checked={selection.region === r.id}
                  disabled={missing}
                  onChange={() => onChange({ ...selection, region: r.id })}
                  aria-describedby={`region-biome-${r.id}${missing || insufficient ? ` ${noteId}` : ""}`}
                />
                <span className="choice-body">
                  <span className="choice-title">{r.name}</span>
                  <span className="choice-text" id={`region-biome-${r.id}`}>
                    {r.biome}
                  </span>
                  <span className="choice-meta">{centre(r)}</span>
                  {insufficient && (
                    <span className="choice-flag" id={noteId}>
                      Insufficient record
                    </span>
                  )}
                  {missing && (
                    <span className="choice-flag" id={noteId}>
                      No case filed for this worry
                    </span>
                  )}
                </span>
              </label>
            );
          })}
        </div>
      </fieldset>
    </div>
  );
}

/** Box centre as text, e.g. "32.5° N, 115.4° W": a locator without network map tiles. */
function centre(r: IndexRegion): string {
  const [s, w, n, e] = r.box;
  const lat = (s + n) / 2;
  const lon = (w + e) / 2;
  const fmt = (v: number, pos: string, neg: string) => `${Math.abs(v).toFixed(1)}° ${v >= 0 ? pos : neg}`;
  return `${fmt(lat, "N", "S")}, ${fmt(lon, "E", "W")}`;
}
