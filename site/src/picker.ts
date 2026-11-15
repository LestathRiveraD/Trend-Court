/**
 * Case picker logic (issue #23): the "What worries you?" copy, the URL hash and
 * the lookup from a (region, variable) pair to its entry in `index.json`.
 *
 * Everything the picker offers comes from `index.json`, so it never offers a
 * case that does not exist. Kept free of React so it can be unit tested.
 */
import type { IndexCase, IndexFile, IndexRegion, IndexVariable } from "./data";

/** A worry in the user's words, for one `index.json` variable. Draft copy, reviewed in Phase 4. */
export interface Worry {
  variable: string;
  label: string;
  definition: string;
}

/** Plain-language copy per variable id. A variable missing here falls back to its index name. */
const WORRY_COPY: Record<string, { label: string; definition: string }> = {
  temperature: {
    label: "Heat",
    definition: "Is it getting hotter? The monthly average air temperature, measured 2 m above the ground.",
  },
  precipitation: {
    label: "Rain",
    definition: "Is it raining more, or less? How much rain and snow falls on an average day, month by month.",
  },
  soil_moisture: {
    label: "Drought / dry soil",
    definition: "Is the ground drying out? How wet the soil is where plant roots drink, from 0 (bone dry) to 1 (soaked).",
  },
};

/** The worries to offer, in `index.json` order, one per variable that has at least one case. */
export function worries(index: IndexFile): Worry[] {
  const filed = new Set(index.cases.map((c) => c.variable));
  return index.variables
    .filter((v) => filed.has(v.id))
    .map((v: IndexVariable) => ({
      variable: v.id,
      label: WORRY_COPY[v.id]?.label ?? v.name,
      definition: WORRY_COPY[v.id]?.definition ?? `${v.name} (${v.units}).`,
    }));
}

/** The regions to offer, in `index.json` order, one per region that has at least one case. */
export function regions(index: IndexFile): IndexRegion[] {
  const filed = new Set(index.cases.map((c) => c.region));
  return index.regions.filter((r) => filed.has(r.id));
}

export function findCase(index: IndexFile, region: string | null, variable: string | null): IndexCase | null {
  if (!region || !variable) return null;
  return index.cases.find((c) => c.region === region && c.variable === variable) ?? null;
}

export interface Selection {
  region: string | null;
  variable: string | null;
}

export const EMPTY: Selection = { region: null, variable: null };

/** `#/<region>/<variable>`, the deep link to one case. Empty when the selection is incomplete. */
export function formatHash(sel: Selection): string {
  return sel.region && sel.variable ? `#/${encodeURIComponent(sel.region)}/${encodeURIComponent(sel.variable)}` : "";
}

export type HashResult =
  | { kind: "empty" }
  | { kind: "case"; selection: Selection; entry: IndexCase }
  | { kind: "unknown"; hash: string };

/**
 * Read `#/<region>/<variable>` against the index. A hash that names no filed
 * case is reported as `unknown` (the caller shows a note) rather than guessed at.
 */
export function parseHash(hash: string, index: IndexFile): HashResult {
  const body = hash.replace(/^#\/?/, "").replace(/\/+$/, "");
  if (!body) return { kind: "empty" };
  const parts = body.split("/");
  if (parts.length === 2) {
    let region: string, variable: string;
    try {
      [region, variable] = parts.map(decodeURIComponent) as [string, string];
    } catch {
      return { kind: "unknown", hash };
    }
    const entry = findCase(index, region, variable);
    if (entry) return { kind: "case", selection: { region, variable }, entry };
  }
  return { kind: "unknown", hash };
}
