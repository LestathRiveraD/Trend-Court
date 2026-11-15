import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { renderServiceWorker } from "./precache";

const template = readFileSync(new URL("./sw-template.js", import.meta.url), "utf8");
const file = (path: string, body = path) => ({ path, content: Buffer.from(body) });

describe("renderServiceWorker", () => {
  const files = [file("index.html"), file("cases/index.json"), file("assets/plotly.min-x.js"), file("sw.js")];

  it("precaches every built file except itself, in a stable order", () => {
    const sw = renderServiceWorker(files, template);
    expect(sw).toContain('const PRECACHE = ["assets/plotly.min-x.js","cases/index.json","index.html"];');
    expect(sw).not.toContain("__PRECACHE__");
    expect(sw).not.toContain("__VERSION__");
  });

  it("is deterministic and changes version when any file's content changes", () => {
    const a = renderServiceWorker(files, template);
    expect(renderServiceWorker([...files].reverse(), template)).toBe(a);
    const changed = files.map((f) => (f.path === "cases/index.json" ? file(f.path, "new data") : f));
    const version = (s: string) => s.match(/const VERSION = "([0-9a-f]+)"/)![1];
    expect(version(renderServiceWorker(changed, template))).not.toBe(version(a));
  });
});
