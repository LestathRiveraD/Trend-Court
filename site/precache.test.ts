import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
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

describe("service worker fetch handler", () => {
  const scope = "https://example.test/Trend-Court/";
  const cached = ["index.html", "cases/index.json", "assets/app.js"];

  /** Run the rendered worker against a fake cache holding `cached`; returns what a request resolves to. */
  function worker() {
    const listeners: Record<string, (e: unknown) => void> = {};
    const store = new Map(cached.map((p) => [new URL(p, scope).href, `cache:${p}`]));
    const cache = { match: async (key: string) => store.get(key), addAll: async () => {} };
    const sandbox = {
      self: { registration: { scope }, addEventListener: (t: string, f: (e: unknown) => void) => (listeners[t] = f) },
      caches: { open: async () => cache },
      fetch: async (req: { url: string }) => `network:${req.url}`,
      URL,
    };
    const sw = renderServiceWorker(cached.map((p) => file(p)), template);
    runInNewContext(sw, sandbox);
    return (url: string, mode = "no-cors", method = "GET") => {
      let response: Promise<unknown> | undefined;
      listeners.fetch({ request: { url, mode, method }, respondWith: (r: Promise<unknown>) => (response = r) });
      return response;
    };
  }

  it("serves a cached file as itself, also when it is opened in a tab", async () => {
    const request = worker();
    await expect(request(`${scope}cases/index.json`, "navigate")).resolves.toBe("cache:cases/index.json");
    await expect(request(`${scope}cases/index.json?x=1#y`)).resolves.toBe("cache:cases/index.json");
    await expect(request(`${scope}assets/app.js`)).resolves.toBe("cache:assets/app.js");
  });

  it("gives the app shell only to navigations that miss the cache", async () => {
    const request = worker();
    await expect(request(scope, "navigate")).resolves.toBe("cache:index.html");
    await expect(request(`${scope}#case=mexicali`, "navigate")).resolves.toBe("cache:index.html");
    await expect(request(`${scope}missing.json`)).resolves.toBe(`network:${scope}missing.json`);
  });

  it("leaves other origins, other paths and non-GET requests to the browser", () => {
    const request = worker();
    expect(request("https://other.test/Trend-Court/index.html")).toBeUndefined();
    expect(request("https://example.test/elsewhere/index.html")).toBeUndefined();
    expect(request(`${scope}index.html`, "no-cors", "POST")).toBeUndefined();
  });
});
