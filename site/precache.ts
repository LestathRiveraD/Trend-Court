import { createHash } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import type { Plugin } from "vite";

const TEMPLATE = new URL("./sw-template.js", import.meta.url);

/** Every file under `dir`, as sorted POSIX paths relative to it. */
export function listFiles(dir: string): string[] {
  const out: string[] = [];
  const walk = (d: string) => {
    for (const e of readdirSync(d, { withFileTypes: true })) {
      const p = join(d, e.name);
      if (e.isDirectory()) walk(p);
      else out.push(relative(dir, p).split(sep).join("/"));
    }
  };
  walk(dir);
  return out.sort();
}

/**
 * Service worker source for a build: the precache list is every file in the build
 * (except the worker itself) and the version is a hash of their contents, so a
 * rebuild with no changes produces an identical sw.js.
 */
export function renderServiceWorker(files: { path: string; content: Buffer }[], template: string): string {
  const precache = files.map((f) => f.path).filter((p) => p !== "sw.js").sort();
  const hash = createHash("sha256");
  for (const p of precache) {
    hash.update(p).update("\0").update(files.find((f) => f.path === p)!.content).update("\0");
  }
  return template
    .replace("__VERSION__", hash.digest("hex").slice(0, 16))
    .replace("__PRECACHE__", JSON.stringify(precache));
}

/** Vite plugin: after the build is written, emit dist/sw.js precaching all of it. */
export function precache(): Plugin {
  let outDir = "dist";
  return {
    name: "trend-court-precache",
    apply: "build",
    configResolved(config) {
      outDir = join(config.root, config.build.outDir);
    },
    closeBundle() {
      const files = listFiles(outDir).map((path) => ({ path, content: readFileSync(join(outDir, path)) }));
      writeFileSync(join(outDir, "sw.js"), renderServiceWorker(files, readFileSync(TEMPLATE, "utf8")));
    },
  };
}
