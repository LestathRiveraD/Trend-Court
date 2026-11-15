/* Trend Court offline cache. Generated into dist/sw.js by precache.ts at build time; not used in dev.
 *
 * The demo has to run with the network off (project.md §6, criterion 5). On install this worker
 * stores every file of the build (the app, Plotly, fonts and all case JSON) and then serves them
 * cache-first, so after one visit the site keeps working offline, including a reload.
 * A new build has a new VERSION, so its worker installs a fresh cache and deletes the old one.
 */
const VERSION = "__VERSION__";
const PRECACHE = __PRECACHE__;
const CACHE = `trend-court-${VERSION}`;
const scope = new URL(self.registration.scope);
const urls = PRECACHE.map((path) => new URL(path, scope).href);
const indexHtml = new URL("index.html", scope).href;

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE)
      .then((cache) => cache.addAll(urls))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith("trend-court-") && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== scope.origin || !url.pathname.startsWith(scope.pathname)) return;
  url.hash = "";
  url.search = "";
  // Any navigation inside the scope gets the app shell (the case is chosen in the hash).
  const key = req.mode === "navigate" ? indexHtml : url.href;
  event.respondWith(
    caches.open(CACHE).then((cache) => cache.match(key).then((hit) => hit ?? fetch(req))),
  );
});
