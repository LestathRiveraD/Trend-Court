# Trend Court

Puts climate-trend claims on trial against NASA data (POWER, MERRA-2, GLDAS, FLDAS, GPCP).

- `trendcourt/clerk.py`: data loading, cleaning, aggregation, deseasonalizing
- `trendcourt/examiner.py`: trend statistics (Theil–Sen, OLS, Mann–Kendall variants)
- `trendcourt/knmi.py`: Gate 1 (c) KNMI Climate Explorer cross-check (upload files, slope comparison)
- `trendcourt/crosscheck.py`: Gate 1 cross-checks against reference series (Giovanni MERRA-2)
- `scripts/gate1_giovanni.py`: reruns Gate 1 (b) and writes `docs/gate1/`
- `trendcourt/benchmark.py`: synthetic series with known truth, run through the same code path as a real case
- `scripts/benchmark.py --experiment fpr`: false-positive rate on trend-free AR(1) noise; writes `docs/benchmark/fpr.json` (about 4–5 min with 12 processes, `--workers` sets the count)
- `scripts/benchmark.py --experiment coverage`: share of Theil–Sen 95% intervals that contain an injected slope, flagging cells below 0.93; writes `docs/benchmark/coverage.json` (about 14 min with 12 processes)
- `scripts/benchmark.py --experiment power`: Hamed–Rao rejection rate by record length, slope / noise SD and AR(1) phi; writes `docs/benchmark/power.json` (about 35 min with 12 processes)
- `scripts/benchmark.py --experiment applied`: the verdict policy as it will be applied (band from each record's own estimated r1) on trend-free AR(1) series; writes `docs/benchmark/applied.json` (about 2.5 min with 12 processes)
- `scripts/benchmark.py --experiment noise`: per real case, how far its noise is from AR(1) (ACF, fitted AR(p) model) and the policy on records simulated from that model; writes `docs/benchmark/noise.json` (about 2.5 min with 12 processes); the caveat flags derived from it go in `thresholds.json` (`noise_caveats`)
- `scripts/benchmark.py --figures`: no simulation; redraws `docs/benchmark/*.png` and the tables in `docs/benchmark.md` from the committed JSON (and, for the no-trend columns of Table 6, the committed records in `data/`); fails if the bands or `noise_caveats` in `thresholds.json` no longer match `fpr.json` / `noise.json`, or if `applied.json` / `noise.json` were run with other thresholds
- `trendcourt/thresholds.json`: calibrated verdict thresholds (alphas, autocorrelation caps, per-variable no-trend margins, relative for precipitation; noise-model caveats per case), reasoning in `docs/benchmark.md`; read only through `trendcourt/thresholds.py`
- `trendcourt/case.py`: `trendcourt.examine(region, variable)`, the case JSON (`python -m trendcourt mexicali temperature`)
- `trendcourt/verdict.py`: `decide(case, thresholds)`, the verdict label, its checks (pass / fail / pending), whether a pending check capped it at Moderate, and a one-line reason; `examine` stores it in the case JSON's `verdict` block (v0: window agreement and witness B are pending until Phase 3)
- `trendcourt/cache.py`, `scripts/build_cache.py`: writes every preset case and `index.json` to `site/public/cases/`, see [Rebuilding the case cache](#rebuilding-the-case-cache)
- `site/`: the static front end (Vite, React, Plotly), see [Site](#site)
- `data/`: raw downloads and region definitions
- `docs/`: method notes; `docs/gate1/` holds the Gate 1 (a) reference case

## Setup

Requires Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

All dependencies are pinned so the case notebooks rerun to identical numbers.
After changing dependencies, run `pip freeze > requirements.txt`.

## Credentials

Earthdata downloads read credentials from `~/.netrc`. Never commit them:
`.netrc` is in `.gitignore`.

## Site

The front end in `site/` is a static React app (Vite, TypeScript, Plotly) that reads the precomputed
case JSON in `site/public/cases/`. It computes no statistics and calls no live service. Plotly and the
fonts are bundled (no CDN), and the production build installs a service worker that caches every file,
so after one visit the site keeps working with the network off. Requires Node 20.19+ or 22.12+.

```bash
cd site
npm ci              # install the pinned dependencies from package-lock.json
npm run dev         # dev server at http://localhost:5173/Trend-Court/
npm test            # loader and service-worker tests (Vitest)
npm run build       # type-check and build to site/dist/
npm run preview     # serve the build at http://localhost:4173/Trend-Court/
```

The build uses `base: "/Trend-Court/"` for GitHub Pages; dev and preview serve under the same path.
The case picker (`src/Picker.tsx`, logic in `src/picker.ts`) offers only what `index.json` lists and keeps the
selection in the URL hash, `#/<region>/<variable>` (for example `#/mexicali/temperature`), so a reload or a
deep link opens the same case.
`src/data.ts` holds the types of the case JSON and `index.json` and refuses files whose
`schema_version` differs from `CASE_SCHEMA_VERSION`, which must equal `SCHEMA_VERSION` in
`trendcourt/case.py`.

### Rebuilding the case cache

`site/public/cases/` holds one case file per preset (`<region>_<variable>.json`, 6 regions × 3
variables) and `index.json` for the picker. They are generated and committed, so the demo runs
with the network off. After changing anything the cases are computed from (`trendcourt/`,
`data/regions.json`, `data/raw/`, `requirements.txt`, `scripts/build_cache.py`), commit that change
first, then rebuild and commit the cache in its own commit:

```bash
python scripts/build_cache.py      # about 10 s; refuses to run if those inputs have uncommitted changes
git add site/public/cases && git commit -m "Rebuild case cache"
```

`index.json` is stamped with the last commit that changed the inputs (not `HEAD`) and the NASA
data access dates from the download manifests, never the wall-clock time. So on a clean checkout
`python scripts/build_cache.py && git diff --exit-code site/public/cases` passes, and
`tests/test_cache.py` fails if the committed cache is stale. `--allow-dirty` builds from
uncommitted inputs for a local preview and stamps `<hash>+dirty`; do not commit that output.
A case that fails the quality gate is still written, with its statistics set to `null`.

