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
- `scripts/benchmark.py --figures`: no simulation; redraws `docs/benchmark/*.png` and the tables in `docs/benchmark.md` from the committed JSON (and, for the no-trend columns of Table 6, the committed records in `data/`); fails if the bands in `thresholds.json` no longer match `fpr.json`
- `trendcourt/thresholds.json`: calibrated verdict thresholds (alphas, autocorrelation caps, per-variable no-trend margins), reasoning in `docs/benchmark.md`; read only through `trendcourt/thresholds.py`
- `trendcourt/case.py`: `trendcourt.examine(region, variable)`, the case JSON (`python -m trendcourt mexicali temperature`)
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
`src/data.ts` holds the types of the case JSON and `index.json` and refuses files whose
`schema_version` differs from `CASE_SCHEMA_VERSION`, which must equal `SCHEMA_VERSION` in
`trendcourt/case.py`. Until the cache build (#21) lands, `site/public/cases/` holds a stand-in:
a copy of `docs/gate1/mexicali_temperature.json` and an `index.json` that lists only that case.

