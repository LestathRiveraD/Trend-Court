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
