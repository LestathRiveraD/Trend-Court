# Trend Court

Puts climate-trend claims on trial against NASA data (POWER, MERRA-2, GLDAS, FLDAS, GPCP).

- `trendcourt/clerk.py`: data loading, cleaning, aggregation, deseasonalizing
- `trendcourt/examiner.py`: trend statistics (Theil–Sen, OLS, Mann–Kendall variants)
- `trendcourt/crosscheck.py`: Gate 1 cross-checks against reference series (Giovanni MERRA-2)
- `scripts/gate1_giovanni.py`: reruns Gate 1 (b) and writes `docs/gate1/`
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
