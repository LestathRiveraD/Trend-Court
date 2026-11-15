# Trend Court

Puts climate-trend claims on trial against NASA data (POWER, MERRA-2, GLDAS, FLDAS, GPCP).

- `trendcourt/clerk.py`: data loading, cleaning, aggregation, deseasonalizing
- `trendcourt/examiner.py`: trend statistics (Theil–Sen, OLS, Mann–Kendall variants)
- `trendcourt/knmi.py`: Gate 1 (c) KNMI Climate Explorer cross-check (upload files, slope comparison)
- `data/`: raw downloads and region definitions
- `docs/`: method notes

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
