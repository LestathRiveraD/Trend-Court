# Gate 1

Gate 1 from `project.md` §5, phase 1:

- **(a)** On the same series, engine output matches `pymannkendall` and `scipy` exactly (issue #7).
- **(b)** Our POWER series tracks Giovanni's MERRA-2 T2M for the same box, with a similar slope (issue #8).
- **(c)** Our OLS slope matches KNMI's within a stated tolerance (issue #9).

## (c) Our OLS slope against KNMI Climate Explorer

KNMI Climate Explorer fits a least-squares trend, not Theil–Sen. So the
like-for-like check compares our `examiner.ols()` slope
(`scipy.stats.linregress`) with KNMI's regression slope on the same series.
A library-parity test cannot catch mistakes in the time axis, units or
per-decade conversion. This check can.

### Route: upload our own series (input identical by construction)

We take the preferred route from issue #9. We upload the exact series our
`ols()` sees to KNMI as a user time series. We do not use a KNMI-served
dataset, because then the input would differ, and the check would test
the data as well as the regression.

- **Cases:** the three cross-check cases from `project.md` §5, temperature
  (POWER `T2M`) in `mexicali`, `iberia` and `amazon`. These are the same
  boxes Gate 1 (b) uses.
- **Series:** the deseasonalized monthly anomalies from `clerk.prepare()`,
  1981-01 to 2025-12, with no missing months. This is the series the
  Examiner fits. We also upload their calendar-year means (45 values) as a
  second, unambiguous check. With an annual series there is no question of
  which months KNMI selects or averages.
- **Files:** `data/knmi/upload/<region>_T2M_anom_{monthly,annual}.dat`,
  written by `python scripts/knmi_export.py`.
- **Format:** KNMI's plain-text time-series format. Lines starting with
  `#` are metadata. Monthly files have one line per year, `yyyy v1 … v12`;
  annual files have `yyyy v`. Missing values are `-999.9` (none occur
  here). Values carry four decimals. `tests/test_knmi.py` reads the
  committed files back and checks that they reproduce our slope to within
  a relative 1e-4. The rounding is therefore far inside the tolerance
  below.
- **Time axis:** our monthly time is mid-month decimal years
  (`year + (month − 0.5)/12`), and our annual time is `year + 0.5`. A
  constant offset in the time axis does not change an OLS slope. KNMI's
  own convention (start or middle of the month) therefore matters only
  through gaps, and these series have none.

### Tolerance (fixed before comparing)

The input is identical, so **KNMI's slope must agree with ours to within
1% (relative), or to within half of KNMI's last displayed digit, whichever
is larger**:

    |KNMI − ours| ≤ max(REL_TOL · |ours|, resolution / 2)    (both per decade)

`REL_TOL = 0.01` is `knmi.REL_TOL`, also stored in
`docs/gate1/knmi_slopes.json`. `resolution` is the place value of the last
digit KNMI shows, recorded with each entry (for example `0.001` if KNMI
shows `0.034`). The display term is there because a correct slope can be
off by up to half a display step after rounding: our iberia slope
0.034367 °C/yr shown as `0.034` is 1.07% off, and amazon's 0.033442 shown
as `0.033` is 1.32% off, yet both are correct. At our slopes (about
0.33–0.39 °C/decade), 1% is about 0.0035 °C/decade. A display of 3
decimals in °C/yr widens that to 0.005 °C/decade; 4 decimals in °C/yr, or
3 in °C/decade, leaves the 1% bound in force. The comparison table says
which bound applied to each row. Apart from display rounding, any
remaining difference can only come from the time axis.

KNMI's standard error is recorded but is not a pass criterion. KNMI may
adjust it for autocorrelation, and `linregress` does not.

### Procedure (manual: KNMI is behind a human check)

KNMI Climate Explorer shows a "verify that I am human" page to scripts.
User-series uploads also need a registered KNMI login. A person
therefore does this step in a browser. The menu names below follow KNMI's
help pages ([season](https://climexp.knmi.nl/help/selectseason.shtml),
[season, monthly data](https://climexp.knmi.nl/help/selectseason12.shtml),
[anomalies](https://climexp.knmi.nl/help/anomalies.shtml),
[detrend](https://climexp.knmi.nl/help/detrend.shtml),
[units](https://climexp.knmi.nl/help/convertunits.shtml)). Correct them here
if the live site differs.

1. Log in to <https://climexp.knmi.nl/> with a registered account.
2. Upload each `data/knmi/upload/*.dat` file as your own time series
   ("upload time series" in the time-series menu). Choose monthly or
   annual to match the file. Give it a name such as
   `tc_iberia_T2M_anom_monthly`.
3. On the uploaded series' page, fit a linear trend against time with
   these settings:

   | KNMI control | Monthly file | Annual file |
   |---|---|---|
   | Starting month | **together** ("all months together, giving a single result"). Not **all**, which gives 12 separate per-month slopes, and not a single calendar month. | not offered / not applicable |
   | Season length | **1** month | not applicable |
   | Averaging vs. selecting months | either (identical at length 1) | not applicable |
   | Anomalies | **unchecked** (the series already has the seasonal cycle removed) | unchecked, if offered |
   | Detrend | **unchecked** | **unchecked** |
   | Begin / end year | **1981** / **2025** | **1981** / **2025** |
   | Units | **original units** (not "convert to standard units") | **original units** |

   Leave every other option (filters, running means, lags) at "none" or
   its default, and note what it was.
4. Note whether KNMI reports the slope per year or per decade. If KNMI
   offers a per-decade or per-century display, prefer it: it shows more
   significant digits.
5. Save the result page (HTML or a screenshot) as
   `docs/gate1/knmi_<region>_<series>.{html,png}`.
6. Fill in that entry of `docs/gate1/knmi_slopes.json`:
   - `slope`: **every digit KNMI shows**, without rounding;
   - `units`: `per_year` or `per_decade` (per century: divide by 10 and
     record `per_decade`, with `resolution` divided by 10 too);
   - `resolution`: the place value of the last displayed digit of `slope`,
     in the same units (KNMI shows `0.0344` → `0.0001`);
   - `stderr`: in the same units, or `null`;
   - `knmi_settings`: every choice from step 3, verbatim as KNMI labels
     it, for example `"starting month: together; season length: 1;
     anomalies: off; detrend: off; years: 1981-2025; units: original"`;
   - `url`, `accessed_utc`, and `evidence` (path of the saved page).

   `knmi_compare.py` refuses a filled-in `slope` without `resolution`,
   `knmi_settings` and `evidence`, and a `slope` with more digits than
   `resolution` allows.
7. Run `python scripts/knmi_compare.py`. It prints the table below and
   exits with:
   - **0**: Gate 1 (c) passes. All six case/series entries are filled in
     and within tolerance.
   - **1**: at least one filled-in entry is outside tolerance (even if
     others are still pending).
   - **2**: nothing is out of tolerance, but some or all entries are still
     pending. Partial results never exit 0.

### Our slopes (awaiting KNMI)

From `python scripts/knmi_compare.py` on the committed data (POWER
accessed 2026-11-14):

| Case | Series | Ours (°C/yr) | Ours (°C/decade) | Ours s.e. (°C/decade) | KNMI |
|---|---|---|---|---|---|
| mexicali | monthly | 0.038987 | 0.3899 | 0.0511 | pending |
| mexicali | annual | 0.039006 | 0.3901 | 0.0648 | pending |
| iberia | monthly | 0.034367 | 0.3437 | 0.0468 | pending |
| iberia | annual | 0.034384 | 0.3438 | 0.0630 | pending |
| amazon | monthly | 0.033442 | 0.3344 | 0.0295 | pending |
| amazon | annual | 0.033458 | 0.3346 | 0.0660 | pending |

Gate 1 (c) passes when every row is filled in and within tolerance
(`knmi_compare.py` exits 0), or when a difference is traced to a named
cause and resolved. The saved KNMI pages
are committed under `docs/gate1/`.
