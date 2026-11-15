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
which bound applied to each row.

**The display allowance is capped at 2%.** If `resolution / 2` exceeds
`MAX_DISPLAY_REL · |ours|` (`knmi.MAX_DISPLAY_REL = 2 · REL_TOL = 0.02`,
per decade), the display is too coarse to tell a correct slope from a wrong
one. That row is **inconclusive**: it is never a pass, and the gate stays
pending until the slope is read again with more digits (a per-decade or
per-century display, or KNMI's raw output). The cap admits a 3-decimal
°C/yr display for all six of our slopes (half a step is 1.28–1.50% of
them) and rejects a 2-decimal one (12.8–15.0%), as well as a mistyped
coarse `resolution` (for example, a KNMI slope of `0.0` with resolution
`0.1` is inconclusive, not a pass). Apart from display rounding, any
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
   - `url`, `accessed_utc`, and `evidence`: the repo-relative path of the
     saved page, which must be an existing file under `docs/gate1/`
     (commit it with the JSON).

   `knmi_compare.py` refuses a filled-in `slope` without `resolution`,
   `knmi_settings` and `evidence`, an `evidence` path that is not an
   existing file under `docs/gate1/`, and a `slope` with more digits than
   `resolution` allows.
7. Run `python scripts/knmi_compare.py`. It prints the table below and
   exits with:
   - **0**: Gate 1 (c) passes. All six case/series entries are filled in
     and within tolerance.
   - **1**: at least one filled-in entry is outside tolerance (even if
     others are still pending).
   - **2**: nothing is out of tolerance, but some or all entries are still
     pending or inconclusive (display too coarse; see the 2% cap above).
     Partial results never exit 0.
   - **3**: `knmi_slopes.json` cannot be used: the file is missing or is
     not valid JSON (for example a trailing comma), a case or series name
     is unknown, an entry lacks `slope` or `units`, a field has the wrong
     type (numbers for `slope`/`resolution`/`stderr`, strings for
     `knmi_settings`/`url`/`accessed_utc`/`evidence`), a filled-in entry
     lacks `resolution`/`knmi_settings`/`evidence`, units or resolution are
     invalid, or the evidence file does not exist. The message says which;
     fix the file and run it again.

### Result: Gate 1 (c) passes

Run on KNMI Climate Explorer on 2026-11-15 (about 02:15–02:20 UTC). Each
uploaded series was regressed on KNMI's system-defined `time` series
("Correlate with other time series", variable **regression**, years
1981–2025, straight-line fit, every other option off; monthly files with
starting month **together** and season **averaging over 1 month**). The
exact settings for each entry are in `docs/gate1/knmi_slopes.json`, and the
saved result pages are `docs/gate1/knmi_<region>_<series>.html` (with
their plots in the matching `_files/` folder).

| Case | Series | Ours (°C/yr) | KNMI `b` (°C/yr) | Ours s.e. | KNMI s.e. | Rel. diff |
|---|---|---|---|---|---|---|
| mexicali | monthly | 0.03898686 | 0.0389869 | 0.005106 | 0.005106 | 0.00% |
| mexicali | annual | 0.03900596 | 0.039006 | 0.006483 | 0.006483 | 0.00% |
| iberia | monthly | 0.03436749 | 0.0343675 | 0.004678 | 0.004678 | 0.00% |
| iberia | annual | 0.03438360 | 0.0343836 | 0.006303 | 0.006303 | 0.00% |
| amazon | monthly | 0.03344156 | 0.0334416 | 0.002952 | 0.002952 | 0.00% |
| amazon | annual | 0.03345810 | 0.0334581 | 0.006596 | 0.006596 | 0.00% |

`python scripts/knmi_compare.py` exits **0**: all six agree to every digit
KNMI displays, far inside the 1% tolerance. The standard errors also
match, so KNMI's fit is plain OLS on our exact series. Only the intercepts
`a` differ, because KNMI measures time from 2000 at mid-period. A constant
time offset moves the intercept and never the slope.

Notes:

- The first attempt at the monthly fits used starting month **all**. That
  gives 12 separate per-calendar-month correlations, not one slope. They
  were re-run with **together**. Only the re-runs are recorded.
- `mexicali_T2M_anom_monthly` was uploaded with type "something else", so
  KNMI labels it "Index" instead of "Temperature". The type only changes
  labels. The fit is identical.
- KNMI erases user series 3 days after last use, so the `url` fields
  point to the form page for the record, not to a live result.

## (b) POWER T2M against Giovanni MERRA-2 T2M

Giovanni does not serve POWER, but POWER's meteorology is derived from
MERRA-2, so the area-averaged POWER T2M should closely track Giovanni's
MERRA-2 T2M area average (`M2TMNXSLV_5_12_4_T2M`) for the same box. If
it does not, our loading or aggregation is wrong.

### Tolerances

Fixed before running the comparison. They are the values suggested in
issue #8, unchanged, and are held in code as `crosscheck.GATE1`.

| Check | Tolerance |
|---|---|
| Pearson r of the monthly anomalies | ≥ 0.95 |
| Absolute mean bias, POWER − MERRA-2 | ≤ 1 °C (loose, because the boxes differ) |
| Theil–Sen slope per decade, POWER − MERRA-2 | ≤ max(0.1 °C, 25% of the POWER slope) |

All three must hold for a region to pass. The raw correlation, RMSE and
whether the two 95% slope intervals overlap are reported but are not
pass criteria.

### Method

- **POWER:** `clerk.prepare(region, "T2M").raw`, the cos(lat)-weighted
  mean of the region's 3 × 3 points.
- **MERRA-2:** `clerk.load_giovanni_csv(path)` on
  `data/raw/giovanni/<region>_M2TMNXSLV_T2M.csv`. It skips the metadata
  header, sets the fill value (1e15) to NaN and converts Kelvin to °C.
- **Shared period:** months where both series have a value.
- **Anomalies:** each series minus its own calendar-month climatology
  over the shared months.
- **Slopes:** `examiner.theil_sen` on the anomalies against mid-month
  decimal years, with 95% intervals.
- **Area Giovanni averaged:** Giovanni's Data Bounding Box lists the
  outermost MERRA-2 cell *centers* inside the user box (centers sit at
  lon = −180 + 0.625·i, lat = −90 + 0.5·j). Each cell extends 0.3125° in
  lon and 0.25° in lat beyond its center, so the averaged area is the data
  box widened by half a cell (`crosscheck.data_area`). As the box check
  from issue #8, POWER is re-averaged over only the points inside that
  area (`crosscheck.power_in_box`).
- **Cell sampling:** each POWER point is mapped to the MERRA-2 cell that
  contains it (`crosscheck.merra2_cell`), points in the same cell are
  averaged into one value, and the cells are cos(lat)-weighted at their
  centers (`crosscheck.power_per_cell`). A cell sampled by several points
  then counts once, as in Giovanni's mean over cells.

### Results

Reproduce with `python scripts/gate1_giovanni.py`. It writes
`docs/gate1/giovanni_crosscheck.json` and one figure per region,
including `docs/gate1/mexicali_power_vs_merra2.png` for the phase 5 table.

| Region | POWER average | Period | Months | r (raw) | r (anomaly) | Bias °C | RMSE °C | POWER slope °C/dec [95%] | MERRA-2 slope °C/dec [95%] | Δ slope (allowed) | Intervals overlap | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mexicali | full box (9 pts) | 1981-01 to 2025-12 | 540 | 0.9999 | 0.9991 | -0.088 | 0.126 | +0.400 [+0.297, +0.502] | +0.396 [+0.292, +0.496] | +0.004 (≤ 0.100) | yes | PASS |
| mexicali | data area (9 pts) | 1981-01 to 2025-12 | 540 | 0.9999 | 0.9991 | -0.088 | 0.126 | +0.400 [+0.297, +0.502] | +0.396 [+0.292, +0.496] | +0.004 (≤ 0.100) | yes | PASS |
| mexicali | per MERRA-2 cell (6 of 6 cells) | 1981-01 to 2025-12 | 540 | 1.0000 | 0.9996 | +0.000 | 0.068 | +0.395 [+0.292, +0.495] | +0.396 [+0.292, +0.496] | -0.001 (≤ 0.100) | yes | PASS |
| iberia | full box (9 pts) | 1981-01 to 2025-12 | 540 | 0.9999 | 0.9982 | +0.216 | 0.265 | +0.329 [+0.232, +0.425] | +0.332 [+0.236, +0.428] | -0.003 (≤ 0.100) | yes | PASS |
| iberia | data area (9 pts) | 1981-01 to 2025-12 | 540 | 0.9999 | 0.9982 | +0.216 | 0.265 | +0.329 [+0.232, +0.425] | +0.332 [+0.236, +0.428] | -0.003 (≤ 0.100) | yes | PASS |
| iberia | per MERRA-2 cell (9 of 20 cells) | 1981-01 to 2025-12 | 540 | 0.9999 | 0.9982 | +0.214 | 0.263 | +0.329 [+0.233, +0.425] | +0.332 [+0.236, +0.428] | -0.003 (≤ 0.100) | yes | PASS |
| amazon | full box (9 pts) | 1981-01 to 2025-12 | 540 | 0.9987 | 0.9978 | +0.120 | 0.139 | +0.268 [+0.221, +0.316] | +0.278 [+0.231, +0.326] | -0.010 (≤ 0.100) | yes | PASS |
| amazon | data area (9 pts) | 1981-01 to 2025-12 | 540 | 0.9987 | 0.9978 | +0.120 | 0.139 | +0.268 [+0.221, +0.316] | +0.278 [+0.231, +0.326] | -0.010 (≤ 0.100) | yes | PASS |
| amazon | per MERRA-2 cell (9 of 15 cells) | 1981-01 to 2025-12 | 540 | 0.9987 | 0.9978 | +0.120 | 0.139 | +0.268 [+0.221, +0.316] | +0.278 [+0.231, +0.326] | -0.010 (≤ 0.100) | yes | PASS |

**Verdict: Gate 1 (b) passes.** The full-box POWER series meets every
tolerance in all three regions, with large margins. Anomaly r is at
least 0.998 (tolerance 0.95), |bias| is at most 0.22 °C (tolerance 1 °C),
and the slopes differ by at most 0.01 °C/decade (tolerance 0.1), with
overlapping intervals.

The box check does not explain the offsets. Read as cell centers,
Giovanni's data box covers all 9 POWER points in every region, so the
"data area" rows are the full-box rows.

What does matter is which MERRA-2 cells our 3 × 3 points sample, and
with what weight:

- **mexicali:** the 9 points cover all 6 of Giovanni's cells, but the lon
  −115.8 and −115.4 columns fall in the same cell (centered at −115.625),
  so that cell column is counted twice. The POWER series of the two
  columns are identical, which fits POWER returning the containing
  MERRA-2 cell's value. With one value per cell the bias falls from
  −0.088 °C to +0.000 °C.
- **amazon and iberia:** each point is in its own cell, so one value per
  cell changes almost nothing (+0.120 °C and +0.214 °C). The points
  sample only 9 of Giovanni's 15 (amazon) and 20 (iberia) cells; the
  cells we miss are the likely source of the offsets. We cannot confirm
  this without the gridded MERRA-2 data or Giovanni's exact cell weights.

Either way the offsets are well within the 1 °C tolerance and do not
affect the anomalies or slopes.
