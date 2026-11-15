# Gate 1

Gate 1 from `project.md` §5, phase 1:

- **(a)** On the same series, engine output matches `pymannkendall` and `scipy` exactly (issue #7).
- **(b)** Our POWER series tracks Giovanni's MERRA-2 T2M for the same box, with a similar slope (issue #8).
- **(c)** Our OLS slope matches KNMI's within a stated tolerance (issue #9).

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
