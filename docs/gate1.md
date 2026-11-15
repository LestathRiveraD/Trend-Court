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
- **Box difference:** Giovanni averages its *data* bounding box (the
  user box snapped to the MERRA-2 grid), not our box. As a check on the
  bias, POWER is also re-averaged over only the points inside the data
  box (`crosscheck.power_in_box`).

### Results

Reproduce with `python scripts/gate1_giovanni.py`. It writes
`docs/gate1/giovanni_crosscheck.json` and one figure per region,
including `docs/gate1/mexicali_power_vs_merra2.png` for the phase 5 table.

| Region | POWER average | Period | Months | r (raw) | r (anomaly) | Bias °C | RMSE °C | POWER slope °C/dec [95%] | MERRA-2 slope °C/dec [95%] | Δ slope (allowed) | Intervals overlap | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mexicali | full box (9 pts) | 1981-01 to 2025-12 | 540 | 0.9999 | 0.9991 | -0.088 | 0.126 | +0.400 [+0.297, +0.502] | +0.396 [+0.292, +0.496] | +0.004 (≤ 0.100) | yes | PASS |
| mexicali | data box (6 pts) | 1981-01 to 2025-12 | 540 | 1.0000 | 0.9996 | +0.000 | 0.068 | +0.395 [+0.292, +0.495] | +0.396 [+0.292, +0.496] | -0.001 (≤ 0.100) | yes | PASS |
| iberia | full box (9 pts) | 1981-01 to 2025-12 | 540 | 0.9999 | 0.9982 | +0.216 | 0.265 | +0.329 [+0.232, +0.425] | +0.332 [+0.236, +0.428] | -0.003 (≤ 0.100) | yes | PASS |
| iberia | data box (9 pts) | 1981-01 to 2025-12 | 540 | 0.9999 | 0.9982 | +0.216 | 0.265 | +0.329 [+0.232, +0.425] | +0.332 [+0.236, +0.428] | -0.003 (≤ 0.100) | yes | PASS |
| amazon | full box (9 pts) | 1981-01 to 2025-12 | 540 | 0.9987 | 0.9978 | +0.120 | 0.139 | +0.268 [+0.221, +0.316] | +0.278 [+0.231, +0.326] | -0.010 (≤ 0.100) | yes | PASS |
| amazon | data box (3 pts) | 1981-01 to 2025-12 | 540 | 0.9975 | 0.9957 | -0.001 | 0.103 | +0.265 [+0.217, +0.314] | +0.278 [+0.231, +0.326] | -0.013 (≤ 0.100) | yes | PASS |

**Verdict: Gate 1 (b) passes.** The full-box POWER series meets every
tolerance in all three regions, with large margins. Anomaly r is at
least 0.998 (tolerance 0.95), |bias| is at most 0.22 °C (tolerance 1 °C),
and the slopes differ by at most 0.01 °C/decade (tolerance 0.1), with
overlapping intervals.

In mexicali and amazon the box difference accounts for the mean
offset: restricting POWER to the points inside Giovanni's data box
brings the bias to about 0.00 °C. In iberia the data box
(`-4.375,39.5,-2.5,41.5`) still contains all 9 POWER points, so the
+0.22 °C there does not come from the box. Its cause is not established.
One candidate is that our 9 points sample the box differently from
Giovanni's mean over every grid cell, which matters more over Iberia's
relief. It is well within the 1 °C tolerance and does not affect the
anomalies or slopes.
