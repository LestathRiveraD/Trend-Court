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
