"""Compare KNMI's OLS slopes (docs/gate1/knmi_slopes.json) with ours.

    python scripts/knmi_compare.py

Prints a Markdown table, plus our slopes for any case still awaiting KNMI.
Exits 1 if any filled-in case is outside the tolerance, 2 if none are filled in.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trendcourt import knmi  # noqa: E402

if __name__ == "__main__":
    slopes = knmi.load_knmi_slopes()
    rows = knmi.compare_all(slopes)
    pending = [
        (region, series)
        for region, by_series in slopes["cases"].items()
        for series, entry in by_series.items()
        if entry.get("slope") is None
    ]
    if rows:
        print(knmi.markdown_table(rows))
    for region, series in pending:
        ours = knmi.series_for(region)["ols"][series]
        print(f"pending KNMI: {region} {series}; ours = {ours.slope:.6f} /yr ({ours.slope_per_decade:.4f} /decade)")
    if not rows:
        sys.exit(2)
    sys.exit(0 if all(r.passed for r in rows) else 1)
