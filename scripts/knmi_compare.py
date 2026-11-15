"""Compare KNMI's OLS slopes (docs/gate1/knmi_slopes.json) with ours.

    python scripts/knmi_compare.py

Prints a Markdown table, plus our slopes for any case still awaiting KNMI.
Exit codes (``knmi.gate_status``):
  0  Gate 1 (c) passes: all six case/series entries filled in and within tolerance
  1  at least one filled-in entry is outside the tolerance
  2  nothing is out of tolerance, but some or all entries still await KNMI
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trendcourt import knmi  # noqa: E402

if __name__ == "__main__":
    slopes = knmi.load_knmi_slopes()
    rows = knmi.compare_all(slopes)
    pending = knmi.pending_entries(slopes)
    if rows:
        print(knmi.markdown_table(rows))
    for region, series in pending:
        ours = knmi.series_for(region)["ols"][series]
        print(f"pending KNMI: {region} {series}; ours = {ours.slope:.6f} /yr ({ours.slope_per_decade:.4f} /decade)")
    status = knmi.gate_status(rows, pending)
    print({knmi.GATE_PASS: "Gate 1 (c): PASS", knmi.GATE_FAIL: "Gate 1 (c): FAIL",
           knmi.GATE_PENDING: f"Gate 1 (c): PENDING ({len(pending)} of {len(knmi.CASES) * len(knmi.SERIES)} entries await KNMI)"}[status])
    sys.exit(status)
