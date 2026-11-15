"""Compare KNMI's OLS slopes (docs/gate1/knmi_slopes.json) with ours.

    python scripts/knmi_compare.py [path/to/knmi_slopes.json]

Prints a Markdown table, plus our slopes for any case still awaiting KNMI.
Exit codes (``knmi.gate_status``, plus 3 for a malformed file):
  0  Gate 1 (c) passes: all six case/series entries filled in and within tolerance
  1  at least one filled-in entry is outside the tolerance
  2  nothing is out of tolerance, but some entries still await KNMI or are
     inconclusive (KNMI's display too coarse to decide; re-read with more digits)
  3  knmi_slopes.json is unreadable, not valid JSON, or has a malformed entry
     (message printed to stderr)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trendcourt import knmi  # noqa: E402

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else knmi.SLOPES_FILE
    try:
        slopes = knmi.load_knmi_slopes(path)
        rows = knmi.compare_all(slopes)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Gate 1 (c): INVALID knmi_slopes.json: {exc}", file=sys.stderr)
        sys.exit(knmi.GATE_INVALID)
    pending = knmi.pending_entries(slopes)
    if rows:
        print(knmi.markdown_table(rows))
    for region, series in pending:
        ours = knmi.series_for(region)["ols"][series]
        print(f"pending KNMI: {region} {series}; ours = {ours.slope:.6f} /yr ({ours.slope_per_decade:.4f} /decade)")
    for r in rows:
        if r.inconclusive:
            print(f"inconclusive: {r.region} {r.series}; KNMI's display is too coarse, record more digits")
    status = knmi.gate_status(rows, pending)
    n = len(knmi.CASES) * len(knmi.SERIES)
    waiting = len(pending) + sum(r.inconclusive for r in rows)
    print({knmi.GATE_PASS: "Gate 1 (c): PASS", knmi.GATE_FAIL: "Gate 1 (c): FAIL",
           knmi.GATE_PENDING: f"Gate 1 (c): PENDING ({waiting} of {n} entries await KNMI or are inconclusive)"}[status])
    sys.exit(status)
