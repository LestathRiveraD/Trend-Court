"""Gate 1 (b): POWER T2M against Giovanni MERRA-2 T2M for the three cross-check cases.

For each region, compares the full-box POWER average and the POWER
average over only the points inside Giovanni's data box, writes
``docs/gate1/<region>_power_vs_merra2.png`` and
``docs/gate1/giovanni_crosscheck.json``, and prints a Markdown table for
``docs/gate1.md``. Tolerances are ``crosscheck.GATE1``.

    python scripts/gate1_giovanni.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trendcourt import clerk, crosscheck  # noqa: E402

REGIONS = ("mexicali", "iberia", "amazon")
OUT_DIR = clerk.REPO_ROOT / "docs" / "gate1"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    rows = []
    for region in REGIONS:
        path = clerk.GIOVANNI_DIR / f"{region}_M2TMNXSLV_T2M.csv"
        header = clerk.read_giovanni_header(path)
        merra2 = clerk.load_giovanni_csv(path)
        prepared = clerk.prepare(region, "T2M")
        full = crosscheck.compare(prepared.raw, merra2)
        boxed_series, n_boxed = crosscheck.power_in_box(region, header.data_bbox)
        boxed = crosscheck.compare(boxed_series, merra2)

        name = clerk.load_regions()["regions"][region]["name"]
        crosscheck.plot_comparison(prepared.raw, merra2, full, OUT_DIR / f"{region}_power_vs_merra2.png",
                                   title=name)
        results[region] = {
            "giovanni_file": str(path.relative_to(clerk.REPO_ROOT)),
            "user_bbox": header.user_bbox,
            "data_bbox": header.data_bbox,
            "full_box": {"n_points": prepared.provenance.n_points, **full.to_dict()},
            "data_box": {"n_points": n_boxed, **boxed.to_dict()},
        }
        for label, n_pts, r in (("full box", prepared.provenance.n_points, full),
                                ("data box", n_boxed, boxed)):
            a, b = r.slope_a, r.slope_b
            rows.append(
                f"| {region} | {label} ({n_pts} pts) | {r.start} to {r.end} | {r.n} | {r.r_raw:.4f} | "
                f"{r.r_anomaly:.4f} | {r.bias:+.3f} | {r.rmse:.3f} | "
                f"{a.slope_per_decade:+.3f} [{a.low_slope_per_decade:+.3f}, {a.high_slope_per_decade:+.3f}] | "
                f"{b.slope_per_decade:+.3f} [{b.low_slope_per_decade:+.3f}, {b.high_slope_per_decade:+.3f}] | "
                f"{r.slope_diff:+.3f} (≤ {r.slope_allowance:.3f}) | {'yes' if r.intervals_overlap else 'no'} | "
                f"{'PASS' if r.passed else 'FAIL'} |"
            )

    (OUT_DIR / "giovanni_crosscheck.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print("| Region | POWER average | Period | Months | r (raw) | r (anomaly) | Bias °C | RMSE °C | "
          "POWER slope °C/dec [95%] | MERRA-2 slope °C/dec [95%] | Δ slope (allowed) | Intervals overlap | Result |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    return 0 if all(results[r]["full_box"]["passed"] for r in REGIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
