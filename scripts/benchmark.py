"""Synthetic benchmark experiments (``project.md`` §3, validation).

    python scripts/benchmark.py --experiment fpr [--replicates 1000] [--workers N] [--out PATH]

``fpr``: rejection rate of the Hamed–Rao and correlated seasonal
Mann–Kendall tests on trend-free AR(1) noise with a seasonal cycle, at
alpha 0.05 and 0.01, over phi in {0, 0.3, 0.6, 0.8, 0.9} and record
lengths of 20, 30 and 45 years, plus the lag-1 autocorrelation of the
18 real cases. Writes ``docs/benchmark/fpr.json`` (see
:func:`trendcourt.benchmark.fpr_experiment`). The numbers do not depend
on ``--workers``. Prints a summary table and the run time to stderr.
"""

import argparse
import json
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from trendcourt import benchmark  # noqa: E402

DEFAULT_OUT = REPO / "docs" / "benchmark" / "fpr.json"


def _fmt(s: dict) -> str:
    flag = " INFLATED" if s["inflated"] else ""
    return f"{s['rate']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]{flag}"


def run_fpr(args) -> int:
    t0 = time.perf_counter()
    result = benchmark.fpr_experiment(args.replicates, seed=args.seed, workers=args.workers)
    result["real_cases"] = benchmark.real_case_autocorr()
    result["versions"] = {pkg: version(pkg) for pkg in ("numpy", "pandas", "scipy", "pymannkendall")}
    result["versions"]["python"] = platform.python_version()
    elapsed = time.perf_counter() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    err = sys.stderr
    print(f"{'phi':>4} {'yrs':>3}  {'HR @0.05':<30}{'HR @0.01':<30}{'SMK @0.05':<30}{'SMK @0.01':<30}fail", file=err)
    for c in result["cells"]:
        r = c["rates"]
        print(f"{c['phi']:>4} {c['n_years']:>3}  "
              f"{_fmt(r['hamed_rao']['0.05']):<30}{_fmt(r['hamed_rao']['0.01']):<30}"
              f"{_fmt(r['seasonal']['0.05']):<30}{_fmt(r['seasonal']['0.01']):<30}{c['n_failed']}", file=err)
    print("\nreal cases, lag-1 autocorrelation of anomalies (detrended):", file=err)
    for row in result["real_cases"]:
        d = "n/a" if row["r1_detrended"] is None else f"{row['r1_detrended']:.3f}"
        print(f"  {row['region']:<12}{row['variable']:<14}{row['r1']:.3f} ({d})", file=err)
    print(f"\nwrote {args.out} in {elapsed:.0f} s with {args.workers} worker(s)", file=err)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=["fpr"])
    ap.add_argument("--replicates", type=int, default=benchmark.FPR_REPLICATES)
    ap.add_argument("--seed", type=int, default=benchmark.FPR_SEED)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    if args.replicates < 1 or args.workers < 1:
        ap.error("--replicates and --workers must be at least 1")
    return {"fpr": run_fpr}[args.experiment](args)


if __name__ == "__main__":
    sys.exit(main())
