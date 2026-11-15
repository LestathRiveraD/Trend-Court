"""Synthetic benchmark experiments (``project.md`` §3, validation).

    python scripts/benchmark.py --experiment {fpr,coverage,power} [--replicates 1000]
                                [--seed S] [--workers N] [--out PATH]

``fpr``: rejection rate of the Hamed–Rao and correlated seasonal
Mann–Kendall tests on trend-free AR(1) noise with a seasonal cycle, at
alpha 0.05 and 0.01, over phi in {0, 0.3, 0.6, 0.8, 0.9} and record
lengths of 20, 30 and 45 years, plus the lag-1 autocorrelation of the
18 real cases. Writes ``docs/benchmark/fpr.json`` (see
:func:`trendcourt.benchmark.fpr_experiment`). The numbers do not depend
on ``--workers``. Prints a summary table and the run time to stderr.

``coverage``: share of replicates whose Theil–Sen 95% interval contains
the injected slope, over phi in {0, 0.3, 0.6, 0.8, 0.9}, slopes of 0.1,
0.3 and 0.5 noise SDs per decade and 20, 30 and 45 years. Cells below
0.93 are flagged, with the phi at which that starts. Writes
``docs/benchmark/coverage.json`` (:func:`trendcourt.benchmark.coverage_experiment`).

``power``: Hamed–Rao rejection rate at alpha 0.05 and 0.01 by record
length (20 to 45 years in steps of 5), slope / noise SD (0.1, 0.2, 0.3,
0.5 per decade) and phi. Writes ``docs/benchmark/power.json``
(:func:`trendcourt.benchmark.power_experiment`).

Every experiment uses ``benchmark.synthetic_series`` and
``benchmark.run_case``; ``--seed`` defaults to the experiment's own
master seed.

    python scripts/benchmark.py --figures

``--figures``: no simulation. Reads the committed ``docs/benchmark/*.json``
and ``trendcourt/thresholds.json``, writes ``fpr_vs_phi.png``,
``coverage_vs_phi.png`` and ``power_vs_years.png`` to ``docs/benchmark/``
and regenerates the tables in ``docs/benchmark.md``
(:mod:`trendcourt.benchmark_report`).
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

OUT_DIR = REPO / "docs" / "benchmark"
DEFAULTS = {
    "fpr": (benchmark.FPR_REPLICATES, benchmark.FPR_SEED),
    "coverage": (benchmark.COVERAGE_REPLICATES, benchmark.COVERAGE_SEED),
    "power": (benchmark.POWER_REPLICATES, benchmark.POWER_SEED),
}


def _stamp(result: dict) -> dict:
    result["versions"] = {pkg: version(pkg) for pkg in ("numpy", "pandas", "scipy", "pymannkendall")}
    result["versions"]["python"] = platform.python_version()
    return result


def _write(result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _ci(s: dict) -> str:
    return f"{s['rate']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"


def _fmt(s: dict) -> str:
    flag = " INFLATED" if s["inflated"] else ""
    return f"{s['rate']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]{flag}"


def run_fpr(args) -> int:
    t0 = time.perf_counter()
    result = benchmark.fpr_experiment(args.replicates, seed=args.seed, workers=args.workers)
    result["real_cases"] = benchmark.real_case_autocorr()
    _stamp(result)
    elapsed = time.perf_counter() - t0
    _write(result, args.out)

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


def run_coverage(args) -> int:
    t0 = time.perf_counter()
    result = _stamp(benchmark.coverage_experiment(args.replicates, seed=args.seed, workers=args.workers))
    elapsed = time.perf_counter() - t0
    _write(result, args.out)

    err = sys.stderr
    print(f"{'phi':>4} {'slope':>5} {'yrs':>3}  {'coverage':<29}{'lo/hi miss':<12}"
          f"{'median slope':>13}{'median width':>13}  fail", file=err)
    for c in result["cells"]:
        flag = " BELOW" if c["below_flag"] else ""
        print(f"{c['phi']:>4} {c['slope_per_decade']:>5} {c['n_years']:>3}  {_ci(c['coverage']) + flag:<29}"
              f"{str(c['miss_low']) + '/' + str(c['miss_high']):<12}"
              f"{c['slope_median_per_decade']:>13.3f}{c['width_median_per_decade']:>13.3f}  {c['n_failed']}",
              file=err)
    print(f"\ncoverage below {result['flag_below']} starts at phi =", file=err)
    for o in result["onset"]:
        print(f"  slope {o['slope_per_decade']:<4} {o['n_years']} yrs: {o['phi']}", file=err)
    print(f"\nwrote {args.out} in {elapsed:.0f} s with {args.workers} worker(s)", file=err)
    return 0


def run_power(args) -> int:
    t0 = time.perf_counter()
    result = _stamp(benchmark.power_experiment(args.replicates, seed=args.seed, workers=args.workers))
    elapsed = time.perf_counter() - t0
    _write(result, args.out)

    err = sys.stderr
    print(f"{'phi':>4} {'slope':>5} {'yrs':>3}  {'reject @0.05':<26}{'reject @0.01':<26}fail", file=err)
    for c in result["cells"]:
        r = c["rates"]
        print(f"{c['phi']:>4} {c['slope_per_decade']:>5} {c['n_years']:>3}  "
              f"{_ci(r['0.05']['rejection']):<26}{_ci(r['0.01']['rejection']):<26}{c['n_failed']}", file=err)
    print(f"\nwrote {args.out} in {elapsed:.0f} s with {args.workers} worker(s)", file=err)
    return 0


def run_figures(args) -> int:
    from trendcourt import benchmark_report, thresholds

    results = benchmark_report.load_results()
    for path in benchmark_report.write_figures(results):
        print(f"wrote {path.relative_to(REPO)}", file=sys.stderr)
    doc = benchmark_report.update_doc(results, thresholds.load())
    print(f"updated tables in {doc.relative_to(REPO)}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--experiment", choices=list(DEFAULTS))
    mode.add_argument("--figures", action="store_true",
                      help="figures and tables from the committed JSON (no simulation)")
    ap.add_argument("--replicates", type=int, default=None, help="default 1000")
    ap.add_argument("--seed", type=int, default=None, help="default: the experiment's master seed")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", type=Path, default=None, help="default docs/benchmark/<experiment>.json")
    args = ap.parse_args(argv)
    if args.figures:
        if any(v is not None for v in (args.replicates, args.seed, args.out)):
            ap.error("--figures takes no --replicates, --seed or --out")
        return run_figures(args)
    replicates, seed = DEFAULTS[args.experiment]
    if args.replicates is None:
        args.replicates = replicates
    if args.seed is None:
        args.seed = seed
    if args.out is None:
        args.out = OUT_DIR / f"{args.experiment}.json"
    if args.replicates < 1 or args.workers < 1:
        ap.error("--replicates and --workers must be at least 1")
    return {"fpr": run_fpr, "coverage": run_coverage, "power": run_power}[args.experiment](args)


if __name__ == "__main__":
    sys.exit(main())
