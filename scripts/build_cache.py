"""Build the site's case cache: ``site/public/cases/*.json`` and ``index.json``.

Runs ``trendcourt.examine()`` for every region in ``data/regions.json`` and
every variable, and writes the results with an index for the case picker
(see :mod:`trendcourt.cache`). Commit the build inputs first, then run this
and commit ``site/public/cases``. On a clean checkout a rebuild changes
nothing::

    python scripts/build_cache.py && git diff --exit-code site/public/cases

``--allow-dirty`` builds with uncommitted input changes and stamps the
commit ``<hash>+dirty``. Use it to preview locally, and do not commit the
result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trendcourt import cache  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="build even if the inputs have uncommitted changes (stamps <hash>+dirty)",
    )
    parser.add_argument(
        "--out", type=Path, default=cache.OUT_DIR, help="output directory (default: %(default)s)"
    )
    args = parser.parse_args(argv)
    try:
        commit = cache.source_commit(allow_dirty=args.allow_dirty)
    except (cache.DirtyInputsError, RuntimeError) as err:
        print(f"build_cache: {err}", file=sys.stderr)
        return 1
    files = cache.build(commit)
    stale = cache.write(files, args.out)
    n_cases = len(files) - 1
    print(f"wrote {n_cases} cases and index.json to {args.out} (inputs at {commit})")
    for name in stale:
        print(f"removed stale {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
