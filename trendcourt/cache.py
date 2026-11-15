"""The cache build: every preset case written out as JSON for the static site.

The site has no backend (``project.md`` §4). It reads
``site/public/cases/<region>_<variable>.json``, one :func:`trendcourt.case.examine`
result per region in ``data/regions.json`` and per variable, and
``site/public/cases/index.json``, which lists everything the case picker
needs. The layout of ``index.json`` is the ``IndexFile`` type in
``site/src/data.ts``.

The output is a pure function of the committed inputs: no wall-clock time,
sorted keys, fixed order. ``index.json`` records where the build came from:

* ``build.commit``: the last commit that changed any of :data:`INPUTS`, not
  ``HEAD``. Committing the generated files then does not change the stamp,
  so ``python scripts/build_cache.py && git diff --exit-code
  site/public/cases`` passes on a clean checkout. This means input changes
  and the regenerated cache go in separate commits: commit the inputs,
  rebuild, then commit ``site/public/cases``.
* ``build.accessed``: the NASA data access dates (``YYYY-MM-DD``) that the
  download manifests record, as carried in each case's provenance.

A case that fails the quality gate is still written (its statistics are
``None``), so the site can show "insufficient record" instead of a missing
case.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from trendcourt import case as case_module
from trendcourt import clerk

#: Version of the ``index.json`` layout. Must equal ``INDEX_SCHEMA_VERSION``
#: in ``site/src/data.ts``.
INDEX_SCHEMA_VERSION = 1

OUT_DIR = clerk.REPO_ROOT / "site" / "public" / "cases"

#: Paths (relative to the repository root) whose last commit stamps the
#: build. Everything the case JSON is computed from.
INPUTS = (
    "trendcourt",
    "data/regions.json",
    "data/raw",
    "scripts/build_cache.py",
    "requirements.txt",
)

_SAFE_ID = re.compile(r"^[a-z0-9_]+$")


class DirtyInputsError(RuntimeError):
    """The build inputs have uncommitted changes, so no commit describes them."""


def _git(*args: str, repo: Path) -> str:
    out = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def source_commit(repo: Path = clerk.REPO_ROOT, allow_dirty: bool = False) -> str:
    """The last commit that changed any of :data:`INPUTS`.

    Raises :class:`DirtyInputsError` when an input has uncommitted changes
    (tracked or untracked), unless ``allow_dirty``, in which case the hash
    gets a ``+dirty`` suffix so the stamp cannot pass for a clean build.
    Raises ``RuntimeError`` in a shallow clone, where ``git log`` cannot see
    the commit that last changed the inputs.
    """
    if _git("rev-parse", "--is-shallow-repository", repo=repo) == "true":
        raise RuntimeError(
            "shallow clone: the commit that last changed the build inputs may be "
            "missing; fetch full history (git fetch --unshallow)"
        )
    commit = _git("log", "-1", "--format=%H", "--", *INPUTS, repo=repo)
    if not commit:
        raise RuntimeError(f"no commit touches the build inputs {INPUTS}")
    dirty = _git("status", "--porcelain", "--", *INPUTS, repo=repo)
    if dirty:
        if not allow_dirty:
            raise DirtyInputsError(
                "build inputs have uncommitted changes; commit them first, or pass "
                "--allow-dirty for a local build that must not be committed:\n" + dirty
            )
        commit += "+dirty"
    return commit


def case_file(region: str, variable: str) -> str:
    return f"{region}_{variable}.json"


def build(commit: str) -> dict[str, str]:
    """Every output file of the cache build, as ``{file name: text}``.

    Runs :func:`trendcourt.case.examine` for each region and variable in
    ``data/regions.json`` (in file order) and returns the canonical case
    JSON plus ``index.json``. Writes nothing.
    """
    regions = clerk.load_regions()
    files: dict[str, str] = {}
    entries = []
    accessed: set[str] = set()
    for region in regions["regions"]:
        for variable in regions["variables"]:
            if not (_SAFE_ID.match(region) and _SAFE_ID.match(variable)):
                raise ValueError(f"unsafe id for a file name: {region!r}, {variable!r}")
            case = case_module.examine(region, variable)
            if case["region"] != region or case["variable"] != variable:
                raise ValueError(
                    f"examine({region!r}, {variable!r}) returned "
                    f"{case['region']!r}/{case['variable']!r}"
                )
            name = case_file(region, variable)
            files[name] = case_module.to_json(case)
            accessed.add(case["provenance"]["accessed_utc"][:10])
            verdict = case.get("verdict")  # added by the verdict module (#20)
            entries.append(
                {
                    "region": region,
                    "variable": variable,
                    "file": name,
                    "verdict": verdict["label"] if verdict else None,
                    "quality": {
                        "passed": case["quality"]["passed"],
                        "verdict": case["quality"]["verdict"],
                    },
                }
            )

    index = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "build": {"commit": commit, "accessed": sorted(accessed)},
        "regions": [
            {"id": rid, "name": r["name"], "biome": r["biome"], "box": r["box"]}
            for rid, r in regions["regions"].items()
        ],
        "variables": [
            {"id": vid, "name": v["name"], "units": v["units"]}
            for vid, v in regions["variables"].items()
        ],
        "cases": entries,
    }
    files["index.json"] = json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return files


def _owned(name: str, variables) -> bool:
    """Whether ``name`` is a file the build writes: ``index.json`` or a case
    file ``<region>_<variable>.json`` for one of ``variables``."""
    if name == "index.json":
        return True
    pattern = "|".join(re.escape(v) for v in variables)
    return re.fullmatch(rf"[a-z0-9_]+_({pattern})\.json", name) is not None


def write(files: dict[str, str], out_dir: Path = OUT_DIR, variables=None) -> list[str]:
    """Write ``files`` into ``out_dir`` and delete any other case file there,
    so a case dropped from ``data/regions.json`` does not linger. Only files
    the build owns are deleted: ``<region>_<variable>.json`` for a variable in
    ``variables`` (default: those in ``data/regions.json``). Any other file,
    ``*.json`` or not, is left alone, so ``--out`` cannot destroy unrelated
    files. Returns the names of the deleted files. Text is written as UTF-8
    with ``\\n`` line endings on every platform."""
    if variables is None:
        variables = list(clerk.load_regions()["variables"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(
        p.name
        for p in out_dir.glob("*.json")
        if p.name not in files and _owned(p.name, variables)
    )
    for name in stale:
        (out_dir / name).unlink()
    for name, text in files.items():
        (out_dir / name).write_bytes(text.encode("utf-8"))
    return stale
