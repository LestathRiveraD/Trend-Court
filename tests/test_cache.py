"""The cache build (``trendcourt/cache.py``, ``scripts/build_cache.py``)."""

from __future__ import annotations

import json
import re
import subprocess

import pytest

from trendcourt import cache, clerk, thresholds
from trendcourt import case as case_module
from trendcourt import verdict as verdict_module

REPO = clerk.REPO_ROOT
DATA_TS = REPO / "site" / "src" / "data.ts"
REGIONS = clerk.load_regions()
EXPECTED = {
    cache.case_file(r, v) for r in REGIONS["regions"] for v in REGIONS["variables"]
}


@pytest.fixture(scope="module")
def built() -> dict[str, str]:
    return cache.build(commit="test")


def _committed() -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(cache.OUT_DIR.glob("*.json"))}


def test_build_twice_is_byte_identical(built, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    cache.write(built, a)
    cache.write(cache.build(commit="test"), b)
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    for name in names:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_build_covers_every_preset(built):
    assert len(EXPECTED) == 18
    assert set(built) == EXPECTED | {"index.json"}


def test_index_contract(built):
    index = json.loads(built["index.json"])
    assert index["schema_version"] == cache.INDEX_SCHEMA_VERSION
    assert index["build"] == {"commit": "test", "accessed": ["2026-11-14"]}
    assert [r["id"] for r in index["regions"]] == list(REGIONS["regions"])
    for r in index["regions"]:
        assert set(r) == {"id", "name", "biome", "box"}
        assert r["box"] == REGIONS["regions"][r["id"]]["box"]
    assert index["variables"] == [
        {"id": "temperature", "name": "Temperature", "units": "degC"},
        {"id": "precipitation", "name": "Precipitation", "units": "mm/day"},
        {"id": "soil_moisture", "name": "Soil moisture", "units": "fraction 0-1"},
    ]
    assert [c["file"] for c in index["cases"]] == [
        cache.case_file(r, v) for r in REGIONS["regions"] for v in REGIONS["variables"]
    ]
    for entry in index["cases"]:
        case = json.loads(built[entry["file"]])
        assert (case["region"], case["variable"]) == (entry["region"], entry["variable"])
        assert entry["quality"] == {
            "passed": case["quality"]["passed"],
            "verdict": case["quality"]["verdict"],
        }
        # #20: every case carries a verdict, and the index lists its label.
        assert entry["verdict"] == case["verdict"]["label"]
        assert entry["verdict"] in verdict_module.LABELS


def test_failed_quality_case_is_still_written(monkeypatch):
    real = case_module.examine

    def failing(region, variable):
        out = real(region, variable)
        if (region, variable) == ("sahel", "precipitation"):
            out["quality"] = {**out["quality"], "passed": False, "verdict": "insufficient record"}
            for k in ("theil_sen", "ols", "mk_hamed_rao", "mk_seasonal", "consistency",
                      "autocorrelation"):
                out[k] = None
            out["verdict"] = verdict_module.decide(out, thresholds.load())
        return out

    monkeypatch.setattr(case_module, "examine", failing)
    files = cache.build(commit="test")
    assert "sahel_precipitation.json" in files
    entry = next(c for c in json.loads(files["index.json"])["cases"]
                 if c["file"] == "sahel_precipitation.json")
    assert entry["quality"] == {"passed": False, "verdict": "insufficient record"}
    assert entry["verdict"] == "insufficient_record"
    assert json.loads(files["sahel_precipitation.json"])["theil_sen"] is None


def test_write_removes_stale_cases_only(tmp_path):
    for name in ("dropped_temperature.json", "dropped_soil_moisture.json",
                 "regions.json", "package.json", "old_case.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "keep.txt").write_text("x")
    files = {"index.json": "{}\n", "mexicali_temperature.json": "{}\n"}
    stale = cache.write(files, tmp_path)
    assert stale == ["dropped_soil_moisture.json", "dropped_temperature.json"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "index.json", "keep.txt", "mexicali_temperature.json", "old_case.json",
        "package.json", "regions.json",
    ]


def test_script_out_dir_keeps_unrelated_json(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import build_cache

    monkeypatch.setattr(cache, "source_commit", lambda allow_dirty=False: "test")
    monkeypatch.setattr(cache, "build", lambda commit: {"index.json": "{}\n"})
    (tmp_path / "regions.json").write_text("{}")
    (tmp_path / "gone_precipitation.json").write_text("{}")
    assert build_cache.main(["--out", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.json", "regions.json"]


def test_schema_versions_match_the_site():
    ts = DATA_TS.read_text()

    def const(name: str) -> int:
        m = re.search(rf"export const {name} = (\d+);", ts)
        assert m, f"{name} not found in {DATA_TS}"
        return int(m.group(1))

    assert const("CASE_SCHEMA_VERSION") == case_module.SCHEMA_VERSION
    assert const("INDEX_SCHEMA_VERSION") == cache.INDEX_SCHEMA_VERSION


def test_committed_files_parse_with_current_schema():
    committed = _committed()
    assert set(committed) == EXPECTED | {"index.json"}
    for name, raw in committed.items():
        data = json.loads(raw)  # strict JSON: NaN would already fail to_json
        want = cache.INDEX_SCHEMA_VERSION if name == "index.json" else case_module.SCHEMA_VERSION
        assert data["schema_version"] == want, name
        if name != "index.json":
            assert raw.decode() == case_module.to_json(data), f"{name} is not canonical"


def _git_skip_reason(allow_dirty: bool = False) -> str | None:
    """Why :func:`cache.source_commit` cannot run here, or ``None``. Called at
    import time, so it must not raise: a shallow clone (the default for
    ``actions/checkout``) or a tree without git would otherwise abort
    collection of every test in this module."""
    try:
        cache.source_commit(allow_dirty=allow_dirty)
    except cache.DirtyInputsError:
        return "build inputs have uncommitted changes"
    except (RuntimeError, OSError, subprocess.CalledProcessError) as err:
        return f"no usable git history: {str(err).splitlines()[0]}"
    return None


_SKIP_CLEAN = _git_skip_reason()
_SKIP_ANY = _git_skip_reason(allow_dirty=True)


@pytest.mark.skipif(_SKIP_CLEAN is not None, reason=_SKIP_CLEAN or "")
def test_committed_cache_is_current(built):
    """The committed cache is what a rebuild at this commit writes. If this
    fails, run ``python scripts/build_cache.py`` and commit site/public/cases."""
    fresh = dict(built)
    index = json.loads(fresh["index.json"])
    index["build"]["commit"] = cache.source_commit()
    fresh["index.json"] = json.dumps(index, indent=2, sort_keys=True) + "\n"
    committed = _committed()
    assert sorted(committed) == sorted(fresh)
    stale = [n for n in fresh if committed[n] != fresh[n].encode()]
    assert not stale, f"stale cache files: {stale}; rerun scripts/build_cache.py"


@pytest.mark.skipif(_SKIP_ANY is not None, reason=_SKIP_ANY or "")
def test_source_commit_is_last_input_commit():
    commit = cache.source_commit(allow_dirty=True)
    expected = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *cache.INPUTS],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert commit.removesuffix("+dirty") == expected


def test_source_commit_refuses_dirty_inputs(tmp_path):
    repo = tmp_path / "repo"
    (repo / "trendcourt").mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "root")
    (repo / "trendcourt" / "x.py").write_text("1\n")
    git("add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "inputs")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True).stdout.strip()
    (repo / "site").mkdir()
    (repo / "site" / "out.json").write_text("{}")
    git("add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "cache")
    # Committing outputs does not move the stamp.
    assert cache.source_commit(repo) == head
    (repo / "trendcourt" / "x.py").write_text("2\n")
    with pytest.raises(cache.DirtyInputsError):
        cache.source_commit(repo)
    assert cache.source_commit(repo, allow_dirty=True) == head + "+dirty"


def test_script_refuses_dirty_inputs_without_flag(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import build_cache

    def dirty(allow_dirty=False):
        raise cache.DirtyInputsError("dirty")

    monkeypatch.setattr(cache, "source_commit", dirty)
    assert build_cache.main([]) == 1
    assert "dirty" in capsys.readouterr().err
